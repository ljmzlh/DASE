/*
 * dase_batch_hnsw: batched HNSW range probes for SemJI (TI) construction.
 *
 * SemJI construction runs one HNSW range query per left row against the right
 * table's index. Each probe descends from the global entry point through the
 * upper layers, then expands layer 0 under the radius gate. This module runs
 * the whole left side through one C function so the probes can share work:
 *
 *   probe       per-row descent + layer-0 range search, the same calls as the
 *               fork's GetRangeScanItems (the existing construction path)
 *   probe_copy  as probe, but layer 0 runs this file's copy of the range search
 *               (sanity check that the copy used by warm costs the same)
 *   share       one descent per group of left rows; every row in the group
 *               starts layer 0 from the group leader's layer-0 entry point
 *   warm        no descent inside a group; each row starts layer 0 from the
 *               nearest nodes the previous row's search ended with
 *
 * Groups and row order come from the caller's left query. It is a separate
 * module, not a change to the vector extension: hnswutils.c is compiled in
 * with hidden visibility, so nothing here can interpose on vector.so.
 */

/* Keep the few exported names in hnswutils.c from colliding with vector.so's */
#define l2_normalize				bh_unused_l2_normalize
#define halfvec_l2_normalize		bh_unused_halfvec_l2_normalize
#define sparsevec_l2_normalize		bh_unused_sparsevec_l2_normalize
#define hnsw_halfvec_support		bh_unused_hnsw_halfvec_support
#define hnsw_bit_support			bh_unused_hnsw_bit_support
#define hnsw_sparsevec_support		bh_unused_hnsw_sparsevec_support
#define hnsw_set_tid_map_table_sql	bh_unused_hnsw_set_tid_map_table_sql

#include "hnswutils.c"

#include "access/relation.h"
#include "catalog/pg_type.h"
#include "commands/defrem.h"
#include "executor/spi.h"
#include "funcapi.h"
#include "portability/instr_time.h"
#include "storage/lmgr.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"

PG_MODULE_MAGIC;

/* Globals hnswutils.c reads; the GUCs themselves live in vector.so */
bool		hnsw_enable_2hop = false;
char	   *hnsw_id_map_table = NULL;

/* Only the vector type with an un-normalized opclass (l2/ip) is supported */
#define BH_UNUSED_STUB(name) \
	PGDLLEXPORT Datum name(PG_FUNCTION_ARGS); \
	Datum name(PG_FUNCTION_ARGS) { elog(ERROR, "dase_batch_hnsw: %s is not supported", #name); PG_RETURN_NULL(); }
BH_UNUSED_STUB(bh_unused_l2_normalize)
BH_UNUSED_STUB(bh_unused_halfvec_l2_normalize)
BH_UNUSED_STUB(bh_unused_sparsevec_l2_normalize)

typedef enum BhMode
{
	BH_PROBE,
	BH_PROBE_COPY,
	BH_SHARE,
	BH_WARM
}			BhMode;

typedef struct BhStats
{
	int64		n_queries;
	int64		n_pairs;
	int64		n_descents;
	int64		nd_descent;		/* elements loaded (~distances) in descent */
	int64		nd_seed;		/* seed entry candidates */
	int64		nd_l0;			/* elements loaded in layer-0 range search */
	double		t_total;
	double		t_fetch;		/* reading left rows through SPI */
	double		t_descent;
	double		t_seed;
	double		t_l0;
	double		t_emit;
}			BhStats;

static BhStats bh_last;
static bool bh_have_last = false;

/* A layer-0 node to start the next row from */
typedef struct BhSeed
{
	BlockNumber blkno;
	OffsetNumber offno;
}			BhSeed;

static inline double
bh_elapsed(instr_time start)
{
	instr_time	now;

	INSTR_TIME_SET_CURRENT(now);
	INSTR_TIME_SUBTRACT(now, start);
	return INSTR_TIME_GET_DOUBLE(now);
}

/*
 * Descent as in GetRangeScanItems: global entry point, greedy ef=1 down to
 * layer 1. Returns the layer-0 entry list.
 */
static List *
bh_descend(Relation index, HnswSupport * support, HnswQuery * q, int *m, int64 *nd)
{
	HnswElement entryPoint;
	List	   *ep;

	HnswGetMetaPageInfo(index, m, &entryPoint);
	if (entryPoint == NULL)
		return NIL;

	ep = list_make1(HnswEntryCandidate(NULL, entryPoint, q, index, support, false));
	(*nd)++;
	for (int lc = entryPoint->level; lc >= 1; lc--)
		ep = HnswSearchLayer(NULL, q, ep, 1, lc, index, support, *m, false, NULL, NULL, NULL, true, nd);
	return ep;
}

/* Entry candidates for q from stored layer-0 nodes */
static List *
bh_seed_candidates(Relation index, HnswSupport * support, HnswQuery * q, BhSeed * seeds, int nseeds, int64 *nd)
{
	List	   *ep = NIL;

	for (int i = 0; i < nseeds; i++)
	{
		HnswElement e = HnswInitElementFromBlock(seeds[i].blkno, seeds[i].offno);

		ep = lappend(ep, HnswEntryCandidate(NULL, e, q, index, support, false));
	}
	*nd += nseeds;
	return ep;
}

/*
 * Copy of the fork's HnswRangeSearchLayer (on-disk, no skip element, no
 * discarded heap), which additionally hands back the nearest nodes left in W
 * so the next row can start from them.
 */
static List *
bh_range_search_layer0(HnswQuery * q, List *ep, int ef, Relation index, HnswSupport * support, int m,
					   int64 *tuples, double radius, BhSeed * seedsOut, int maxSeeds, int *nSeedsOut)
{
	List	   *R = NIL;
	pairingheap *C = pairingheap_allocate(CompareNearestCandidates, NULL);
	pairingheap *W = pairingheap_allocate(CompareFurthestCandidates, NULL);
	int			wlen = 0;
	visited_hash v;
	ListCell   *lc2;
	int			lm = HnswGetLayerM(m, 0);
	HnswUnvisited *unvisited = palloc(lm * sizeof(HnswUnvisited));
	int			unvisitedLength;
	const double radius_thre = radius * radius;
	const double radius_cap = radius_thre * 1.1;

	InitVisited(NULL, &v, false, ef, m);

	foreach(lc2, ep)
	{
		HnswSearchCandidate *sc = (HnswSearchCandidate *) lfirst(lc2);
		bool		found;

		AddToVisited(NULL, &v, sc->element, false, &found);
		if (found)
			continue;			/* duplicate seed */
		(*tuples)++;

		pairingheap_add(C, &sc->c_node);
		pairingheap_add(W, &sc->w_node);
		wlen++;

		if (sc->distance <= radius_thre)
			R = lappend(R, sc);
	}

	while (!pairingheap_is_empty(C))
	{
		HnswSearchCandidate *c = HnswGetSearchCandidate(c_node, pairingheap_remove_first(C));
		HnswSearchCandidate *w = HnswGetSearchCandidate(w_node, pairingheap_first(W));
		HnswElement cElement;

		if (c->distance > w->distance && c->distance > radius_cap)
			break;

		cElement = HnswPtrPointer(c->element);

		HnswLoadUnvisited(false, NULL, index, cElement, unvisited, &unvisitedLength,
						  &v, 0, NULL, 0, m, lm);
		(*tuples) += unvisitedLength;

		for (int i = 0; i < unvisitedLength; i++)
		{
			HnswElement eElement = NULL;
			HnswSearchCandidate *e;
			double		eDistance;
			bool		alwaysAdd = wlen < ef;
			double		gate;
			ItemPointer indextid = &unvisited[i].indextid;

			w = HnswGetSearchCandidate(w_node, pairingheap_first(W));
			gate = Max(radius_cap, w->distance);

			HnswLoadElementImpl(ItemPointerGetBlockNumber(indextid), ItemPointerGetOffsetNumber(indextid),
								&eDistance, q, index, support, false, alwaysAdd ? NULL : &gate, &eElement);

			if (eElement == NULL || (eDistance > gate && !alwaysAdd))
				continue;

			e = HnswInitSearchCandidate(NULL, eElement, eDistance);
			pairingheap_add(C, &e->c_node);
			pairingheap_add(W, &e->w_node);

			if (eDistance <= radius_thre)
				R = lappend(R, e);

			wlen++;
			if (wlen > ef)
				pairingheap_remove_first(W);
		}
	}

	/* W pops furthest first; the last maxSeeds popped are the nearest */
	if (seedsOut != NULL)
	{
		int			n = 0;

		while (!pairingheap_is_empty(W))
		{
			HnswSearchCandidate *sc = HnswGetSearchCandidate(w_node, pairingheap_remove_first(W));
			HnswElement e = HnswPtrPointer(sc->element);

			seedsOut[n % maxSeeds].blkno = e->blkno;
			seedsOut[n % maxSeeds].offno = e->offno;
			n++;
		}
		*nSeedsOut = Min(n, maxSeeds);
	}

	return R;
}

static int
bh_cmp_int64(const void *a, const void *b)
{
	int64		x = *(const int64 *) a;
	int64		y = *(const int64 *) b;

	return (x > y) - (x < y);
}

static BhMode
bh_parse_mode(const char *s)
{
	if (strcmp(s, "probe") == 0)
		return BH_PROBE;
	if (strcmp(s, "probe_copy") == 0)
		return BH_PROBE_COPY;
	if (strcmp(s, "share") == 0)
		return BH_SHARE;
	if (strcmp(s, "warm") == 0)
		return BH_WARM;
	ereport(ERROR, (errmsg("dase_batch_hnsw: unknown mode \"%s\"", s),
					errhint("Use probe, probe_copy, share or warm.")));
	return BH_PROBE;			/* keep compiler quiet */
}

/*
 * Probe every row of left_sql against the index and put (left_id, right_tid,
 * dist) rows into the materialized result: every pair when sampleIds is NULL,
 * otherwise only the pairs of left rows in sampleIds (sorted). Counts and
 * timings go to bh_last_stats().
 */
static void
bh_run(FunctionCallInfo fcinfo, const char *leftSql, Oid indexOid, double radius, int ef,
	   BhMode mode, int maxSeeds, const int64 *sampleIds, int nSample)
{
	ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
	Relation	index;
	HnswSupport support;
	MemoryContext qctx;
	MemoryContext oldctx;
	SPIPlanPtr	plan;
	Portal		portal;
	BhStats		st = {0};
	instr_time	tTotal;
	BhSeed	   *seeds = palloc(maxSeeds * sizeof(BhSeed));
	int			nSeeds = 0;
	BhSeed		shared = {0};
	bool		haveShared = false;
	int			prevGrp = 0;
	bool		first = true;
	int			m = 0;

	INSTR_TIME_SET_CURRENT(tTotal);
	InitMaterializedSRF(fcinfo, 0);

	index = index_open(indexOid, AccessShareLock);
	if (strcmp(get_am_name(index->rd_rel->relam), "hnsw") != 0)
		ereport(ERROR, (errmsg("dase_batch_hnsw: \"%s\" is not an hnsw index",
							   RelationGetRelationName(index))));
	HnswInitSupport(&support, index);
	if (support.normprocinfo != NULL)
		ereport(ERROR, (errmsg("dase_batch_hnsw: normalized opclasses (cosine) are not supported")));

	qctx = AllocSetContextCreate(CurrentMemoryContext, "bh per-row", ALLOCSET_DEFAULT_SIZES);

	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "dase_batch_hnsw: SPI_connect failed");
	plan = SPI_prepare(leftSql, 0, NULL);
	if (plan == NULL)
		elog(ERROR, "dase_batch_hnsw: cannot prepare left query: %s", SPI_result_code_string(SPI_result));
	portal = SPI_cursor_open(NULL, plan, NULL, NULL, true);

	for (;;)
	{
		instr_time	tFetch;
		uint64		nrows;

		INSTR_TIME_SET_CURRENT(tFetch);
		SPI_cursor_fetch(portal, true, 512);
		nrows = SPI_processed;
		st.t_fetch += bh_elapsed(tFetch);
		if (nrows == 0)
			break;

		for (uint64 r = 0; r < nrows; r++)
		{
			HeapTuple	tup = SPI_tuptable->vals[r];
			TupleDesc	desc = SPI_tuptable->tupdesc;
			bool		isnull;
			int64		leftId;
			int			grp;
			Datum		vec;
			HnswQuery	q;
			List	   *ep = NIL;
			List	   *R;
			ListCell   *lc;
			bool		newGroup;
			bool		emit;
			instr_time	t0;

			CHECK_FOR_INTERRUPTS();

			leftId = DatumGetInt64(SPI_getbinval(tup, desc, 1, &isnull));
			vec = SPI_getbinval(tup, desc, 2, &isnull);
			if (isnull)
				continue;
			grp = DatumGetInt32(SPI_getbinval(tup, desc, 3, &isnull));
			newGroup = first || grp != prevGrp;
			first = false;
			prevGrp = grp;

			MemoryContextReset(qctx);
			oldctx = MemoryContextSwitchTo(qctx);

			INSTR_TIME_SET_CURRENT(t0);
			q.value = PointerGetDatum(PG_DETOAST_DATUM(vec));
			st.t_fetch += bh_elapsed(t0);

			LockPage(index, HNSW_SCAN_LOCK, ShareLock);

			/* Where layer 0 starts */
			INSTR_TIME_SET_CURRENT(t0);
			if (mode == BH_PROBE || mode == BH_PROBE_COPY ||
				(mode == BH_SHARE && (newGroup || !haveShared)) ||
				(mode == BH_WARM && (newGroup || nSeeds == 0)))
			{
				ep = bh_descend(index, &support, &q, &m, &st.nd_descent);
				st.n_descents++;
				st.t_descent += bh_elapsed(t0);
				if (mode == BH_SHARE && ep != NIL)
				{
					HnswElement e = HnswPtrPointer(((HnswSearchCandidate *) linitial(ep))->element);

					shared.blkno = e->blkno;
					shared.offno = e->offno;
					haveShared = true;
				}
			}
			else if (mode == BH_SHARE)
			{
				ep = bh_seed_candidates(index, &support, &q, &shared, 1, &st.nd_seed);
				st.t_seed += bh_elapsed(t0);
			}
			else
			{
				ep = bh_seed_candidates(index, &support, &q, seeds, nSeeds, &st.nd_seed);
				st.t_seed += bh_elapsed(t0);
			}

			/* Layer-0 range search */
			INSTR_TIME_SET_CURRENT(t0);
			if (ep == NIL)
				R = NIL;
			else if (mode == BH_PROBE || mode == BH_SHARE)
				R = HnswRangeSearchLayer(NULL, &q, ep, ef, 0, index, &support, m, false, NULL,
										 NULL, NULL, true, &st.nd_l0, radius);
			else
				R = bh_range_search_layer0(&q, ep, ef, index, &support, m, &st.nd_l0, radius,
										   mode == BH_WARM ? seeds : NULL, maxSeeds, &nSeeds);
			st.t_l0 += bh_elapsed(t0);

			UnlockPage(index, HNSW_SCAN_LOCK, ShareLock);

			/* Count pairs; return all of them, or those of sample rows */
			INSTR_TIME_SET_CURRENT(t0);
			emit = sampleIds == NULL ||
				(nSample > 0 &&
				 bsearch(&leftId, sampleIds, nSample, sizeof(int64), bh_cmp_int64) != NULL);
			foreach(lc, R)
			{
				HnswSearchCandidate *sc = (HnswSearchCandidate *) lfirst(lc);
				HnswElement e = HnswPtrPointer(sc->element);

				st.n_pairs += e->heaptidsLength;
				if (emit)
				{
					for (int i = 0; i < e->heaptidsLength; i++)
					{
						Datum		values[3];
						bool		nulls[3] = {false, false, false};

						values[0] = Int64GetDatum(leftId);
						values[1] = PointerGetDatum(&e->heaptids[i]);
						values[2] = Float8GetDatum(sqrt(sc->distance));
						tuplestore_putvalues(rsinfo->setResult, rsinfo->setDesc, values, nulls);
					}
				}
			}
			st.t_emit += bh_elapsed(t0);
			st.n_queries++;

			MemoryContextSwitchTo(oldctx);
		}
		SPI_freetuptable(SPI_tuptable);
	}

	SPI_cursor_close(portal);
	SPI_finish();
	MemoryContextDelete(qctx);
	index_close(index, AccessShareLock);

	st.t_total = bh_elapsed(tTotal);
	bh_last = st;
	bh_have_last = true;
}

/*
 * bh_range_join(left_sql text, right_index regclass, radius float8, ef int4,
 *               mode text, n_seeds int4, sample_ids int8[])
 *   RETURNS TABLE (left_id int8, right_tid tid, dist float8)
 *
 * left_sql returns (id int8, vec vector, grp int4) in processing order. Every
 * row is probed; only rows whose id is in sample_ids have their pairs
 * returned (benchmarking: recall without the cost of returning every pair).
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(bh_range_join);
Datum
bh_range_join(PG_FUNCTION_ARGS)
{
	ArrayType  *sampleArr = PG_GETARG_ARRAYTYPE_P(6);
	Datum	   *sampleDatums;
	int			nSample;
	int64	   *sampleIds;

	deconstruct_array(sampleArr, INT8OID, sizeof(int64), true, TYPALIGN_DOUBLE,
					  &sampleDatums, NULL, &nSample);
	sampleIds = palloc(Max(nSample, 1) * sizeof(int64));
	for (int i = 0; i < nSample; i++)
		sampleIds[i] = DatumGetInt64(sampleDatums[i]);
	qsort(sampleIds, nSample, sizeof(int64), bh_cmp_int64);

	bh_run(fcinfo, text_to_cstring(PG_GETARG_TEXT_PP(0)), PG_GETARG_OID(1), PG_GETARG_FLOAT8(2),
		   PG_GETARG_INT32(3), bh_parse_mode(text_to_cstring(PG_GETARG_TEXT_PP(4))),
		   Max(PG_GETARG_INT32(5), 1), sampleIds, nSample);
	return (Datum) 0;
}

/*
 * bh_range_join_all(left_sql text, right_index regclass, radius float8, ef int4,
 *                   mode text, n_seeds int4)
 *   RETURNS TABLE (left_id int8, right_tid tid, dist float8)
 *
 * As bh_range_join, returning every pair (TI construction).
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(bh_range_join_all);
Datum
bh_range_join_all(PG_FUNCTION_ARGS)
{
	bh_run(fcinfo, text_to_cstring(PG_GETARG_TEXT_PP(0)), PG_GETARG_OID(1), PG_GETARG_FLOAT8(2),
		   PG_GETARG_INT32(3), bh_parse_mode(text_to_cstring(PG_GETARG_TEXT_PP(4))),
		   Max(PG_GETARG_INT32(5), 1), NULL, 0);
	return (Datum) 0;
}

/* JSON with the counts and timings of this backend's last bh_range_join call */
PGDLLEXPORT PG_FUNCTION_INFO_V1(bh_last_stats);
Datum
bh_last_stats(PG_FUNCTION_ARGS)
{
	BhStats    *s = &bh_last;

	if (!bh_have_last)
		PG_RETURN_NULL();

	PG_RETURN_TEXT_P(cstring_to_text(psprintf(
											  "{\"n_queries\": " INT64_FORMAT ", \"n_pairs\": " INT64_FORMAT
											  ", \"n_descents\": " INT64_FORMAT
											  ", \"nd_descent\": " INT64_FORMAT ", \"nd_seed\": " INT64_FORMAT
											  ", \"nd_l0\": " INT64_FORMAT
											  ", \"t_total\": %.6f, \"t_fetch\": %.6f, \"t_descent\": %.6f"
											  ", \"t_seed\": %.6f, \"t_l0\": %.6f, \"t_emit\": %.6f}",
											  s->n_queries, s->n_pairs, s->n_descents,
											  s->nd_descent, s->nd_seed, s->nd_l0,
											  s->t_total, s->t_fetch, s->t_descent,
											  s->t_seed, s->t_l0, s->t_emit)));
}
