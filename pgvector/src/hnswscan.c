#include "postgres.h"

#include "access/relscan.h"
#include "hnsw.h"
#include "pgstat.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "utils/bytea.h"
#include "utils/float.h"
#include "utils/memutils.h"

typedef struct __attribute__((packed)) HnswBloomFilterHeader
{
	char		magic[4];
	uint32		nHashes;
	uint32		modNumber;
	uint64		seed1;
	uint64		seed2;
}			HnswBloomFilterHeader;

#define HNSW_FILTER_BLOOM_MAGIC "HBLM"

StaticAssertDecl(sizeof(HnswBloomFilterHeader) == 28,
				 "unexpected bloom filter header size");

static void HnswInitEmptyFilterSpec(HnswFilterSpec *spec);
static void HnswInitFilterSpecFromBytes(HnswScanOpaque so, Size len);

/*
 * Algorithm 5 from paper
 */

static List *
GetScanItems(IndexScanDesc scan, Datum value)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	Relation	index = scan->indexRelation;
	HnswSupport *support = &so->support;
	List	   *ep;
	List	   *w;
	int			m;
	HnswElement entryPoint;
	char	   *base = NULL;
	HnswQuery  *q = &so->q;
	ListCell   *lc2;
	int			cnt, len;


	/* Get m and entry point */
	HnswGetMetaPageInfo(index, &m, &entryPoint);

	q->value = value;
	so->m = m;

	if (entryPoint == NULL)
		return NIL;

	ep = list_make1(HnswEntryCandidate(base, entryPoint, q, index, support, false));


	for (int lc = entryPoint->level; lc >= 1; lc--)
	{
		elog(LOG, "ep to level %d: %d", lc, list_length(ep));
		foreach(lc2, ep)
		{
			HnswSearchCandidate *sc = (HnswSearchCandidate *) lfirst(lc2);
			elog(LOG, "element: %p, distance: %f", sc->element, sc->distance);
		}
		w = HnswSearchLayer(base, q, ep, 1, lc, index, support, m, false, NULL, NULL, NULL, true, NULL);
		ep = w;
	}

	elog(LOG, "ep to base level: %d", list_length(ep));
	foreach(lc2, ep)
	{
		HnswSearchCandidate *sc = (HnswSearchCandidate *) lfirst(lc2);
		elog(LOG, "element: %p, distance: %f", sc->element, sc->distance);
	}

	w = HnswSearchLayer(base, q, ep, hnsw_ef_search, 0, index, support, m, false, NULL, &so->v, hnsw_iterative_scan != HNSW_ITERATIVE_SCAN_OFF ? &so->discarded : NULL, true, &so->tuples);
	
	// output last 5
	elog(LOG, "Got %d results. Top 5 are:", list_length(w));
	cnt = 0;
	len = list_length(w);
	foreach(lc2, w)
	{
		if (cnt >= len - 5)
		{
			HnswSearchCandidate *sc = (HnswSearchCandidate *) lfirst(lc2);
			HnswElement element = HnswPtrAccess(base, sc->element);
			if (element != NULL && element->heaptidsLength > 0)
			{
				ItemPointerData *tid = &element->heaptids[element->heaptidsLength - 1];
				elog(LOG, "heap blkno: %u, offno: %u, distance: %f",
					 ItemPointerGetBlockNumber(tid),
					 ItemPointerGetOffsetNumber(tid),
					 sqrt(sc->distance));
			}
			else
			{
				elog(LOG, "heap tid: <none>, distance: %f", sqrt(sc->distance));
			}
		}
		cnt++;
	}




	elog(LOG, "hnswscan.c: GetScanItems finished");
	return w;
}



/* --- filter predicate support (strategy = HNSW_STRAT_FILTER) --- */
static bool
hnsw_parse_filter(IndexScanDesc scan, Datum *qval, Oid *subtype)
{
	elog(LOG, "hnsw_parse_filter: numberOfKeys = %d", scan->numberOfKeys);
	for (int i = 0; i < scan->numberOfKeys; i++)
	{
		ScanKey sk = &scan->keyData[i];

		if (sk->sk_strategy == HNSW_STRAT_FILTER)
		{
			*qval = sk->sk_argument;
			if (subtype != NULL)
				*subtype = sk->sk_subtype;
			return true;
		}
	}

	return false;
}


/* --- radius predicate support (strategy = HNSW_STRAT_RADIUS) --- */
static bool
hnsw_parse_radius(IndexScanDesc scan, Datum *qval, double *radius)
{
	for (int i = 0; i < scan->numberOfKeys; i++) {
        ScanKey sk = &scan->keyData[i];
		if (sk->sk_strategy == HNSW_STRAT_RADIUS) {
            *qval = sk->sk_argument;   /* 右参：查询向量 */
            *radius = hnsw_radius;     /* 阈值：GUC */
            return true;
        }
    }
    return false;
}

static List *
GetRangeScanItems(IndexScanDesc scan, Datum value, double radius)
{
    HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
    Relation index = scan->indexRelation;
    HnswSupport *support = &so->support;
    List *ep, *w;
    int m;
    HnswElement entryPoint;
    char *base = NULL;
    HnswQuery *q = &so->q;


    /* 和 GetScanItems 同样的入口逻辑 */
    HnswGetMetaPageInfo(index, &m, &entryPoint);
    q->value = value;
    so->m = m;

    if (entryPoint == NULL)
        return NIL;

    ep = list_make1(HnswEntryCandidate(base, entryPoint, q, index, support, false));

    for (int lc = entryPoint->level; lc >= 1; lc--) {
        w = HnswSearchLayer(base, q, ep, 1, lc, index, support, m,
                            false, NULL, NULL, NULL, true, NULL);
        ep = w;
    }


    return HnswRangeSearchLayer(base, q, ep, hnsw_ef_search, 0, index, support, m,
                                false, NULL, &so->v,
                                hnsw_iterative_scan != HNSW_ITERATIVE_SCAN_OFF ? &so->discarded : NULL,
                                true, &so->tuples, radius);

}

static void
HnswInitEmptyFilterSpec(HnswFilterSpec *spec)
{
	if (spec == NULL)
		return;

	spec->filterType = HNSW_FILTER_TYPE_NONE;
	MemSet(&spec->data, 0, sizeof(spec->data));
}

static void
HnswInitFilterSpecFromBytes(HnswScanOpaque so, Size len)
{
	HnswFilterSpec *spec = &so->filterSpec;
	Size		headerSize = sizeof(HnswBloomFilterHeader);
	uint8	   *raw = so->filterData;

	elog(LOG, "HnswInitFilterSpecFromBytes: len=%zu, headerSize=%zu, raw=%p",
		 len, headerSize, (void *) raw);

	if (so == NULL)
		elog(ERROR, "HnswInitFilterSpecFromBytes: so is NULL");

	if (so->filterData == NULL || len == 0)
	{
		elog(LOG, "HnswInitFilterSpecFromBytes: empty filter, filterData=%p, len=%zu",
			 (void *) so->filterData, len);
		HnswInitEmptyFilterSpec(spec);
		return;
	}

	if (len < 4)
		elog(ERROR, "HnswInitFilterSpecFromBytes: len too small: %zu", len);

	elog(LOG, "HnswInitFilterSpecFromBytes: magic bytes = %02x %02x %02x %02x",
		 raw[0], raw[1], raw[2], raw[3]);

	if(len >= headerSize && memcmp(raw, HNSW_FILTER_BLOOM_MAGIC, 4) == 0)
	{
		Size		payloadBytes = len - headerSize;
		uint32		nHashes;
		uint32		modNumber;
		uint64		seed1;
		uint64		seed2;

		elog(LOG, "HnswInitFilterSpecFromBytes: detected bloom filter, payloadBytes=%zu", payloadBytes);

		/* 使用 memcpy 安全读取 packed 结构体，避免未对齐访问 */
		memcpy(&nHashes, raw + 4, sizeof(uint32));
		memcpy(&modNumber, raw + 8, sizeof(uint32));
		memcpy(&seed1, raw + 12, sizeof(uint64));
		memcpy(&seed2, raw + 20, sizeof(uint64));

		elog(LOG, "HnswInitFilterSpecFromBytes: read values: nHashes=%u, modNumber=%u, seed1=%lu, seed2=%lu",
			 nHashes, modNumber, (unsigned long) seed1, (unsigned long) seed2);

		if (nHashes == 0)
			elog(ERROR, "invalid bloom filter: nHashes must be greater than zero");

		if (modNumber == 0)
			elog(ERROR, "invalid bloom filter: modNumber must be greater than zero");

		if (payloadBytes == 0)
			elog(ERROR, "invalid bloom filter: missing bitset payload");

		if ((uint64) modNumber > (uint64) payloadBytes * 8)
			elog(ERROR, "invalid bloom filter: modNumber(%u) exceeds bitset capacity(%zu * 8 = %zu)",
				 modNumber, payloadBytes, payloadBytes * 8);

		spec->filterType = HNSW_FILTER_TYPE_BLOOM;
		spec->data.bloom.bitset = raw + headerSize;
		spec->data.bloom.byteLength = payloadBytes;
		spec->data.bloom.modNumber = modNumber;
		spec->data.bloom.nHashes = nHashes;
		spec->data.bloom.seed1 = seed1;
		spec->data.bloom.seed2 = seed2;

		elog(LOG, "bloom filter parsed OK: nHashes=%u, modNumber=%u, byteLength=%zu",
			 nHashes, modNumber, payloadBytes);
		
		/* 打印 bitset 前 8 个字节用于调试 */
		if (payloadBytes >= 8)
			elog(LOG, "bitset first 8 bytes: %02x %02x %02x %02x %02x %02x %02x %02x",
				 spec->data.bloom.bitset[0], spec->data.bloom.bitset[1],
				 spec->data.bloom.bitset[2], spec->data.bloom.bitset[3],
				 spec->data.bloom.bitset[4], spec->data.bloom.bitset[5],
				 spec->data.bloom.bitset[6], spec->data.bloom.bitset[7]);
	}
	else
	{
		elog(LOG, "HnswInitFilterSpecFromBytes: detected bitmap filter, size=%zu", len);
		spec->filterType = HNSW_FILTER_TYPE_BITMAP;
		MemSet(&spec->data, 0, sizeof(spec->data));
		spec->data.bitmap.bitset = so->filterData;
		spec->data.bitmap.size = len;
	}
}

static void
HnswResetFilter(HnswScanOpaque so)
{
	if (so->filterData != NULL)
	{
		pfree(so->filterData);
		so->filterData = NULL;
	}
	so->filterDataSize = 0;
	HnswInitEmptyFilterSpec(&so->filterSpec);
}

static void
HnswSetFilter(HnswScanOpaque so, Datum filterDatum)
{
	bytea	   *data;
	Size		len;

	elog(LOG, "HnswSetFilter: enter, so=%p, filterDatum=%p",
		 (void *) so, DatumGetPointer(filterDatum));

	if (so == NULL)
		elog(ERROR, "HnswSetFilter: so is NULL");

	if (DatumGetPointer(filterDatum) == NULL)
	{
		elog(LOG, "HnswSetFilter: filterDatum is NULL, resetting");
		HnswResetFilter(so);
		return;
	}

	data = DatumGetByteaPP(filterDatum);
	if (data == NULL)
		elog(ERROR, "HnswSetFilter: data is NULL after DatumGetByteaPP");

	len = VARSIZE_ANY_EXHDR(data);
	elog(LOG, "HnswSetFilter: data=%p, len=%zu", (void *) data, len);

	if (len == 0)
	{
		elog(LOG, "HnswSetFilter: len is 0, resetting");
		HnswResetFilter(so);
		return;
	}

	if (so->filterData != NULL)
	{
		elog(LOG, "HnswSetFilter: freeing old filterData=%p", (void *) so->filterData);
		pfree(so->filterData);
	}

	if (so->tmpCtx == NULL)
		elog(ERROR, "HnswSetFilter: so->tmpCtx is NULL");

	elog(LOG, "HnswSetFilter: allocating %zu bytes in tmpCtx", len);
	so->filterData = (uint8 *) MemoryContextAlloc(so->tmpCtx, len);
	if (so->filterData == NULL)
		elog(ERROR, "HnswSetFilter: MemoryContextAlloc returned NULL");

	elog(LOG, "HnswSetFilter: copying data to filterData=%p", (void *) so->filterData);
	memcpy(so->filterData, VARDATA_ANY(data), len);
	so->filterDataSize = len;

	elog(LOG, "HnswSetFilter: calling HnswInitFilterSpecFromBytes");
	HnswInitFilterSpecFromBytes(so, len);
	elog(LOG, "HnswSetFilter: done");
}



static List *
GetFilterScanItems(IndexScanDesc scan, Datum value)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	Relation	index = scan->indexRelation;
	HnswSupport *support = &so->support;
	List	   *ep;
	List	   *w;
	ListCell   *lc;
	ListCell   *lc2;
	int			m, cnt, len;
	HnswElement entryPoint;
	char	   *base = NULL;
	HnswQuery  *q = &so->q;

	elog(LOG, "GetFilterScanItems: ENTER");
	elog(LOG, "GetFilterScanItems: filterSpec.filterType=%d", so->filterSpec.filterType);

	if (so->filterSpec.filterType == HNSW_FILTER_TYPE_BLOOM)
	{
		elog(LOG, "GetFilterScanItems: bloom filter - modNumber=%lu, nHashes=%u, byteLength=%zu, seed1=%lu, seed2=%lu, bitset=%p",
			 (unsigned long) so->filterSpec.data.bloom.modNumber,
			 so->filterSpec.data.bloom.nHashes,
			 so->filterSpec.data.bloom.byteLength,
			 (unsigned long) so->filterSpec.data.bloom.seed1,
			 (unsigned long) so->filterSpec.data.bloom.seed2,
			 (void *) so->filterSpec.data.bloom.bitset);
	}

	/* Get m and entry point */
	HnswGetMetaPageInfo(index, &m, &entryPoint);

	q->value = value;
	so->m = m;

	if (entryPoint == NULL)
		return NIL;

	ep = list_make1(HnswEntryCandidate(base, entryPoint, q, index, support, false));

	for (int lc = entryPoint->level; lc >= 1; lc--)
	{
		elog(LOG, "ep to level %d: %d", lc, list_length(ep));
		foreach(lc2, ep)
		{
			HnswSearchCandidate *sc = (HnswSearchCandidate *) lfirst(lc2);
			elog(LOG, "element: %p, distance: %f", sc->element, sc->distance);
		}

		w = HnswFilterSearchLayer(base, q, ep, 1, lc, index, support, m,
								  false, NULL, NULL, NULL, true, NULL,
								  &so->filterSpec);
		ep = w;
	}

	elog(LOG, "ep to base level: %d", list_length(ep));
	foreach(lc, ep)
	{
		HnswSearchCandidate *sc = (HnswSearchCandidate *) lfirst(lc);
		elog(LOG, "element: %p, distance: %f", sc->element, sc->distance);
	}

	w = HnswFilterSearchLayer(base, q, ep, hnsw_ef_search, 0, index, support, m,
							   false, NULL, &so->v,
							   hnsw_iterative_scan != HNSW_ITERATIVE_SCAN_OFF ? &so->discarded : NULL,
							   true, &so->tuples,
							   &so->filterSpec);

	/* 确认 filter 版本与原始版本一致。 包括长度和内容 */

	// output last 5
	/*
	elog(LOG, "Got %d results. Top 5 are:", list_length(w));
	cnt = 0;
	len = list_length(w);
	foreach(lc, w)
	{
		if (cnt >= len - 5)
		{
			HnswSearchCandidate *sc = (HnswSearchCandidate *) lfirst(lc);
			HnswElement element = HnswPtrAccess(base, sc->element);
			if (element != NULL && element->heaptidsLength > 0)
			{
				ItemPointerData *tid = &element->heaptids[element->heaptidsLength - 1];
				elog(LOG, "heap blkno: %u, offno: %u, distance: %f",
					 ItemPointerGetBlockNumber(tid),
					 ItemPointerGetOffsetNumber(tid),
					 sqrt(sc->distance));
			}
			else
			{
				elog(LOG, "heap tid: <none>, distance: %f", sqrt(sc->distance));
			}
		}
		cnt++;
	}
	*/

	elog(LOG, "hnswscan.c: GetFilterScanItems finished");
	return w;
}






/*
 * Resume scan at ground level with discarded candidates
 */
static List *
ResumeScanItems(IndexScanDesc scan)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	Relation	index = scan->indexRelation;
	List	   *ep = NIL;
	char	   *base = NULL;
	int			batch_size = hnsw_ef_search;

	if (pairingheap_is_empty(so->discarded))
		return NIL;

	/* Get next batch of candidates */
	for (int i = 0; i < batch_size; i++)
	{
		HnswSearchCandidate *sc;

		if (pairingheap_is_empty(so->discarded))
			break;

		sc = HnswGetSearchCandidate(w_node, pairingheap_remove_first(so->discarded));

		ep = lappend(ep, sc);
	}

	return HnswSearchLayer(base, &so->q, ep, batch_size, 0, index, &so->support, so->m, false, NULL, &so->v, &so->discarded, false, &so->tuples);
}

/*
 * Get scan value
 */
static Datum
GetScanValue(IndexScanDesc scan)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	Datum		value;

	if (scan->orderByData->sk_flags & SK_ISNULL)
		value = PointerGetDatum(NULL);
	else
	{
		value = scan->orderByData->sk_argument;

		/* Value should not be compressed or toasted */
		Assert(!VARATT_IS_COMPRESSED(DatumGetPointer(value)));
		Assert(!VARATT_IS_EXTENDED(DatumGetPointer(value)));

		/* Normalize if needed */
		if (so->support.normprocinfo != NULL)
			value = HnswNormValue(so->typeInfo, so->support.collation, value);
	}

	return value;
}

#if defined(HNSW_MEMORY)
/*
 * Show memory usage
 */
static void
ShowMemoryUsage(HnswScanOpaque so)
{
	elog(INFO, "memory: %zu KB, tuples: " INT64_FORMAT, MemoryContextMemAllocated(so->tmpCtx, false) / 1024, so->tuples);
}
#endif

/*
 * Prepare for an index scan
 */
IndexScanDesc
hnswbeginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan;
	HnswScanOpaque so;
	double		maxMemory;

	scan = RelationGetIndexScan(index, nkeys, norderbys);

	so = (HnswScanOpaque) palloc(sizeof(HnswScanOpaqueData));
	so->typeInfo = HnswGetTypeInfo(index);
	so->filterData = NULL;
	so->filterDataSize = 0;
	HnswInitEmptyFilterSpec(&so->filterSpec);
	so->filterActive = false;

	/* Set support functions */
	HnswInitSupport(&so->support, index);

	/*
	 * Use a lower max allocation size than default to allow scanning more
	 * tuples for iterative search before exceeding work_mem
	 */
	so->tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
									   "Hnsw scan temporary context",
									   0, 8 * 1024, 256 * 1024);

	/* Calculate max memory */
	/* Add 256 extra bytes to fill last block when close */
	maxMemory = (double) work_mem * hnsw_scan_mem_multiplier * 1024.0 + 256;
	so->maxMemory = Min(maxMemory, (double) SIZE_MAX);

	scan->opaque = so;

	return scan;
}

/*
 * Start or restart an index scan
 */
void
hnswrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;

	so->first = true;
	so->filterActive = false;
	/* v and discarded are allocated in tmpCtx */
	so->v.tids = NULL;
	so->discarded = NULL;
	so->tuples = 0;
	so->previousDistance = -get_float8_infinity();
	MemoryContextReset(so->tmpCtx);
	/* MemoryContextReset 已释放 filterData，必须先置 NULL 避免 double-free */
	so->filterData = NULL;
	HnswResetFilter(so);

	if (keys && scan->numberOfKeys > 0)
		memmove(scan->keyData, keys, scan->numberOfKeys * sizeof(ScanKeyData));

	if (orderbys && scan->numberOfOrderBys > 0)
		memmove(scan->orderByData, orderbys, scan->numberOfOrderBys * sizeof(ScanKeyData));
}

/*
 * Fetch the next tuple in the given scan
 */
bool
hnswgettuple(IndexScanDesc scan, ScanDirection dir)
{
	HnswScanOpaque so;
	MemoryContext oldCtx;

	elog(LOG, "hnswgettuple: ENTER, scan=%p", (void *) scan);

	if (scan == NULL)
		elog(ERROR, "hnswgettuple: scan is NULL");

	so = (HnswScanOpaque) scan->opaque;
	if (so == NULL)
		elog(ERROR, "hnswgettuple: scan->opaque is NULL");

	elog(LOG, "hnswgettuple: so=%p, so->tmpCtx=%p", (void *) so, (void *) so->tmpCtx);

	if (so->tmpCtx == NULL)
		elog(ERROR, "hnswgettuple: so->tmpCtx is NULL");

	oldCtx = MemoryContextSwitchTo(so->tmpCtx);
	elog(LOG, "hnswgettuple: switched memory context");

	/*
	 * Index can be used to scan backward, but Postgres doesn't support
	 * backward scan on operators
	 */
	Assert(ScanDirectionIsForward(dir));

	if (so->first)
	{
		Datum		orderValue = PointerGetDatum(NULL);
		Datum      qval = 0;       /* 半径谓词的查询向量 */
		Datum      filterVal = 0;  /* 过滤谓词的查询参数 */
		double     radius = 0.0;   /* 半径阈值（来自 GUC） */
		bool       has_radius = false;
		bool       has_filter = false;
		Oid        filterType = InvalidOid;
		bool       haveOrderBy = scan->orderByData != NULL;

		if (haveOrderBy)
			orderValue = GetScanValue(scan);

		/* Count index scan for stats */
		pgstat_count_index_scan(scan->indexRelation);


		/* 先探测是否有 filter 谓词（strategy = HNSW_STRAT_FILTER；对应 <->#） */
		has_filter = hnsw_parse_filter(scan, &filterVal, &filterType);
		/* 再探测是否有半径谓词（strategy = HNSW_STRAT_RADIUS；对应 <->@） */
		has_radius = hnsw_parse_radius(scan, &qval, &radius);

		so->filterActive = has_filter;


		/* debug */
		const char *orderFlag = haveOrderBy ? "t" : "f";
		const char *filterFlag = has_filter ? "t" : "f";
		const char *radiusFlag = has_radius ? "t" : "f";

		elog(LOG,
			 "hnswscan begin: orderBy=%s filter=%s (type=%u) radius=%s",
			 orderFlag, filterFlag, filterType, radiusFlag);

		/* Safety check */
		if (scan->orderByData == NULL && !has_radius && !has_filter)
        	elog(ERROR, "cannot scan hnsw index without ORDER BY, <->@ or <-># predicate");

		/* Requires MVCC-compliant snapshot as not able to maintain a pin */
		/* https://www.postgresql.org/docs/current/index-locking.html */
		if (!IsMVCCSnapshot(scan->xs_snapshot))
			elog(ERROR, "non-MVCC snapshots are not supported with hnsw");

		/*
		 * Get a shared lock. This allows vacuum to ensure no in-flight scans
		 * before marking tuples as deleted.
		 */
		LockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

		elog(LOG, "in hnswgettuple");

		if (has_filter)
		{
			elog(LOG, "has_filter type: %u", filterType);
			elog(LOG, "bytea oid: %u", BYTEAOID);
			if (filterType == BYTEAOID)
			{
				if (!haveOrderBy)
					elog(ERROR, "<->#(vector, bytea) requires ORDER BY clause");

				if (hnsw_id_map_table == NULL || hnsw_id_map_table[0] == '\0')
					ereport(ERROR,
							(errmsg("hnsw.id_map_table must be set before running filter search"),
							 errhint("Run SET hnsw.id_map_table = '<schema.table>' in this session.")));

				HnswSetFilter(so, filterVal);
				so->w = GetFilterScanItems(scan, orderValue);
			}
			else
			{
				elog(ERROR, "<-># predicate currently supports only bytea filters");
			}
		}
		else if (has_radius) {
			HnswResetFilter(so);
			/* 半径查询路径：WHERE embedding <->@ qvec */
			so->w = GetRangeScanItems(scan, qval, radius);
		}
		else {
			HnswResetFilter(so);
			/* 原 kNN 路径：ORDER BY <-> */
			so->w = GetScanItems(scan, orderValue);
		}

		/* Release shared lock */
		UnlockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

		so->first = false;

#if defined(HNSW_MEMORY)
		ShowMemoryUsage(so);
#endif
	}

	for (;;)
	{
		char	   *base = NULL;
		HnswSearchCandidate *sc;
		HnswElement element;
		ItemPointer heaptid;

		//elog(LOG, "Looping bitch %d", list_length(so->w));

		if (list_length(so->w) == 0)
		{
			if (hnsw_iterative_scan == HNSW_ITERATIVE_SCAN_OFF)
				break;

			/* Empty index */
			if (so->discarded == NULL)
				break;

			/* Reached max number of tuples or memory limit */
			if (so->tuples >= hnsw_max_scan_tuples || MemoryContextMemAllocated(so->tmpCtx, false) > so->maxMemory)
			{
				if (pairingheap_is_empty(so->discarded))
					break;

				/* Return remaining tuples */
				so->w = lappend(so->w, HnswGetSearchCandidate(w_node, pairingheap_remove_first(so->discarded)));
			}
			else
			{
				/*
				 * Locking ensures when neighbors are read, the elements they
				 * reference will not be deleted (and replaced) during the
				 * iteration.
				 *
				 * Elements loaded into memory on previous iterations may have
				 * been deleted (and replaced), so when reading neighbors, the
				 * element version must be checked.
				 */
				LockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

				so->w = ResumeScanItems(scan);

				UnlockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

#if defined(HNSW_MEMORY)
				ShowMemoryUsage(so);
#endif
			}

			if (list_length(so->w) == 0)
				break;
		}

		sc = llast(so->w);
		element = HnswPtrAccess(base, sc->element);

		/* Move to next element if no valid heap TIDs */
		if (element->heaptidsLength == 0)
		{
			so->w = list_delete_last(so->w);

			/* Mark memory as free for next iteration */
			if (hnsw_iterative_scan != HNSW_ITERATIVE_SCAN_OFF)
			{
				pfree(element);
				pfree(sc);
			}

			continue;
		}

		heaptid = &element->heaptids[--element->heaptidsLength];

		if (so->filterActive)
		{
			if (so->filterData == NULL)
				elog(ERROR, "filterData is NULL");

			if (!HnswHeapTidPassesFilter(heaptid, &so->filterSpec))
				continue;

			/* Only need one valid heap TID per element when filtering */
			element->heaptidsLength = 0;
		}

		if (hnsw_iterative_scan == HNSW_ITERATIVE_SCAN_STRICT)
		{
			if (sc->distance < so->previousDistance)
				continue;

			so->previousDistance = sc->distance;
		}

		MemoryContextSwitchTo(oldCtx);

		scan->xs_heaptid = *heaptid;
		//elog(LOG, "emit tid blkno=%u offno=%u distance=%f",
		//	 ItemPointerGetBlockNumber(heaptid),
		//	 ItemPointerGetOffsetNumber(heaptid),
		//	 sqrt(sc->distance));
		scan->xs_recheck = false;
		scan->xs_recheckorderby = false;
		return true;
	}

	MemoryContextSwitchTo(oldCtx);
	return false;
}

/*
 * End a scan and release resources
 */
void
hnswendscan(IndexScanDesc scan)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;

	MemoryContextDelete(so->tmpCtx);

	pfree(so);
	scan->opaque = NULL;
}
