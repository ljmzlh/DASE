#include "postgres.h"

#include <math.h>

#include "access/generic_xlog.h"
#include "catalog/pg_type.h"
#include "catalog/pg_type_d.h"
#include "common/hashfn.h"
#include "executor/spi.h"
#include "fmgr.h"
#include "hnsw.h"
#include "miscadmin.h"
#include "lib/pairingheap.h"
#include "lib/stringinfo.h"
#include "sparsevec.h"
#include "storage/bufmgr.h"
#include "utils/builtins.h"
#include "utils/datum.h"
#include "utils/hsearch.h"
#include "utils/memdebug.h"
#include "utils/memutils.h"
#include "utils/rel.h"



#if PG_VERSION_NUM < 170000
static inline uint64
murmurhash64(uint64 data)
{
	uint64		h = data;

	h ^= h >> 33;
	h *= 0xff51afd7ed558ccd;
	h ^= h >> 33;
	h *= 0xc4ceb9fe1a85ec53;
	h ^= h >> 33;

	return h;
}
#endif

/* TID hash table */
static uint32
hash_tid(ItemPointerData tid)
{
	union
	{
		uint64		i;
		ItemPointerData tid;
	}			x;

	/* Initialize unused bytes */
	x.i = 0;
	x.tid = tid;

	return murmurhash64(x.i);
}

#define SH_PREFIX		tidhash
#define SH_ELEMENT_TYPE	TidHashEntry
#define SH_KEY_TYPE		ItemPointerData
#define	SH_KEY			tid
#define SH_HASH_KEY(tb, key)	hash_tid(key)
#define SH_EQUAL(tb, a, b)		ItemPointerEquals(&a, &b)
#define	SH_SCOPE		extern
#define SH_DEFINE
#include "lib/simplehash.h"

typedef struct HnswTidMapEntry
{
	ItemPointerData tid;
	int			realid;
}			HnswTidMapEntry;

typedef struct HnswTidMapCache
{
	char	   *table_name;
	HTAB	   *entries;
}			HnswTidMapCache;

static HnswTidMapCache HnswTidMap = {NULL, NULL};

static void ClearTidMapCache(void);
static void AppendQualifiedTableName(StringInfo buf, const char *tableName);
static void LoadTidMapCache(void);
static bool LookupRealIdFromMap(BlockNumber blkno, OffsetNumber offno, int *realid);

void
HnswSetTidMapTable(const char *tableName)
{
	bool success;

	ClearTidMapCache();

	if (tableName == NULL || tableName[0] == '\0')
	{
		/* RESET/SET LOCAL 可能传入 NULL，只需清缓存即可 */
		return;
	}

	HnswTidMap.table_name = MemoryContextStrdup(TopMemoryContext, tableName);
  
	if (!IsUnderPostmaster || !IsNormalProcessingMode())
		elog(ERROR, "not under postmaster or normal processing mode");

	elog(LOG, "trying to set tid table %s", tableName);

	success = false;
	PG_TRY();
	{
		LoadTidMapCache();
		success = true;
	}
	PG_CATCH();
	{
		ClearTidMapCache();
		PG_RE_THROW();
		success = false;
	}
	PG_END_TRY();

	if (!success)
		elog(ERROR, "failed to set tid table %s", tableName);
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hnsw_set_tid_map_table_sql);
PGDLLEXPORT Datum hnsw_set_tid_map_table_sql(PG_FUNCTION_ARGS);

Datum
hnsw_set_tid_map_table_sql(PG_FUNCTION_ARGS)
{
	char	   *tableName = NULL;

	if (!PG_ARGISNULL(0))
		tableName = text_to_cstring(PG_GETARG_TEXT_PP(0));

	HnswSetTidMapTable(tableName);

	if (tableName != NULL)
		pfree(tableName);

	PG_RETURN_VOID();
}

static void
ClearTidMapCache(void)
{
	if (HnswTidMap.entries != NULL)
	{
		hash_destroy(HnswTidMap.entries);
		HnswTidMap.entries = NULL;
	}

	if (HnswTidMap.table_name != NULL)
	{
		pfree(HnswTidMap.table_name);
		HnswTidMap.table_name = NULL;
	}
}

static void
AppendQualifiedTableName(StringInfo buf, const char *tableName)
{
	if (tableName == NULL || tableName[0] == '\0')
		ereport(ERROR,
				(errmsg("mapping table name must be provided before loading TID map")));

	appendStringInfoString(buf, tableName);
}

static void
LoadTidMapCache(void)
{
	HASHCTL		ctl;
	int			ret;
	StringInfoData query;

	if (HnswTidMap.table_name == NULL || HnswTidMap.table_name[0] == '\0')
		ereport(ERROR,
				(errmsg("mapping table name has not been set"),
				 errhint("Run SET hnsw.id_map_table = '<schema.table>' in this session.")));

	if (HnswTidMap.entries != NULL)
		return;

	elog(LOG, "loading tid map cache from %s", HnswTidMap.table_name);

	MemSet(&ctl, 0, sizeof(ctl));
	ctl.keysize = sizeof(ItemPointerData);
	ctl.entrysize = sizeof(HnswTidMapEntry);
	ctl.hcxt = TopMemoryContext;

	HnswTidMap.entries = hash_create("hnsw_tid_map_cache",
									 1024,
									 &ctl,
									 HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);

	ret = SPI_connect();
	if (ret != SPI_OK_CONNECT)
		ereport(ERROR,
				(errmsg("SPI_connect failed while loading mapping table"),
				 errdetail("Error code %d", ret)));

	initStringInfo(&query);
	appendStringInfo(&query,
					 "SELECT blkno, offno, realid FROM %s",
					 HnswTidMap.table_name);

	ret = SPI_execute(query.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		SPI_finish();
		pfree(query.data);
		ereport(ERROR,
				(errmsg("failed to load mapping table \"%s\"", HnswTidMap.table_name),
				 errdetail("SPI_execute returned %d", ret)));
	}

	for (uint64 i = 0; i < SPI_processed; i++)
	{
		bool		isnull;
		Datum		blkDatum;
		Datum		offDatum;
		Datum		idDatum;
		ItemPointerData tid;
		HnswTidMapEntry *entry;

		blkDatum = SPI_getbinval(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 1, &isnull);
		if (isnull)
			continue;
		offDatum = SPI_getbinval(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 2, &isnull);
		if (isnull)
			continue;
		idDatum = SPI_getbinval(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 3, &isnull);
		if (isnull)
			continue;

		ItemPointerSet(&tid,
					   (BlockNumber) DatumGetInt32(blkDatum),
					   (OffsetNumber) DatumGetInt32(offDatum));

		entry = hash_search(HnswTidMap.entries, &tid, HASH_ENTER, NULL);
		entry->realid = DatumGetInt32(idDatum);
	}

	SPI_finish();
	pfree(query.data);
}

static bool
LookupRealIdFromMap(BlockNumber blkno, OffsetNumber offno, int *realid)
{
	ItemPointerData tid;
	HnswTidMapEntry *entry;

	/*
	 * If the SQL helper hasn't been called yet, fall back to the current GUC
	 * value so existing workflows that only run "SET hnsw.id_map_table = ..."
	 * continue to work. Also handle the case where the user points the GUC to
	 * a different table after we have already cached another one.
	 */
	if (hnsw_id_map_table != NULL && hnsw_id_map_table[0] != '\0')
	{
		if (HnswTidMap.table_name == NULL ||
			strcmp(HnswTidMap.table_name, hnsw_id_map_table) != 0)
			HnswSetTidMapTable(hnsw_id_map_table);
	}

	if (HnswTidMap.table_name == NULL)
		return false;

	if (HnswTidMap.entries == NULL)
	{
		elog(LOG, "tid map missing, loading from disk");
		LoadTidMapCache();
	}

	ItemPointerSet(&tid, blkno, offno);
	entry = hash_search(HnswTidMap.entries, &tid, HASH_FIND, NULL);

	if (entry == NULL)
		return false;

	if (realid != NULL)
		*realid = entry->realid;

	return true;
}

static int bitmap_check_count = 0;

static inline bool
BitmapFilterAllows(const HnswBitmapFilterData *bitmap, int realid)
{
	Size		byteIdx;
	int			bitOffset;
	bool		result;

	if (bitmap == NULL || bitmap->bitset == NULL)
		elog(ERROR, "bitmap filter payload is NULL");

	if (realid < 0)
		return false;

	byteIdx = ((Size) realid) >> 3;
	if (byteIdx >= bitmap->size)
		elog(ERROR, "bitmap filter payload is too small");

	bitOffset = realid & 0x07;

	result = (bitmap->bitset[byteIdx] & (1 << bitOffset)) != 0;

	if (bitmap_check_count < 100)
	{
		elog(LOG, "BitmapFilterAllows: realid=%d, byteIdx=%zu, bitOffset=%d, result=%s",
			 realid, byteIdx, bitOffset, result ? "PASS" : "FAIL");
		bitmap_check_count++;
	}

	return result;
}

static inline uint64
BloomHashValue(uint64 key, uint64 seed)
{
	return murmurhash64(key ^ seed);
}

static int bloom_check_count = 0;

static bool
BloomFilterAllows(const HnswBloomFilterData *bloom, int realid)
{
	uint64		modNumber;
	uint64		h1;
	uint64		h2;
	bool		result = true;

	/* 严格检查所有参数 */
	if (bloom == NULL)
		elog(ERROR, "BloomFilterAllows: bloom is NULL");

	if (bloom->bitset == NULL)
		elog(ERROR, "BloomFilterAllows: bloom->bitset is NULL");

	if (bloom->modNumber == 0)
		elog(ERROR, "BloomFilterAllows: bloom->modNumber is 0");

	if (bloom->nHashes == 0)
		elog(ERROR, "BloomFilterAllows: bloom->nHashes is 0");

	if (bloom->byteLength == 0)
		elog(ERROR, "BloomFilterAllows: bloom->byteLength is 0");

	if (realid < 0)
		elog(ERROR, "BloomFilterAllows: realid is negative: %d", realid);

	modNumber = bloom->modNumber;

	h1 = BloomHashValue((uint64) realid, bloom->seed1) % modNumber;
	h2 = BloomHashValue((uint64) realid, bloom->seed2) % modNumber;
	if (h2 == 0)
		h2 = 1;

	for (uint32 i = 0; i < bloom->nHashes; i++)
	{
		uint64		idx = (h1 + i * h2) % modNumber;
		Size		byteIdx = (Size) (idx >> 3);
		uint8		mask = (uint8) (1U << (idx & 0x07));

		if (byteIdx >= bloom->byteLength)
			elog(ERROR, "BloomFilterAllows: byteIdx out of bounds: byteIdx=%zu, byteLength=%zu, idx=%lu, modNumber=%lu",
				 byteIdx, bloom->byteLength, (unsigned long) idx, (unsigned long) modNumber);

		if ((bloom->bitset[byteIdx] & mask) == 0)
		{
			result = false;
			break;
		}
	}

	/* 只打印前100次检查的结果 */
	if (bloom_check_count < 100)
	{
		elog(LOG, "BloomFilterAllows: realid=%d, h1=%lu, h2=%lu, modNumber=%lu, nHashes=%u, result=%s",
			 realid, (unsigned long) h1, (unsigned long) h2, 
			 (unsigned long) modNumber, bloom->nHashes,
			 result ? "PASS" : "FAIL");
		bloom_check_count++;
	}

	return result;
}

static bool
FilterSpecAllowsRealId(const HnswFilterSpec *filterSpec, int realid)
{
	if (filterSpec == NULL || filterSpec->filterType == HNSW_FILTER_TYPE_NONE)
		elog(ERROR, "filterSpec is NULL");

	switch (filterSpec->filterType)
	{
		case HNSW_FILTER_TYPE_BITMAP:
			return BitmapFilterAllows(&filterSpec->data.bitmap, realid);
		case HNSW_FILTER_TYPE_BLOOM:
			return BloomFilterAllows(&filterSpec->data.bloom, realid);
		default:
			elog(ERROR, "unsupported filter type: %d", filterSpec->filterType);
	}

	return false;				/* keep compiler quiet */
}

bool
HnswHeapTidPassesFilter(const ItemPointerData *heaptid, const HnswFilterSpec *filterSpec)
{
	BlockNumber blkno;
	OffsetNumber offno;
	int			realid = 0;
	bool		bo = false;

	if (heaptid == NULL)
		elog(ERROR, "heaptid is NULL");

	if (filterSpec == NULL || filterSpec->filterType == HNSW_FILTER_TYPE_NONE)
		elog(ERROR, "filterSpec is NULL");

	blkno = ItemPointerGetBlockNumber(heaptid);
	offno = ItemPointerGetOffsetNumber(heaptid);

	if (!LookupRealIdFromMap(blkno, offno, &realid))
		elog(ERROR, "No realid (scan) blkno=%d, offno=%d", blkno, offno);

	bo = FilterSpecAllowsRealId(filterSpec, realid);
	
	return bo;
}



/* Pointer hash table */
static uint32
hash_pointer(uintptr_t ptr)
{
#if SIZEOF_VOID_P == 8
	return murmurhash64((uint64) ptr);
#else
	return murmurhash32((uint32) ptr);
#endif
}

#define SH_PREFIX		pointerhash
#define SH_ELEMENT_TYPE	PointerHashEntry
#define SH_KEY_TYPE		uintptr_t
#define	SH_KEY			ptr
#define SH_HASH_KEY(tb, key)	hash_pointer(key)
#define SH_EQUAL(tb, a, b)		(a == b)
#define	SH_SCOPE		extern
#define SH_DEFINE
#include "lib/simplehash.h"

/* Offset hash table */
static uint32
hash_offset(Size offset)
{
#if SIZEOF_SIZE_T == 8
	return murmurhash64((uint64) offset);
#else
	return murmurhash32((uint32) offset);
#endif
}

#define SH_PREFIX		offsethash
#define SH_ELEMENT_TYPE	OffsetHashEntry
#define SH_KEY_TYPE		Size
#define	SH_KEY			offset
#define SH_HASH_KEY(tb, key)	hash_offset(key)
#define SH_EQUAL(tb, a, b)		(a == b)
#define	SH_SCOPE		extern
#define SH_DEFINE
#include "lib/simplehash.h"

/*
 * Get the max number of connections in an upper layer for each element in the index
 */
int
HnswGetM(Relation index)
{
	HnswOptions *opts = (HnswOptions *) index->rd_options;

	if (opts)
		return opts->m;

	return HNSW_DEFAULT_M;
}

/*
 * Get the size of the dynamic candidate list in the index
 */
int
HnswGetEfConstruction(Relation index)
{
	HnswOptions *opts = (HnswOptions *) index->rd_options;

	if (opts)
		return opts->efConstruction;

	return HNSW_DEFAULT_EF_CONSTRUCTION;
}

/*
 * Get proc
 */
FmgrInfo *
HnswOptionalProcInfo(Relation index, uint16 procnum)
{
	if (!OidIsValid(index_getprocid(index, 1, procnum)))
		return NULL;

	return index_getprocinfo(index, 1, procnum);
}

/*
 * Init support functions
 */
void
HnswInitSupport(HnswSupport * support, Relation index)
{
	support->procinfo = index_getprocinfo(index, 1, HNSW_DISTANCE_PROC);
	support->collation = index->rd_indcollation[0];
	support->normprocinfo = HnswOptionalProcInfo(index, HNSW_NORM_PROC);
}

/*
 * Normalize value
 */
Datum
HnswNormValue(const HnswTypeInfo * typeInfo, Oid collation, Datum value)
{
	return DirectFunctionCall1Coll(typeInfo->normalize, collation, value);
}

/*
 * Check if non-zero norm
 */
bool
HnswCheckNorm(HnswSupport * support, Datum value)
{
	return DatumGetFloat8(FunctionCall1Coll(support->normprocinfo, support->collation, value)) > 0;
}

/*
 * New buffer
 */
Buffer
HnswNewBuffer(Relation index, ForkNumber forkNum)
{
	Buffer		buf = ReadBufferExtended(index, forkNum, P_NEW, RBM_NORMAL, NULL);

	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	return buf;
}

/*
 * Init page
 */
void
HnswInitPage(Buffer buf, Page page)
{
	PageInit(page, BufferGetPageSize(buf), sizeof(HnswPageOpaqueData));
	HnswPageGetOpaque(page)->nextblkno = InvalidBlockNumber;
	HnswPageGetOpaque(page)->page_id = HNSW_PAGE_ID;
}

/*
 * Allocate a neighbor array
 */
HnswNeighborArray *
HnswInitNeighborArray(int lm, HnswAllocator * allocator)
{
	HnswNeighborArray *a = HnswAlloc(allocator, HNSW_NEIGHBOR_ARRAY_SIZE(lm));

	a->length = 0;
	a->closerSet = false;
	return a;
}

/*
 * Allocate neighbors
 */
void
HnswInitNeighbors(char *base, HnswElement element, int m, HnswAllocator * allocator)
{
	int			level = element->level;
	HnswNeighborArrayPtr *neighborList = (HnswNeighborArrayPtr *) HnswAlloc(allocator, sizeof(HnswNeighborArrayPtr) * (level + 1));

	HnswPtrStore(base, element->neighbors, neighborList);

	for (int lc = 0; lc <= level; lc++)
		HnswPtrStore(base, neighborList[lc], HnswInitNeighborArray(HnswGetLayerM(m, lc), allocator));
}

/*
 * Allocate memory from the allocator
 */
void *
HnswAlloc(HnswAllocator * allocator, Size size)
{
	if (allocator)
		return (*(allocator)->alloc) (size, (allocator)->state);

	return palloc(size);
}

/*
 * Allocate an element
 */
HnswElement
HnswInitElement(char *base, ItemPointer heaptid, int m, double ml, int maxLevel, HnswAllocator * allocator)
{
	HnswElement element = HnswAlloc(allocator, sizeof(HnswElementData));

	int			level = (int) (-log(RandomDouble()) * ml);

	/* Cap level */
	if (level > maxLevel)
		level = maxLevel;

	element->heaptidsLength = 0;
	HnswAddHeapTid(element, heaptid);

	element->level = level;
	element->deleted = 0;
	/* Start at one to make it easier to find issues */
	element->version = 1;

	HnswInitNeighbors(base, element, m, allocator);

	HnswPtrStore(base, element->value, (Pointer) NULL);

	return element;
}

/*
 * Add a heap TID to an element
 */
void
HnswAddHeapTid(HnswElement element, ItemPointer heaptid)
{
	element->heaptids[element->heaptidsLength++] = *heaptid;
}

/*
 * Allocate an element from block and offset numbers
 */
HnswElement
HnswInitElementFromBlock(BlockNumber blkno, OffsetNumber offno)
{
	HnswElement element = palloc(sizeof(HnswElementData));
	char	   *base = NULL;

	element->blkno = blkno;
	element->offno = offno;
	HnswPtrStore(base, element->neighbors, (HnswNeighborArrayPtr *) NULL);
	HnswPtrStore(base, element->value, (Pointer) NULL);
	return element;
}

/*
 * Get the metapage info
 */
void
HnswGetMetaPageInfo(Relation index, int *m, HnswElement * entryPoint)
{
	Buffer		buf;
	Page		page;
	HnswMetaPage metap;

	buf = ReadBuffer(index, HNSW_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	metap = HnswPageGetMeta(page);

	if (unlikely(metap->magicNumber != HNSW_MAGIC_NUMBER))
		elog(ERROR, "hnsw index is not valid");

	if (m != NULL)
		*m = metap->m;

	if (entryPoint != NULL)
	{
		if (BlockNumberIsValid(metap->entryBlkno))
		{
			*entryPoint = HnswInitElementFromBlock(metap->entryBlkno, metap->entryOffno);
			(*entryPoint)->level = metap->entryLevel;
		}
		else
			*entryPoint = NULL;
	}

	UnlockReleaseBuffer(buf);
}

/*
 * Get the entry point
 */
HnswElement
HnswGetEntryPoint(Relation index)
{
	HnswElement entryPoint;

	HnswGetMetaPageInfo(index, NULL, &entryPoint);

	return entryPoint;
}

/*
 * Update the metapage info
 */
static void
HnswUpdateMetaPageInfo(Page page, int updateEntry, HnswElement entryPoint, BlockNumber insertPage)
{
	HnswMetaPage metap = HnswPageGetMeta(page);

	if (updateEntry)
	{
		if (entryPoint == NULL)
		{
			metap->entryBlkno = InvalidBlockNumber;
			metap->entryOffno = InvalidOffsetNumber;
			metap->entryLevel = -1;
		}
		else if (entryPoint->level > metap->entryLevel || updateEntry == HNSW_UPDATE_ENTRY_ALWAYS)
		{
			metap->entryBlkno = entryPoint->blkno;
			metap->entryOffno = entryPoint->offno;
			metap->entryLevel = entryPoint->level;
		}
	}

	if (BlockNumberIsValid(insertPage))
		metap->insertPage = insertPage;
}

/*
 * Update the metapage
 */
void
HnswUpdateMetaPage(Relation index, int updateEntry, HnswElement entryPoint, BlockNumber insertPage, ForkNumber forkNum, bool building)
{
	Buffer		buf;
	Page		page;
	GenericXLogState *state;

	buf = ReadBufferExtended(index, forkNum, HNSW_METAPAGE_BLKNO, RBM_NORMAL, NULL);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	if (building)
	{
		state = NULL;
		page = BufferGetPage(buf);
	}
	else
	{
		state = GenericXLogStart(index);
		page = GenericXLogRegisterBuffer(state, buf, 0);
	}

	HnswUpdateMetaPageInfo(page, updateEntry, entryPoint, insertPage);

	if (building)
		MarkBufferDirty(buf);
	else
		GenericXLogFinish(state);
	UnlockReleaseBuffer(buf);
}

/*
 * Form index value
 */
bool
HnswFormIndexValue(Datum *out, Datum *values, bool *isnull, const HnswTypeInfo * typeInfo, HnswSupport * support)
{
	/* Detoast once for all calls */
	Datum		value = PointerGetDatum(PG_DETOAST_DATUM(values[0]));

	/* Check value */
	if (typeInfo->checkValue != NULL)
		typeInfo->checkValue(DatumGetPointer(value));

	/* Normalize if needed */
	if (support->normprocinfo != NULL)
	{
		if (!HnswCheckNorm(support, value))
			return false;

		value = HnswNormValue(typeInfo, support->collation, value);
	}

	*out = value;

	return true;
}

/*
 * Set element tuple, except for neighbor info
 */
void
HnswSetElementTuple(char *base, HnswElementTuple etup, HnswElement element)
{
	Pointer		valuePtr = HnswPtrAccess(base, element->value);

	etup->type = HNSW_ELEMENT_TUPLE_TYPE;
	etup->level = element->level;
	etup->deleted = 0;
	etup->version = element->version;
	for (int i = 0; i < HNSW_HEAPTIDS; i++)
	{
		if (i < element->heaptidsLength)
			etup->heaptids[i] = element->heaptids[i];
		else
			ItemPointerSetInvalid(&etup->heaptids[i]);
	}
	memcpy(&etup->data, valuePtr, VARSIZE_ANY(valuePtr));
}

/*
 * Set neighbor tuple
 */
void
HnswSetNeighborTuple(char *base, HnswNeighborTuple ntup, HnswElement e, int m)
{
	int			idx = 0;

	ntup->type = HNSW_NEIGHBOR_TUPLE_TYPE;

	for (int lc = e->level; lc >= 0; lc--)
	{
		HnswNeighborArray *neighbors = HnswGetNeighbors(base, e, lc);
		int			lm = HnswGetLayerM(m, lc);

		for (int i = 0; i < lm; i++)
		{
			ItemPointer indextid = &ntup->indextids[idx++];

			if (i < neighbors->length)
			{
				HnswCandidate *hc = &neighbors->items[i];
				HnswElement hce = HnswPtrAccess(base, hc->element);

				ItemPointerSet(indextid, hce->blkno, hce->offno);
			}
			else
				ItemPointerSetInvalid(indextid);
		}
	}

	ntup->count = idx;
	ntup->version = e->version;
}

/*
 * Load an element from a tuple
 */
void
HnswLoadElementFromTuple(HnswElement element, HnswElementTuple etup, bool loadHeaptids, bool loadVec)
{
	element->level = etup->level;
	element->deleted = etup->deleted;
	element->version = etup->version;
	element->neighborPage = ItemPointerGetBlockNumber(&etup->neighbortid);
	element->neighborOffno = ItemPointerGetOffsetNumber(&etup->neighbortid);
	element->heaptidsLength = 0;

	if (loadHeaptids)
	{
		for (int i = 0; i < HNSW_HEAPTIDS; i++)
		{
			/* Can stop at first invalid */
			if (!ItemPointerIsValid(&etup->heaptids[i]))
				break;

			HnswAddHeapTid(element, &etup->heaptids[i]);
		}
	}

	if (loadVec)
	{
		char	   *base = NULL;
		Datum		value = datumCopy(PointerGetDatum(&etup->data), false, -1);

		HnswPtrStore(base, element->value, DatumGetPointer(value));
	}
}

/*
 * Calculate the distance between values
 */
static inline double
HnswGetDistance(Datum a, Datum b, HnswSupport * support)
{
	return DatumGetFloat8(FunctionCall2Coll(support->procinfo, support->collation, a, b));
}

/*
 * Load an element and optionally get its distance from q
 */
static void
HnswLoadElementImpl(BlockNumber blkno, OffsetNumber offno, double *distance, HnswQuery * q, Relation index, HnswSupport * support, bool loadVec, double *maxDistance, HnswElement * element)
{
	Buffer		buf;
	Page		page;
	HnswElementTuple etup;

	/* Read vector */
	buf = ReadBuffer(index, blkno);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);

	etup = (HnswElementTuple) PageGetItem(page, PageGetItemId(page, offno));

	Assert(HnswIsElementTuple(etup));

	/* Calculate distance */
	if (distance != NULL)
	{
		if (DatumGetPointer(q->value) == NULL)
			*distance = 0;
		else
			*distance = HnswGetDistance(q->value, PointerGetDatum(&etup->data), support);
	}

	/* Load element */
	if (distance == NULL || maxDistance == NULL || *distance < *maxDistance)
	{
		if (*element == NULL)
			*element = HnswInitElementFromBlock(blkno, offno);

		HnswLoadElementFromTuple(*element, etup, true, loadVec);
	}

	UnlockReleaseBuffer(buf);
}

/*
 * Load an element and optionally get its distance from q
 */
void
HnswLoadElement(HnswElement element, double *distance, HnswQuery * q, Relation index, HnswSupport * support, bool loadVec, double *maxDistance)
{
	HnswLoadElementImpl(element->blkno, element->offno, distance, q, index, support, loadVec, maxDistance, &element);
}

/*
 * Get the distance for an element
 */
static double
GetElementDistance(char *base, HnswElement element, HnswQuery * q, HnswSupport * support)
{
	Datum		value = HnswGetValue(base, element);

	return HnswGetDistance(q->value, value, support);
}

/*
 * Allocate a search candidate
 */
static HnswSearchCandidate *
HnswInitSearchCandidate(char *base, HnswElement element, double distance)
{
	HnswSearchCandidate *sc = palloc(sizeof(HnswSearchCandidate));

	HnswPtrStore(base, sc->element, element);
	sc->distance = distance;
	return sc;
}

/*
 * Create a candidate for the entry point
 */
HnswSearchCandidate *
HnswEntryCandidate(char *base, HnswElement entryPoint, HnswQuery * q, Relation index, HnswSupport * support, bool loadVec)
{
	bool		inMemory = index == NULL;
	double		distance;

	if (inMemory)
		distance = GetElementDistance(base, entryPoint, q, support);
	else
		HnswLoadElement(entryPoint, &distance, q, index, support, loadVec, NULL);

	return HnswInitSearchCandidate(base, entryPoint, distance);
}

/*
 * Compare candidate distances
 */
static int
CompareNearestCandidates(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	if (HnswGetSearchCandidateConst(c_node, a)->distance < HnswGetSearchCandidateConst(c_node, b)->distance)
		return 1;

	if (HnswGetSearchCandidateConst(c_node, a)->distance > HnswGetSearchCandidateConst(c_node, b)->distance)
		return -1;

	return 0;
}

/*
 * Compare discarded candidate distances
 */
static int
CompareNearestDiscardedCandidates(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	if (HnswGetSearchCandidateConst(w_node, a)->distance < HnswGetSearchCandidateConst(w_node, b)->distance)
		return 1;

	if (HnswGetSearchCandidateConst(w_node, a)->distance > HnswGetSearchCandidateConst(w_node, b)->distance)
		return -1;

	return 0;
}

/*
 * Compare candidate distances
 */
static int
CompareFurthestCandidates(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	if (HnswGetSearchCandidateConst(w_node, a)->distance < HnswGetSearchCandidateConst(w_node, b)->distance)
		return -1;

	if (HnswGetSearchCandidateConst(w_node, a)->distance > HnswGetSearchCandidateConst(w_node, b)->distance)
		return 1;

	return 0;
}

/*
 * Init visited
 * 
 * 使用 TopMemoryContext 的子 context 来隔离 tidhash 分配，
 * 完全避免与 tmpCtx 中的对象（如 ep list）发生地址冲突。
 * 这个 context 会在函数结束后被显式删除。
 */
static MemoryContext visitedHashCtx = NULL;

static inline void
InitVisited(char *base, visited_hash * v, bool inMemory, int ef, int m)
{
	Size hashSize = ef * m * 2;
	
	/* 删除上一次的 context（如果存在） */
	if (visitedHashCtx != NULL)
	{
		MemoryContextDelete(visitedHashCtx);
		visitedHashCtx = NULL;
	}
	
	/* 在 TopMemoryContext 下创建新的 context，完全独立于 tmpCtx */
	visitedHashCtx = AllocSetContextCreate(TopMemoryContext,
										   "VisitedHashContext",
										   ALLOCSET_DEFAULT_SIZES);
	
	if (!inMemory)
	{
		v->tids = tidhash_create(visitedHashCtx, hashSize, NULL);
	}
	else if (base != NULL)
	{
		v->offsets = offsethash_create(visitedHashCtx, hashSize, NULL);
	}
	else
	{
		v->pointers = pointerhash_create(visitedHashCtx, hashSize, NULL);
	}
}

/*
 * Add to visited
 */
static inline void
AddToVisited(char *base, visited_hash * v, HnswElementPtr elementPtr, bool inMemory, bool *found)
{
	if (!inMemory)
	{
		HnswElement element = HnswPtrAccess(base, elementPtr);
		ItemPointerData indextid;

		ItemPointerSet(&indextid, element->blkno, element->offno);
		tidhash_insert(v->tids, indextid, found);
	}
	else if (base != NULL)
	{
		HnswElement element = HnswPtrAccess(base, elementPtr);

		offsethash_insert_hash(v->offsets, HnswPtrOffset(elementPtr), element->hash, found);
	}
	else
	{
		HnswElement element = HnswPtrAccess(base, elementPtr);

		pointerhash_insert_hash(v->pointers, (uintptr_t) HnswPtrPointer(elementPtr), element->hash, found);
	}
}

/*
 * Count element towards ef
 */
static inline bool
CountElement(HnswElement skipElement, HnswElement e)
{
	if (skipElement == NULL)
		return true;

	/* Ensure does not access heaptidsLength during in-memory build */
	pg_memory_barrier();

	/* Keep scan-build happy on Mac x86-64 */
	Assert(e);

	return e->heaptidsLength != 0;
}

static void
HnswInitCandidates(char *base, List *ep, visited_hash *v, bool inMemory,
				   int64 *tuples, pairingheap *C, pairingheap *W, int *wlen,
				   HnswElement skipElement, bool initVisited)
{
	ListCell   *lc;
	int count = 0;

	elog(LOG, "HnswInitCandidates: ep_len=%d, C=%p, W=%p", list_length(ep), (void*)C, (void*)W);

	foreach(lc, ep)
	{
		HnswSearchCandidate *sc = (HnswSearchCandidate *) lfirst(lc);
		bool		found;

		elog(LOG, "HnswInitCandidates: processing candidate %d, sc=%p", count, (void*)sc);
		count++;

		if (initVisited)
		{
			AddToVisited(base, v, sc->element, inMemory, &found);

			/* OK to count elements instead of tuples */
			if (tuples != NULL)
				(*tuples)++;
		}

		pairingheap_add(C, &sc->c_node);
		pairingheap_add(W, &sc->w_node);
		elog(LOG, "HnswInitCandidates: added to C and W, C_empty=%d", pairingheap_is_empty(C));

		/*
		 * Do not count elements being deleted towards ef when vacuuming. It
		 * would be ideal to do this for inserts as well, but this could
		 * affect insert performance.
		 */
		if (CountElement(skipElement, HnswPtrAccess(base, sc->element)) && wlen != NULL)
			(*wlen)++;
	}
}

/*
 * Helper to obtain an element pointer for 2-hop expansion.
 *
 * - In memory mode we already have the element pointer in unvisited entry.
 * - In disk mode we try to reuse the element if it's already loaded;
 *   otherwise initialize a lightweight element handle from the stored TID.
 */
static inline HnswElement
GetHop1Element(bool inMemory, HnswUnvisited *unvisitedEntry, bool loadedElement,
			   HnswElement loaded, ItemPointerData *indextid_local)
{
	if (inMemory)
		return unvisitedEntry->element;

	if (loadedElement && loaded != NULL)
		return loaded;

	if (indextid_local != NULL)
	{
		BlockNumber blkno = ItemPointerGetBlockNumber(indextid_local);
		OffsetNumber offno = ItemPointerGetOffsetNumber(indextid_local);

		return HnswInitElementFromBlock(blkno, offno);
	}

	return NULL;
}

static inline visited_hash *
InitFilterSearchContext(char *base, visited_hash *v, visited_hash *vh_storage,
						bool *initVisited, bool inMemory, int ef, int m, int lm,
						pairingheap **discarded,
						HnswNeighborArray **localNeighborhood,
						Size *neighborhoodSize,
						HnswNeighborArray **localNeighborhood2,
						Size *neighborhoodSize2,
						HnswUnvisited **unvisited2)
{
	if (v == NULL)
	{
		v = vh_storage;
		*initVisited = true;
	}

	if (*initVisited)
	{
		InitVisited(base, v, inMemory, ef, m);

		if (discarded != NULL)
			*discarded = pairingheap_allocate(CompareNearestDiscardedCandidates, NULL);
	}

	if (inMemory)
	{
		*neighborhoodSize = HNSW_NEIGHBOR_ARRAY_SIZE(lm);
		*localNeighborhood = palloc(*neighborhoodSize);

		*neighborhoodSize2 = HNSW_NEIGHBOR_ARRAY_SIZE(lm);
		*localNeighborhood2 = palloc(*neighborhoodSize2);
		*unvisited2 = palloc(lm * sizeof(HnswUnvisited));
	}
	else
	{
		*unvisited2 = palloc(lm * sizeof(HnswUnvisited));
	}

	return v;
}

/*
 * Load unvisited neighbors from memory
 */
static void
HnswLoadUnvisitedFromMemory(char *base, HnswElement element, HnswUnvisited * unvisited, int *unvisitedLength, visited_hash * v, int lc, HnswNeighborArray * localNeighborhood, Size neighborhoodSize)
{
	/* Get the neighborhood at layer lc */
	HnswNeighborArray *neighborhood = HnswGetNeighbors(base, element, lc);

	/* Copy neighborhood to local memory */
	LWLockAcquire(&element->lock, LW_SHARED);
	memcpy(localNeighborhood, neighborhood, neighborhoodSize);
	LWLockRelease(&element->lock);

	*unvisitedLength = 0;

	for (int i = 0; i < localNeighborhood->length; i++)
	{
		HnswCandidate *hc = &localNeighborhood->items[i];
		bool		found;

		AddToVisited(base, v, hc->element, true, &found);

		if (!found)
			unvisited[(*unvisitedLength)++].element = HnswPtrAccess(base, hc->element);
	}
}

/*
 * Load neighbor index TIDs
 */
bool
HnswLoadNeighborTids(HnswElement element, ItemPointerData *indextids, Relation index, int m, int lm, int lc)
{
	Buffer		buf;
	Page		page;
	HnswNeighborTuple ntup;
	int			start;

	buf = ReadBuffer(index, element->neighborPage);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);

	ntup = (HnswNeighborTuple) PageGetItem(page, PageGetItemId(page, element->neighborOffno));

	/*
	 * Ensure the neighbor tuple has not been deleted or replaced between
	 * index scan iterations
	 */
	if (ntup->version != element->version || ntup->count != (element->level + 2) * m)
	{
		UnlockReleaseBuffer(buf);
		return false;
	}

	/* Copy to minimize lock time */
	start = (element->level - lc) * m;
	memcpy(indextids, ntup->indextids + start, lm * sizeof(ItemPointerData));

	UnlockReleaseBuffer(buf);
	return true;
}

/*
 * Load unvisited neighbors from disk
 */
static void
HnswLoadUnvisitedFromDisk(HnswElement element, HnswUnvisited * unvisited, int *unvisitedLength, visited_hash * v, Relation index, int m, int lm, int lc)
{
	ItemPointerData indextids[HNSW_MAX_M * 2];

	*unvisitedLength = 0;

	if (!HnswLoadNeighborTids(element, indextids, index, m, lm, lc))
		return;

	for (int i = 0; i < lm; i++)
	{
		ItemPointer indextid = &indextids[i];
		bool		found;

		if (!ItemPointerIsValid(indextid))
			break;

		tidhash_insert(v->tids, *indextid, &found);

		if (!found)
			unvisited[(*unvisitedLength)++].indextid = *indextid;
	}
}

static inline void
HnswLoadUnvisited(bool inMemory,
				  char *base,
				  Relation index,
				  HnswElement element,
				  HnswUnvisited *unvisited,
				  int *unvisitedLength,
				  visited_hash *v,
				  int lc,
				  HnswNeighborArray *localNeighborhood,
				  Size neighborhoodSize,
				  int m,
				  int lm)
{
	if (inMemory)
		HnswLoadUnvisitedFromMemory(base, element, unvisited, unvisitedLength, v, lc, localNeighborhood, neighborhoodSize);
	else
		HnswLoadUnvisitedFromDisk(element, unvisited, unvisitedLength, v, index, m, lm, lc);
}

static bool
HnswLoadCandidateElement(bool inMemory,
						 char *base,
						 HnswUnvisited *entry,
						 HnswElement *element,
						 double *distance,
						 HnswQuery *q,
						 Relation index,
						 HnswSupport *support,
						 bool inserting,
						 bool alwaysAdd,
						 HnswSearchCandidate *f,
						 pairingheap **discarded)
{
	if (inMemory)
	{
		*element = entry->element;
		*distance = GetElementDistance(base, *element, q, support);
		return true;
	}

	ItemPointer indextid = &entry->indextid;
	BlockNumber blkno = ItemPointerGetBlockNumber(indextid);
	OffsetNumber offno = ItemPointerGetOffsetNumber(indextid);

	/* Avoid allocations if we won't add the candidate */
	*element = NULL;
	double *maxDistance = (alwaysAdd || discarded != NULL) ? NULL : &f->distance;

	HnswLoadElementImpl(blkno, offno, distance, q, index, support, inserting, maxDistance, element);

	return *element != NULL;
}

typedef struct HnswSortedNeighbor
{
	HnswElement element;
	double		distance;
}			HnswSortedNeighbor;

static int
CompareHnswSortedNeighbor(const void *a, const void *b)
{
	const HnswSortedNeighbor *sa = (const HnswSortedNeighbor *) a;
	const HnswSortedNeighbor *sb = (const HnswSortedNeighbor *) b;

	if (sa->distance < sb->distance)
		return -1;
	if (sa->distance > sb->distance)
		return 1;
	return 0;
}

static int
HnswLoadUnvisitedSorted(bool inMemory,
						char *base,
						Relation index,
						HnswElement element,
						HnswUnvisited *unvisited,
						HnswSortedNeighbor *sortedNeighbors,
						int *unvisitedLength,
						visited_hash *v,
						int lc,
						HnswNeighborArray *localNeighborhood,
						Size neighborhoodSize,
						int m,
						int lm,
						HnswQuery *q,
						HnswSupport *support,
						bool inserting)
{
	int			rawLen = 0;
	int			sortedLen = 0;

	HnswLoadUnvisited(inMemory, base, index, element, unvisited, &rawLen,
					  v, lc, localNeighborhood, neighborhoodSize, m, lm);

	if (unvisitedLength != NULL)
		*unvisitedLength = rawLen;

	for (int i = 0; i < rawLen; i++)
	{
		HnswElement eElement;
		double		eDistance;

		if (!HnswLoadCandidateElement(inMemory, base, &unvisited[i], &eElement,
									  &eDistance, q, index, support, inserting,
									  true, NULL, NULL))
			continue;

		sortedNeighbors[sortedLen].element = eElement;
		sortedNeighbors[sortedLen].distance = eDistance;
		sortedLen++;
	}

	if (sortedLen > 1)
		qsort(sortedNeighbors, sortedLen, sizeof(HnswSortedNeighbor),
			  CompareHnswSortedNeighbor);

	return sortedLen;
}

/*
 * Algorithm 2 from paper
 */
 List *
 HnswSearchLayer(char *base, HnswQuery * q, List *ep, int ef, int lc, Relation index, HnswSupport * support, int m, bool inserting, HnswElement skipElement, visited_hash * v, pairingheap **discarded, bool initVisited, int64 *tuples)
 {
	 List	   *w = NIL;
	 pairingheap *C = pairingheap_allocate(CompareNearestCandidates, NULL);
	 pairingheap *W = pairingheap_allocate(CompareFurthestCandidates, NULL);
	 int			wlen = 0;
	 visited_hash vh;
	 ListCell   *lc2;
	 HnswNeighborArray *localNeighborhood = NULL;
	 Size		neighborhoodSize = 0;
	 int			lm = HnswGetLayerM(m, lc);
	 HnswUnvisited *unvisited = palloc(lm * sizeof(HnswUnvisited));
	 int			unvisitedLength;
	 bool		inMemory = index == NULL;
 
	 if (v == NULL)
	 {
		 v = &vh;
		 initVisited = true;
	 }
 
	 if (initVisited)
	 {
		 InitVisited(base, v, inMemory, ef, m);
 
		 if (discarded != NULL)
			 *discarded = pairingheap_allocate(CompareNearestDiscardedCandidates, NULL);
	 }
 
	 /* Create local memory for neighborhood if needed */
	 if (inMemory)
	 {
		 neighborhoodSize = HNSW_NEIGHBOR_ARRAY_SIZE(lm);
		 localNeighborhood = palloc(neighborhoodSize);
	 }
 
	 /* Add entry points to v, C, and W */
	 foreach(lc2, ep)
	 {
		 HnswSearchCandidate *sc = (HnswSearchCandidate *) lfirst(lc2);
		 bool		found;
 
		 if (initVisited)
		 {
			 AddToVisited(base, v, sc->element, inMemory, &found);
 
			 /* OK to count elements instead of tuples */
			 if (tuples != NULL)
				 (*tuples)++;
		 }
 
		 pairingheap_add(C, &sc->c_node);
		 pairingheap_add(W, &sc->w_node);
 
		 /*
		  * Do not count elements being deleted towards ef when vacuuming. It
		  * would be ideal to do this for inserts as well, but this could
		  * affect insert performance.
		  */
		 if (CountElement(skipElement, HnswPtrAccess(base, sc->element)))
			 wlen++;
	 }
 
	 while (!pairingheap_is_empty(C))
	 {
		 HnswSearchCandidate *c = HnswGetSearchCandidate(c_node, pairingheap_remove_first(C));
		 HnswSearchCandidate *f = HnswGetSearchCandidate(w_node, pairingheap_first(W));
		 HnswElement cElement;
 
		 if (c->distance > f->distance)
			 break;

		
		 cElement = HnswPtrAccess(base, c->element);
 
		 if (inMemory)
			 HnswLoadUnvisitedFromMemory(base, cElement, unvisited, &unvisitedLength, v, lc, localNeighborhood, neighborhoodSize);
		 else
			 HnswLoadUnvisitedFromDisk(cElement, unvisited, &unvisitedLength, v, index, m, lm, lc);
 
		 /* OK to count elements instead of tuples */
		 if (tuples != NULL)
			 (*tuples) += unvisitedLength;
 
		 for (int i = 0; i < unvisitedLength; i++)
		 {
			 HnswElement eElement;
			 HnswSearchCandidate *e;
			 double		eDistance;
			 bool		alwaysAdd = wlen < ef;
 
			 f = HnswGetSearchCandidate(w_node, pairingheap_first(W));
 
			 if (inMemory)
			 {
				 eElement = unvisited[i].element;
				 eDistance = GetElementDistance(base, eElement, q, support);
			 }
			 else
			 {
				 ItemPointer indextid = &unvisited[i].indextid;
				 BlockNumber blkno = ItemPointerGetBlockNumber(indextid);
				 OffsetNumber offno = ItemPointerGetOffsetNumber(indextid);
 
				 /* Avoid any allocations if not adding */
				 eElement = NULL;
				 HnswLoadElementImpl(blkno, offno, &eDistance, q, index, support, inserting, alwaysAdd || discarded != NULL ? NULL : &f->distance, &eElement);
 
				 if (eElement == NULL)
					 continue;
			 }


 
			 if (eElement == NULL || !(eDistance < f->distance || alwaysAdd))
			 {
				 if (discarded != NULL)
				 {
					 /* Create a new candidate */
					 e = HnswInitSearchCandidate(base, eElement, eDistance);
					 pairingheap_add(*discarded, &e->w_node);
				 }
 
				 continue;
			 }
 
			 /* Make robust to issues */
			 if (eElement->level < lc)
				 continue;
 
			 /* Create a new candidate */
			 e = HnswInitSearchCandidate(base, eElement, eDistance);
			 pairingheap_add(C, &e->c_node);
			 pairingheap_add(W, &e->w_node);
 
			 /*
			  * Do not count elements being deleted towards ef when vacuuming.
			  * It would be ideal to do this for inserts as well, but this
			  * could affect insert performance.
			  */
			 if (CountElement(skipElement, eElement))
			 {
				 wlen++;
 
				 /* No need to decrement wlen */
				 if (wlen > ef)
				 {
					 HnswSearchCandidate *d = HnswGetSearchCandidate(w_node, pairingheap_remove_first(W));
 
					 if (discarded != NULL)
						 pairingheap_add(*discarded, &d->w_node);
				 }
			 }
		 }
	 }
 
	 /* Add each element of W to w */
	 while (!pairingheap_is_empty(W))
	 {
		 HnswSearchCandidate *sc = HnswGetSearchCandidate(w_node, pairingheap_remove_first(W));
 
		 w = lappend(w, sc);
	 }

	 return w;
 }





/////////////////////////////////////////////////////
/*
 * Range Algorithm 2 from paper
 */
 List *
 HnswRangeSearchLayer(char *base, HnswQuery * q, List *ep, int ef, int lc, 
						Relation index, HnswSupport * support, int m, bool inserting, 
						HnswElement skipElement, visited_hash * v, pairingheap **discarded, 
						bool initVisited, int64 *tuples, double radius)
 {
	 List	   *R = NIL;
	 pairingheap *C = pairingheap_allocate(CompareNearestCandidates, NULL);
	 pairingheap *W = pairingheap_allocate(CompareFurthestCandidates, NULL);
	 int			wlen = 0;
	 visited_hash vh;
	 ListCell   *lc2;
	 HnswNeighborArray *localNeighborhood = NULL;
	 Size		neighborhoodSize = 0;
	 int			lm = HnswGetLayerM(m, lc);
	 HnswUnvisited *unvisited = palloc(lm * sizeof(HnswUnvisited));
	 int			unvisitedLength;
	 bool		inMemory = index == NULL;

	 const double radius_thre = radius * radius; // inside is squared l2
	 const double radius_cap = radius_thre * 1.1;

 
	 if (v == NULL)
	 {
		 v = &vh;
		 initVisited = true;
	 }
 
	 if (initVisited)
	 {
		 InitVisited(base, v, inMemory, ef, m);
 
		 if (discarded != NULL)
			 *discarded = pairingheap_allocate(CompareNearestDiscardedCandidates, NULL);
	 }
 
	 /* Create local memory for neighborhood if needed */
	 if (inMemory)
	 {
		 neighborhoodSize = HNSW_NEIGHBOR_ARRAY_SIZE(lm);
		 localNeighborhood = palloc(neighborhoodSize);
	 }
	 
	 /* Add entry points to v, C, and W */
	 foreach(lc2, ep)
	 {
		 HnswSearchCandidate *sc = (HnswSearchCandidate *) lfirst(lc2);
		 bool		found;
		 double epDist;   // 先声明
		 HnswElement eElement = NULL;
 
		 if (initVisited)
		 {
			 AddToVisited(base, v, sc->element, inMemory, &found);
 
			 /* OK to count elements instead of tuples */
			 if (tuples != NULL)
				 (*tuples)++;
		 }
 
		 pairingheap_add(C, &sc->c_node);
		 pairingheap_add(W, &sc->w_node);
 
		 /*
		  * Do not count elements being deleted towards ef when vacuuming. It
		  * would be ideal to do this for inserts as well, but this could
		  * affect insert performance.
		  */
		 if (CountElement(skipElement, HnswPtrAccess(base, sc->element)))
			 wlen++;

		epDist = sc->distance;
		if (epDist <= radius_thre) {
			R = lappend(R, sc);
		}	
	 }

 
	 while (!pairingheap_is_empty(C))
	 {
		 HnswSearchCandidate *c = HnswGetSearchCandidate(c_node, pairingheap_remove_first(C));
		 HnswSearchCandidate *w = HnswGetSearchCandidate(w_node, pairingheap_first(W));
		 HnswElement cElement;

		 if (c->distance > w->distance && c->distance > radius_cap)
		 	break;
 
		 cElement = HnswPtrAccess(base, c->element);
 
		 HnswLoadUnvisited(inMemory, base, index, cElement, unvisited, &unvisitedLength,
						   v, lc, localNeighborhood, neighborhoodSize, m, lm);
 
		 /* OK to count elements instead of tuples */
		 if (tuples != NULL)
			 (*tuples) += unvisitedLength;
 
		 for (int i = 0; i < unvisitedLength; i++)
		 {
			 HnswElement eElement;
			 HnswSearchCandidate *e;
			 double		eDistance;
			 bool		alwaysAdd = wlen < ef;
			 double gate;
			 
 
			 w = HnswGetSearchCandidate(w_node, pairingheap_first(W));
			 gate=Max(radius_cap, w->distance);
 
			 if (inMemory)
			 {
				 eElement = unvisited[i].element;
				 eDistance = GetElementDistance(base, eElement, q, support);
			 }
			 else
			 {
				 ItemPointer indextid = &unvisited[i].indextid;
				 BlockNumber blkno = ItemPointerGetBlockNumber(indextid);
				 OffsetNumber offno = ItemPointerGetOffsetNumber(indextid);
 
				 /* Avoid any allocations if not adding */
				 eElement = NULL;
				 HnswLoadElementImpl(blkno, offno, &eDistance, q, index, support, inserting, 
							alwaysAdd || discarded != NULL ? NULL : &gate, &eElement);
 
				 if (eElement == NULL)
					 continue;
			 }
 
			 if (eElement == NULL || (eDistance >gate && !alwaysAdd))
			 {
				 if (discarded != NULL)
				 {
					 /* Create a new candidate */
					 e = HnswInitSearchCandidate(base, eElement, eDistance);
					 pairingheap_add(*discarded, &e->w_node);
				 }
 
				 continue;
			 }
 
			 /* Make robust to issues */
			 if (eElement->level < lc)
				 continue;
 
			 /* Create a new candidate */
			 e = HnswInitSearchCandidate(base, eElement, eDistance);
			 pairingheap_add(C, &e->c_node);
			 pairingheap_add(W, &e->w_node);

			 if (eDistance <= radius_thre) {
				R = lappend(R, e);	
			 }

 
			 /*
			  * Do not count elements being deleted towards ef when vacuuming.
			  * It would be ideal to do this for inserts as well, but this
			  * could affect insert performance.
			  */
			 if (CountElement(skipElement, eElement))
			 {
				 wlen++;
 
				 /* No need to decrement wlen */
				 if (wlen > ef)
				 {
					 HnswSearchCandidate *d = HnswGetSearchCandidate(w_node, pairingheap_remove_first(W));
 
					 if (discarded != NULL)
						 pairingheap_add(*discarded, &d->w_node);
				 }
			 }
		 }
	 }
 
	 return R;
 }

 //////////////////////////////////////////


/////////////////////////////////////////////////////
/*
 * Filter version (ACORN) of Algorithm 2 from paper
 */

 static void expand_candidate(char *base,
	HnswElement eElement,
	double eDistance,
	HnswSearchCandidate *f,
	bool alwaysAdd,
	pairingheap **discarded,
	int lc,
	pairingheap *C,
	pairingheap *W,
	HnswElement skipElement,
	int ef,
	int *wlen)
{
	HnswSearchCandidate *e;

	if (eElement == NULL || !(eDistance < f->distance || alwaysAdd))
	{
		if (discarded != NULL)
		{
		e = HnswInitSearchCandidate(base, eElement, eDistance);
		pairingheap_add(*discarded, &e->w_node);
		}

		return;
	}

	/* Make robust to issues */
	if (eElement->level < lc) return;

	/* Create a new candidate */
	e = HnswInitSearchCandidate(base, eElement, eDistance);
	pairingheap_add(C, &e->c_node);
	pairingheap_add(W, &e->w_node);

	if (CountElement(skipElement, eElement))
	{
		(*wlen)++;

		if (*wlen > ef)
		{
			HnswSearchCandidate *d = HnswGetSearchCandidate(w_node, pairingheap_remove_first(W));

			if (discarded != NULL) pairingheap_add(*discarded, &d->w_node);
		}
	}
}


/*
 * Bitmap-filter variant, with optional 2-hop expansion:
 * - 仅当 filter_test(base, element, filter_map, filter_map_bytes, ctx) 为真时计分入堆
 * - enableTwoHop 为真时：对于一跳未通过过滤的节点，尝试加载其邻居进行“二跳扩展”，
 *   仅对二跳里通过过滤的元素计分入堆（gate/ef 逻辑同 HnswSearchLayer）
 */


 static bool
 ElementPassesFilter(HnswElement element, const HnswFilterSpec *filterSpec)
 {
	 if (element->heaptidsLength <= 0)
		 elog(ERROR, "eElement has no heap tids");
 
	 for (int j = 0; j < element->heaptidsLength; j++)
	 {
		 ItemPointerData *tid = &element->heaptids[j];
 
		if (HnswHeapTidPassesFilter(tid, filterSpec))
			 return true;
	 }
 
	 return false;
 }




List *
HnswFilterSearchLayer(char *base, HnswQuery * q, List *ep, int ef, int lc,
	Relation index, HnswSupport * support, int m, bool inserting, HnswElement skipElement,
	visited_hash * v, pairingheap **discarded, bool initVisited, int64 *tuples,
	const HnswFilterSpec *filterSpec)
{
	List	   *w = NIL;
	pairingheap *C;
	pairingheap *W;
	int			wlen = 0;
	visited_hash vh;
	ListCell   *lc2;
	int			lm;

	elog(LOG, "HnswFilterSearchLayer: ENTER lc=%d, ef=%d, m=%d", lc, ef, m);
	elog(LOG, "HnswFilterSearchLayer: ep=%p", (void*)ep);
	
	if (ep == NIL)
		elog(ERROR, "HnswFilterSearchLayer: ep is NIL");
	
	elog(LOG, "HnswFilterSearchLayer: checking ep->length");
	int ep_len_check = list_length(ep);
	elog(LOG, "HnswFilterSearchLayer: ep_len=%d", ep_len_check);
	
	if (ep_len_check == 0)
		elog(ERROR, "HnswFilterSearchLayer: ep is empty (unexpected)");

	bool		inMemory = index == NULL;

	if (v == NULL)
	{
		v = &vh;
		initVisited = true;
	}

	/* InitVisited 使用专用的子 MemoryContext，不会和 ep 冲突 */
	if (initVisited)
	{
		InitVisited(base, v, inMemory, ef, m);

		if (discarded != NULL)
			*discarded = pairingheap_allocate(CompareNearestDiscardedCandidates, NULL);
	}

	lm = HnswGetLayerM(m, lc);

	C = pairingheap_allocate(CompareNearestCandidates, NULL);
	W = pairingheap_allocate(CompareFurthestCandidates, NULL);

	HnswNeighborArray *localNeighborhood = NULL;
	Size		neighborhoodSize = 0;
	HnswUnvisited *unvisited = palloc(lm * sizeof(HnswUnvisited));

	HnswNeighborArray *localNeighborhood2 = NULL;
	Size		neighborhoodSize2 = 0;
	HnswUnvisited *unvisited2 = palloc(lm * sizeof(HnswUnvisited));

	int			unvisitedLength, unvisitedLength2;
	HnswSortedNeighbor *sortedNeighbors = NULL;
	HnswSortedNeighbor *sortedNeighbors2 = NULL;
	bool		pass_filter = true;

	if (hnsw_enable_2hop)
	{
		sortedNeighbors = palloc(lm * sizeof(HnswSortedNeighbor));
		sortedNeighbors2 = palloc(lm * sizeof(HnswSortedNeighbor));
	}

	/* Create local memory for neighborhood if needed;
	   also create local memory for 2-hop neighbors */
	if (inMemory)
	{
		neighborhoodSize = HNSW_NEIGHBOR_ARRAY_SIZE(lm);
		localNeighborhood = palloc(neighborhoodSize);
		neighborhoodSize2 = HNSW_NEIGHBOR_ARRAY_SIZE(lm);
		localNeighborhood2 = palloc(neighborhoodSize2);
		elog(LOG, "TRACE: after localNeighborhood alloc, ep_len=%d", list_length(ep));
	}

	/* Add entry points to v, C, and W */
	elog(LOG, "HnswFilterSearchLayer: before HnswInitCandidates, ep=%p, ep_len=%d", (void*)ep, list_length(ep));
	HnswInitCandidates(base, ep, v, inMemory, tuples, C, W, &wlen, skipElement, initVisited);
	elog(LOG, "HnswFilterSearchLayer: after HnswInitCandidates, ep_len=%d", list_length(ep));


	elog(LOG, "HnswFilterSearchLayer: C is_empty=%d", pairingheap_is_empty(C));

	while (!pairingheap_is_empty(C))
	{
		HnswSearchCandidate *c = HnswGetSearchCandidate(c_node, pairingheap_remove_first(C));
		HnswSearchCandidate *f = HnswGetSearchCandidate(w_node, pairingheap_first(W));
		HnswElement cElement;
		// jianzhi: num_found
		int num_found = 0;
		int			neighborCount = 0;
		
		elog(LOG, "HnswFilterSearchLayer: c->distance=%f, f->distance=%f", c->distance, f->distance);

		if (c->distance > f->distance)
		{
			elog(LOG, "HnswFilterSearchLayer: breaking early");
			break;
		}

		cElement = HnswPtrAccess(base, c->element);
		if (cElement == NULL)
			elog(ERROR, "HnswFilterSearchLayer: cElement is NULL");

		if (!hnsw_enable_2hop)
		{
			HnswLoadUnvisited(inMemory, base, index, cElement, unvisited, &unvisitedLength,
							  v, lc, localNeighborhood, neighborhoodSize, m, lm);
			neighborCount = unvisitedLength;
		}
		else
		{
			if (sortedNeighbors == NULL)
				elog(ERROR, "sortedNeighbors is NULL while 2-hop is enabled");

			neighborCount = HnswLoadUnvisitedSorted(inMemory, base, index, cElement,
													unvisited, sortedNeighbors,
													&unvisitedLength, v, lc,
													localNeighborhood, neighborhoodSize,
													m, lm, q, support, inserting);
		}

		elog(LOG, "HnswFilterSearchLayer: neighborCount=%d, unvisitedLength=%d", neighborCount, unvisitedLength);

		/* OK to count elements instead of tuples */
		if (tuples != NULL)
			(*tuples) += unvisitedLength;

		for (int i = 0; i < neighborCount; i++)
		{
			HnswElement eElement;
			HnswSearchCandidate *e;
			double		eDistance;
			bool		alwaysAdd = wlen < ef;

			f = HnswGetSearchCandidate(w_node, pairingheap_first(W));
			
			if (!hnsw_enable_2hop)
			{
				if (!HnswLoadCandidateElement(inMemory, base, &unvisited[i], &eElement, &eDistance, q, index, support, inserting, alwaysAdd, f, discarded))
					continue;
			}
			else
			{
				eElement = sortedNeighbors[i].element;
				eDistance = sortedNeighbors[i].distance;

				if (eElement == NULL)
					continue;
			}

			/* filter test */
			pass_filter = ElementPassesFilter(eElement, filterSpec);

			if(pass_filter) 
			{
				num_found++;
				expand_candidate(base, eElement, eDistance, f, alwaysAdd, discarded, lc, C, W, skipElement, ef, &wlen);
				//if(num_found > 2*m) break;
				// making this tweak to improve recall
			}
			// explore 2-hop neighbors
			// this is bugged because the neigbors are not sorted in pgvector
			
			if(hnsw_enable_2hop == false) continue;

			
			
			if (sortedNeighbors2 == NULL)
				elog(ERROR, "sortedNeighbors2 is NULL while 2-hop is enabled");

			HnswLoadUnvisitedSorted(inMemory, base, index, eElement,
										unvisited2, sortedNeighbors2,
										&unvisitedLength2, v, lc,
										localNeighborhood2, neighborhoodSize2,
										m, lm, q, support, inserting);
			
			for (int i2 = 0; i2 < unvisitedLength2; i2++)
			{
				HnswElement eElement2;
				HnswSearchCandidate *e2;
				double		eDistance2;
				int			realid_local2 = 0;
				bool		haveRealId2 = false;
				BlockNumber blkno2 = InvalidBlockNumber;
				OffsetNumber offno2 = InvalidOffsetNumber;

				
				eElement2 = sortedNeighbors2[i2].element;
				eDistance2 = sortedNeighbors2[i2].distance;

				if (eElement2 == NULL)
					continue;

				pass_filter = ElementPassesFilter(eElement2, filterSpec);

				if(pass_filter) 
				{
					num_found++;
					expand_candidate(base, eElement2, eDistance2, f, alwaysAdd, discarded, lc, C, W, skipElement, ef, &wlen);
					if(num_found > 2*m) break;
				}

			}
			
		}
	}



	/* Add each element of W to w */
	elog(LOG, "HnswFilterSearchLayer: wlen=%d before extracting W", wlen);

	while (!pairingheap_is_empty(W))
	{
		HnswSearchCandidate *sc = HnswGetSearchCandidate(w_node, pairingheap_remove_first(W));
		w = lappend(w, sc);
	}

	elog(LOG, "HnswFilterSearchLayer: returning %d results", list_length(w));

	
	return w;
}












/*
This is a piece of code from ACORN (based on FAISS)
For hybrid/filter search


entry:
if (upper_beam == 1) { // common branch
        debug("%s\n", "reached upper beam == 1");

        //  greedy search on upper levels
        storage_idx_t nearest = entry_point;
        float d_nearest = qdis(nearest);

        debug_search("-starting at ep: %d, d: %f, metadata: %d\n", nearest, d_nearest, metadata[nearest]);

        int ndis_upper = 0;
        for (int level = max_level; level >= 1; level--) {
            debug_search("-at level %d, searching for greedy nearest from current nearest: %d, dist: %f, metadata: %d\n", level, nearest, d_nearest, metadata[nearest]);
            ndis_upper += hybrid_greedy_update_nearest(*this, qdis, filter_map, level, nearest, d_nearest);
            // ndis_upper += hybrid_greedy_update_nearest(*this, qdis, filter, op, regex, level, nearest, d_nearest);
            debug_search("-at level %d, new nearest: %d, d: %f, metadata: %d\n", level, nearest, d_nearest, metadata[nearest]);
        }
        stats.n3 += ndis_upper;

        int ef = std::max(efSearch, k);
        if (search_bounded_queue) { // this is the most common branch
            debug("%s\n", "reached search bounded queue");

            MinimaxHeap candidates(ef);

            candidates.push(nearest, d_nearest);
            debug_search("-starting BFS at level 0 with ef: %d, nearest: %d, d: %f, metadata: %d\n", ef, nearest, d_nearest, metadata[nearest]);
            hybrid_search_from_candidates(
                    *this, qdis, filter_map, k, I, D, candidates, vt, stats, 0, 0, params);
            

        } else {
            // TODO
            printf("UNIMPLEMENTED BRANCH for hybid search\n");
            debug("%s\n", "reached search_bounded_queue == False");
            throw FaissException("UNIMPLEMENTED search unbounded queue");

            
        }

        vt.advance();
}


// for upper level search
hybrid_greedy_update_nearest:
int hybrid_greedy_update_nearest(
        const ACORN& hnsw,
        DistanceComputer& qdis,
        char* filter_map,
        // int filter,
        // Operation op,
        // std::string regex,
        int level,
        storage_idx_t& nearest,
        float& d_nearest) {
    debug("%s\n", "reached"); 
    // printf("hybrid_greedy_update_nearest called with parameters: filter: %d, op: %d, regex: %s, level: %d\n", filter, op, regex.c_str(), level);
    int ndis = 0;
    for (;;) {
        int num_found = 0;
        storage_idx_t prev_nearest = nearest;
        debug_search("----hybrid_greedy_update visists current nearest: %d, d_nearest: %f\n", nearest, d_nearest);

        size_t begin, end;
        hnsw.neighbor_range(nearest, level, &begin, &end);
        debug_search("%s", "--------checking neighbors: \n");
        
        // for debugging, collect all neighbors looked at in a vector
        std::vector<std::pair<storage_idx_t, int>> neighbors_checked;
        bool keep_expanding = true;

        for (size_t i = begin; i < end; i++) {
            auto v = hnsw.neighbors[i];
            
            if (v < 0)
                break;
                
            // note that this slows down search significantly but can be useful for debugging
            // if (debugSearchFlag) {
            //     neighbors_checked.push_back(std::make_pair(v, metadata)); 
            //     debug_search("------------checking neighbor: %d, metadata: %d, metadata & filter: %d\n", v, metadata, metadata & filter);
            // }

            // filter
            // printf("---at first filter: op: %d, metadata: %s, regex: %s, check_regex result: %d\n", op, hnsw.metadata_strings[v].c_str(), regex.c_str(), CHECK_REGEX(hnsw.metadata_strings[v], regex));
            if (filter_map[v]) {
                num_found = num_found + 1;
            } else {
                // not filter & gamma > 1
                if (hnsw.gamma > 1) continue;
            }
            

        
            
            // check if filter pass
            if (filter_map[v]) {
    
                float dis = qdis(v);
                ndis += 1;
                if (dis < d_nearest || !filter_map[nearest]) {
                
                    nearest = v;
                    d_nearest = dis;
                    // debug_search("----------------new nearest: %d, d_nearest: %f\n", nearest, d_nearest);
                }
                if (num_found >= hnsw.M) {
                    // debug_search("----found %d neighbors with filter %d, returning\n", num_found, filter);
                    break;
                }
            }            

          

            // expand neighbor list if gamma=1
            if (hnsw.gamma == 1) {
                size_t begin2, end2;
                hnsw.neighbor_range(v, level, &begin2, &end2);
                for (size_t j = begin2; j < end2; j++) {
                    auto v2 = hnsw.neighbors[j];
                   

                    if (v2 < 0)
                        break;


                    // check filter pass
                    if (filter_map[v2]) {
                        num_found = num_found + 1;
                        float dis2 = qdis(v2);
                        ndis += 1;
                        // debug_search("------------found: %d, metadata: %d distance to v: %f\n", v2, metadata2, dis2);
          
                        if (dis2 < d_nearest || !filter_map[nearest]) {
                            nearest = v2;
                            d_nearest = dis2;
                            // debug_search("----------------new nearest: %d, d_nearest: %f\n", nearest, d_nearest);
                        }
                        if (num_found >= hnsw.M) {
                            break;
                        }
                    } 
                   
                }
            }
        }       

        if (nearest == prev_nearest) {
            return ndis;
        }
    }
    return ndis;
}







// has a filter arg for hybrid search, this only gets called on level 0
int hybrid_search_from_candidates(
        const ACORN& hnsw,
        DistanceComputer& qdis,
        char* filter_map,
        // int filter,
        // Operation op,
        // std::string regex,
        int k,
        idx_t* I,
        float* D,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        ACORNStats& stats,
        int level,
        int nres_in = 0,
        const SearchParametersACORN* params = nullptr) {
    // debug("%s\n", "reached");
    // printf("----hybrid_search_from_candidates called with filter: %d, k: %d, op: %d, regex: %s\n", filter, k, op, regex.c_str());
    // debug_search("----hybrid_search_from_candidates called with filter: %d, k: %d\n", filter, k);
    int nres = nres_in;
    int ndis = 0;

    // can be overridden by search params
    bool do_dis_check = params ? params->check_relative_distance
                               : hnsw.check_relative_distance;
    int efSearch = params ? params->efSearch : hnsw.efSearch;
    const IDSelector* sel = params ? params->sel : nullptr;

    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        FAISS_ASSERT(v1 >= 0);
        if (!sel || sel->is_member(v1)) {
            if (nres < k) {
                faiss::maxheap_push(++nres, D, I, d, v1);
            } else if (d < D[0]) {
                faiss::maxheap_replace_top(nres, D, I, d, v1);
            }
        }
        vt.set(v1);
    }

    int nstep = 0;


    // timing variables
    double t1_candidates_loop = elapsed();
    
    while (candidates.size() > 0) { // candidates is heap of size max(efs, k)
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);
        // debug_search("--------visiting v0: %d, d0: %f, candidates_size: %d\n", v0, d0, candidates.size());

        if (do_dis_check) {
            // tricky stopping condition: there are more that ef
            // distances that are processed already that are smaller
            // than d0
            int n_dis_below = candidates.count_below(d0);
            if (n_dis_below >= efSearch) {
                // debug("--------%s\n", "n_dis_below >= efSearch BREAK cond reached");
                // debug_search("--------n_dis_below: %d, efSearch: %d - triggers break\n", n_dis_below, efSearch);
                break;
            }
        }

        size_t begin, end;
        hnsw.neighbor_range(v0, level, &begin, &end);

        // variable to keep track of search expansion
        int num_found = 0;
        int num_new = 0;
        bool keep_expanding = true;

        // for debugging, collect all neighbors looked at in a vector
        std::vector<std::pair<storage_idx_t, int>> neighbors_checked;

        double t1_neighbors_loop = elapsed();
        for (size_t j = begin; j < end; j++) {
            // auto [v1, metadata] = hnsw.neighbors[j];
            bool promising = 0;
            bool outerskip = false;

            auto v1 = hnsw.neighbors[j];
            // auto metadata = hnsw.metadata[v1];
            // debug_search("------------visiting neighbor (%ld) - %d, metadata: %d\n", j-begin, v1, metadata);


            if (v1 < 0) {
                break;
            }

            // note that this slows down search performance significantly
            // if (debugSearchFlag) {
            //     neighbors_checked.push_back(std::make_pair(v1, metadata)); // for debugging
            // }
            if (filter_map[v1]) {
               num_found = num_found + 1; // increment num found
            }
            
            if (vt.get(v1)) {
                continue;
            }


            // filter
            if (filter_map[v1]) {
                vt.set(v1);
                num_new = num_new + 1; // increment num new
                ndis++;
                float d = qdis(v1);
                // debug_search("------------new candidate %d, distance: %f\n", v1, d);

                if (!sel || sel->is_member(v1)) {
                    if (nres < k) {
                        // debug_search("-----------------pushing new candidate, nres: %d (to be incrd)\n", nres);
                        faiss::maxheap_push(++nres, D, I, d, v1);
                        // debug_search("-----------------pushed new candidate, nres: %d\n", nres);
                        promising = 1;
                    } else if (d < D[0]) {
                        // debug_search("-----------------replacing top, nres: %d\n", nres);
                        faiss::maxheap_replace_top(nres, D, I, d, v1);
                        promising =1;
                    }
                }
                candidates.push(v1, d);

                if (num_found >= hnsw.M * 2) {
                    // debug_search("------------num_found: %d, M: %d - triggered outer brea, skpping to M_beta=%d neighbork\n", num_found, hnsw.M * 2, hnsw.M_beta);
                    keep_expanding = false;
                    break;
                }
            }    
            
            if (((j - begin >= hnsw.M_beta) && keep_expanding) || hnsw.gamma == 1) {
                debug_search("------------expanding neighbor list for %d; neighbor %ld, hnsw.M_beta: %d\n", v1, j-begin, hnsw.M_beta);
                size_t begin2, end2;
                hnsw.neighbor_range(v1, level, &begin2, &end2);
                // try to parallelize neighbor expansion
                for (size_t j2 = begin2; j2 < end2; j2+=1) {
                    
                    auto v2 = hnsw.neighbors[j2];

                    // note that this slows down search performance significantly when flag is on
                    // if (debugSearchFlag) {
                    //     neighbors_checked.push_back(std::make_pair(v2, metadata2)); // for debugging
                    // }
                    if (v2 < 0) {
                        // continue;
                        break;
                    }

                    // if (metadata2 == filter) {
                    if (filter_map[v2]) {
                        num_found = num_found + 1; // increment num found
                    } else {
                        continue;
                    }

        

                    if (vt.get(v2)) {
                        continue;
                    }
                    
                    vt.set(v2);
                    ndis++;
  
                    float d2 = qdis(v2);
                    // debug_search("------------new candidate from expansion %d, distance: %f\n", v2, d2);
                    if (!sel || sel->is_member(v2)) {
                        if (nres < k) {
                            // debug_search("-----------------pushing new candidate, nres: %d (to be incrd)\n", nres);
                            faiss::maxheap_push(++nres, D, I, d2, v2);
                            // debug_search("-----------------pushed new candidate, nres: %d\n", nres);

                        } else if (d2 < D[0]) {
                            // debug_search("-----------------replacing top, nres: %d\n", nres);
                            faiss::maxheap_replace_top(nres, D, I, d2, v2);
                        }
                    }
                    candidates.push(v2, d2);
                    if (num_found >= hnsw.M * 2) {
    
                        // debug_search("------------num_found: %d, 2M: %d - triggers break\n", num_found, hnsw.M * 2);
                        keep_expanding = false;
                        break;
                    }
                }


    
            }
        
            
        }

     
        

        nstep++; 
        if (!do_dis_check && nstep > efSearch) {
            break;
        }
    }

    if (level == 0) {
        stats.n1++;
        if (candidates.size() == 0) {
            stats.n2++;
        }
        stats.n3 += ndis;
    }


    return nres;
}

*/






/*
 * Compare candidate distances with pointer tie-breaker
 */
static int
CompareCandidateDistances(const ListCell *a, const ListCell *b)
{
	HnswCandidate *hca = lfirst(a);
	HnswCandidate *hcb = lfirst(b);

	if (hca->distance < hcb->distance)
		return 1;

	if (hca->distance > hcb->distance)
		return -1;

	if (HnswPtrPointer(hca->element) < HnswPtrPointer(hcb->element))
		return 1;

	if (HnswPtrPointer(hca->element) > HnswPtrPointer(hcb->element))
		return -1;

	return 0;
}

/*
 * Compare candidate distances with offset tie-breaker
 */
static int
CompareCandidateDistancesOffset(const ListCell *a, const ListCell *b)
{
	HnswCandidate *hca = lfirst(a);
	HnswCandidate *hcb = lfirst(b);

	if (hca->distance < hcb->distance)
		return 1;

	if (hca->distance > hcb->distance)
		return -1;

	if (HnswPtrOffset(hca->element) < HnswPtrOffset(hcb->element))
		return 1;

	if (HnswPtrOffset(hca->element) > HnswPtrOffset(hcb->element))
		return -1;

	return 0;
}

/*
 * Check if an element is closer to q than any element from R
 */
static bool
CheckElementCloser(char *base, HnswCandidate * e, List *r, HnswSupport * support)
{
	HnswElement eElement = HnswPtrAccess(base, e->element);
	Datum		eValue = HnswGetValue(base, eElement);
	ListCell   *lc2;

	foreach(lc2, r)
	{
		HnswCandidate *ri = lfirst(lc2);
		HnswElement riElement = HnswPtrAccess(base, ri->element);
		Datum		riValue = HnswGetValue(base, riElement);
		float		distance = HnswGetDistance(eValue, riValue, support);

		if (distance <= e->distance)
			return false;
	}

	return true;
}

/*
 * Algorithm 4 from paper
 */
static List *
SelectNeighbors(char *base, List *c, int lm, HnswSupport * support, bool *closerSet, HnswCandidate * newCandidate, HnswCandidate * *pruned, bool sortCandidates)
{
	List	   *r = NIL;
	List	   *w = list_copy(c);
	HnswCandidate **wd;
	int			wdlen = 0;
	int			wdoff = 0;
	bool		mustCalculate = !(*closerSet);
	List	   *added = NIL;
	bool		removedAny = false;

	if (list_length(w) <= lm)
		return w;

	wd = palloc(sizeof(HnswCandidate *) * list_length(w));

	/* Ensure order of candidates is deterministic for closer caching */
	if (sortCandidates)
	{
		if (base == NULL)
			list_sort(w, CompareCandidateDistances);
		else
			list_sort(w, CompareCandidateDistancesOffset);
	}

	while (list_length(w) > 0 && list_length(r) < lm)
	{
		/* Assumes w is already ordered desc */
		HnswCandidate *e = llast(w);

		w = list_delete_last(w);

		/* Use previous state of r and wd to skip work when possible */
		if (mustCalculate)
			e->closer = CheckElementCloser(base, e, r, support);
		else if (list_length(added) > 0)
		{
			/* Keep Valgrind happy for in-memory, parallel builds */
			if (base != NULL)
				VALGRIND_MAKE_MEM_DEFINED(&e->closer, 1);

			/*
			 * If the current candidate was closer, we only need to compare it
			 * with the other candidates that we have added.
			 */
			if (e->closer)
			{
				e->closer = CheckElementCloser(base, e, added, support);

				if (!e->closer)
					removedAny = true;
			}
			else
			{
				/*
				 * If we have removed any candidates from closer, a candidate
				 * that was not closer earlier might now be.
				 */
				if (removedAny)
				{
					e->closer = CheckElementCloser(base, e, r, support);
					if (e->closer)
						added = lappend(added, e);
				}
			}
		}
		else if (e == newCandidate)
		{
			e->closer = CheckElementCloser(base, e, r, support);
			if (e->closer)
				added = lappend(added, e);
		}

		/* Keep Valgrind happy for in-memory, parallel builds */
		if (base != NULL)
			VALGRIND_MAKE_MEM_DEFINED(&e->closer, 1);

		if (e->closer)
			r = lappend(r, e);
		else
			wd[wdlen++] = e;
	}

	/* Cached value can only be used in future if sorted deterministically */
	*closerSet = sortCandidates;

	/* Keep pruned connections */
	while (wdoff < wdlen && list_length(r) < lm)
		r = lappend(r, wd[wdoff++]);

	/* Return pruned for update connections */
	if (pruned != NULL)
	{
		if (wdoff < wdlen)
			*pruned = wd[wdoff];
		else
			*pruned = linitial(w);
	}

	return r;
}

/*
 * Add connections
 */
static void
AddConnections(char *base, HnswElement element, List *neighbors, int lc)
{
	ListCell   *lc2;
	HnswNeighborArray *a = HnswGetNeighbors(base, element, lc);

	foreach(lc2, neighbors)
		a->items[a->length++] = *((HnswCandidate *) lfirst(lc2));
}

/*
 * Update connections
 */
void
HnswUpdateConnection(char *base, HnswNeighborArray * neighbors, HnswElement newElement, float distance, int lm, int *updateIdx, Relation index, HnswSupport * support)
{
	HnswCandidate newHc;

	HnswPtrStore(base, newHc.element, newElement);
	newHc.distance = distance;

	if (neighbors->length < lm)
	{
		neighbors->items[neighbors->length++] = newHc;

		/* Track update */
		if (updateIdx != NULL)
			*updateIdx = -2;
	}
	else
	{
		/* Shrink connections */
		List	   *c = NIL;
		HnswCandidate *pruned = NULL;

		/* Add candidates */
		for (int i = 0; i < neighbors->length; i++)
			c = lappend(c, &neighbors->items[i]);
		c = lappend(c, &newHc);

		SelectNeighbors(base, c, lm, support, &neighbors->closerSet, &newHc, &pruned, true);

		/* Should not happen */
		if (pruned == NULL)
			return;

		/* Find and replace the pruned element */
		for (int i = 0; i < neighbors->length; i++)
		{
			if (HnswPtrEqual(base, neighbors->items[i].element, pruned->element))
			{
				neighbors->items[i] = newHc;

				/* Track update */
				if (updateIdx != NULL)
					*updateIdx = i;

				break;
			}
		}
	}
}

/*
 * Remove elements being deleted or skipped
 */
static List *
RemoveElements(char *base, List *w, HnswElement skipElement)
{
	ListCell   *lc2;
	List	   *w2 = NIL;

	/* Ensure does not access heaptidsLength during in-memory build */
	pg_memory_barrier();

	foreach(lc2, w)
	{
		HnswCandidate *hc = (HnswCandidate *) lfirst(lc2);
		HnswElement hce = HnswPtrAccess(base, hc->element);

		/* Skip self for vacuuming update */
		if (skipElement != NULL && hce->blkno == skipElement->blkno && hce->offno == skipElement->offno)
			continue;

		if (hce->heaptidsLength != 0)
			w2 = lappend(w2, hc);
	}

	return w2;
}

/*
 * Precompute hash
 */
static void
PrecomputeHash(char *base, HnswElement element)
{
	HnswElementPtr ptr;

	HnswPtrStore(base, ptr, element);

	if (base == NULL)
		element->hash = hash_pointer((uintptr_t) HnswPtrPointer(ptr));
	else
		element->hash = hash_offset(HnswPtrOffset(ptr));
}

/*
 * Algorithm 1 from paper
 */
void
HnswFindElementNeighbors(char *base, HnswElement element, HnswElement entryPoint, Relation index, HnswSupport * support, int m, int efConstruction, bool existing)
{
	List	   *ep;
	List	   *w;
	int			level = element->level;
	int			entryLevel;
	HnswQuery	q;
	HnswElement skipElement = existing ? element : NULL;
	bool		inMemory = index == NULL;

	q.value = HnswGetValue(base, element);

	/* Precompute hash */
	if (inMemory)
		PrecomputeHash(base, element);

	/* No neighbors if no entry point */
	if (entryPoint == NULL)
		return;

	/* Get entry point and level */
	ep = list_make1(HnswEntryCandidate(base, entryPoint, &q, index, support, true));
	entryLevel = entryPoint->level;

	/* 1st phase: greedy search to insert level */
	for (int lc = entryLevel; lc >= level + 1; lc--)
	{
		w = HnswSearchLayer(base, &q, ep, 1, lc, index, support, m, true, skipElement, NULL, NULL, true, NULL);
		ep = w;
	}

	if (level > entryLevel)
		level = entryLevel;

	/* Add one for existing element */
	if (existing)
		efConstruction++;

	/* 2nd phase */
	for (int lc = level; lc >= 0; lc--)
	{
		int			lm = HnswGetLayerM(m, lc);
		List	   *neighbors;
		List	   *lw = NIL;
		ListCell   *lc2;

		w = HnswSearchLayer(base, &q, ep, efConstruction, lc, index, support, m, true, skipElement, NULL, NULL, true, NULL);

		/* Convert search candidates to candidates */
		foreach(lc2, w)
		{
			HnswSearchCandidate *sc = lfirst(lc2);
			HnswCandidate *hc = palloc(sizeof(HnswCandidate));

			hc->element = sc->element;
			hc->distance = sc->distance;

			lw = lappend(lw, hc);
		}

		/* Elements being deleted or skipped can help with search */
		/* but should be removed before selecting neighbors */
		if (!inMemory)
			lw = RemoveElements(base, lw, skipElement);

		/*
		 * Candidates are sorted, but not deterministically. Could set
		 * sortCandidates to true for in-memory builds to enable closer
		 * caching, but there does not seem to be a difference in performance.
		 */
		neighbors = SelectNeighbors(base, lw, lm, support, &HnswGetNeighbors(base, element, lc)->closerSet, NULL, NULL, false);

		AddConnections(base, element, neighbors, lc);

		ep = w;
	}
}

PGDLLEXPORT Datum l2_normalize(PG_FUNCTION_ARGS);
PGDLLEXPORT Datum halfvec_l2_normalize(PG_FUNCTION_ARGS);
PGDLLEXPORT Datum sparsevec_l2_normalize(PG_FUNCTION_ARGS);

static void
SparsevecCheckValue(Pointer v)
{
	SparseVector *vec = (SparseVector *) v;

	if (vec->nnz > HNSW_MAX_NNZ)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("sparsevec cannot have more than %d non-zero elements for hnsw index", HNSW_MAX_NNZ)));
}

/*
 * Get type info
 */
const		HnswTypeInfo *
HnswGetTypeInfo(Relation index)
{
	FmgrInfo   *procinfo = HnswOptionalProcInfo(index, HNSW_TYPE_INFO_PROC);

	if (procinfo == NULL)
	{
		static const HnswTypeInfo typeInfo = {
			.maxDimensions = HNSW_MAX_DIM,
			.normalize = l2_normalize,
			.checkValue = NULL
		};

		return (&typeInfo);
	}
	else
		return (const HnswTypeInfo *) DatumGetPointer(FunctionCall0Coll(procinfo, InvalidOid));
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hnsw_halfvec_support);
Datum
hnsw_halfvec_support(PG_FUNCTION_ARGS)
{
	static const HnswTypeInfo typeInfo = {
		.maxDimensions = HNSW_MAX_DIM * 2,
		.normalize = halfvec_l2_normalize,
		.checkValue = NULL
	};

	PG_RETURN_POINTER(&typeInfo);
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hnsw_bit_support);
Datum
hnsw_bit_support(PG_FUNCTION_ARGS)
{
	static const HnswTypeInfo typeInfo = {
		.maxDimensions = HNSW_MAX_DIM * 32,
		.normalize = NULL,
		.checkValue = NULL
	};

	PG_RETURN_POINTER(&typeInfo);
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hnsw_sparsevec_support);
Datum
hnsw_sparsevec_support(PG_FUNCTION_ARGS)
{
	static const HnswTypeInfo typeInfo = {
		.maxDimensions = SPARSEVEC_MAX_DIM,
		.normalize = sparsevec_l2_normalize,
		.checkValue = SparsevecCheckValue
	};

	PG_RETURN_POINTER(&typeInfo);
}
