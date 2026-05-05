"""
Filtered-Score Streamer (FSS) — core data-access operator for DASE.

FSS(table, P, s) → Stream((entry, score))

Given a table, a set of predicates P, and a scoring signal s, produces a
sorted stream of entries that pass P in ascending score (best-first) order.

Scoring signal types:
  1. Semantic:    dist(query_vec, row_embedding)
  2. Relational:  "dis" column (precomputed join distance in TI table)
  3. Attribute:   scalar column value

Execution strategies (auto-selected by joint predicate selectivity σ):
  Semantic signals:
    - native_where_orderby: σ < sigma_low → single SQL `WHERE P ORDER BY score LIMIT K`,
      letting pgvector planner pick bitmap scan or iterative HNSW.
    - score_first:          σ ≥ sigma_low → HNSW scan by score (ef_search), post-filter in Python.
  Non-semantic signals (relational/attribute):
    - attribute_first:  low σ  → filter via B-tree then brute-force score
    - score_first:      high σ → scan by score, post-filter in Python
    - predicate_aware:  mid σ  → filtered HNSW traversal (bitmap) [semantic only]

Caller is responsible for ensuring all predicates can be applied on the table.
"""

import time as _time
from typing import Any, Dict, List, Optional, Tuple

import psycopg
from psycopg.rows import dict_row
from pgvector import Vector

from ours.utils import (
    METRIC_OP, SQL_OPS,
    quote as _quote,
    check_predicate as _check_predicate,
    resolve_predicate_ids,
    get_max_id,
    make_bitmap,
    filtered_hnsw_search,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Vector (embedding) column names for the active dataset. Set by
# ours.sys.main() based on workload["dataset"]. light_mode excludes these
# from SELECT to avoid pulling large embeddings.
DATASET_VEC_COLS: set = {"title_emb", "plot_emb", "actor_director_emb"}

# Default selectivity thresholds (overridden by table_meta.json fss_strategy)
SIGMA_LOW_DEFAULT = 0.05
SIGMA_HIGH_DEFAULT = 0.80

# Score-first: initial and max probe sizes
SCORE_FIRST_INIT = 200
SCORE_FIRST_MAX = 100_000
HNSW_EF_SEARCH_MAX = 1000

# Default internal batch size
DEFAULT_BATCH = 1000


# ---------------------------------------------------------------------------
# Table metadata (loaded from <dataset>_data/table_meta.json, falls back to DB)
# ---------------------------------------------------------------------------
_table_meta: Optional[Dict[str, Any]] = None
_active_dataset: Optional[str] = None
_table_size_cache: Dict[str, int] = {}
_selectivity_cache: Dict[str, float] = {}  # key: "table|pred_fingerprint"


def _load_table_meta(dataset: Optional[str] = None) -> Dict[str, Any]:
    """Load precomputed table metadata for the given dataset (default: imdb).

    Cached per-dataset; switching datasets clears the size/selectivity caches.
    """
    global _table_meta, _active_dataset
    if dataset is None:
        dataset = _active_dataset or "imdb"
    if _table_meta is not None and _active_dataset == dataset:
        return _table_meta
    _table_size_cache.clear()
    _selectivity_cache.clear()
    _active_dataset = dataset
    import os, json
    meta_path = os.path.join(os.path.dirname(__file__), "..", f"{dataset}_data", "table_meta.json")
    meta_path = os.path.normpath(meta_path)
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            _table_meta = json.load(f)
        for tname, sz in _table_meta.get("table_sizes", {}).items():
            _table_size_cache[tname] = sz
    else:
        _table_meta = {}
    return _table_meta


# ---------------------------------------------------------------------------
# FilteredScoreStreamer
# ---------------------------------------------------------------------------

class FilteredScoreStreamer:
    """
    FSS(table, P, s) → Stream((entry, score))

    Produces entries from `table` satisfying predicates P, ordered by
    scoring signal s.

    Interface:
        next()       → (entry_dict, score) | None
        peek_score() → float | None
        fetch_topk(k) → list[(entry_dict, score)]

    Strategy is chosen internally during __init__ based on estimated
    joint selectivity of P. The caller does NOT control strategy.
    """

    def __init__(
        self,
        conn: psycopg.Connection,
        table: str,
        scoring_signal: Dict[str, Any],
        predicates: Optional[List[Dict[str, Any]]] = None,
        init_stream_fetchK: Optional[int] = None,
        precomputed_selectivity: Optional[float] = None,
        batch_size: Optional[int] = None,
        light_mode: bool = False,
        query_k: int = 20,
        fss_pf: float = 1.0,
        keep_vec_cols: Optional[List[str]] = None,
        id_col: str = "id",
        text_col: str = "title",
    ):
        """
        Args:
            conn: Active psycopg connection (pgvector registered).

            table: The table to scan. Can be an entity table ("imdb_t1")
                or a TI table ("ti_imdb_t1_imdb_t2_0.5"). Caller ensures
                all predicates and the scoring signal apply to this table.

            scoring_signal: Describes how to compute the score.
                Semantic:
                    {"type": "semantic", "field": "plot_emb",
                     "query_embed": [...], "metric": "l2"}
                Relational:
                    {"type": "relational", "field": "dis"}
                Attribute:
                    {"type": "attribute", "field": "rating",
                     "direction": "asc"}

            predicates: List of predicate dicts. Each dict:
                {"attribute": "year", "operator": ">=", "value": 1985}
                {"attribute": "id", "operator": "in", "value": [1,2,3]}
                Caller ensures all predicates can be applied on `table`.

            precomputed_selectivity: If provided, skip selectivity estimation
                and use this value directly for strategy selection.

            query_k: The query top-K, used for IVFFlat probe calculation.

            fss_pf: Probe factor multiplier: n_probes = probe_rule(K) * fss_pf.
        """
        self.conn = conn
        self.table = table
        self.scoring_signal = scoring_signal
        self.predicates = predicates or []
        self._init_stream_fetchK = init_stream_fetchK
        self._precomputed_selectivity = precomputed_selectivity
        self._batch_size = batch_size or DEFAULT_BATCH
        self._light_mode = light_mode  # True → exclude vector cols from SELECT
        self._keep_vec_cols = set(keep_vec_cols or [])
        self._query_k = query_k
        self._fss_pf = fss_pf
        self._id_col = id_col
        self._text_col = text_col

        self._sig_type = scoring_signal["type"]

        # Internal stream buffer
        self._buffer: List[Tuple[Dict[str, Any], float]] = []
        self._buf_pos: int = 0
        self._exhausted: bool = False

        # Non-vector column names (cached for attribute_first SELECT)
        self._non_vec_cols: Optional[str] = None

        # Strategy-specific state (set by _init_strategy)
        self._strategy: str = ""
        self._offset: int = 0           # SQL OFFSET for attribute_first
        self._fetch_K: int = 0       # current probe size for score_first
        self._score_first_offset: int = 0  # rows already returned in score_first

        # Profiling accumulators
        self.profile = {
            "init_table_size": 0.0,
            "init_selectivity": 0.0,
            "fetch_db_query": 0.0,
            "fetch_postfilter": 0.0,
            "fetch_n_batches": 0,
            "unfilter_fetch_k": 0,            # last LIMIT used on unfiltered stream
            "filtered_stream_length": 0,       # total filtered rows produced
            "cursor_on_filtered_stream": 0,    # filtered rows consumed by caller
        }

        # Load precomputed metadata (no-op after first call)
        _load_table_meta()

        # Choose strategy based on selectivity
        self._init_strategy()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def strategy(self) -> str:
        """The execution strategy selected for this FSS instance."""
        return self._strategy

    def next(self) -> Optional[Tuple[Dict[str, Any], float]]:
        """Return next (entry, score), or None if exhausted."""
        if self._buf_pos < len(self._buffer):
            item = self._buffer[self._buf_pos]
            self._buf_pos += 1
            self.profile["cursor_on_filtered_stream"] += 1
            return item
        while not self._exhausted:
            self._fetch_more()
            if self._buffer:
                self._buf_pos = 1
                self.profile["cursor_on_filtered_stream"] += 1
                return self._buffer[0]
        return None

    def peek_score(self) -> Optional[float]:
        """Score of the next entry without consuming it. None if exhausted."""
        if self._buf_pos < len(self._buffer):
            return self._buffer[self._buf_pos][1]
        while not self._exhausted:
            self._fetch_more()
            if self._buffer:
                return self._buffer[0][1]
        return None

    def fetch_topk(self, k: int) -> List[Tuple[Dict[str, Any], float]]:
        """Pull up to k entries from the stream."""
        results = []
        while len(results) < k:
            item = self.next()
            if item is None:
                break
            results.append(item)
        return results

    # ------------------------------------------------------------------
    # Strategy selection
    # ------------------------------------------------------------------

    def _init_strategy(self):
        """Estimate joint selectivity and pick execution strategy."""
        if self._precomputed_selectivity is not None:
            sigma = self._precomputed_selectivity
        else:
            sigma = self._estimate_selectivity()
        self.profile["sigma"] = float(sigma)
        self.profile["n_rows"] = self._get_table_size()

        # Read dataset-specific thresholds from table_meta, fall back to defaults
        fss_cfg = (_table_meta or {}).get("fss_strategy", {})
        sigma_low = fss_cfg.get("sigma_low", SIGMA_LOW_DEFAULT)
        sigma_high = fss_cfg.get("sigma_high", SIGMA_HIGH_DEFAULT)

        if self._sig_type == "semantic":
            # W2-style binary dispatch at sigma_low:
            #   σ < sigma_low → native WHERE+ORDER BY (pgvector picks bitmap/HNSW iter)
            #   σ ≥ sigma_low → HNSW scan + Python post-filter (score_first_hnsw)
            if sigma < sigma_low:
                self._strategy = "native_where_orderby"
            else:
                self._strategy = "score_first"
                self._fetch_K = self._init_stream_fetchK or SCORE_FIRST_INIT
                # Keep HNSW table (no IVFFlat conversion)
        else:
            # Non-semantic (relational/attribute): legacy 3-way dispatch
            if sigma < sigma_low:
                self._strategy = "attribute_first"
            elif sigma > sigma_high:
                self._strategy = "score_first"
                self._fetch_K = self._init_stream_fetchK or SCORE_FIRST_INIT
            else:
                self._strategy = "attribute_first"
            # TI (join) tables have composite key (t1.id, t2.id) — no single `id`
            # column for attribute_first's 2-step fetch. Use native_where_orderby
            # which runs a single WHERE+ORDER BY over SELECT *.
            # NOTE: For TI+relational, callers should prefer `TIScoreStream`
            # (ours.ti_stream) which streams in dis-btree range chunks and
            # avoids PG's top-N heapsort pathology. This branch remains as a
            # safety fallback.
            if self._strategy == "attribute_first" and self.table.startswith("ti_"):
                self._strategy = "native_where_orderby"

        self.profile["strategy"] = self._strategy

    @staticmethod
    def _to_ivf_table(table: str) -> str:
        """Map table name to IVFFlat variant (e.g., imdb_t1_hnsw -> imdb_t1_ivf)."""
        base = table.replace("_hnsw", "").replace("_ivf", "")
        return f"{base}_ivf"

    def _estimate_selectivity(self) -> float:
        """
        Estimate the joint selectivity σ of all predicates on the table.

        σ = (rows passing all P) / N

        For "in" predicates on id, we use |list|/N without querying.
        For structural predicates, we run a COUNT(*) with WHERE.
        """
        n = self._get_table_size()
        if n == 0:
            return 0.0

        if not self.predicates:
            return 1.0

        structural_preds = []
        id_in_sel = 1.0

        for p in self.predicates:
            if p["operator"] == "in" and p["attribute"] == "id":
                id_in_sel *= len(p["value"]) / n
            else:
                structural_preds.append(p)

        if structural_preds:
            # Build cache key from table + sorted predicate fingerprint
            cache_key = self.table + "|" + "|".join(
                f"{p['attribute']}{p['operator']}{p['value']}" for p in sorted(structural_preds, key=lambda p: p["attribute"])
            )
            if cache_key in _selectivity_cache:
                struct_sel = _selectivity_cache[cache_key]
            else:
                # Try precomputed TI selectivity from metadata
                struct_sel = self._lookup_meta_selectivity(structural_preds)
                if struct_sel is None:
                    where_parts, params = self._build_where(structural_preds)
                    sql = f"SELECT count(*) FROM {_quote(self.table)} WHERE {' AND '.join(where_parts)}"
                    t0 = _time.perf_counter()
                    with self.conn.cursor(row_factory=psycopg.rows.tuple_row) as cur:
                        cur.execute(sql, params)
                        struct_sel = cur.fetchone()[0] / n
                    self.profile["init_selectivity"] += _time.perf_counter() - t0
                _selectivity_cache[cache_key] = struct_sel
        else:
            struct_sel = 1.0

        return struct_sel * id_in_sel

    def _lookup_meta_selectivity(self, preds: List[Dict[str, Any]]) -> Optional[float]:
        """Look up precomputed selectivity from table_meta.json for TI dis predicates."""
        if _table_meta is None:
            return None
        ti_sel = _table_meta.get("ti_selectivity", {}).get(self.table)
        if ti_sel is None:
            return None
        non_dis_preds = [p for p in preds if p["attribute"] != "dis"]
        if non_dis_preds:
            return None
        for p in preds:
            if p["attribute"] == "dis" and p["operator"] == "<=":
                tau_key = str(p["value"])
                if tau_key in ti_sel:
                    return ti_sel[tau_key]
        return None

    def _get_table_size(self) -> int:
        if self.table in _table_size_cache:
            return _table_size_cache[self.table]
        t0 = _time.perf_counter()
        with self.conn.cursor(row_factory=psycopg.rows.tuple_row) as cur:
            cur.execute(f"SELECT count(*) FROM {_quote(self.table)}")
            result = cur.fetchone()[0]
        self.profile["init_table_size"] += _time.perf_counter() - t0
        _table_size_cache[self.table] = result
        return result

    # ------------------------------------------------------------------
    # WHERE clause builder
    # ------------------------------------------------------------------

    def _build_where(
        self, preds: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Build SQL WHERE parts from predicates."""
        if preds is None:
            preds = self.predicates
        parts: List[str] = []
        params: Dict[str, Any] = {}
        for i, p in enumerate(preds):
            attr, op, val = p["attribute"], p["operator"], p["value"]
            key = f"p{i}"
            if op == "in":
                parts.append(f'{_quote(attr)} = ANY(%({key})s)')
                params[key] = list(val) if not isinstance(val, list) else val
            elif op in SQL_OPS:
                parts.append(f'{_quote(attr)} {op} %({key})s')
                params[key] = val
            else:
                raise ValueError(f"Unsupported predicate operator: {op}")
        return parts, params

    def _score_first_check_predicates(self, row: Dict[str, Any]) -> bool:
        """Python-side predicate check (used by score_first post-filter)."""
        for p in self.predicates:
            attr = p["attribute"]
            if not _check_predicate(row, attr, p["operator"], p["value"]):
                return False
        return True

    # ------------------------------------------------------------------
    # _fetch_more: refills the buffer according to strategy
    # ------------------------------------------------------------------

    def _fetch_more(self):
        """Fetch the next batch into self._buffer. Dispatches by strategy."""
        self._buffer = []
        self._buf_pos = 0

        dispatch = {
            "attribute_first": self._fetch_attribute_first,
            "score_first": self._fetch_score_first,
            "predicate_aware": self._fetch_predicate_aware,
            "native_where_orderby": self._fetch_native_where_orderby,
        }
        dispatch[self._strategy]()
        self.profile["fetch_n_batches"] += 1

    # -- attribute_first: push predicates to SQL WHERE, ORDER BY score -

    def _fetch_attribute_first(self):
        # Step 1: filter via B-tree — fetch only IDs
        where_parts, where_params = self._build_where()
        where_sql = " AND ".join(where_parts) if where_parts else "TRUE"

        sql_ids = (
            f"SELECT {_quote(self._id_col)} FROM {_quote(self.table)} "
            f"WHERE {where_sql}"
        )
        t0 = _time.perf_counter()
        with self.conn.cursor(row_factory=psycopg.rows.tuple_row) as cur:
            cur.execute(sql_ids, where_params)
            id_rows = cur.fetchall()
        self.profile["fetch_postfilter"] += _time.perf_counter() - t0

        if not id_rows:
            self._exhausted = True
            return

        ids = [r[0] for r in id_rows]

        # Step 2: score + sort only the filtered rows (fetch ids + scores first)
        score_expr, score_params = self._score_expression()
        score_params["_af_ids"] = ids
        score_params["lim"] = self._batch_size
        score_params["off"] = self._offset

        if self._light_mode:
            # Exclude vector columns to avoid pulling large embeddings.
            # keep_vec_cols overrides exclusion for specific vectors (e.g. the
            # join embedding is needed downstream for cross-product distance).
            if self._non_vec_cols is None:
                _vec_cols = DATASET_VEC_COLS - self._keep_vec_cols
                with self.conn.cursor(row_factory=psycopg.rows.tuple_row) as cur:
                    cur.execute(
                        "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
                        [self.table],
                    )
                    all_cols = [r[0] for r in cur.fetchall()]
                self._non_vec_cols = ", ".join(_quote(c) for c in all_cols if c not in _vec_cols)
            select_expr = self._non_vec_cols
        else:
            select_expr = "*"

        sql_score = (
            f"SELECT {select_expr}, {score_expr} AS _score "
            f"FROM {_quote(self.table)} "
            f"WHERE {_quote(self._id_col)} = ANY(%(_af_ids)s) "
            f"ORDER BY {score_expr} "
            f"LIMIT %(lim)s OFFSET %(off)s"
        )
        t0 = _time.perf_counter()
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SET LOCAL enable_indexscan = off")
            cur.execute(sql_score, score_params)
            rows = cur.fetchall()
        self.profile["fetch_db_query"] += _time.perf_counter() - t0

        self._buffer = [(dict(r), float(r["_score"])) for r in rows]
        self._offset += len(rows)
        self.profile["unfilter_fetch_k"] = len(ids)  # all filtered IDs = no unfiltered scan
        self.profile["filtered_stream_length"] += len(rows)
        if len(rows) < self._batch_size:
            self._exhausted = True

    # -- native_where_orderby: single-SQL WHERE P ORDER BY score LIMIT --

    def _fetch_native_where_orderby(self):
        """Single SQL: `WHERE P ORDER BY score LIMIT K`. Planner picks plan.

        For low-σ semantic queries, pgvector's planner typically chooses a
        Bitmap Heap Scan on B-tree predicate indexes + top-N sort, which is
        faster than a 2-step filter_first (no redundant round-trip). At the
        boundary, the planner may switch to iterative HNSW on its own.
        """
        where_parts, params = self._build_where()
        where_sql = " AND ".join(where_parts) if where_parts else "TRUE"
        score_expr, score_params = self._score_expression()
        params.update(score_params)
        params["lim"] = self._batch_size
        params["off"] = self._offset

        if self._light_mode:
            if self._non_vec_cols is None:
                _vec_cols = DATASET_VEC_COLS - self._keep_vec_cols
                with self.conn.cursor(row_factory=psycopg.rows.tuple_row) as cur:
                    cur.execute(
                        "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
                        [self.table],
                    )
                    all_cols = [r[0] for r in cur.fetchall()]
                self._non_vec_cols = ", ".join(_quote(c) for c in all_cols if c not in _vec_cols)
            select_expr = self._non_vec_cols
        else:
            select_expr = "*"

        sql = (
            f"SELECT {select_expr}, {score_expr} AS _score "
            f"FROM {_quote(self.table)} "
            f"WHERE {where_sql} "
            f"ORDER BY {score_expr} "
            f"LIMIT %(lim)s OFFSET %(off)s"
        )
        t0 = _time.perf_counter()
        with self.conn.cursor(row_factory=dict_row) as cur:
            # Hint HNSW ef_search in case planner picks iterative HNSW
            ef = min(max(self._batch_size, self._query_k), HNSW_EF_SEARCH_MAX)
            cur.execute(f"SET LOCAL hnsw.ef_search = {ef}")
            cur.execute(sql, params)
            rows = cur.fetchall()
        self.profile["fetch_db_query"] += _time.perf_counter() - t0

        self._buffer = [(dict(r), float(r["_score"])) for r in rows]
        self._offset += len(rows)
        self.profile["unfilter_fetch_k"] = len(rows)
        self.profile["filtered_stream_length"] += len(rows)
        if len(rows) < self._batch_size:
            self._exhausted = True

    # -- score_first: index scan by score, post-filter in Python -------

    def _fetch_score_first(self):
        score_expr, params = self._score_expression()

        params["lim"] = self._fetch_K
        if self._light_mode:
            # Keep score_first close to rerank_w4: fetch only id/title/predicate cols.
            # keep_vec_cols lets the caller retain specific vector columns
            # (e.g., the W7 join embedding needed for cross-product distance).
            pred_cols = []
            seen = set()
            for p in self.predicates:
                attr = p["attribute"]
                if attr in (self._id_col, self._text_col) or attr in seen:
                    continue
                seen.add(attr)
                pred_cols.append(attr)
            extra_cols_list = pred_cols + [c for c in self._keep_vec_cols if c not in pred_cols]
            extra_cols = ", ".join(f"{_quote(c)}" for c in extra_cols_list)
            select_expr = f'{_quote(self._id_col)}, {_quote(self._text_col)}{(", " + extra_cols) if extra_cols else ""}'
        else:
            select_expr = "*"
        sql = (
            f"SELECT {select_expr}, {score_expr} AS _score "
            f"FROM {_quote(self.table)} "
            f"ORDER BY {score_expr} "
            f"LIMIT %(lim)s"
        )
        t0 = _time.perf_counter()
        with self.conn.cursor(row_factory=dict_row) as cur:
            if self._sig_type == "semantic":
                if self.table.endswith("_hnsw"):
                    # HNSW: ef_search = fetch_K (clamped to cap). Note: enable_seqscan=off
                    # must be set by caller at conn init — without it the planner picks
                    # Seq Scan + top-N heapsort (cheaper cost model at LIMIT~200) which
                    # runs ~240ms vs HNSW's ~15ms due to heap I/O on wide rows.
                    ef = min(max(self._fetch_K, self._query_k), HNSW_EF_SEARCH_MAX)
                    cur.execute(f"SET hnsw.ef_search = {ef}")
                else:
                    # IVFFlat: n_probes = probe_rule(probe_size) * fss_pf
                    from baselines import compute_ivfflat_probes, assert_ivfflat_exists
                    probes = compute_ivfflat_probes(self._fetch_K, probe_factor=self._fss_pf)
                    cur.execute(f"SET ivfflat.probes = {probes}")
                    if self.table.endswith("_ivf"):
                        field = self.scoring_signal["field"]
                        assert_ivfflat_exists(self.conn, self.table.replace("_ivf", ""), field)
                        idx_name = f"idx_ivf_{self.table}_{field}"
                        sql = f"/*+ IndexScan(t {idx_name}) */ {sql}"
            cur.execute(sql, params)
            rows = cur.fetchall()
        self.profile["fetch_db_query"] += _time.perf_counter() - t0

        # Skip already-returned rows, only buffer new ones
        t0 = _time.perf_counter()
        buf = []
        for r in rows[self._score_first_offset:]:
            if not self._score_first_check_predicates(r):
                continue
            buf.append((dict(r), float(r["_score"])))
        self.profile["fetch_postfilter"] += _time.perf_counter() - t0

        self._score_first_offset = len(rows)
        self._buffer = buf
        self.profile["unfilter_fetch_k"] = self._fetch_K
        self.profile["filtered_stream_length"] += len(buf)

        if len(rows) < self._fetch_K:
            self._exhausted = True
        else:
            cap = HNSW_EF_SEARCH_MAX if (self._sig_type == "semantic" and self.table.endswith("_hnsw")) else SCORE_FIRST_MAX
            if self._fetch_K >= cap:
                self._exhausted = True
            else:
                self._fetch_K = min(self._fetch_K * 2, cap)

        # Fallback: score_first exhausted without producing ANY predicate-matching
        # row. This happens when predicate direction is anti-correlated with the
        # scoring signal (the top-cap nearest neighbors all violate predicates).
        # Switch to native_where_orderby: WHERE preds ORDER BY score — exact top-K
        # over the filtered subset via planner's choice (bitmap/seqscan + top-N).
        if self._exhausted and self.profile["filtered_stream_length"] == 0:
            self._strategy = "native_where_orderby"
            self._exhausted = False
            self._offset = 0
            self.profile["strategy"] = "score_first->native_where_orderby"

    # -- predicate_aware: filtered HNSW traversal ----------------------

    def _fetch_predicate_aware(self):
        assert False, "predicate_aware is dead code: _init_strategy never selects it"
        """
        Filtered HNSW via the custom <-># operator.

        Delegates to the filter search module in utils:
          1. resolve_predicate_ids → valid ID set
          2. make_bitmap → bitmap filter
          3. filtered_hnsw_search → filtered HNSW query

        Results are already sorted by score; no post-filtering needed
        (bitmap has no false positives).
        """
        if self._sig_type != "semantic":
            self._fetch_attribute_first()
            return

        sig = self.scoring_signal

        t0 = _time.perf_counter()
        valid_ids = resolve_predicate_ids(self.conn, self.table, self.predicates)
        max_id = get_max_id(self.conn, self.table)
        bitmap = make_bitmap(valid_ids, max_id)
        self.profile["fetch_postfilter"] += _time.perf_counter() - t0

        limit = max(DEFAULT_BATCH, SCORE_FIRST_INIT)
        t0 = _time.perf_counter()
        self._buffer = filtered_hnsw_search(
            self.conn, self.table, sig["field"], sig["query_embed"],
            sig.get("metric", "l2"), bitmap, limit, self._offset,
        )
        self.profile["fetch_db_query"] += _time.perf_counter() - t0

        self._offset += len(self._buffer)
        self.profile["unfilter_fetch_k"] = limit  # filtered HNSW, no unfiltered scan
        self.profile["filtered_stream_length"] += len(self._buffer)
        if len(self._buffer) < limit:
            self._exhausted = True

    # ------------------------------------------------------------------
    # Score expression builder (unified across strategies)
    # ------------------------------------------------------------------

    def _score_expression(self) -> Tuple[str, Dict[str, Any]]:
        """
        Build the SQL expression for the scoring signal.

        Returns (expr_str, params) where expr_str is a SQL fragment
        and params is a dict of query parameters it depends on.
        """
        sig = self.scoring_signal
        params: Dict[str, Any] = {}

        if self._sig_type == "semantic":
            field = sig["field"]
            metric = sig.get("metric", "l2")
            dist_op = METRIC_OP[metric]
            if metric == "jaccard":
                qe = sig["query_embed"]
                params["qv"] = "".join(str(int(x)) for x in qe)
                cast = f"::bit({len(qe)})"
                return f'({_quote(field)} {dist_op.replace("%", "%%")} %(qv)s{cast})', params
            params["qv"] = Vector(sig["query_embed"])
            return f'({_quote(field)} {dist_op} %(qv)s)', params

        elif self._sig_type == "relational":
            field = sig.get("field", "dis")
            return _quote(field), params

        elif self._sig_type == "attribute":
            field = sig["field"]
            direction = sig.get("direction", "asc").upper()
            # Direction is handled in ORDER BY, not in the expression
            # But for score_first we need consistent ordering
            if direction == "DESC":
                return f'(-1 * {_quote(field)})', params
            return _quote(field), params

        raise ValueError(f"Unknown signal type: {self._sig_type}")
