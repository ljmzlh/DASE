"""
DASE System — main entry point.

Routes workload queries to the appropriate execution pattern:

  Group A: Intra-Entity (single table, no join)
    W1(s, K)       = TopK{FSS(∅, s)}
    W2(s, P, K)    = TopK{FSS(P, s)}
    W3(S, K)       = MSA({FSS(∅, s_i)}, score_f, K)
    W4(S, P, K)    = MSA({FSS(P, s_i)}, score_f, K)

  Group B: Inter-Entity (cross-table join via TI)
    W5(s, J, K)    = TopK{FSS(P_J, s)}
    W6(s, P, J, K) = TopK{FSS(P_seed [∧ P_J if σ_target low], s)} + synthesize
    W7(S, J, K)    = MSA({FSS(P_J, s_i)}, score_f, K)
    W8(S, P, J, K) = MSA({FSS(P ∪ P_J, s_i)}, score_f, K)

Usage (from /dase/):
    python -m ours.sys <workload_path>
"""

import heapq
import json
import math
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# W3 adaptive TA: cold-start + lazy two-point normal fit (O(1) per stream per round)
W3_COLD_M = 20
W3_LAZY_STEP = 20

import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector

from ours.fss import FilteredScoreStreamer
from ours.jim import jim_get_valid_ids
from ours.msa import build_msa_from_query, synthesize
from ours.utils import find_ti_table, quote as _quote, METRIC_OP

DATABASE_URL = None  # set by main() from workload["dataset"]
DATASET = None  # set by main() from workload["dataset"]; used by setting loaders
ID_COL = "id"
TEXT_COL = "title"
VEC_COLS: set = set()

DATASET_DB_URLS = {
    "imdb": os.environ.get("IMDB_DATABASE_URL", "postgresql://localhost/imdb"),
    "molecule": os.environ.get("MOLECULE_DATABASE_URL", "postgresql://localhost/molecule"),
}
DATASET_COLS = {
    "imdb": ("id", "title"),
    "molecule": ("fact_id", "fact_text"),
}
DATASET_VEC_COLS = {
    "imdb": {"title_emb", "plot_emb", "actor_director_emb"},
    "molecule": {"fact_text_emb"},
}
DATASET_TABLE_PK_COLS = {
    "imdb": {"imdb_t1": "id", "imdb_t2": "id"},
    "molecule": {"facts_50k": "fact_id", "paper": "id"},
}
DATASET_TABLE_TEXT_COLS = {
    "imdb": {"imdb_t1": "title", "imdb_t2": "title"},
    "molecule": {"facts_50k": "fact_text", "paper": "title"},
}


def _table_pk(table: str) -> str:
    """Return the PK column name for a given table within the current dataset."""
    assert DATASET is not None
    base = table.lower().replace("_hnsw", "").replace("_ivf", "").replace("_1000", "")
    return DATASET_TABLE_PK_COLS[DATASET].get(base, ID_COL)


def _table_text(table: str) -> str:
    """Return the display/text column name for a given table within the current dataset."""
    assert DATASET is not None
    base = table.lower().replace("_hnsw", "").replace("_ivf", "").replace("_1000", "")
    return DATASET_TABLE_TEXT_COLS[DATASET].get(base, TEXT_COL)


# ---------------------------------------------------------------------------
# Execution patterns
# ---------------------------------------------------------------------------

def run_w1(conn: psycopg.Connection, query: Dict[str, Any], **kwargs) -> Tuple[List[Tuple], Dict[str, Any]]:
    """W1(s, K) = TopK{FSS(∅, s)} — direct SQL on HNSW table."""
    from pgvector import Vector
    from ours.utils import METRIC_OP

    prof: Dict[str, Any] = {}
    t_total = time.perf_counter()

    scoring = query["scoring"]
    sig = scoring["signals"][0]
    k = int(query["K"])

    table_rel = _resolve_hnsw_table(sig["table"])
    field = sig["field"]
    dist_op = METRIC_OP.get(sig.get("metric", "l2"), "<->")

    sql = (
        f'SELECT t.id, NULL::bigint AS id_t2,'
        f' t.title, NULL::text AS title_t2,'
        f' (t.{_quote(field)} {dist_op} %(query_embed)s) AS score_dist'
        f' FROM {_quote(table_rel)} t'
        f' ORDER BY score_dist'
        f' LIMIT %(limit_k)s'
    )
    params = {"query_embed": Vector(sig["query_embed"]), "limit_k": k}

    t0 = time.perf_counter()
    with conn.cursor(row_factory=psycopg.rows.tuple_row) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    prof["ann_query"] = time.perf_counter() - t0
    prof["n_results"] = len(rows)
    prof["total"] = time.perf_counter() - t_total

    return [(r[0], r[2], float(r[4])) for r in rows], prof


def run_w2(conn: psycopg.Connection, query: Dict[str, Any], **kwargs) -> Tuple[List[Tuple], Dict[str, Any]]:
    """W2(s, P, K) = TopK{FSS(P, s)}"""
    prof: Dict[str, Any] = {}
    t_total = time.perf_counter()

    scoring = query["scoring"]
    sig = scoring["signals"][0]
    k = int(query["K"])
    predicates = query["predicates"]

    # Filter predicates for this table
    table = sig["table"]
    table_preds = [p for p in predicates if p["table"].lower() == table.lower()]

    t0 = time.perf_counter()
    fss = FilteredScoreStreamer(
        conn=conn,
        table=_resolve_hnsw_table(table),
        scoring_signal={
            "type": "semantic",
            "field": sig["field"],
            "query_embed": sig["query_embed"],
            "metric": sig.get("metric", "l2"),
        },
        predicates=table_preds,
        light_mode=True,
        id_col=ID_COL,
        text_col=TEXT_COL,
    )
    prof["fss_init"] = time.perf_counter() - t0
    prof["fss_init.table_size"] = fss.profile["init_table_size"]
    prof["fss_init.selectivity"] = fss.profile["init_selectivity"]
    prof["fss_strategy"] = fss.strategy

    t0 = time.perf_counter()
    results = fss.fetch_topk(k)
    prof["fss_fetch"] = time.perf_counter() - t0
    prof["fss_fetch.db_query"] = fss.profile["fetch_db_query"]
    prof["fss_fetch.postfilter"] = fss.profile["fetch_postfilter"]
    prof["fss_fetch.probe_size"] = fss.profile.get("fetch_probe_size")
    prof["fss_fetch.n_batches"] = fss.profile.get("fetch_n_batches", 0)
    prof["n_results"] = len(results)
    prof["total"] = time.perf_counter() - t_total

    return _format_single_table(results), prof


def _w3_two_point_mu_sigma(
    n: int, t: int, t0: int, x_t: float, x_t0: float
) -> Tuple[float, float]:
    """O(1) μ, σ from two order statistics: ranks t and t0 (1-based), values x_t, x_t0.

    Z_t = Φ^{-1}(t/(n+1)), σ ≈ (x_t - x_t0)/(Z_t - Z_t0), μ ≈ x_t - σ Z_t.
    """
    p_t = min(max(t / (n + 1), 1e-12), 1.0 - 1e-12)
    p_t0 = min(max(t0 / (n + 1), 1e-12), 1.0 - 1e-12)
    z_t = statistics.NormalDist().inv_cdf(p_t)
    z_t0 = statistics.NormalDist().inv_cdf(p_t0)
    dz = z_t - z_t0
    if abs(dz) < 1e-9:
        sigma = max(abs(x_t - x_t0), 1e-9)
    else:
        sigma = (x_t - x_t0) / dz
    sigma = max(sigma, 1e-9)
    mu = x_t - sigma * z_t
    return mu, sigma


def _w3_stream_normal_params(dists: List[float], n: int) -> Tuple[float, float]:
    """Two-point fit: t < m → ranks (t,1); t ≥ m → ranks (t, max(1,t-LAZY_STEP)) (cold = t=20,t0=1)."""
    t = len(dists)
    if t < 2:
        return (dists[0], 1.0) if t == 1 else (0.0, 1.0)
    t0 = 1 if t < W3_COLD_M else max(1, t - W3_LAZY_STEP)
    x_t = dists[t - 1]
    x_t0 = dists[t0 - 1]
    return _w3_two_point_mu_sigma(n, t, t0, x_t, x_t0)


def _w3_g_score(weight: float, x_bound: float, mu: float, sigma: float) -> float:
    """G = weight / PDF(x_bound); larger → extend this stream next."""
    sigma = max(sigma, 1e-9)
    pdf = statistics.NormalDist(mu, sigma).pdf(x_bound)
    pdf = max(pdf, 1e-300)
    return weight / pdf


def _signal_dist_expr(sig: Dict[str, Any]) -> Tuple[str, Any, str]:
    """Build (dist_op_escaped, param_value, cast_suffix) for a scoring signal.

    Handles metric→operator mapping, param type adaptation, and %-escaping for
    psycopg parameterized SQL.

    jaccard → op="<%%>", param=bitstring text, cast="::bit(N)"
    l2/ip/cos → op="<->"/"<#>"/"<=>", param=Vector(...), cast=""
    """
    from pgvector import Vector as _Vec
    metric = sig.get("metric", "l2")
    qe = sig["query_embed"]
    if metric == "jaccard":
        dist_op = "<%>"
        param_val = "".join(str(int(x)) for x in qe)
        cast = f"::bit({len(qe)})"
    else:
        dist_op = METRIC_OP.get(metric, "<->")
        param_val = _Vec(qe)
        cast = ""
    return dist_op.replace("%", "%%"), param_val, cast


def run_w3(
    conn: psycopg.Connection,
    query: Dict[str, Any],
    adaptive: bool = True,
    eps: float = 0.0,
    **kwargs,
) -> Tuple[List[Tuple], Dict[str, Any]]:
    """W3: multi-signal single-table k-NN via TA with random access.

    Each signal gets an ordered stream (HNSW index scan).  Candidates
    discovered from any stream get their missing signal distances via
    random access (PK lookup).

    adaptive=True  — After cold start (m points per stream), pick the stream
                     with largest G_j = w_j / PDF(x_j | μ_j, σ_j), where
                     μ_j, σ_j come from O(1) two-point normal order-stat
                     estimates (ranks t and t0 with Z = Φ^{-1}(·/(n+1))).
    adaptive=False — all streams extended equally each round (vanilla TA).

    eps — TA slack: stop when kth_score <= threshold + eps (same units as
          joint score).  Larger eps → earlier stop → faster, lower accuracy.
          CLI passes --eps (default 0.01); use 0 for strict TA.
    """
    from pgvector import Vector
    from ours.utils import METRIC_OP

    prof: Dict[str, Any] = {}
    t_total = time.perf_counter()

    scoring = query["scoring"]
    signals = scoring["signals"]
    agg = scoring["aggregation"]
    weights = scoring["weights"]
    k = int(query["K"])
    qid = query.get("query_id", "?")
    n_sig = len(signals)

    table = signals[0]["table"]
    table_rel = _resolve_hnsw_table(table)
    score_f = _build_score_f(agg, weights)

    t_count = time.perf_counter()
    with conn.cursor(row_factory=psycopg.rows.tuple_row) as cur:
        cur.execute(f"SELECT COUNT(*) FROM {_quote(table_rel)}")
        n_table = int(cur.fetchone()[0])
    prof["w3_count_sec"] = time.perf_counter() - t_count

    INIT_LIMIT = max(k * 4, W3_COLD_M)
    MAX_LIMIT = 50_000

    stream_limits = [INIT_LIMIT] * n_sig
    candidate_info: Dict[int, Dict] = {}
    stream_bounds = [0.0] * n_sig
    stream_dists: List[List[float]] = [[] for _ in range(n_sig)]

    prof["stream_fetch"] = 0.0
    prof["random_access"] = 0.0
    prof["w3_heap_sec"] = 0.0
    prof["w3_adaptive_g_sec"] = 0.0
    prof["w3_ta_eps"] = float(eps)
    extensions: List = []

    # ------------------------------------------------------------------
    HNSW_EF_MAX = 1000
    ivf_table_rel = table_rel.replace("_hnsw", "_ivf")

    def _fetch_stream(sig_idx: int):
        sig = signals[sig_idx]
        field = sig["field"]
        metric = sig.get("metric", "l2")
        dist_op, qe_val, cast = _signal_dist_expr(sig)
        lim = stream_limits[sig_idx]

        use_ivf = lim > HNSW_EF_MAX
        use_table = ivf_table_rel if use_ivf else table_rel

        sql = (
            f"SELECT t.{_quote(ID_COL)}, t.{_quote(TEXT_COL)},"
            f" (t.{_quote(field)} {dist_op} %(qe)s{cast}) AS score_dist"
            f" FROM {_quote(use_table)} t"
            f" ORDER BY score_dist LIMIT %(lim)s"
        )
        t0 = time.perf_counter()
        with conn.cursor(row_factory=dict_row) as cur:
            if use_ivf:
                if metric != "jaccard":
                    from baselines_molecule import compute_ivfflat_probes
                    probes = compute_ivfflat_probes(lim)
                    cur.execute(f"SET ivfflat.probes = {probes}")
            else:
                ef = min(max(lim, k), HNSW_EF_MAX)
                cur.execute(f"SET hnsw.ef_search = {ef}")
            cur.execute(sql, {"qe": qe_val, "lim": lim})
            rows = cur.fetchall()
        prof["stream_fetch"] += time.perf_counter() - t0

        if rows:
            stream_bounds[sig_idx] = float(rows[-1]["score_dist"])
            stream_dists[sig_idx] = [float(r["score_dist"]) for r in rows]
        for r in rows:
            rid = r[ID_COL]
            if rid not in candidate_info:
                candidate_info[rid] = {TEXT_COL: r[TEXT_COL]}
            candidate_info[rid][sig_idx] = float(r["score_dist"])

    def _random_access_missing():
        t0 = time.perf_counter()
        for sig_idx, sig in enumerate(signals):
            missing_ids = [rid for rid, info in candidate_info.items()
                           if sig_idx not in info]
            if not missing_ids:
                continue
            field = sig["field"]
            dist_op, qe_val, cast = _signal_dist_expr(sig)
            sql = (
                f"SELECT t.{_quote(ID_COL)},"
                f" (t.{_quote(field)} {dist_op} %(qe)s{cast}) AS score_dist"
                f" FROM {_quote(table_rel)} t WHERE t.{_quote(ID_COL)} = ANY(%(ids)s)"
            )
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, {"qe": qe_val, "ids": missing_ids})
                for r in cur.fetchall():
                    candidate_info[r[ID_COL]][sig_idx] = float(r["score_dist"])
        prof["random_access"] += time.perf_counter() - t0
    # ------------------------------------------------------------------

    # Initial fetch — all streams
    for j in range(n_sig):
        _fetch_stream(j)
    _random_access_missing()

    n_rounds = 0
    while True:
        n_rounds += 1

        # --- Build top-K heap ---
        t_heap = time.perf_counter()
        answer_heap: list = []
        for rid, info in candidate_info.items():
            dists = [info.get(i) for i in range(n_sig)]
            if any(d is None for d in dists):
                continue
            joint = score_f(dists)
            heapq.heappush(answer_heap, (-joint, rid, (rid, info[TEXT_COL], joint)))
            if len(answer_heap) > k:
                heapq.heappop(answer_heap)
        prof["w3_heap_sec"] += time.perf_counter() - t_heap

        # --- Threshold stopping ---
        threshold = score_f(stream_bounds)
        kth_score = float("inf")
        if len(answer_heap) >= k:
            kth_score = -answer_heap[0][0]

        if len(answer_heap) >= k and kth_score <= threshold + eps:
            break
        if all(sl >= MAX_LIMIT for sl in stream_limits):
            break
        if time.perf_counter() - t_total > 30.0:
            break

        # --- Decide which stream(s) to extend ---
        if adaptive and n_sig > 1:
            # G_j = w_j / PDF(x_j) with (μ_j, σ_j) from two-point order-stat fit (O(1)).
            t_g = time.perf_counter()
            g_scores: List[float] = []
            for j in range(n_sig):
                if stream_limits[j] >= MAX_LIMIT:
                    g_scores.append(float("-inf"))
                    continue
                dists = stream_dists[j]
                if not dists:
                    g_scores.append(float("-inf"))
                    continue
                x_bound = float(dists[-1])
                mu_j, sigma_j = _w3_stream_normal_params(dists, n_table)
                g_scores.append(_w3_g_score(weights[j], x_bound, mu_j, sigma_j))
            prof["w3_adaptive_g_sec"] += time.perf_counter() - t_g

            best_j = int(max(range(n_sig), key=lambda j: g_scores[j]))
            if g_scores[best_j] == float("-inf"):
                break
            stream_limits[best_j] = min(stream_limits[best_j] * 2, MAX_LIMIT)
            extensions.append(best_j)
            _fetch_stream(best_j)
        else:
            for j in range(n_sig):
                stream_limits[j] = min(stream_limits[j] * 2, MAX_LIMIT)
            extensions.append("all")
            for j in range(n_sig):
                _fetch_stream(j)

        _random_access_missing()

    top = sorted(answer_heap, key=lambda x: -x[0])
    results = [(rid, title, sc) for _, _, (rid, title, sc) in top]

    prof["total"] = time.perf_counter() - t_total
    prof["n_candidates"] = len(candidate_info)
    prof["n_rounds"] = n_rounds
    prof["adaptive"] = adaptive
    prof["stream_limits_final"] = list(stream_limits)
    prof["extensions"] = extensions
    prof["n_extensions"] = len(extensions)
    prof["n_results"] = len(results)

    tot = prof["total"]
    if tot > 0:
        prof["w3_pct_stream_fetch"] = round(100.0 * prof["stream_fetch"] / tot, 4)
        prof["w3_pct_random_access"] = round(100.0 * prof["random_access"] / tot, 4)
        prof["w3_pct_heap"] = round(100.0 * prof["w3_heap_sec"] / tot, 4)
        prof["w3_pct_adaptive_g"] = round(100.0 * prof["w3_adaptive_g_sec"] / tot, 4)
        prof["w3_pct_count"] = round(100.0 * prof.get("w3_count_sec", 0.0) / tot, 4)
        rest = tot - prof["stream_fetch"] - prof["random_access"] - prof["w3_heap_sec"]
        rest -= prof["w3_adaptive_g_sec"] + prof.get("w3_count_sec", 0.0)
        prof["w3_pct_other_python"] = round(100.0 * max(rest, 0.0) / tot, 4)

    return results, prof


def _estimate_w4_predicate_sigma(
    conn: psycopg.Connection,
    query: Dict[str, Any],
    table_preds: List[Dict[str, Any]],
    table_display: str,
) -> Tuple[float, Optional[int]]:
    """
    σ = fraction of table rows satisfying predicates on this table.
    Uses query[\"predicate_selectivity\"] when present; else COUNT(*) on *_ivf
    (same relation as baselines.filter_first.build_filter_id_sql).

    Returns (sigma, n_passing_rows or None if sigma came only from query metadata).
    """
    raw = query.get("predicate_selectivity")
    if raw is not None:
        try:
            return float(raw), None
        except (TypeError, ValueError):
            pass
    if DATASET == "molecule":
        from baselines_molecule.filter_first import build_filter_id_sql
    else:
        from baselines.filter_first import build_filter_id_sql

    sql, params = build_filter_id_sql(table_display, table_preds)
    count_sql = sql.replace("SELECT id", "SELECT COUNT(*)", 1)
    rel_ivf = table_display.lower() + "_ivf"
    with conn.cursor(row_factory=psycopg.rows.tuple_row) as cur:
        cur.execute(f"SELECT COUNT(*) FROM {_quote(rel_ivf)}")
        n_total = int(cur.fetchone()[0])
    if n_total <= 0:
        return 0.0, 0
    with conn.cursor(row_factory=psycopg.rows.tuple_row) as cur:
        cur.execute(count_sql, params)
        n_pass = int(cur.fetchone()[0])
    return n_pass / n_total, n_pass


def _run_w4_filter_first_ra(
    conn: psycopg.Connection,
    query: Dict[str, Any],
    table_resolved: str,
    table_preds: List[Dict[str, Any]],
    table_display: str,
    scoring: Dict[str, Any],
    signals: List[Dict[str, Any]],
    k: int,
    sigma: float,
    t0: float,
) -> Tuple[List[Tuple], Dict]:
    """
    W4 bypass: same execution as baselines.filter_first.run_filter_first_query_w4 —
    filter on *_ivf, one SQL with weighted multi-signal distance + ORDER BY + LIMIT k
    (IVFFlat, enable_indexscan off). Avoids multiple HNSW round-trips per signal.
    """
    if DATASET == "molecule":
        from baselines_molecule.filter_first import run_filter_first_query_w4
    else:
        from baselines.filter_first import run_filter_first_query_w4

    n_sig = len(signals)
    t1 = time.perf_counter()
    raw_rows, n_ids = run_filter_first_query_w4(conn, query)
    t_db = time.perf_counter() - t1

    if not raw_rows:
        prof = {
            "w4_mode": "filter_first_ra",
            "w4_predicate_sigma": round(sigma, 6),
            "fss_init_sec": 0.0,
            "ta_sec": round(t_db, 6),
            "total_sec": round(time.perf_counter() - t0, 6),
            "n_random_access": 0,
            "n_candidates": n_ids,
        }
        for i in range(n_sig):
            prof[f"stream_{i}_strategy"] = "filter_first_ra"
            prof[f"stream_{i}_table"] = f"{table_display.lower()}_ivf"
            prof[f"stream_{i}_selectivity"] = round(sigma, 6)
            prof[f"stream_{i}_db_query_sec"] = round(t_db / max(n_sig, 1), 6)
            prof[f"stream_{i}_postfilter_sec"] = 0.0
            prof[f"stream_{i}_n_batches"] = 1
            prof[f"stream_{i}_unfilter_fetch_k"] = 0
            prof[f"stream_{i}_filtered_stream_length"] = n_ids
            prof[f"stream_{i}_cursor_on_filtered_stream"] = 0
        return [], prof

    # imdb row shape: (id, id_t2, title, title_t2, score_dist)
    # molecule row shape: (fact_id, score_dist)
    if DATASET == "molecule":
        rows = [(r[0], "", float(r[1])) for r in raw_rows]
    else:
        rows = [(r[0], r[2], float(r[4])) for r in raw_rows]

    prof = {
        "w4_mode": "filter_first_ra",
        "w4_predicate_sigma": round(sigma, 6),
        "fss_init_sec": 0.0,
        "ta_sec": round(t_db, 6),
        "total_sec": round(time.perf_counter() - t0, 6),
        "n_random_access": 0,
        "n_candidates": n_ids,
    }
    table_ivf = f"{table_display.lower()}_ivf"
    for i in range(n_sig):
        prof[f"stream_{i}_strategy"] = "filter_first_ra"
        prof[f"stream_{i}_table"] = table_ivf
        prof[f"stream_{i}_selectivity"] = round(sigma, 6)
        prof[f"stream_{i}_db_query_sec"] = round(t_db / max(n_sig, 1), 6)
        prof[f"stream_{i}_postfilter_sec"] = 0.0
        prof[f"stream_{i}_n_batches"] = 1
        prof[f"stream_{i}_unfilter_fetch_k"] = n_ids
        prof[f"stream_{i}_filtered_stream_length"] = n_ids
        prof[f"stream_{i}_cursor_on_filtered_stream"] = min(k, len(rows))

    return rows, prof


def _w4_kw_first_present(kwargs: Dict[str, Any], keys: Tuple[str, ...], default: Any = None) -> Any:
    """Return kwargs[k] for the first key k that appears in kwargs (value may be None)."""
    for k in keys:
        if k in kwargs:
            return kwargs[k]
    return default


# W4 FilterFirst vs TA: kwargs aliases (most intuitive names first).
_W4_FF_TA_THRESHOLD_KEYS = (
    "w4_FilterFirst_TA_threshold",
    "w4_filter_first_ta_match_fraction_threshold",
    "w4_if_matching_row_fraction_at_most",
    "w4_ff_sigma_max",
)
def run_w4(conn: psycopg.Connection, query: Dict[str, Any], **kwargs) -> Tuple[List[Tuple], Dict]:
    """W4(S, P, K) = TA with random access over FSS streams, or filter-first + batched RA when σ is very low."""
    from pgvector import Vector
    from ours.utils import METRIC_OP

    fss_pf = kwargs.get("fss_pf", 1.0)
    eps = float(kwargs.get("w4_ta_eps", kwargs.get("eps", 0.0)) or 0.0)
    _th_raw = _w4_kw_first_present(kwargs, _W4_FF_TA_THRESHOLD_KEYS, default=0)
    if _th_raw is None:
        _th_raw = 0
    ff_match_fraction_threshold = float(_th_raw)
    t0 = time.perf_counter()

    scoring = query["scoring"]
    signals = scoring["signals"]
    agg = scoring["aggregation"]
    weights = scoring["weights"]
    k = int(query["K"])
    predicates = query.get("predicates", [])
    n_sig = len(signals)

    table = signals[0]["table"].lower()
    table_display = signals[0]["table"]
    table_resolved = _resolve_hnsw_table(table)
    table_preds = [p for p in predicates if p["table"].lower() == table]

    sigma, _ = _estimate_w4_predicate_sigma(conn, query, table_preds, table_display)
    # σ <= threshold → FilterFirst path; threshold 0 ⇒ only σ==0 (e.g. no matches) uses it, else always TA.
    use_ff = sigma <= ff_match_fraction_threshold

    if use_ff:
        return _run_w4_filter_first_ra(
            conn,
            query,
            table_resolved,
            table_preds,
            table_display,
            scoring,
            signals,
            k,
            sigma,
            t0,
        )

    # Build FSS streams
    assert sigma > 0, f"W4 requires predicate_selectivity > 0, got {sigma}"
    # Align W4 TA.score_first cold-start with rerank_w4:
    # high-σ queries start from a small top-N instead of the generic 200-row probe.
    w4_init_stream_fetchK = min(max(int(math.ceil(20.0 / sigma)), 20), 100_000)

    streams = []
    for i, sig in enumerate(signals):
        fss_signal = {"type": sig["type"], "field": sig["field"]}
        if sig["type"] == "semantic":
            fss_signal["query_embed"] = sig["query_embed"]
            fss_signal["metric"] = sig["metric"]
        elif sig["type"] == "attribute":
            fss_signal["direction"] = sig["direction"]
        fss = FilteredScoreStreamer(
            conn=conn, table=table_resolved, scoring_signal=fss_signal,
            predicates=table_preds, query_k=k, fss_pf=fss_pf,
            init_stream_fetchK=w4_init_stream_fetchK,
            precomputed_selectivity=sigma,
            light_mode=True,
            id_col=ID_COL,
            text_col=TEXT_COL,
        )
        streams.append(fss)

    t_fss_init = time.perf_counter() - t0

    # Batch random access: fill missing signal distances for a set of eids
    def _batch_random_access(sig_idx: int, eids: List[int]) -> Dict[int, float]:
        if not eids:
            return {}
        sig = signals[sig_idx]
        field = sig["field"]
        dist_op, qe_val, cast = _signal_dist_expr(sig)
        sql = (
            f"SELECT t.{_quote(ID_COL)},"
            f" (t.{_quote(field)} {dist_op} %(qe)s{cast}) AS score_dist"
            f" FROM {_quote(table_resolved)} t WHERE t.{_quote(ID_COL)} = ANY(%(ids)s)"
        )
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, {"qe": qe_val, "ids": eids})
            return {r[ID_COL]: float(r["score_dist"]) for r in cur.fetchall()}

    score_f = _build_score_f(agg, weights)

    # TA with batched random access
    t1 = time.perf_counter()
    last_scores = [0.0] * n_sig
    candidates: Dict[int, Dict] = {}
    heap: List = []
    in_heap: Dict[int, float] = {}
    tie = 0
    stream_idx = 0
    n_ra = 0
    BATCH_SIZE = 20  # consume this many items per round before batch RA

    while True:
        # Consume a batch of items from streams (round-robin)
        batch_eids: List[int] = []
        for _ in range(BATCH_SIZE):
            # Find non-exhausted stream
            attempts = 0
            while attempts < n_sig:
                if streams[stream_idx].peek_score() is not None:
                    break
                stream_idx = (stream_idx + 1) % n_sig
                attempts += 1
            else:
                break

            fss = streams[stream_idx]
            item = fss.next()
            if item is None:
                stream_idx = (stream_idx + 1) % n_sig
                continue

            entry, score = item
            eid = entry[ID_COL]
            last_scores[stream_idx] = score

            if eid not in candidates:
                candidates[eid] = {TEXT_COL: entry.get(TEXT_COL, "")}
            candidates[eid][stream_idx] = score
            batch_eids.append(eid)
            stream_idx = (stream_idx + 1) % n_sig
        else:
            # Batch random access for all missing signals
            for si in range(n_sig):
                missing = [eid for eid in batch_eids if si not in candidates[eid]]
                if not missing:
                    continue
                results = _batch_random_access(si, missing)
                for eid, dist in results.items():
                    candidates[eid][si] = dist
                    n_ra += 1

        # If inner for-loop broke (all streams exhausted), do final RA
        if not batch_eids:
            break

        # Update heap for all batch candidates
        for eid in batch_eids:
            dists = [candidates[eid].get(si) for si in range(n_sig)]
            if any(d is None for d in dists):
                continue
            total = score_f(dists)

            if eid in in_heap:
                pass
            elif len(heap) < k:
                heapq.heappush(heap, (-total, tie, (eid, candidates[eid][TEXT_COL], total)))
                in_heap[eid] = total
                tie += 1
            elif total < -heap[0][0]:
                _, _, (old_id, _, _) = heapq.heapreplace(
                    heap, (-total, tie, (eid, candidates[eid][TEXT_COL], total)))
                del in_heap[old_id]
                in_heap[eid] = total
                tie += 1

        # Check TA termination
        if len(heap) >= k:
            bound = score_f([
                streams[j].peek_score() if streams[j].peek_score() is not None else last_scores[j]
                for j in range(n_sig)
            ])
            if -heap[0][0] <= bound + eps:
                break

    t_ta = time.perf_counter() - t1

    results = sorted(heap, key=lambda x: x[0])
    rows = [(eid, title, total) for _, _, (eid, title, total) in results]

    # Profile
    prof = {
        "fss_init_sec": round(t_fss_init, 6),
        "ta_sec": round(t_ta, 6),
        "total_sec": round(time.perf_counter() - t0, 6),
        "w4_ta_eps": eps,
        "n_random_access": n_ra,
        "n_candidates": len(candidates),
    }
    for i, fss in enumerate(streams):
        prof[f"stream_{i}_strategy"] = fss.strategy
        prof[f"stream_{i}_table"] = fss.table
        prof[f"stream_{i}_selectivity"] = round(fss.profile.get("init_selectivity", 0.0), 6)
        prof[f"stream_{i}_db_query_sec"] = round(fss.profile["fetch_db_query"], 6)
        prof[f"stream_{i}_postfilter_sec"] = round(fss.profile["fetch_postfilter"], 6)
        prof[f"stream_{i}_n_batches"] = fss.profile.get("fetch_n_batches", 0)
        prof[f"stream_{i}_unfilter_fetch_k"] = fss.profile["unfilter_fetch_k"]
        prof[f"stream_{i}_filtered_stream_length"] = fss.profile["filtered_stream_length"]
        prof[f"stream_{i}_cursor_on_filtered_stream"] = fss.profile["cursor_on_filtered_stream"]

    return rows, prof


LIMIT_OUTER_MAX = 100_000


def _load_setting(wtype: str) -> Dict[str, Any]:
    """Load ours/{DATASET}_{wtype}_setting.json (set by main from workload.dataset)."""
    assert DATASET is not None, "DATASET not set; must be initialized by main() before loading settings"
    p = os.path.join(os.path.dirname(__file__), f"{DATASET}_{wtype.lower()}_setting.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def _load_w2_setting() -> Dict[str, Any]: return _load_setting("w2")
def _load_w3_setting() -> Dict[str, Any]: return _load_setting("w3")
def _load_w4_setting() -> Dict[str, Any]: return _load_setting("w4")
def _load_w5_setting() -> Dict[str, Any]: return _load_setting("w5")
def _load_w7_setting() -> Dict[str, Any]: return _load_setting("w7")
def _load_w8_setting() -> Dict[str, Any]: return _load_setting("w8")


W5_SELECT_DISTINCT_RATE_THRESHOLD = 0.35


def _run_w5_select_distinct(
    conn: psycopg.Connection,
    syn_conn: psycopg.Connection,
    query: Dict[str, Any],
    ti_table: str,
    seed_table: str,
    partner_table: str,
    seed_side: str,
    tau: float,
    prof: Dict[str, Any],
) -> List[Tuple]:
    """W5 strategy=select_distinct: SELECT DISTINCT valid seeds from TI, then random-access score + top-K."""
    scoring = query["scoring"]
    sig = scoring["signals"][0]
    k = int(query["K"])
    field = sig["field"]
    dist_op, qe_val, cast = _signal_dist_expr(sig)
    seed_rel = _resolve_hnsw_table(seed_table)
    seed_pk = _table_pk(seed_table)
    partner_pk = _table_pk(partner_table)
    seed_id_col = f"{seed_table}.{seed_pk}"
    partner_id_col = f"{partner_table}.{partner_pk}"

    # Step 1: get all valid seed IDs from TI
    t0 = time.perf_counter()
    with syn_conn.cursor() as cur:
        cur.execute(
            f'SELECT DISTINCT "{seed_id_col}" FROM "{ti_table}" WHERE "dis" <= %(tau)s',
            {"tau": tau},
        )
        valid_seeds = [r[0] for r in cur.fetchall()]
    prof["select_distinct_time"] = time.perf_counter() - t0
    prof["select_distinct_n_valid"] = len(valid_seeds)

    if not valid_seeds:
        prof["score_query"] = 0.0
        prof["ti_query"] = 0.0
        prof["n_scanned"] = 0
        prof["n_ti_queries"] = 0
        return []

    # Step 2: random-access score computation on valid seeds, sorted top-K
    score_sql = (
        f'SELECT t.{_quote(seed_pk)}, t.{_quote(TEXT_COL)},'
        f' (t.{_quote(field)} {dist_op} %(query_embed)s{cast}) AS score_dist'
        f' FROM {_quote(seed_rel)} t'
        f' WHERE t.{_quote(seed_pk)} = ANY(%(valid_ids)s)'
        f' ORDER BY score_dist'
        f' LIMIT %(limit_k)s'
    )
    score_params = {
        "query_embed": qe_val,
        "valid_ids": valid_seeds,
        "limit_k": k,
    }

    t0 = time.perf_counter()
    with conn.cursor(row_factory=psycopg.rows.tuple_row) as cur:
        cur.execute("SET LOCAL enable_indexscan = off")
        cur.execute("SET LOCAL enable_indexonlyscan = off")
        cur.execute(score_sql, score_params)
        candidates = cur.fetchall()
    prof["score_query"] = time.perf_counter() - t0

    # Step 3: get partner IDs for top-K seeds via TI
    top_ids = [c[0] for c in candidates]
    ti_check_sql = (
        f'SELECT s.id AS seed_id, t."{partner_id_col}" AS partner_id '
        f'FROM unnest(%(ids)s) AS s(id) '
        f'CROSS JOIN LATERAL ('
        f'  SELECT "{partner_id_col}" FROM "{ti_table}" '
        f'  WHERE "{seed_id_col}" = s.id AND "dis" <= %(tau)s '
        f'  LIMIT 1'
        f') t'
    )
    t0 = time.perf_counter()
    with syn_conn.cursor() as cur:
        cur.execute(ti_check_sql, {"ids": top_ids, "tau": tau})
        ti_rows = cur.fetchall()
    prof["ti_query"] = time.perf_counter() - t0

    partner_map = {s_id: p_id for s_id, p_id in ti_rows}

    out: List[Tuple] = []
    for cand_id, cand_title, score_dist in candidates:
        if cand_id in partner_map:
            pid = partner_map[cand_id]
            if seed_side == "left":
                out.append((cand_id, pid, cand_title, None, float(score_dist)))
            else:
                out.append((pid, cand_id, None, cand_title, float(score_dist)))

    prof["n_scanned"] = len(valid_seeds)
    prof["n_ti_queries"] = 1
    return out


def _run_w5_index_topk(
    conn: psycopg.Connection,
    syn_conn: psycopg.Connection,
    query: Dict[str, Any],
    ti_table: str,
    seed_table: str,
    partner_table: str,
    seed_side: str,
    tau: float,
    partner_rate: float,
    pf: float,
    prof: Dict[str, Any],
) -> List[Tuple]:
    """W5 strategy=index_topk: HNSW top-K (starting at K/partner_rate) + batch TI post-filter, double on failure."""
    import math

    scoring = query["scoring"]
    sig = scoring["signals"][0]
    k = int(query["K"])
    field = sig["field"]
    dist_op, qe_val, cast = _signal_dist_expr(sig)
    seed_rel = _resolve_hnsw_table(seed_table)
    seed_pk = _table_pk(seed_table)
    partner_pk = _table_pk(partner_table)
    seed_id_col = f"{seed_table}.{seed_pk}"
    partner_id_col = f"{partner_table}.{partner_pk}"

    ann_sql = (
        f'SELECT t.{_quote(seed_pk)}, t.{_quote(TEXT_COL)},'
        f' (t.{_quote(field)} {dist_op} %(query_embed)s{cast}) AS score_dist'
        f' FROM {_quote(seed_rel)} t'
        f' ORDER BY score_dist'
        f' LIMIT %(limit_outer)s'
    )
    ann_params: Dict[str, Any] = {"query_embed": qe_val}

    ti_check_sql = (
        f'SELECT s.id AS seed_id, t."{partner_id_col}" AS partner_id '
        f'FROM unnest(%(ids)s) AS s(id) '
        f'CROSS JOIN LATERAL ('
        f'  SELECT "{partner_id_col}" FROM "{ti_table}" '
        f'  WHERE "{seed_id_col}" = s.id AND "dis" <= %(tau)s '
        f'  LIMIT 1'
        f') t'
    )

    limit_outer = int(math.ceil(k / max(partner_rate, 0.01)))
    out: List[Tuple] = []
    checked = 0
    prof["ann_query"] = 0.0
    prof["ti_query"] = 0.0
    n_ti_queries = 0

    while len(out) < k and limit_outer <= LIMIT_OUTER_MAX:
        ef = max(20, int(round(limit_outer * pf)))
        t0 = time.perf_counter()
        with conn.cursor(row_factory=psycopg.rows.tuple_row) as cur:
            cur.execute(f"SET hnsw.ef_search = {ef}")
            cur.execute("SET LOCAL enable_seqscan = off")
            ann_params["limit_outer"] = limit_outer
            cur.execute(ann_sql, ann_params)
            candidates = cur.fetchall()
        prof["ann_query"] += time.perf_counter() - t0

        new_cands = candidates[checked:]
        if not new_cands:
            break

        new_ids = [c[0] for c in new_cands]
        t0 = time.perf_counter()
        with syn_conn.cursor() as cur:
            cur.execute(ti_check_sql, {"ids": new_ids, "tau": tau})
            ti_rows = cur.fetchall()
        prof["ti_query"] += time.perf_counter() - t0
        n_ti_queries += 1

        partner_map: Dict[int, int] = {}
        for s_id, p_id in ti_rows:
            partner_map[s_id] = p_id

        for cand_id, cand_title, score_dist in new_cands:
            if cand_id in partner_map:
                pid = partner_map[cand_id]
                if seed_side == "left":
                    out.append((cand_id, pid, cand_title, None, float(score_dist)))
                else:
                    out.append((pid, cand_id, None, cand_title, float(score_dist)))
                if len(out) >= k:
                    break

        checked = len(candidates)
        if len(out) < k:
            limit_outer *= 2

    prof["n_scanned"] = checked
    prof["n_ti_queries"] = n_ti_queries
    prof["limit_outer_final"] = limit_outer
    return out


def run_w5(conn: psycopg.Connection, query: Dict[str, Any], **kwargs) -> Tuple[List[Tuple], Dict[str, Any]]:
    """W5(s, J, K) — ANN-first on HNSW + TI join check.

    Two strategies selected by partner_rate:
      - select_distinct (partner_rate < 0.35): SELECT DISTINCT valid seeds from TI,
        random-access score computation, return top-K.
      - index_topk (partner_rate >= 0.35): HNSW index top-K + batch TI post-filter,
        double limit_outer on failure.
    """
    w5_cfg = kwargs.get("w5_cfg") or _load_w5_setting()
    pf = float(w5_cfg.get("pf", 1.5))
    rate_threshold = float(w5_cfg.get("select_distinct_threshold", W5_SELECT_DISTINCT_RATE_THRESHOLD))

    prof: Dict[str, Any] = {"pf": pf}
    t_total = time.perf_counter()

    scoring = query["scoring"]
    sig = scoring["signals"][0]
    join_spec = query["join"]
    k = int(query["K"])

    seed_table = sig["table"].lower()
    t_left = join_spec["table_left"].lower()
    t_right = join_spec["table_right"].lower()
    seed_side = "left" if seed_table == t_left else "right"
    partner_table = t_right if seed_side == "left" else t_left
    tau = float(join_spec["distance_threshold"])

    # Find TI table
    t0 = time.perf_counter()
    ti_table = find_ti_table(conn, t_left, t_right, tau)
    prof["ti_lookup"] = time.perf_counter() - t0

    # Choose strategy
    partner_rate = float(query.get("partner_rate", 1.0))
    strategy = "select_distinct" if partner_rate < rate_threshold else "index_topk"
    prof["strategy"] = strategy
    prof["partner_rate"] = partner_rate

    shared_syn = kwargs.get("syn_conn")
    if shared_syn is not None:
        syn_conn = shared_syn
        owns_syn = False
    else:
        syn_conn = psycopg.connect(DATABASE_URL, row_factory=psycopg.rows.tuple_row, autocommit=True)
        owns_syn = True

    try:
        if strategy == "select_distinct":
            out = _run_w5_select_distinct(
                conn, syn_conn, query, ti_table, seed_table, partner_table,
                seed_side, tau, prof,
            )
        else:
            out = _run_w5_index_topk(
                conn, syn_conn, query, ti_table, seed_table, partner_table,
                seed_side, tau, partner_rate, pf, prof,
            )
    finally:
        if owns_syn:
            syn_conn.close()

    prof["total"] = time.perf_counter() - t_total

    def _round(d):
        return {kk: (round(vv, 6) if isinstance(vv, (int, float)) else vv) for kk, vv in d.items()}
    return out, _round(prof)


def _synthesize_w6(
    conn: psycopg.Connection,
    seed_id: int,
    ti_ladder: List[Tuple[str, float]],
    seed_table: str,
    partner_table: str,
    tau: float,
    partner_preds: List[Dict[str, Any]],
) -> Optional[int]:
    """
    W6 synthesize: given a seed ID, find any valid partner via the TI ladder.

    TI tables are cumulative (ti_0.6 ⊂ ti_0.7), and per-seed partner sets grow
    fast with tau. So we walk the ladder ascending: try the smallest ti_tau
    first; any hit there has dis ≤ ti_tau ≤ query_tau, so it's globally
    closest. Only escalate to a larger table when the smaller one has no
    matching partner.

    Returns partner ID, or None if no valid partner exists in any rung.
    """
    seed_id_col = f"{seed_table}.{_table_pk(seed_table)}"
    partner_id_col = f"{partner_table}.{_table_pk(partner_table)}"

    base_where = [
        f'"{seed_id_col}" = %(seed_id)s',
        '"dis" <= %(tau)s',
    ]
    base_params: Dict[str, Any] = {}
    for i, p in enumerate(partner_preds):
        attr = p["attribute"]
        op = p["operator"]
        val = p["value"]
        ti_col = f"{partner_table}.{attr}"
        key = f"sp{i}"
        if op == "in":
            base_where.append(f'"{ti_col}" = ANY(%({key})s)')
            base_params[key] = list(val) if not isinstance(val, list) else val
        else:
            base_where.append(f'"{ti_col}" {op} %({key})s')
            base_params[key] = val

    with conn.cursor(row_factory=psycopg.rows.tuple_row) as cur:
        for ti_table, ti_tau in ti_ladder:
            effective_tau = min(tau, ti_tau)
            sql = (
                f'SELECT "{partner_id_col}" '
                f'FROM "{ti_table}" '
                f'WHERE {" AND ".join(base_where)} '
                f'ORDER BY "dis" LIMIT 1'
            )
            cur.execute(sql, {"seed_id": seed_id, "tau": effective_tau, **base_params})
            row = cur.fetchone()
            if row is not None:
                return row[0]
    return None


def _run_w6_strategy_c(
    conn: psycopg.Connection,
    query: Dict[str, Any],
    ti_table: str,
    seed_table: str,
    partner_table: str,
    seed_side: str,
    tau: float,
    seed_preds: List[Dict[str, Any]],
    partner_preds: List[Dict[str, Any]],
    sig: Dict[str, Any],
    k: int,
    prof: Dict[str, Any],
) -> List[Tuple]:
    """
    Strategy C: direct enumeration when expected_valid_tuples is tiny.

    1. Query TI table for all matching tuples (seed_preds + partner_preds + dis ≤ τ)
    2. Batch compute semantic scores for distinct seed IDs
    3. For each seed, keep the closest partner (smallest dis)
    4. Return top-K by score

    Note: we tried chunked dis-btree streaming here (like W7/W8 TIScoreStream)
    and it regressed. The no-ORDER-BY filter lets PG pick bitmap-on-predicate-
    index which is optimal for selective predicates; forcing a dis-btree walk
    via enable_bitmapscan=off was 5-30× slower when predicate selectivity is
    high. See run 20260414_164823.
    """
    seed_pk = _table_pk(seed_table)
    seed_id_col = f"{seed_table}.{seed_pk}"
    partner_id_col = f"{partner_table}.{_table_pk(partner_table)}"

    where_parts: List[str] = ['"dis" <= %(tau)s']
    params: Dict[str, Any] = {"tau": tau}

    all_preds = [(seed_table, p) for p in seed_preds] + [(partner_table, p) for p in partner_preds]
    for i, (tbl, p) in enumerate(all_preds):
        attr = p["attribute"]
        op = p["operator"]
        val = p["value"]
        ti_col = f"{tbl}.{attr}"
        key = f"cp{i}"
        if op == "in":
            where_parts.append(f'"{ti_col}" = ANY(%({key})s)')
            params[key] = list(val) if not isinstance(val, list) else val
        else:
            where_parts.append(f'"{ti_col}" {op} %({key})s')
            params[key] = val

    t0 = time.perf_counter()
    sql = (
        f'SELECT "{seed_id_col}", "{partner_id_col}", "dis" '
        f'FROM "{ti_table}" '
        f'WHERE {" AND ".join(where_parts)}'
    )
    with conn.cursor(row_factory=psycopg.rows.tuple_row) as cur:
        cur.execute(sql, params)
        ti_rows = cur.fetchall()
    prof["c_ti_query"] = time.perf_counter() - t0
    prof["c_ti_rows"] = len(ti_rows)

    if not ti_rows:
        return []

    # For each seed_id, keep the partner with smallest distance
    best_partner: Dict[int, Tuple[int, float]] = {}
    for seed_id, partner_id, dis in ti_rows:
        if seed_id not in best_partner or dis < best_partner[seed_id][1]:
            best_partner[seed_id] = (partner_id, dis)

    seed_ids = list(best_partner.keys())
    prof["c_n_seeds"] = len(seed_ids)

    # Batch compute semantic scores for all seed IDs
    t0 = time.perf_counter()
    dist_op, qe_val, cast = _signal_dist_expr(sig)
    field = sig["field"]
    actual_table = _resolve_hnsw_table(seed_table)
    score_sql = (
        f'SELECT {_quote(seed_pk)}, ({_quote(field)} {dist_op} %(qv)s{cast}) AS score '
        f'FROM {_quote(actual_table)} WHERE {_quote(seed_pk)} = ANY(%(ids)s)'
    )
    with conn.cursor(row_factory=psycopg.rows.tuple_row) as cur:
        cur.execute(score_sql, {"qv": qe_val, "ids": seed_ids})
        scores = {row[0]: float(row[1]) for row in cur.fetchall()}
    prof["c_batch_score"] = time.perf_counter() - t0

    # Build (seed_id, partner_id, score) and sort by score (ascending = best)
    scored = []
    for seed_id, (partner_id, _dis) in best_partner.items():
        if seed_id in scores:
            scored.append((seed_id, partner_id, scores[seed_id]))
    scored.sort(key=lambda x: x[2])

    out = []
    for seed_id, partner_id, score in scored[:k]:
        if seed_side == "left":
            out.append((seed_id, partner_id, score))
        else:
            out.append((partner_id, seed_id, score))
    return out


def run_w6(conn: psycopg.Connection, query: Dict[str, Any], **kwargs) -> Tuple[List[Tuple], Dict[str, Any]]:
    """
    W6(s, P, J, K) — single scoring signal + predicates + join.

    Strategy based on expected_valid_tuples (TI_size × σ_partner × σ_threshold):

    Case C — expected_valid_tuples < threshold (default 100):
      Direct enumeration: query TI for all matching tuples, batch score
      the seed IDs, return top-K. No FSS or synthesize needed.

    Case B — expected_valid_tuples ≥ threshold:
      Post-filter: FSS(P_seed, s) → synthesize with P_partner + join.

    Synthesize queries TI table directly (B-tree indexed) instead of
    computing vector range distances on the raw partner table.
    """
    prof: Dict[str, Any] = {}
    t_total = time.perf_counter()

    scoring = query["scoring"]
    sig = scoring["signals"][0]
    join_spec = query["join"]
    predicates = query.get("predicates", [])  # no predicates = select all
    k = int(query["K"])

    seed_table = sig["table"].lower()
    seed_side = "left" if seed_table == join_spec["table_left"].lower() else "right"

    seed_preds = [p for p in predicates if p["table"].lower() == seed_table]
    partner_preds = [p for p in predicates if p["table"].lower() != seed_table]

    partner_table = join_spec["table_right"].lower() if seed_side == "left" else join_spec["table_left"].lower()
    tau = float(join_spec["distance_threshold"])

    # Find TI table (single = largest rung, for strategy A/B cost model and
    # direct-enumeration path). The ladder lets synthesize try small tables
    # first and escalate only if needed.
    from ours.utils import find_ti_table, find_ti_table_ladder
    t0 = time.perf_counter()
    ti_table = find_ti_table(conn, join_spec["table_left"].lower(),
                             join_spec["table_right"].lower(), tau)
    ti_ladder = find_ti_table_ladder(conn, join_spec["table_left"].lower(),
                                     join_spec["table_right"].lower(), tau)
    prof["ti_lookup"] = time.perf_counter() - t0
    prof["ti_ladder_size"] = len(ti_ladder)

    # Partner selectivity: use precomputed value if available
    precomputed_sel = query.get("precomputed_selectivity", {})
    t0 = time.perf_counter()
    if "sel_partner" in precomputed_sel:
        sigma_partner = precomputed_sel["sel_partner"]
    else:
        sigma_partner = _estimate_selectivity(conn, _resolve_hnsw_table(partner_table), partner_preds)
    prof["selectivity_estimation"] = time.perf_counter() - t0
    prof["sigma_partner"] = sigma_partner

    # Strategy A/B decision based on expected_valid_tuples.
    # expected_valid_tuples = TI_size × σ_partner × σ_threshold
    # (expected matching rows in the TI table, not distinct seed IDs)
    # When small (< threshold), JIM lookup is cheap → use A (pre-filter).
    # When large, JIM lookup dominates → use B (post-filter).
    EVT_THRESHOLD = float(query.get("w6_evt_threshold", 10000))
    evt = precomputed_sel.get("expected_valid_tuples")
    if evt is None:
        ti_table_size = precomputed_sel.get("ti_table_size")
        if ti_table_size is None:
            ti_table_size = _reltuples(conn, ti_table)
        sel_seed = precomputed_sel.get("sel_seed", 1.0)
        n_left = _reltuples(conn, _resolve_hnsw_table(join_spec["table_left"]))
        n_right = _reltuples(conn, _resolve_hnsw_table(join_spec["table_right"]))
        denom = sel_seed * sigma_partner * n_left * n_right
        expected_candidates = query.get("expected_candidate_tuples", 0)
        sigma_threshold = expected_candidates / denom if denom > 0 else 0
        evt = ti_table_size * sigma_partner * sigma_threshold
    prof["expected_valid_tuples"] = evt
    prof["evt_threshold"] = EVT_THRESHOLD

    # Strategy dispatch
    if evt < EVT_THRESHOLD:
        # Case C: few tuples → direct enumeration, no FSS needed
        prof["strategy"] = "C"
        out = _run_w6_strategy_c(
            conn, query, ti_table=ti_table,
            seed_table=seed_table, partner_table=partner_table,
            seed_side=seed_side, tau=tau,
            seed_preds=seed_preds, partner_preds=partner_preds,
            sig=sig, k=k, prof=prof,
        )
        prof["total"] = time.perf_counter() - t_total
        def _round(d):
            return {k: (_round(v) if isinstance(v, dict) else round(v, 6) if isinstance(v, (int, float)) else v) for k, v in d.items()}
        prof = _round(prof)
        return out, prof

    # Case B: expected_valid_tuples large → post-filter
    prof["strategy"] = "B"
    fss_preds = list(seed_preds)

    t0 = time.perf_counter()
    fss = FilteredScoreStreamer(
        conn=conn,
        table=_resolve_hnsw_table(seed_table),
        scoring_signal={
            "type": "semantic",
            "field": sig["field"],
            "query_embed": sig["query_embed"],
            "metric": sig.get("metric", "l2"),
        },
        predicates=fss_preds,
        precomputed_selectivity=precomputed_sel.get("sel_seed"),
        light_mode=True,
        query_k=k,
        id_col=_table_pk(seed_table),
        text_col=TEXT_COL,
    )
    prof["fss_init"] = time.perf_counter() - t0
    prof["fss_init.table_size"] = fss.profile["init_table_size"]
    prof["fss_init.selectivity"] = fss.profile["init_selectivity"]

    # pgvector HNSW leaves dirty sort state on the connection, so
    # synthesize (ORDER BY on TI table) must use a separate connection.
    syn_conn = psycopg.connect(DATABASE_URL, row_factory=dict_row, autocommit=True)

    out: List[Tuple] = []
    seen_seeds: set = set()
    prof["synthesize"] = 0.0
    n_synthesize_calls = 0
    n_synthesize_miss = 0
    seed_pk = _table_pk(seed_table)
    try:
        while len(out) < k:
            item = fss.next()
            if item is None:
                break
            entry, score = item

            seed_id = entry[seed_pk]
            if seed_id in seen_seeds:
                continue
            seen_seeds.add(seed_id)

            t0 = time.perf_counter()
            partner_id = _synthesize_w6(
                syn_conn, seed_id,
                ti_ladder=ti_ladder,
                seed_table=seed_table,
                partner_table=partner_table,
                tau=tau,
                partner_preds=partner_preds,
            )
            prof["synthesize"] += time.perf_counter() - t0
            n_synthesize_calls += 1
            if partner_id is None:
                n_synthesize_miss += 1
                continue

            if seed_side == "left":
                out.append((seed_id, partner_id, score))
            else:
                out.append((partner_id, seed_id, score))
    finally:
        syn_conn.close()

    prof["fss_seed.fetch_db_query"] = fss.profile["fetch_db_query"]
    prof["fss_seed.fetch_postfilter"] = fss.profile["fetch_postfilter"]
    prof["fss_seed"] = (
        prof["fss_seed.fetch_db_query"] + prof["fss_seed.fetch_postfilter"]
    )
    prof["n_seeds_scanned"] = len(seen_seeds)
    prof["n_synthesize_calls"] = n_synthesize_calls
    prof["n_synthesize_miss"] = n_synthesize_miss
    prof["total"] = time.perf_counter() - t_total

    def _round(d):
        return {k: (_round(v) if isinstance(v, dict) else round(v, 6) if isinstance(v, (int, float)) else v) for k, v in d.items()}
    prof = _round(prof)

    return out, prof


def run_w7(conn: psycopg.Connection, query: Dict[str, Any], eps: float = 0.1,
           ti_chunk_step: float = 0.05, time_budget: float = 30.0,
           **kwargs) -> List[Tuple]:
    """W7: Explicit threshold algorithm with differential stream sizing.

    Best-effort deadline: returns whatever top-K has been found so far.
    """
    prof = {}
    t_total = time.perf_counter()
    deadline = t_total + time_budget

    predicates = query.get("predicates", [])
    join_spec = query.get("join", {})
    scoring = query.get("scoring", {})
    k = int(query.get("K", 20))

    signals = scoring.get("signals", [])
    agg = scoring.get("aggregation", "identity")
    weights = scoring.get("weights", [1.0] * len(signals))

    t_left = join_spec["table_left"]
    t_right = join_spec["table_right"]
    tau = float(join_spec["distance_threshold"])

    # Determine stream batch sizes: S for entity, S^2 for join
    S = max(int(k ** 0.5) + 5, 20)
    S_join = S * S

    # Build streams (sub-profiled internally)
    streams = _build_w7_streams(conn, signals, predicates, join_spec, prof,
                                init_stream_fetchKs={"left": S, "right": S, "join": S_join},
                                precomputed_selectivity=query.get("precomputed_selectivity"),
                                ti_chunk_step=ti_chunk_step)

    # Build score function
    score_f = _build_score_f(agg, weights)

    # Run threshold algorithm (sub-profiled internally)
    results = _run_w7_threshold(conn, streams, k, S, S_join, score_f, join_spec,
                                predicates, prof, eps=eps, deadline=deadline)

    prof["total"] = time.perf_counter() - t_total

    # Round all values recursively
    def _round(d):
        return {k: (_round(v) if isinstance(v, dict) else round(v, 6) if isinstance(v, (int, float)) else v) for k, v in d.items()}
    prof = _round(prof)

    return _format_join_results(results, t_left, t_right), prof


def _build_w7_streams(
    conn: psycopg.Connection,
    signals: List[Dict[str, Any]],
    predicates: List[Dict[str, Any]],
    join_spec: Dict[str, Any],
    prof: Dict[str, Any] = None,
    init_stream_fetchKs: Optional[Dict[str, int]] = None,
    join_conn: Optional[psycopg.Connection] = None,
    precomputed_selectivity: Optional[Dict[str, float]] = None,
    ti_chunk_step: float = 0.05,
    light_mode: bool = False,
) -> Dict[str, Any]:
    """Build FSS streams for W7 by role: left, right, join."""
    from ours.fss import FilteredScoreStreamer
    from ours.ti_stream import TIScoreStream
    from ours.utils import find_ti_table_ladder
    if prof is None:
        prof = {}
    if init_stream_fetchKs is None:
        init_stream_fetchKs = {}
    if precomputed_selectivity is None:
        precomputed_selectivity = {}

    def norm_table(name: str) -> str:
        n = name.lower()
        if n.endswith("_hnsw"):
            n = n[:-5]
        return n

    t_left = norm_table(join_spec["table_left"])
    t_right = norm_table(join_spec["table_right"])
    tau = float(join_spec["distance_threshold"])

    t0 = time.perf_counter()
    ti_ladder = find_ti_table_ladder(conn, join_spec["table_left"], join_spec["table_right"], tau)
    prof["stream_build.ti_lookup"] = time.perf_counter() - t0
    prof["stream_build.ti_ladder"] = [t for _, t in ti_ladder]

    streams = {"left": None, "right": None, "join": None}
    stream_weights = {"left": None, "right": None, "join": None}

    for i, sig in enumerate(signals):
        sig_type = sig["type"]

        if sig_type == "semantic":
            table = norm_table(sig["table"])
            role = "left" if table == t_left else "right"
            table_preds = [p for p in predicates if norm_table(p["table"]) == table]

            sel_role = precomputed_selectivity.get(f"sel_{role}")
            assert sel_role is not None, (
                f"w7 requires precomputed sel_{role} in query['precomputed_selectivity']; "
                f"runtime estimation disabled"
            )

            t0 = time.perf_counter()
            fss = FilteredScoreStreamer(
                conn=conn,
                table=_resolve_hnsw_table(sig["table"]),
                scoring_signal={
                    "type": "semantic",
                    "field": sig["field"],
                    "query_embed": sig["query_embed"],
                    "metric": sig.get("metric", "l2"),
                },
                predicates=table_preds,
                init_stream_fetchK=init_stream_fetchKs.get(role),
                precomputed_selectivity=sel_role,
                light_mode=light_mode,
                id_col=_table_pk(sig["table"]),
                text_col=_table_text(sig["table"]),
            )
            prof[f"stream_build.fss_init_{role}"] = time.perf_counter() - t0
            prof[f"stream_build.fss_init_{role}.table_size"] = fss.profile["init_table_size"]
            prof[f"stream_build.fss_init_{role}.selectivity"] = fss.profile["init_selectivity"]
            prof[f"fss_{role}.sigma"] = fss.profile.get("sigma")
            prof[f"fss_{role}.strategy"] = fss.profile.get("strategy")
            prof[f"fss_{role}.n_rows"] = fss.profile.get("n_rows")
            streams[role] = fss
            stream_weights[role] = i

        elif sig_type == "join_distance":
            ti_preds = [{
                "attribute": f"{norm_table(p['table'])}.{p['attribute']}",
                "operator": p["operator"],
                "value": p["value"],
            } for p in predicates]

            left_pk = f"{t_left}.{_table_pk(t_left)}"
            right_pk = f"{t_right}.{_table_pk(t_right)}"
            t0 = time.perf_counter()
            fss = TIScoreStream(
                conn=join_conn or conn,
                tables=ti_ladder,
                tau=tau,
                predicates=ti_preds,
                pk_cols=[left_pk, right_pk],
                select_cols=[left_pk, right_pk],
                init_fetch_K=init_stream_fetchKs.get("join"),
            )
            prof["stream_build.fss_init_join"] = time.perf_counter() - t0
            prof["stream_build.fss_init_join.table_size"] = fss.profile["init_table_size"]
            prof["stream_build.fss_init_join.selectivity"] = fss.profile["init_selectivity"]
            prof["fss_join.strategy"] = fss.strategy
            streams["join"] = fss
            stream_weights["join"] = i

    # Store signal info for direct score computation on join-stream entities
    signal_info = {}
    for i, sig in enumerate(signals):
        if sig["type"] == "semantic":
            table = norm_table(sig["table"])
            role = "left" if table == t_left else "right"
            signal_info[role] = {
                "query_embed": sig["query_embed"],
                "field": sig["field"],
                "metric": sig.get("metric", "l2"),
                "table": sig["table"].lower(),
            }

    return {"streams": streams, "weights": stream_weights, "signal_info": signal_info}


def _run_w7_threshold(
    conn: psycopg.Connection,
    stream_info: Dict[str, Any],
    k: int,
    S: int,
    S_join: int,
    score_f: Any,
    join_spec: Dict[str, Any],
    predicates: List[Dict[str, Any]],
    prof: Dict[str, float] = None,
    eps: float = 0.0,
    deadline: float = None,
) -> List[Tuple[Dict, Dict, float]]:
    """
    Threshold algorithm for W7.

    Each iteration: next() one item from left and right streams.
    Join stream grows quadratically: k→k+1 needs 2k+1 more join pairs.
    Stops when worst_in_heap <= upper_bound + eps, streams exhausted,
    or deadline is reached (best-effort).
    """
    import heapq
    if prof is None:
        prof = {}

    streams = stream_info["streams"]
    weights = stream_info["weights"]
    n_signals = max(v for v in weights.values() if v is not None) + 1

    t_left = join_spec["table_left"]
    t_right = join_spec["table_right"]
    tau = float(join_spec["distance_threshold"])

    assert streams["left"] is not None, "W7 requires left semantic stream"
    assert streams["right"] is not None, "W7 requires right semantic stream"
    assert streams["join"] is not None, "W7 requires join_distance stream"

    heap: List[Tuple[float, int, Tuple]] = []
    seen_pairs: set = set()
    tie_breaker = 0

    # Signal info for direct score computation on join-stream entities
    sig_info = stream_info.get("signal_info", {})

    # Entity + score caches (populated from both entity streams and join-stream lookups)
    entity_cache_left: Dict[int, Dict] = {}
    entity_cache_right: Dict[int, Dict] = {}
    score_cache_left: Dict[int, float] = {}
    score_cache_right: Dict[int, float] = {}

    # All seen entities from left/right streams (ordered)
    all_left: List[Tuple[Dict, float]] = []
    all_right: List[Tuple[Dict, float]] = []

    # Profiling accumulators
    prof["random_access"] = 0.0
    prof["distance_compute"] = 0.0

    iteration = 0
    left_exhausted = False
    right_exhausted = False
    join_exhausted = False

    while True:
        if deadline is not None and time.perf_counter() >= deadline:
            prof["early_stop"] = "deadline"
            break

        iteration += 1

        left_pk = _table_pk(t_left)
        right_pk = _table_pk(t_right)

        # --- Pull one item from left stream ---
        new_left_entry = None
        if not left_exhausted:
            item = streams["left"].next()
            if item is None:
                left_exhausted = True
            else:
                new_left_entry = item
                entry, score = item
                eid = entry.get(left_pk)
                if eid is not None:
                    entity_cache_left[eid] = entry
                    score_cache_left[eid] = score
                all_left.append(item)

        # --- Pull one item from right stream ---
        new_right_entry = None
        if not right_exhausted:
            item = streams["right"].next()
            if item is None:
                right_exhausted = True
            else:
                new_right_entry = item
                entry, score = item
                eid = entry.get(right_pk)
                if eid is not None:
                    entity_cache_right[eid] = entry
                    score_cache_right[eid] = score
                all_right.append(item)

        # --- Pull from join stream: need (k+1)^2 - k^2 = 2k+1 new pairs ---
        n_prev = iteration - 1
        join_need = 2 * n_prev + 1
        new_join_pairs = []
        if not join_exhausted:
            for _ in range(join_need):
                item = streams["join"].next()
                if item is None:
                    join_exhausted = True
                    break
                ti_entry, score_dis = item
                left_id = ti_entry.get(f"{t_left.lower()}.{_table_pk(t_left)}")
                right_id = ti_entry.get(f"{t_right.lower()}.{_table_pk(t_right)}")
                if left_id is None or right_id is None:
                    continue
                pair_key = (left_id, right_id)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                new_join_pairs.append((left_id, right_id, score_dis))

        # --- Batch score join pairs via SQL distance ---
        if new_join_pairs:
            missing_left_ids = list({lid for lid, _, _ in new_join_pairs if lid not in score_cache_left})
            missing_right_ids = list({rid for _, rid, _ in new_join_pairs if rid not in score_cache_right})

            t0 = time.perf_counter()
            left_ids_q = missing_left_ids if ("left" in sig_info) else []
            right_ids_q = missing_right_ids if ("right" in sig_info) else []
            if left_ids_q or right_ids_q:
                l_scores, r_scores = _batch_score_sql_two_sides(
                    conn,
                    t_left, sig_info.get("left", {}), left_ids_q,
                    t_right, sig_info.get("right", {}), right_ids_q,
                )
                score_cache_left.update(l_scores)
                score_cache_right.update(r_scores)
            prof["random_access"] += time.perf_counter() - t0

            for left_id, right_id, score_dis in new_join_pairs:
                if left_id not in score_cache_left or right_id not in score_cache_right:
                    continue
                scores_list = [0.0] * n_signals
                if weights["left"] is not None:
                    scores_list[weights["left"]] = score_cache_left[left_id]
                if weights["right"] is not None:
                    scores_list[weights["right"]] = score_cache_right[right_id]
                if weights["join"] is not None:
                    scores_list[weights["join"]] = score_dis
                total_score = score_f(scores_list)

                if len(heap) < k:
                    heapq.heappush(heap, (-total_score, tie_breaker, (left_id, right_id, total_score)))
                    tie_breaker += 1
                elif total_score < -heap[0][0]:
                    heapq.heapreplace(heap, (-total_score, tie_breaker, (left_id, right_id, total_score)))
                    tie_breaker += 1

        # --- Process entity cross-product: new_left × all_right + old_left × new_right ---
        # Cross-product only discovers pairs where BOTH entities are in their
        # semantic streams.  Valid unseen pairs fall into three cases:
        #   (1) t1_unseen, t2_unseen, sj_unseen  → discoverable by join stream
        #   (2) t1_unseen, t2_seen,   sj_unseen  → NOT covered by cross-product
        #   (3) t1_seen,   t2_unseen, sj_unseen  → NOT covered by cross-product
        # Note: (t1_seen, t2_seen) pairs are always scored by cross-product, so
        # no valid unseen pair can have both sides seen.  Case (1) is dominated
        # by (2) and (3) in the bound, so the correct termination bound is:
        #   min(w·top_left  + w·peek_right + w·peek_join,   // case 3
        #       w·peek_left + w·top_right  + w·peek_join)   // case 2
        # where top_left/top_right = best score among SEEN entities (<=peek).
        # This is tighter than the naive peek_left+peek_right+peek_join bound,
        # causing the algorithm to run longer and discover more pairs.
        if new_left_entry is not None:
            left_entry, left_score = new_left_entry
            for right_entry, right_score in all_right:
                _w7_try_entity_pair(
                    left_entry, left_score, right_entry, right_score,
                    join_spec, tau, weights, n_signals, score_f,
                    seen_pairs, heap, k, tie_breaker, prof,
                )
                tie_breaker += 1

        if new_right_entry is not None:
            right_entry, right_score = new_right_entry
            # Exclude the new left entry (already covered above)
            for left_entry, left_score in all_left[:-1] if new_left_entry else all_left:
                _w7_try_entity_pair(
                    left_entry, left_score, right_entry, right_score,
                    join_spec, tau, weights, n_signals, score_f,
                    seen_pairs, heap, k, tie_breaker, prof,
                )
                tie_breaker += 1

        # --- Join stream exhausted: all valid candidates have been scored ---
        if join_exhausted:
            prof["early_stop"] = "join_exhausted"
            break

        # --- Check termination bound ---
        peek_left = streams["left"].peek_score()
        peek_right = streams["right"].peek_score()
        peek_join = streams["join"].peek_score()

        pl = peek_left if peek_left is not None else (all_left[-1][1] if all_left else 0.0)
        pr = peek_right if peek_right is not None else (all_right[-1][1] if all_right else 0.0)
        pj = peek_join if peek_join is not None else 0.0
        tl = all_left[0][1] if all_left else pl
        tr = all_right[0][1] if all_right else pr

        # Case 2: t1_unseen, t2_seen → use top_right (best seen) instead of peek_right
        bound2 = [0.0] * n_signals
        if weights["left"] is not None:
            bound2[weights["left"]] = pl
        if weights["right"] is not None:
            bound2[weights["right"]] = tr
        if weights["join"] is not None:
            bound2[weights["join"]] = pj

        # Case 3: t1_seen, t2_unseen → use top_left (best seen) instead of peek_left
        bound3 = [0.0] * n_signals
        if weights["left"] is not None:
            bound3[weights["left"]] = tl
        if weights["right"] is not None:
            bound3[weights["right"]] = pr
        if weights["join"] is not None:
            bound3[weights["join"]] = pj

        upper_bound = min(score_f(bound2), score_f(bound3))

        kth_best = -heap[0][0] if len(heap) >= k else float("inf")

        # --- Debug: print every 100 iterations ---
        if iteration % 100 == 0:
            worst = -heap[0][0] if heap else float("inf")
            print(f"  iter={iteration:>4d}  kth={kth_best:.4f}  bound={upper_bound:.4f}  heap={len(heap)}  seen={len(seen_pairs)}  L={len(all_left)} R={len(all_right)}",
                  file=sys.stderr)
            
        if len(heap) >= k and kth_best <= upper_bound + eps:
            break

        # All streams exhausted and nothing new
        if left_exhausted and right_exhausted and join_exhausted:
            break
        if new_left_entry is None and new_right_entry is None and join_exhausted:
            break

    # --- Profiling ---
    prof["fss_left"] = streams["left"].profile["fetch_db_query"] + streams["left"].profile["fetch_postfilter"]
    prof["fss_right"] = streams["right"].profile["fetch_db_query"] + streams["right"].profile["fetch_postfilter"]
    prof["fss_join"] = streams["join"].profile["fetch_db_query"] + streams["join"].profile["fetch_postfilter"]
    prof["stream_len_left"] = len(all_left)
    prof["stream_len_right"] = len(all_right)
    prof["n_seen_pairs"] = len(seen_pairs)
    prof["iterations"] = iteration

    # --- Materialize final K results: batch-fetch missing entities ---
    final_left_ids = set()
    final_right_ids = set()
    for _, _, item in heap:
        left_v, right_v, _ = item
        if isinstance(left_v, int):
            if left_v not in entity_cache_left:
                final_left_ids.add(left_v)
            if right_v not in entity_cache_right:
                final_right_ids.add(right_v)

    t0 = time.perf_counter()
    if final_left_ids:
        entity_cache_left.update(_fetch_entities_batch(conn, t_left, list(final_left_ids)))
    if final_right_ids:
        entity_cache_right.update(_fetch_entities_batch(conn, t_right, list(final_right_ids)))
    prof["random_access"] += time.perf_counter() - t0

    results_sorted = sorted(heap, key=lambda x: -x[0])
    final = []
    for _, _, item in results_sorted:
        left_v, right_v, sc = item
        if isinstance(left_v, int):
            left_ent = entity_cache_left.get(left_v)
            right_ent = entity_cache_right.get(right_v)
            if not left_ent or not right_ent:
                continue
            final.append((left_ent, right_ent, sc))
        else:
            final.append((left_v, right_v, sc))
    return final


def _w7_try_entity_pair(
    left_entry, left_score, right_entry, right_score,
    join_spec, tau, weights, n_signals, score_f,
    seen_pairs, heap, k, tie_breaker, prof,
):
    """Try a (left, right) entity pair from the cross-product streams."""
    left_pk = _table_pk(join_spec["table_left"])
    right_pk = _table_pk(join_spec["table_right"])
    left_id = left_entry.get(left_pk)
    right_id = right_entry.get(right_pk)
    if left_id is None or right_id is None:
        return
    pair_key = (left_id, right_id)
    if pair_key in seen_pairs:
        return
    seen_pairs.add(pair_key)

    left_emb = left_entry.get(join_spec["embed_left"])
    right_emb = right_entry.get(join_spec["embed_right"])
    if left_emb is None or right_emb is None:
        return

    t0 = time.perf_counter()
    dist = _compute_distance(left_emb, right_emb, join_spec.get("metric", "l2"))
    prof["distance_compute"] += time.perf_counter() - t0
    if dist > tau:
        return

    scores_list = [0.0] * n_signals
    if weights["left"] is not None:
        scores_list[weights["left"]] = left_score
    if weights["right"] is not None:
        scores_list[weights["right"]] = right_score
    if weights["join"] is not None:
        scores_list[weights["join"]] = dist
    total_score = score_f(scores_list)

    if len(heap) < k:
        heapq.heappush(heap, (-total_score, tie_breaker, (left_entry, right_entry, total_score)))
    elif total_score < -heap[0][0]:
        heapq.heapreplace(heap, (-total_score, tie_breaker, (left_entry, right_entry, total_score)))


def _batch_score_sql(
    conn: psycopg.Connection, table: str, sig_info: Dict[str, Any], ids: List[int],
) -> Dict[int, float]:
    """Compute semantic distances in SQL. Returns {id: distance}."""
    if not ids:
        return {}
    from ours.utils import METRIC_OP
    from pgvector import Vector
    actual_table = _resolve_hnsw_table(table)
    pk = _table_pk(table)
    metric = sig_info["metric"]
    dist_op = METRIC_OP[metric]
    field = sig_info["field"]
    if metric == "jaccard":
        qe = sig_info["query_embed"]
        qv_param = "".join(str(int(x)) for x in qe)
        cast = f"::bit({len(qe)})"
        op_sql = dist_op.replace("%", "%%")
        sql = (
            f'SELECT {_quote(pk)}, ({_quote(field)} {op_sql} %(qv)s{cast}) AS score '
            f'FROM {_quote(actual_table)} WHERE {_quote(pk)} = ANY(%(ids)s)'
        )
    else:
        qv_param = Vector(sig_info["query_embed"])
        sql = (
            f'SELECT {_quote(pk)}, ({_quote(field)} {dist_op} %(qv)s) AS score '
            f'FROM {_quote(actual_table)} WHERE {_quote(pk)} = ANY(%(ids)s)'
        )
    with conn.cursor(row_factory=psycopg.rows.tuple_row) as cur:
        cur.execute(sql, {"qv": qv_param, "ids": list(ids)})
        return {row[0]: float(row[1]) for row in cur.fetchall()}


def _batch_score_sql_two_sides(
    conn: psycopg.Connection,
    left_table: str, left_sig: Dict[str, Any], left_ids: List[int],
    right_table: str, right_sig: Dict[str, Any], right_ids: List[int],
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """One-round-trip UNION ALL variant: score left-side and right-side ids in a
    single SQL, returns (left_scores, right_scores).

    The W7 inner loop calls this ~n_iter times; batching both sides halves DB
    round-trips vs calling _batch_score_sql twice.
    """
    if not left_ids and not right_ids:
        return {}, {}
    from ours.utils import METRIC_OP
    from pgvector import Vector

    def _side_sql(
        tag: str, table: str, sig: Dict[str, Any], ids_key: str, qv_key: str,
    ) -> Tuple[str, Any]:
        actual_table = _resolve_hnsw_table(table)
        pk = _table_pk(table)
        metric = sig["metric"]
        dist_op = METRIC_OP[metric]
        field = sig["field"]
        if metric == "jaccard":
            qe = sig["query_embed"]
            qv_value = "".join(str(int(x)) for x in qe)
            cast = f"::bit({len(qe)})"
            op_sql = dist_op.replace("%", "%%")
            frag = (
                f"SELECT '{tag}'::text AS side, {_quote(pk)} AS id, "
                f"({_quote(field)} {op_sql} %({qv_key})s{cast}) AS score "
                f"FROM {_quote(actual_table)} WHERE {_quote(pk)} = ANY(%({ids_key})s)"
            )
        else:
            qv_value = Vector(sig["query_embed"])
            frag = (
                f"SELECT '{tag}'::text AS side, {_quote(pk)} AS id, "
                f"({_quote(field)} {dist_op} %({qv_key})s) AS score "
                f"FROM {_quote(actual_table)} WHERE {_quote(pk)} = ANY(%({ids_key})s)"
            )
        return frag, qv_value

    frags: List[str] = []
    params: Dict[str, Any] = {}
    if left_ids:
        frag, qv = _side_sql("L", left_table, left_sig, "lids", "lqv")
        frags.append(frag)
        params["lids"] = list(left_ids)
        params["lqv"] = qv
    if right_ids:
        frag, qv = _side_sql("R", right_table, right_sig, "rids", "rqv")
        frags.append(frag)
        params["rids"] = list(right_ids)
        params["rqv"] = qv

    sql = " UNION ALL ".join(frags)
    left_out: Dict[int, float] = {}
    right_out: Dict[int, float] = {}
    with conn.cursor(row_factory=psycopg.rows.tuple_row) as cur:
        cur.execute(sql, params)
        for side, rid, score in cur.fetchall():
            if side == "L":
                left_out[rid] = float(score)
            else:
                right_out[rid] = float(score)
    return left_out, right_out


def _resolve_hnsw_table(table: str) -> str:
    """Map base table name to HNSW variant (e.g., imdb_T1 -> imdb_t1_hnsw)."""
    base_name = table.lower().replace("_hnsw", "").replace("_ivf", "").replace("_1000", "")
    return f"{base_name}_hnsw"


def _reltuples(conn: psycopg.Connection, table: str) -> int:
    """Return pg_class.reltuples for a table (cheap size estimate, no COUNT)."""
    with conn.cursor(row_factory=psycopg.rows.tuple_row) as cur:
        cur.execute(
            "SELECT reltuples::bigint FROM pg_class WHERE relname = %s",
            (table.lower(),),
        )
        row = cur.fetchone()
    return int(row[0]) if row and row[0] else 0


def _fetch_entity(conn: psycopg.Connection, table: str, entity_id: int) -> Optional[Dict]:
    """Fetch a single entity by ID from the HNSW base table."""
    actual_table = _resolve_hnsw_table(table)
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(f'SELECT * FROM {_quote(actual_table)} WHERE id = %s', (entity_id,))
        return cur.fetchone()


def _fetch_entities_batch(
    conn: psycopg.Connection, table: str, ids: List[int],
) -> Dict[int, Dict]:
    """Batch-fetch entities by IDs. Returns {id: entity_dict}."""
    if not ids:
        return {}
    actual_table = _resolve_hnsw_table(table)
    pk = _table_pk(table)
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(
            f'SELECT * FROM {_quote(actual_table)} WHERE {_quote(pk)} = ANY(%s)',
            (list(ids),),
        )
        rows = cur.fetchall()
    return {r[pk]: dict(r) for r in rows}


def _compute_distance(emb1: Any, emb2: Any, metric: str) -> float:
    """Compute distance between two embeddings."""
    from pgvector import Vector
    import numpy as np

    v1 = np.array(emb1) if not isinstance(emb1, np.ndarray) else emb1
    v2 = np.array(emb2) if not isinstance(emb2, np.ndarray) else emb2

    if metric == "l2":
        return float(np.linalg.norm(v1 - v2))
    elif metric == "cos":
        return 1.0 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    elif metric == "ip":
        return -float(np.dot(v1, v2))
    return 0.0


def _run_w8_strategy_c(
    conn: psycopg.Connection,
    query: Dict[str, Any],
    ti_table: str,
    tau: float,
    predicates: List[Dict[str, Any]],
    signals: List[Dict[str, Any]],
    score_f: Any,
    k: int,
    prof: Dict[str, Any],
) -> List[Tuple]:
    """W8 Case C: direct TI enumeration for small expected_valid_tuples.

    1. SELECT (left_id, right_id, dis) FROM ti WHERE predicates AND dis<=tau
    2. Batch score distinct left_ids and right_ids on their semantic signals
    3. For each (l, r, dis) compute score_f([left_score, right_score, dis]) and
       keep top-K

    Mirrors _run_w6_strategy_c but 3-signal (left semantic + right semantic +
    join distance) instead of single-signal. No ORDER BY in the TI query —
    lets planner pick bitmap-on-predicate-index which is optimal for selective
    predicates.
    """
    join_spec = query["join"]
    t_left = join_spec["table_left"].lower()
    t_right = join_spec["table_right"].lower()
    left_pk_col = f"{t_left}.{_table_pk(t_left)}"
    right_pk_col = f"{t_right}.{_table_pk(t_right)}"

    where_parts: List[str] = ['"dis" <= %(tau)s']
    params: Dict[str, Any] = {"tau": tau}
    for i, p in enumerate(predicates):
        tbl = p["table"].lower()
        attr = p["attribute"]
        op = p["operator"]
        val = p["value"]
        ti_col = f"{tbl}.{attr}"
        key = f"cp{i}"
        if op == "in":
            where_parts.append(f'"{ti_col}" = ANY(%({key})s)')
            params[key] = list(val) if not isinstance(val, list) else val
        else:
            where_parts.append(f'"{ti_col}" {op} %({key})s')
            params[key] = val

    t0 = time.perf_counter()
    sql = (
        f'SELECT "{left_pk_col}", "{right_pk_col}", "dis" '
        f'FROM "{ti_table}" '
        f'WHERE {" AND ".join(where_parts)}'
    )
    with conn.cursor(row_factory=psycopg.rows.tuple_row) as cur:
        cur.execute(sql, params)
        ti_rows = cur.fetchall()
    prof["c_ti_query"] = time.perf_counter() - t0
    prof["c_ti_rows"] = len(ti_rows)

    if not ti_rows:
        return []

    # Resolve signal → role (left / right / join), mirroring _build_w7_streams.
    # weights[role] = signal index in the scoring signals list.
    roles: Dict[str, Dict[str, Any]] = {}  # role -> sig
    role_idx: Dict[str, int] = {}           # role -> signal index
    for i, sig in enumerate(signals):
        if sig["type"] == "semantic":
            tbl = sig["table"].lower()
            role = "left" if tbl == t_left else "right"
            roles[role] = sig
            role_idx[role] = i
        elif sig["type"] == "join_distance":
            role_idx["join"] = i
    n_signals = len(signals)

    # Batch score distinct left/right ids via one UNION ALL SQL (same helper
    # as W7 inner loop).
    left_ids = list({r[0] for r in ti_rows})
    right_ids = list({r[1] for r in ti_rows})

    t0 = time.perf_counter()
    l_scores, r_scores = _batch_score_sql_two_sides(
        conn,
        t_left, roles.get("left", {}), left_ids if "left" in roles else [],
        t_right, roles.get("right", {}), right_ids if "right" in roles else [],
    )
    prof["c_batch_score"] = time.perf_counter() - t0

    # Score every (l, r, dis) tuple.
    scored: List[Tuple[int, int, float]] = []
    for lid, rid, dis in ti_rows:
        scores_list = [0.0] * n_signals
        if "left" in role_idx:
            if lid not in l_scores:
                continue
            scores_list[role_idx["left"]] = l_scores[lid]
        if "right" in role_idx:
            if rid not in r_scores:
                continue
            scores_list[role_idx["right"]] = r_scores[rid]
        if "join" in role_idx:
            scores_list[role_idx["join"]] = float(dis)
        total = score_f(scores_list)
        scored.append((lid, rid, total))
    prof["c_n_scored"] = len(scored)

    # Top-K by lowest total score (smaller = better for L2/jaccard-style signals).
    scored.sort(key=lambda x: x[2])
    return scored[:k]


def run_w8(conn: psycopg.Connection, query: Dict[str, Any], eps: float = 0.1,
           time_budget: float = 30.0, ti_chunk_step: float = 0.05,
           evt_threshold: float = 10000.0, **kwargs) -> List[Tuple]:
    """W8(S, P, J, K): multi-signal + predicates + join.

    Two strategies, chosen by `expected_valid_tuples` (evt = TI rows matching
    predicates AND dis<=tau — taken from query.expected_candidate_tuples or
    computed from precomputed_selectivity):

    Case C — evt < evt_threshold (default 10k): direct TI enumeration in a
      single SQL, batch score, top-K. Cheap for small evt — avoids the full
      threshold algorithm overhead.

    Case A — evt ≥ evt_threshold: threshold algorithm with left/right/join
      streams (same as W7 but with predicates). Uses fresh per-query connections
      to isolate HNSW GUCs from the btree TI stream. Best-effort deadline.
    """
    prof = {}
    t_total = time.perf_counter()
    deadline = t_total + time_budget

    predicates = query.get("predicates", [])
    join_spec = query.get("join", {})
    scoring = query.get("scoring", {})
    k = int(query.get("K", 20))

    signals = scoring.get("signals", [])
    agg = scoring.get("aggregation", "identity")
    weights = scoring.get("weights", [1.0] * len(signals))
    score_f = _build_score_f(agg, weights)

    # --- Case C dispatch ---
    evt = query.get("expected_candidate_tuples")
    prof["expected_valid_tuples"] = evt
    prof["evt_threshold"] = evt_threshold

    if evt is not None and evt < evt_threshold:
        prof["strategy"] = "C"
        tau = float(join_spec["distance_threshold"])
        from ours.utils import find_ti_table
        ti_table = find_ti_table(conn, join_spec["table_left"].lower(),
                                 join_spec["table_right"].lower(), tau)
        top_triples = _run_w8_strategy_c(
            conn, query, ti_table=ti_table, tau=tau,
            predicates=predicates, signals=signals, score_f=score_f,
            k=k, prof=prof,
        )

        # Case C returns (lid, rid, score); fetch text cols for display on top-K only.
        t_left = join_spec["table_left"]
        t_right = join_spec["table_right"]
        l_text_col = _table_text(t_left)
        r_text_col = _table_text(t_right)
        l_pk = _table_pk(t_left)
        r_pk = _table_pk(t_right)

        results: List[Tuple] = []
        if top_triples:
            left_ids = list({t[0] for t in top_triples})
            right_ids = list({t[1] for t in top_triples})
            with conn.cursor(row_factory=psycopg.rows.tuple_row) as cur:
                cur.execute(
                    f'SELECT {_quote(l_pk)}, {_quote(l_text_col)} FROM '
                    f'{_quote(_resolve_hnsw_table(t_left))} WHERE {_quote(l_pk)} = ANY(%s)',
                    (left_ids,),
                )
                l_txt = {r[0]: r[1] for r in cur.fetchall()}
                cur.execute(
                    f'SELECT {_quote(r_pk)}, {_quote(r_text_col)} FROM '
                    f'{_quote(_resolve_hnsw_table(t_right))} WHERE {_quote(r_pk)} = ANY(%s)',
                    (right_ids,),
                )
                r_txt = {r[0]: r[1] for r in cur.fetchall()}
            for lid, rid, sc in top_triples:
                results.append((lid, rid, l_txt.get(lid), r_txt.get(rid), sc))

        prof["total"] = time.perf_counter() - t_total

        def _round(d):
            return {k: (_round(v) if isinstance(v, dict) else round(v, 6) if isinstance(v, (int, float)) else v) for k, v in d.items()}
        prof = _round(prof)
        return results, prof

    # --- Case A: threshold algorithm ---
    prof["strategy"] = "A"
    S = max(int(k ** 0.5) + 5, 20)
    S_join = S * S

    # Fresh connections: entity streams and join stream are fully isolated
    entity_conn = psycopg.connect(DATABASE_URL, row_factory=dict_row, autocommit=True)
    register_vector(entity_conn)
    entity_conn.execute("SET enable_seqscan = off")
    join_conn = psycopg.connect(DATABASE_URL, row_factory=dict_row, autocommit=True)
    register_vector(join_conn)
    join_conn.execute("SET enable_seqscan = off")

    streams = _build_w7_streams(entity_conn, signals, predicates, join_spec, prof,
                                init_stream_fetchKs={"left": S, "right": S, "join": S_join},
                                join_conn=join_conn,
                                precomputed_selectivity=query.get("precomputed_selectivity"),
                                ti_chunk_step=ti_chunk_step,
                                light_mode=True)

    results = _run_w7_threshold(entity_conn, streams, k, S, S_join, score_f,
                                join_spec, predicates, prof, eps=eps,
                                deadline=deadline)

    entity_conn.close()
    join_conn.close()

    prof["total"] = time.perf_counter() - t_total

    def _round(d):
        return {k: (_round(v) if isinstance(v, dict) else round(v, 6) if isinstance(v, (int, float)) else v) for k, v in d.items()}
    prof = _round(prof)

    return _format_join_results(results, join_spec["table_left"], join_spec["table_right"]), prof


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estimate_selectivity(
    conn: psycopg.Connection, table: str, predicates: List[Dict[str, Any]]
) -> float:
    """
    Estimate the fraction of rows in `table` passing all predicates.
    Returns σ in [0, 1]. If no predicates, returns 1.0.
    """
    if not predicates:
        return 1.0

    where_parts = []
    params: Dict[str, Any] = {}
    for i, p in enumerate(predicates):
        attr = p["attribute"]
        op = p["operator"]
        val = p["value"]
        key = f"sel{i}"
        if op == "in":
            where_parts.append(f'"{attr}" = ANY(%({key})s)')
            params[key] = list(val) if not isinstance(val, list) else val
        else:
            where_parts.append(f'"{attr}" {op} %({key})s')
            params[key] = val

    where_sql = " AND ".join(where_parts)
    sql = (
        f'SELECT COUNT(*) FILTER (WHERE {where_sql}) AS pass, '
        f'COUNT(*) AS total FROM "{table}"'
    )
    with conn.cursor(row_factory=psycopg.rows.tuple_row) as cur:
        cur.execute(sql, params)
        passed, total = cur.fetchone()
    return passed / total if total > 0 else 1.0


def _make_join_row(
    seed: Dict, partner: Dict, score: float, side: str
) -> Tuple:
    """Format a join result as (id_left, id_right, title_left, title_right, score)."""
    if side == "left":
        return (seed["id"], partner["id"], seed["title"], partner["title"], score)
    else:
        return (partner["id"], seed["id"], partner["title"], seed["title"], score)


# ---------------------------------------------------------------------------
# Synthesis helpers for W5/W6
# ---------------------------------------------------------------------------

def _synthesize_join_results(
    conn: psycopg.Connection,
    fss_results: List[Tuple[Dict, float]],
    join_spec: Dict[str, Any],
    other_predicates: List[Dict],
    side: str,
) -> List[Tuple]:
    """
    For each entry from FSS, find the best join partner.
    Returns list of (id_left, id_right, title_left, title_right, score).
    """
    t_left = join_spec["table_left"].lower()
    t_right = join_spec["table_right"].lower()
    embed_left = join_spec["embed_left"]
    embed_right = join_spec["embed_right"]
    tau = float(join_spec["distance_threshold"])
    metric = join_spec["metric"]

    out = []
    for entry, score in fss_results:
        if side == "left":
            seed_emb = entry[embed_left]
            result = synthesize(
                conn, entry, seed_emb,
                partner_table=t_right,
                partner_join_field=embed_right,
                tau=tau, metric=metric,
                predicates=other_predicates,
            )
            if result is None:
                continue
            partner, _ = result
            out.append((
                entry["id"], partner["id"],
                entry["title"], partner["title"],
                score,
            ))
        else:
            seed_emb = entry[embed_right]
            result = synthesize(
                conn, entry, seed_emb,
                partner_table=t_left,
                partner_join_field=embed_left,
                tau=tau, metric=metric,
                predicates=other_predicates,
            )
            if result is None:
                continue
            partner, _ = result
            out.append((
                partner["id"], entry["id"],
                partner["title"], entry["title"],
                score,
            ))
    return out


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_single_table(results: List[Tuple[Dict, float]]) -> List[Tuple]:
    """Format single-table FSS results → (id, title, score)."""
    return [(e[ID_COL], e[TEXT_COL], s) for e, s in results]


def _format_join_results(
    answer: List[Tuple[Dict, Dict, float]],
    t_left: str,
    t_right: str,
) -> List[Tuple]:
    """Format MSA join results → (id_left, id_right, title_left, title_right, score)."""
    lpk, ltx = _table_pk(t_left), _table_text(t_left)
    rpk, rtx = _table_pk(t_right), _table_text(t_right)
    return [
        (tl[lpk], tr[rpk], tl[ltx], tr[rtx], sc)
        for tl, tr, sc in answer
    ]


def _build_score_f(agg: str, weights: List[float]):
    """Build score aggregation function from spec."""
    if agg == "weighted_sum":
        def f(scores):
            return sum(w * s for w, s in zip(weights, scores))
        return f
    elif agg == "sum":
        return lambda scores: sum(scores)
    elif agg == "min":
        return lambda scores: min(scores)
    elif agg == "max":
        return lambda scores: max(scores)
    else:
        raise ValueError(f"Unknown aggregation: {agg}")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

DISPATCH = {
    "W1": run_w1,
    "W2": run_w2,
    "W3": run_w3,
    "W4": run_w4,
    "W5": run_w5,
    "W6": run_w6,
    "W7": run_w7,
    "W8": run_w8,
}


#W7_HARD_IDS = {"w7_070", "w7_077", "w7_099"}

TEST_QUERY_IDS = {}
SKIP_QUERY_IDS = {}

def run_query(conn: psycopg.Connection, query: Dict[str, Any], wtype: str, **kwargs):
    """Execute a single query. Returns (rows, profile) where profile may be None."""
    qid = query.get("query_id")
    if TEST_QUERY_IDS:
        if qid not in TEST_QUERY_IDS:
            return [], None
    elif qid in SKIP_QUERY_IDS:
        return [], None
    handler = DISPATCH[wtype.upper()]
    result = handler(conn, query, **kwargs)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return result  # (rows, profile)
    return result, None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _generate_run_id() -> str:
    """Generate a run ID from current timestamp: YYYYMMDD_HHMMSS."""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    import argparse
    from imdb_data.workload.load import load_workload, get_queries

    parser = argparse.ArgumentParser()
    parser.add_argument("workload", nargs="?", default=os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "imdb_data", "workload", "w6_queries_100.json")))
    parser.add_argument(
        "--eps",
        type=float,
        default=0.01,
        help="TA slack: kth_score <= threshold + eps (W3; also W7/W8). 0 = strict. Default 0.01.",
    )
    parser.add_argument("--no-adaptive", action="store_true",
                        help="Disable adaptive stream extension for W3")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Only run first N queries.")
    args = parser.parse_args()

    path = args.workload
    data = load_workload(path)
    dataset = data.get("dataset")
    assert dataset in DATASET_DB_URLS, (
        f"workload[{path!r}] missing/unknown 'dataset' field: {dataset!r} "
        f"(expected one of {sorted(DATASET_DB_URLS)})"
    )
    global DATABASE_URL, DATASET, ID_COL, TEXT_COL, VEC_COLS
    DATABASE_URL = DATASET_DB_URLS[dataset]
    DATASET = dataset
    ID_COL, TEXT_COL = DATASET_COLS[dataset]
    VEC_COLS = DATASET_VEC_COLS[dataset]
    from ours import fss as _fss
    _fss.DATASET_VEC_COLS = VEC_COLS
    _fss._load_table_meta(dataset)
    queries = get_queries(data)
    if not queries:
        print("No queries.", file=sys.stderr)
        return

    if args.limit:
        queries = queries[:args.limit]
    wtype = data["workload"]
    run_id = _generate_run_id()

    # Build run_kwargs from workload-specific settings
    run_kwargs = {"eps": args.eps, "adaptive": not args.no_adaptive}
    setting_snapshot = {
        "run_id": run_id,
        "workload": wtype,
        "workload_path": path,
        "eps": args.eps,
        "adaptive": not args.no_adaptive,
    }

    if wtype.upper() in ("W2", "W3"):
        w_cfg = _load_setting(wtype.lower())
        if "eps" in w_cfg:
            run_kwargs["eps"] = float(w_cfg["eps"])
        if "adaptive" in w_cfg:
            run_kwargs["adaptive"] = bool(w_cfg["adaptive"])
        setting_snapshot[f"{wtype.lower()}_setting"] = w_cfg

    if wtype.upper() == "W2":
        w2_cfg = _load_w2_setting()
        fss_cfg = w2_cfg.get("fss", {})
        from ours.fss import _load_table_meta
        meta = _load_table_meta()
        meta["fss_strategy"] = {
            "sigma_low": fss_cfg.get("sigma_low", 0.05),
            "sigma_high": fss_cfg.get("sigma_high", 0.80),
        }
        setting_snapshot["w2_setting"] = w2_cfg

    if wtype.upper() == "W4":
        w4_cfg = _load_w4_setting()
        fss_cfg = w4_cfg.get("fss", {})
        fss_pf = w4_cfg.get("fss_pf", 1.0)
        run_kwargs["fss_pf"] = fss_pf
        if "ta_eps" in w4_cfg:
            run_kwargs["w4_ta_eps"] = w4_cfg["ta_eps"]
        _w4_cfg_keys_to_run = _W4_FF_TA_THRESHOLD_KEYS
        for _k in _w4_cfg_keys_to_run:
            if _k in w4_cfg:
                run_kwargs[_k] = w4_cfg[_k]
        # Push FSS thresholds into table_meta so FSS picks them up
        from ours.fss import _load_table_meta
        meta = _load_table_meta()
        meta["fss_strategy"] = {
            "sigma_low": fss_cfg.get("sigma_low", 0.05),
            "sigma_high": fss_cfg.get("sigma_high", 0.80),
        }
        setting_snapshot["w4_setting"] = w4_cfg

    # Shared TI connection for W5 (avoids 8ms connect overhead per query)
    syn_conn = None
    if wtype.upper() == "W5":
        syn_conn = psycopg.connect(DATABASE_URL, row_factory=psycopg.rows.tuple_row, autocommit=True)
        run_kwargs["syn_conn"] = syn_conn
        setting_snapshot["w5_setting"] = _load_w5_setting()

    if wtype.upper() == "W7":
        w7_cfg = _load_w7_setting()
        if "eps" in w7_cfg:
            run_kwargs["eps"] = float(w7_cfg["eps"])
        if "time_budget" in w7_cfg:
            run_kwargs["time_budget"] = float(w7_cfg["time_budget"])
        if "ti_chunk_step" in w7_cfg:
            run_kwargs["ti_chunk_step"] = float(w7_cfg["ti_chunk_step"])
        setting_snapshot["w7_setting"] = w7_cfg

    if wtype.upper() == "W8":
        w8_cfg = _load_w8_setting()
        if "eps" in w8_cfg:
            run_kwargs["eps"] = float(w8_cfg["eps"])
        if "time_budget" in w8_cfg:
            run_kwargs["time_budget"] = float(w8_cfg["time_budget"])
        if "ti_chunk_step" in w8_cfg:
            run_kwargs["ti_chunk_step"] = float(w8_cfg["ti_chunk_step"])
        if "evt_threshold" in w8_cfg:
            run_kwargs["evt_threshold"] = float(w8_cfg["evt_threshold"])
        setting_snapshot["w8_setting"] = w8_cfg

    print(f"Workload: {wtype}, {len(queries)} queries, run_id={run_id}", file=sys.stderr)

    conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
    register_vector(conn)
    # W2 score_first + W4 rerank use HNSW ORDER BY. Force the planner onto the
    # HNSW index: otherwise cost model picks Seq Scan + top-N heapsort at
    # LIMIT~200 which is 10x slower than HNSW on wide rows.
    # work_mem: W6 direct enumeration ANDs 5+ bitmap indexes on 63M-row TI;
    # default 4MB forces lossy bitmap → 3.5M-row heap recheck (~40s).
    with conn.cursor() as _cur:
        _cur.execute("SET enable_seqscan = off")
        _cur.execute("SET work_mem = '1GB'")
    conn.commit()

    results_out = []
    t0 = time.perf_counter()

    for qi, q in enumerate(queries):
        qid = q["query_id"]
        tq = time.perf_counter()

        try:
            rows, profile = run_query(conn, q, wtype, **run_kwargs)
        except Exception:
            conn.rollback()
            raise

        elapsed_q = time.perf_counter() - tq
        print(f"  {qid}: {len(rows)} results in {elapsed_q:.3f}s", file=sys.stderr)

        # End transaction so SET LOCAL from baselines.filter_first (e.g.
        # enable_indexscan = off) does not leak across queries. Baseline uses a
        # new connection per query; without commit/rollback here one FilterFirst
        # W4 query disables index scans for the entire remaining run.
        conn.commit()

        # Serialize results
        serialized = []
        for row in rows:
            serialized.append([
                v if isinstance(v, (int, float, str, bool, type(None)))
                else float(v) if hasattr(v, "__float__") else str(v)
                for v in row
            ])

        entry = {
            "query_id": qid,
            "answer": serialized,
            "elapsed_sec": round(elapsed_q, 6),
            "n_rows": len(rows),
            "K": q["K"],
        }
        if profile is not None:
            entry["profile"] = profile
        results_out.append(entry)

    elapsed = time.perf_counter() - t0
    conn.close()
    if syn_conn is not None:
        syn_conn.close()

    qps = len(queries) / elapsed if elapsed > 0 else 0
    print(f"--- {len(queries)} queries in {elapsed:.3f}s ({qps:.2f} QPS) ---", file=sys.stderr)

    # Save results
    workload_stem = os.path.splitext(os.path.basename(path))[0]
    dataset_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(path))))
    results_dir = os.path.join(dataset_dir, "results", wtype.lower())
    os.makedirs(results_dir, exist_ok=True)

    out_path = os.path.join(results_dir, f"results_dase_{workload_stem}_{run_id}.json")
    out_data = {
        "method": "dase",
        "run_id": run_id,
        "workload_path": path,
        "workload_name": wtype,
        "total_elapsed_sec": round(elapsed, 6),
        "n_queries": len(results_out),
        "qps": round(qps, 4),
        "results": results_out,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {out_path}", file=sys.stderr)

    # Save setting snapshot to global logs dir
    logs_dir = os.path.join(dataset_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    setting_snapshot["total_elapsed_sec"] = round(elapsed, 6)
    setting_snapshot["n_queries"] = len(results_out)
    setting_snapshot["result_file"] = out_path
    setting_path = os.path.join(logs_dir, f"{run_id}.setting.json")
    with open(setting_path, "w", encoding="utf-8") as f:
        json.dump(setting_snapshot, f, ensure_ascii=False, indent=2)
    print(f"Settings saved to {setting_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
