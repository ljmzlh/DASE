#!/usr/bin/env -S python -u
"""
Ecomm Q7 cascade — semantic self-join on text emb (same category + same brand).

NL: pairs of products priced≤500 of same category and same brand, judged from
    text descriptions.
GT: 173 GT positive pairs over 89-row scope.
Eval: F1 over pair ids "id1-id2".

Refactored to use dase_cascade.PairCosineSignal. Operator (paper Table 3): J.

Pipeline:
  1. dase: PairCosineSignal on text embeddings of 89 in-scope products.
  2. Two absolute thresholds:
       sim ≥ TAU_HIGH (0.95) → confident positive (skip BQ, emit pair)
       sim ≤ TAU_LOW  (0.79) → drop
       else → uncertain → BQ AI.IF
  3. Stage 1 CTAS staging pair table; Stage 2 verbatim AI.IF on staging.
  4. cascade_pairs = confident_pos ∪ bq_returned.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import (
    PairCosineSignal,
    bq_client, run_query,
    f1_set, build_profile, write_profile, print_summary,
)
from dase_cascade.calibration import _sum_tokens, _to_cost
from google.cloud import bigquery

ECOMM_DIR = os.path.abspath(os.path.join(_HERE, ".."))
PRODUCTS_PARQUET = os.path.join(ECOMM_DIR, "data", "products_text.parquet")
STYLES_PARQUET = os.path.join(ECOMM_DIR, "cache", "sf_500", "styles_details.parquet")
PROFILE_PATH = os.path.join(ECOMM_DIR, "outputs", "Q7.json")
BASELINE_CACHE_PATH = os.path.join(ECOMM_DIR, "outputs", "Q7_baseline_cache.json")

PROJECT = os.environ.get("GCP_PROJECT", "")
DATASET = "fashion_product_images"
STAGING_TABLE = f"{DATASET}.q7_uncertain_pairs"

PRICE_LIMIT = 500
TAU_HIGH = 0.95
TAU_LOW = 0.79
PAPER_BQ_Q7 = {"score_f1": 0.83, "latency_s": 45.4, "cost_usd": 0.86}
PAPER_DASE_NN_Q7 = {"score_f1": 0.78, "latency_s": 2e-3, "cost_usd": 1e-9}

JOIN_PROMPT_HEADER = (
    "\n     You will be given two product descriptions. Do both product descriptions describe products of the same category from the same brand, e.g., both are t-shirts from Adidas?\n     \n     The first product description is:\n     "
)
JOIN_PROMPT_MID = "\n     The second product description is:\n     "


def _q7_baseline_sql():
    return f"""
WITH product_selection AS (
  SELECT *
  FROM {DATASET}.STYLES_DETAILS styles_details
  WHERE true
    AND price <= {PRICE_LIMIT}
)
SELECT
  CONCAT(CAST(p1.id AS STRING), '-', CAST(p2.id AS STRING)) AS id
FROM product_selection p1
JOIN product_selection p2
  ON AI.IF(('''{JOIN_PROMPT_HEADER}''', p1.productDisplayName, ' - ', p1.productDescriptors.description.value,
           '''{JOIN_PROMPT_MID}''', p2.productDisplayName, ' - ', p2.productDescriptors.description.value
          ),
    connection_id => 'us.connection',
    endpoint => 'gemini-2.5-flash'
  )
"""


def _stage2_sql():
    return f"""
SELECT
  CONCAT(CAST(p1.id AS STRING), '-', CAST(p2.id AS STRING)) AS id
FROM {STAGING_TABLE} pairs
JOIN {DATASET}.STYLES_DETAILS p1 ON p1.id = pairs.left_id
JOIN {DATASET}.STYLES_DETAILS p2 ON p2.id = pairs.right_id
WHERE AI.IF(('''{JOIN_PROMPT_HEADER}''', p1.productDisplayName, ' - ', p1.productDescriptors.description.value,
             '''{JOIN_PROMPT_MID}''', p2.productDisplayName, ' - ', p2.productDescriptors.description.value
            ),
    connection_id => 'us.connection',
    endpoint => 'gemini-2.5-flash'
  )
"""


def _create_pairs_table(client, pairs):
    if not pairs:
        return run_query(client, f"CREATE OR REPLACE TABLE {STAGING_TABLE} (left_id INT64, right_id INT64)")
    structs = ",".join(f"STRUCT({int(l)} AS left_id, {int(r)} AS right_id)" for l, r in pairs)
    return run_query(client, f"CREATE OR REPLACE TABLE {STAGING_TABLE} AS SELECT left_id, right_id FROM UNNEST([{structs}])")


# Bespoke pair calibration (2 STRING params per pair).
def per_row_pair_calibration_q7(client, sample_pairs, k=10):
    THINKING = "JSON '{\"generation_config\":{\"thinking_config\":{\"thinking_budget\":0}}}'"
    selects, params = [], []
    for i, (t1, t2) in enumerate(sample_pairs[:k]):
        selects.append(f"""
        SELECT AI.GENERATE_BOOL(
          ('''{JOIN_PROMPT_HEADER}''', @t1_{i},
           '''{JOIN_PROMPT_MID}''', @t2_{i}),
          connection_id => 'us.connection',
          endpoint => 'gemini-2.5-flash',
          model_params => {THINKING}
        ) AS verdict""")
        params += [
            bigquery.ScalarQueryParameter(f"t1_{i}", "STRING", t1),
            bigquery.ScalarQueryParameter(f"t2_{i}", "STRING", t2),
        ]
    sql = " UNION ALL ".join(selects)
    cfg = bigquery.QueryJobConfig(query_parameters=params, use_query_cache=False)
    import time as _t
    t0 = _t.time()
    df = client.query(sql, job_config=cfg).result().to_dataframe()
    elapsed = _t.time() - t0
    p_other, p_audio, out, thoughts = _sum_tokens(df["verdict"])
    n = len(df)
    cost = _to_cost(p_other, p_audio, out, thoughts)
    return {
        "method": "AI.GENERATE_BOOL with Q7 pair prompt + thinking_budget=0",
        "n_sample": n,
        "tokens_total": {"prompt_other": p_other, "prompt_audio": p_audio,
                         "output": out, "thoughts": thoughts},
        "sample_cost_usd": cost,
        "per_row_cost_usd": cost / n if n else 0.0,
        "elapsed_s": elapsed,
    }


def main():
    profile = build_profile(
        scenario="ecomm", query_id=7, scale_factor=500,
        params={"price_limit": PRICE_LIMIT, "tau_high": TAU_HIGH, "tau_low": TAU_LOW},
        cascade_form=(
            "J-cascade: PairCosineSignal on text emb (89×89); two-threshold partition; "
            "CTAS staging pair table; verbatim AI.IF on staging-driver join. "
            "cascade_pairs = confident_pos (sim≥τ_high) ∪ bq_returned."
        ),
    )

    print("Loading + computing PairCosineSignal on price≤500 scope ...")
    sdf = pd.read_parquet(STYLES_PARQUET)
    sdf = sdf[sdf["price"] <= PRICE_LIMIT].copy()
    sdf["atype"] = sdf["articleType"].apply(lambda x: x.get("typeName") if isinstance(x, dict) else None)
    sdf["id"] = sdf["id"].astype(np.int64)
    keep_ids = sdf["id"].tolist()
    keep_set = set(keep_ids)
    brand_cat = {int(r["id"]): (r["brandName"], r["atype"]) for _, r in sdf.iterrows()}

    pdf = pd.read_parquet(PRODUCTS_PARQUET)
    pdf["Id"] = pdf["Id"].astype(np.int64)
    pdf = pdf[pdf["Id"].isin(keep_set)].copy()
    pdf = pdf.set_index("Id").loc[keep_ids].reset_index()
    n = len(pdf)
    if n != len(sdf):
        raise RuntimeError(f"mismatch: styles={len(sdf)} but only {n} have embeddings")

    emb = np.stack(pdf["embedding"].tolist()).astype(np.float32)
    ids = pdf["Id"].to_numpy()

    import time as _t
    t0 = _t.time()
    pair_sig = PairCosineSignal(embeddings_left=emb)  # self-join
    # Compute full self-sim matrix (small: n=89), then partition by 2 thresholds.
    L = np.arange(n, dtype=np.int64)
    sim = pair_sig._left @ pair_sig._left.T  # use pre-normalized embeddings
    confident_pos, uncertain = [], []
    n_drop = 0
    for i in range(n):
        for j in range(n):
            s = float(sim[i, j])
            if s >= TAU_HIGH:
                confident_pos.append((i, j))
            elif s <= TAU_LOW:
                n_drop += 1
            else:
                uncertain.append((i, j))
    t_dase = _t.time() - t0

    # GT
    gt_pair_ids = set()
    for i in range(n):
        bi, ci = brand_cat[int(ids[i])]
        for j in range(n):
            bj, cj = brand_cat[int(ids[j])]
            if bi == bj and ci == cj:
                gt_pair_ids.add(f"{int(ids[i])}-{int(ids[j])}")
    n_gt = len(gt_pair_ids)
    print(f"  scope: {n} products, total candidate pairs: {n*n}, GT positive pairs: {n_gt}")
    profile["data"] = {"n_products_in_scope": n, "n_candidate_pairs": n*n,
                       "n_gt_positive_pairs": n_gt, "scope_filter": f"price <= {PRICE_LIMIT}"}

    confident_pos_pair_ids = set(f"{int(ids[i])}-{int(ids[j])}" for (i, j) in confident_pos)
    uncertain_pairs_idlist = [(int(ids[i]), int(ids[j])) for (i, j) in uncertain]
    n_conf = len(confident_pos_pair_ids); n_unc = len(uncertain_pairs_idlist)
    print(f"  TAU_HIGH={TAU_HIGH}, TAU_LOW={TAU_LOW}")
    print(f"  confident_pos={n_conf}, uncertain (→BQ)={n_unc}, drop={n_drop}")

    profile["dase_breakdown"] = {"dase_compute_s": t_dase, "total_s": t_dase}
    profile["dase_partition"] = {"n_confident_pos": n_conf, "n_uncertain": n_unc,
                                  "n_drop": n_drop, "tau_high": TAU_HIGH, "tau_low": TAU_LOW}

    client = bq_client(PROJECT)

    print("\n=== Per-row cost calibration (pair AI.IF) ===")
    sample_pairs = []
    for k in range(min(10, n)):
        i, j = k, (k + 1) % n
        t1 = f"{pdf.iloc[i]['Title']} - {pdf.iloc[i]['Description']}"
        t2 = f"{pdf.iloc[j]['Title']} - {pdf.iloc[j]['Description']}"
        sample_pairs.append((t1, t2))
    cal = per_row_pair_calibration_q7(client, sample_pairs, k=10)
    per_row = cal["per_row_cost_usd"]
    print(f"  per_row=${per_row:.6f}, sample_cost=${cal['sample_cost_usd']:.6f}, elapsed={cal['elapsed_s']:.1f}s")
    profile["calibration"] = cal

    # Baseline: cached or fresh
    if os.path.isfile(BASELINE_CACHE_PATH):
        print(f"\n=== Baseline (cached from {BASELINE_CACHE_PATH}) ===")
        with open(BASELINE_CACHE_PATH) as f:
            cache = json.load(f)
        bres_pair_ids = set(cache["result_pair_ids"])
        bwall = cache["wall_s"]; bslot = cache.get("slot_ms")
    else:
        print("\n=== Baseline (sembench q7.sql verbatim) ===")
        bdf, bwall, bslot, _ = run_query(client, _q7_baseline_sql())
        bres_pair_ids = set(str(x) for x in bdf["id"])
        with open(BASELINE_CACHE_PATH, "w") as f:
            json.dump({"result_pair_ids": sorted(list(bres_pair_ids)),
                      "wall_s": bwall, "slot_ms": bslot}, f, indent=2)
        print(f"  cached to {BASELINE_CACHE_PATH}")

    bp, br, b_f1 = f1_set(bres_pair_ids, gt_pair_ids)
    bcalls = n * n
    bcost = per_row * bcalls
    print(f"  returned {len(bres_pair_ids)} pairs; P={bp:.4f} R={br:.4f} F1={b_f1:.4f}")
    print(f"  wall={bwall:.2f}s slot={bslot} n_calls={bcalls} cost=${bcost:.6f}")
    profile["baseline"] = {
        "method": "sembench bigquery/q7.sql verbatim", "sql": _q7_baseline_sql().strip(),
        "n_returned": len(bres_pair_ids),
        "score": {"precision": bp, "recall": br, "f1_score": b_f1},
        "latency_breakdown": {"wall_s": bwall, "slot_ms": bslot},
        "cost_breakdown": {"n_llm_calls": bcalls,
                           "n_llm_calls_method": "n_scope^2 (Cartesian self-join)",
                           "per_row_cost_usd": per_row, "total_cost_usd": bcost},
    }

    # Cascade Stage 1 + 2
    print(f"\n=== Cascade Stage 1: CTAS {STAGING_TABLE} from {n_unc} uncertain pairs ===")
    s1_df, s1_wall, s1_slot, s1_sql = _create_pairs_table(client, uncertain_pairs_idlist)
    print(f"  wall={s1_wall:.2f}s slot={s1_slot}")

    print(f"\n=== Cascade Stage 2: verbatim AI.IF on staging ===")
    if n_unc == 0:
        bq_pair_ids = set(); s2_wall, s2_slot = 0.0, 0
    else:
        s2_df, s2_wall, s2_slot, _ = run_query(client, _stage2_sql())
        bq_pair_ids = set(str(x) for x in s2_df["id"])
    s2_calls = n_unc
    cascade_cost = per_row * s2_calls
    print(f"  BQ returned {len(bq_pair_ids)} positive pairs")
    print(f"  wall={s2_wall:.2f}s slot={s2_slot} n_calls={s2_calls} cost=${cascade_cost:.6f}")

    cascade_pair_ids = confident_pos_pair_ids | bq_pair_ids
    cp, cr, c_f1 = f1_set(cascade_pair_ids, gt_pair_ids)
    print(f"  cascade {len(cascade_pair_ids)} pairs; P={cp:.4f} R={cr:.4f} F1={c_f1:.4f}")

    cascade_total_wall = t_dase + s1_wall + s2_wall
    profile["cascade"] = {
        "method": ("J-cascade: PairCosineSignal 89×89 → 2-thresh partition → "
                   "Stage1 staging pairs → Stage2 verbatim AI.IF on uncertain"),
        "stage1_ctas": {"sql": s1_sql.strip()[:1500] + ("..." if len(s1_sql.strip()) > 1500 else ""),
                        "latency_breakdown": {"wall_s": s1_wall, "slot_ms": s1_slot}, "cost_usd": 0.0},
        "stage2_run": {"sql": _stage2_sql().strip(), "n_returned": len(bq_pair_ids),
                       "latency_breakdown": {"wall_s": s2_wall, "slot_ms": s2_slot},
                       "cost_breakdown": {"n_llm_calls": s2_calls,
                                          "n_llm_calls_method": "|uncertain pairs|",
                                          "per_row_cost_usd": per_row, "total_cost_usd": cascade_cost}},
        "score": {"precision": cp, "recall": cr, "f1_score": c_f1},
        "totals": {
            "wall_s": cascade_total_wall,
            "wall_breakdown_s": {"dase": t_dase, "bq_stage1_ctas": s1_wall, "bq_stage2_aiif": s2_wall},
            "slot_ms_bq_total": s1_slot + s2_slot,
            "cost_usd": cascade_cost, "n_llm_calls": s2_calls,
        },
    }

    paper_n_calls = round(PAPER_BQ_Q7["cost_usd"] / per_row) if per_row else None
    profile["comparison"] = {
        "score": {"paper_BQ": PAPER_BQ_Q7["score_f1"], "paper_DASE_NN": PAPER_DASE_NN_Q7["score_f1"],
                  "ours_BQ": b_f1, "ours_cascade": c_f1},
        "wall_s": {"paper_BQ": PAPER_BQ_Q7["latency_s"], "paper_DASE_NN": PAPER_DASE_NN_Q7["latency_s"],
                   "ours_BQ": bwall, "ours_cascade": cascade_total_wall},
        "cost_usd": {"paper_BQ": PAPER_BQ_Q7["cost_usd"], "paper_DASE_NN": PAPER_DASE_NN_Q7["cost_usd"],
                     "ours_BQ": bcost, "ours_cascade": cascade_cost},
        "n_llm_calls": {"paper_BQ": paper_n_calls, "paper_DASE_NN": 0,
                        "ours_BQ": bcalls, "ours_cascade": s2_calls},
    }

    write_profile(profile, PROFILE_PATH)

    print_summary(
        f"Ecomm Q7",
        columns=["paper BQ", "DASE+NN", "ours BQ", "ours cascade"],
        rows=[
            ("F1",         [PAPER_BQ_Q7["score_f1"], PAPER_DASE_NN_Q7["score_f1"], b_f1, c_f1], ".2f"),
            ("wall (s)",   [PAPER_BQ_Q7["latency_s"], PAPER_DASE_NN_Q7["latency_s"], bwall, cascade_total_wall], ".2f"),
            ("cost ($)",   [PAPER_BQ_Q7["cost_usd"], PAPER_DASE_NN_Q7["cost_usd"], bcost, cascade_cost], ".4f"),
            ("#LLM calls", [paper_n_calls, 0, bcalls, s2_calls], "d"),
        ],
    )


if __name__ == "__main__":
    main()
