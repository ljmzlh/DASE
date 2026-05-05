#!/usr/bin/env -S python -u
"""
Ecomm Q9 cascade — image-to-image self-join via image-cap emb cosine, 2-threshold partition.

NL: pairs of products price<800 in 6 base colours (mono colour1/colour2='') depicting
    same category + same dominant surface color. Image-to-image AI.IF.
GT: 28 in-scope, 46 GT positive (excl self-pairs).
Eval: F1 over pair ids "{p1_id}-{p2_id}".

Refactored to use dase_cascade.PairCosineSignal. Operator (paper Table 3): J.
Pattern is identical to Q7 (self-join + 2-threshold pair partition + verbatim
AI.IF on staging-driver join), just with image embeddings + the Q9 prompt.
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
PRODUCTS_IMAGE_PARQUET = os.path.join(ECOMM_DIR, "data", "products_image.parquet")
STYLES_PARQUET = os.path.join(ECOMM_DIR, "cache", "sf_500", "styles_details.parquet")
PROFILE_PATH = os.path.join(ECOMM_DIR, "outputs", "Q9.json")
BASELINE_CACHE_PATH = os.path.join(ECOMM_DIR, "outputs", "Q9_baseline_cache.json")

PROJECT = os.environ.get("GCP_PROJECT", "")
DATASET = "fashion_product_images"
GCS_BUCKET = f"{PROJECT}-mmb-fashion-product-images-bucket"
STAGING_TABLE = f"{DATASET}.q9_uncertain_pairs"

PRICE_LIMIT = 800
BASE_COLOURS = ["Black", "Blue", "Red", "White", "Orange", "Green"]
TAU_HIGH = 0.93
TAU_LOW = 0.78
PAPER_BQ_Q9 = {"score_f1": 0.58, "latency_s": 48.6, "cost_usd": 0.21}
PAPER_DASE_NN_Q9 = {"score_f1": 0.41, "latency_s": 1e-3, "cost_usd": 1e-9}

JOIN_PROMPT = (
    "\n     Determine whether both images display objects of the same category\n"
    "     (e.g., both are shoes, both are bags, etc.) and whether these objects\n"
    "     share the same dominant surface color. Disregard any logos, text, or\n"
    "     printed graphics on the objects. There might be other objects in the\n"
    "     images. Only focus on the main object. Base your comparison solely on\n"
    "     object type and overall surface color."
)


def _colour_in_clause():
    return "(" + ", ".join(f"'{c}'" for c in BASE_COLOURS) + ")"


PRODUCT_SELECTION_CTE = f"""
WITH product_selection AS (
  SELECT images.*
  FROM {DATASET}.STYLES_DETAILS styles_details
  JOIN {DATASET}.IMAGE_MAPPING mapping
    ON styles_details.styleImages.default.imageURL = mapping.link
  JOIN EXTERNAL_OBJECT_TRANSFORM(TABLE `{DATASET}.IMAGES`, ['SIGNED_URL']) as images
    ON ARRAY_LAST(SPLIT(images.uri, '/')) = mapping.filename
  WHERE true
    AND styles_details.baseColour IN {_colour_in_clause()}
    AND styles_details.colour1 = ''
    AND styles_details.colour2 = ''
    AND price < {PRICE_LIMIT}
)
"""


def _q9_baseline_sql():
    return f"""
{PRODUCT_SELECTION_CTE}
SELECT
  CONCAT(
    ARRAY_FIRST(SPLIT(ARRAY_LAST(SPLIT(p1.uri, '/')), '.')),
    '-',
    ARRAY_FIRST(SPLIT(ARRAY_LAST(SPLIT(p2.uri, '/')), '.'))
  ) AS id
FROM product_selection p1
LEFT OUTER JOIN product_selection p2
  ON p1.uri != p2.uri
  AND AI.IF(
    ('''{JOIN_PROMPT}''', p1.ref, p2.ref),
    connection_id => 'us.connection',
    endpoint => 'gemini-2.5-flash'
  )
"""


def _stage2_sql():
    return f"""
{PRODUCT_SELECTION_CTE}
SELECT
  CONCAT(
    ARRAY_FIRST(SPLIT(ARRAY_LAST(SPLIT(p1.uri, '/')), '.')),
    '-',
    ARRAY_FIRST(SPLIT(ARRAY_LAST(SPLIT(p2.uri, '/')), '.'))
  ) AS id
FROM {STAGING_TABLE} pairs
JOIN product_selection p1
  ON CAST(ARRAY_FIRST(SPLIT(ARRAY_LAST(SPLIT(p1.uri, '/')), '.')) AS INT64) = pairs.left_id
JOIN product_selection p2
  ON CAST(ARRAY_FIRST(SPLIT(ARRAY_LAST(SPLIT(p2.uri, '/')), '.')) AS INT64) = pairs.right_id
WHERE p1.uri != p2.uri
  AND AI.IF(
    ('''{JOIN_PROMPT}''', p1.ref, p2.ref),
    connection_id => 'us.connection',
    endpoint => 'gemini-2.5-flash'
  )
"""


def _create_pairs_table(client, pairs):
    if not pairs:
        return run_query(client, f"CREATE OR REPLACE TABLE {STAGING_TABLE} (left_id INT64, right_id INT64)")
    structs = ",".join(f"STRUCT({int(l)} AS left_id, {int(r)} AS right_id)" for l, r in pairs)
    return run_query(client, f"CREATE OR REPLACE TABLE {STAGING_TABLE} AS SELECT left_id, right_id FROM UNNEST([{structs}])")


def per_row_pair_calibration_q9(client, sample_uris_pairs, k=10):
    THINKING = "JSON '{\"generation_config\":{\"thinking_config\":{\"thinking_budget\":0}}}'"
    selects, params = [], []
    for i, (u1, u2) in enumerate(sample_uris_pairs[:k]):
        selects.append(f"""
        SELECT AI.GENERATE_BOOL(
          ('''{JOIN_PROMPT}''', img1.ref, img2.ref),
          connection_id => 'us.connection',
          endpoint => 'gemini-2.5-flash',
          model_params => {THINKING}
        ) AS verdict
        FROM EXTERNAL_OBJECT_TRANSFORM(TABLE {DATASET}.IMAGES, ['SIGNED_URL']) AS img1
        JOIN EXTERNAL_OBJECT_TRANSFORM(TABLE {DATASET}.IMAGES, ['SIGNED_URL']) AS img2
          ON img1.uri = @u1_{i} AND img2.uri = @u2_{i}""")
        params += [
            bigquery.ScalarQueryParameter(f"u1_{i}", "STRING", u1),
            bigquery.ScalarQueryParameter(f"u2_{i}", "STRING", u2),
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
        "method": "AI.GENERATE_BOOL on Q9 image-pair prompt + thinking_budget=0",
        "n_sample": n,
        "tokens_total": {"prompt_other": p_other, "prompt_audio": p_audio,
                         "output": out, "thoughts": thoughts},
        "sample_cost_usd": cost,
        "per_row_cost_usd": cost / n if n else 0.0,
        "elapsed_s": elapsed,
    }


def main():
    profile = build_profile(
        scenario="ecomm", query_id=9, scale_factor=500,
        params={"price_limit": PRICE_LIMIT, "base_colours": BASE_COLOURS,
                "tau_high": TAU_HIGH, "tau_low": TAU_LOW},
        cascade_form=(
            "J-cascade: PairCosineSignal on image-cap emb (28×28); two-threshold partition; "
            "Stage1 CTAS uncertain pairs; Stage2 verbatim AI.IF on staging-driver join."
        ),
    )

    print("Loading + computing PairCosineSignal on prefiltered scope ...")
    sdf = pd.read_parquet(STYLES_PARQUET)
    sdf = sdf[
        sdf["baseColour"].isin(BASE_COLOURS)
        & (sdf["colour1"] == "")
        & (sdf["colour2"] == "")
        & (sdf["price"] < PRICE_LIMIT)
    ].copy()
    sdf["atype"] = sdf["articleType"].apply(
        lambda x: x.get("typeName") if isinstance(x, dict) else None)
    sdf["id"] = sdf["id"].astype(np.int64)
    keep_ids = sdf["id"].tolist()
    keep_set = set(keep_ids)
    color_cat = {int(r["id"]): (r["baseColour"], r["atype"]) for _, r in sdf.iterrows()}

    pdf = pd.read_parquet(PRODUCTS_IMAGE_PARQUET)
    pdf["Id"] = pdf["Id"].astype(np.int64)
    pdf = pdf[pdf["Id"].isin(keep_set)].copy()
    pdf = pdf.set_index("Id").loc[keep_ids].reset_index()
    n = len(pdf)
    if n != len(sdf):
        raise RuntimeError(f"mismatch: styles={len(sdf)} but only {n} have image embeddings")

    emb = np.stack(pdf["embedding"].tolist()).astype(np.float32)
    ids = pdf["Id"].to_numpy()

    import time as _t
    t0 = _t.time()
    pair_sig = PairCosineSignal(embeddings_left=emb)
    sim = pair_sig._left @ pair_sig._left.T
    confident_pos, uncertain = [], []
    n_drop = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            s = float(sim[i, j])
            if s >= TAU_HIGH:
                confident_pos.append((i, j))
            elif s <= TAU_LOW:
                n_drop += 1
            else:
                uncertain.append((i, j))
    t_dase = _t.time() - t0

    # GT (excl self pairs)
    gt_pair_ids = set()
    for i in range(n):
        bi, ci = color_cat[int(ids[i])]
        for j in range(n):
            if i == j: continue
            bj, cj = color_cat[int(ids[j])]
            if bi == bj and ci == cj:
                gt_pair_ids.add(f"{int(ids[i])}-{int(ids[j])}")
    n_gt = len(gt_pair_ids)
    print(f"  scope: {n} products, candidate pairs (excl self): {n*(n-1)}, GT positive: {n_gt}")

    profile["data"] = {"n_products_in_scope": n, "n_candidate_pairs": n * (n - 1),
                       "n_gt_positive_pairs": n_gt,
                       "scope_filter": (f"baseColour IN {BASE_COLOURS}, colour1='', "
                                        f"colour2='', price<{PRICE_LIMIT}")}

    confident_pos_pair_ids = set(f"{int(ids[i])}-{int(ids[j])}" for (i, j) in confident_pos)
    uncertain_pairs_idlist = [(int(ids[i]), int(ids[j])) for (i, j) in uncertain]
    n_conf = len(confident_pos_pair_ids); n_unc = len(uncertain_pairs_idlist)
    print(f"  TAU_HIGH={TAU_HIGH} TAU_LOW={TAU_LOW}")
    print(f"  confident_pos={n_conf}, uncertain (→BQ)={n_unc}, drop={n_drop}")

    profile["dase_breakdown"] = {"dase_compute_s": t_dase, "total_s": t_dase}
    profile["dase_partition"] = {"n_confident_pos": n_conf, "n_uncertain": n_unc, "n_drop": n_drop,
                                  "tau_high": TAU_HIGH, "tau_low": TAU_LOW}

    client = bq_client(PROJECT)

    print("\n=== Per-row cost calibration (image-pair AI.IF) ===")
    sample_pairs = []
    for k in range(min(10, n)):
        i, j = k, (k + 1) % n
        u1 = f"gs://{GCS_BUCKET}/{int(ids[i])}.jpg"
        u2 = f"gs://{GCS_BUCKET}/{int(ids[j])}.jpg"
        sample_pairs.append((u1, u2))
    cal = per_row_pair_calibration_q9(client, sample_pairs, k=10)
    per_row = cal["per_row_cost_usd"]
    print(f"  per_row=${per_row:.6f}, sample_cost=${cal['sample_cost_usd']:.6f}, elapsed={cal['elapsed_s']:.1f}s")
    profile["calibration"] = cal

    if os.path.isfile(BASELINE_CACHE_PATH):
        print(f"\n=== Baseline (cached from {BASELINE_CACHE_PATH}) ===")
        with open(BASELINE_CACHE_PATH) as f:
            cache = json.load(f)
        bres_pair_ids = set(cache["result_pair_ids"])
        bwall = cache["wall_s"]; bslot = cache.get("slot_ms")
    else:
        print("\n=== Baseline (sembench q9.sql verbatim) ===")
        bdf, bwall, bslot, _ = run_query(client, _q9_baseline_sql())
        bres_pair_ids = set(str(x) for x in bdf["id"] if x is not None and "-" in str(x))
        bres_pair_ids = {pid for pid in bres_pair_ids if not pid.endswith("-")}
        with open(BASELINE_CACHE_PATH, "w") as f:
            json.dump({"result_pair_ids": sorted(list(bres_pair_ids)),
                      "wall_s": bwall, "slot_ms": bslot}, f, indent=2)
        print(f"  cached to {BASELINE_CACHE_PATH}")

    bp, br, b_f1 = f1_set(bres_pair_ids, gt_pair_ids)
    bcalls = n * (n - 1)
    bcost = per_row * bcalls
    print(f"  returned {len(bres_pair_ids)} pairs; P={bp:.4f} R={br:.4f} F1={b_f1:.4f}")
    print(f"  wall={bwall:.2f}s slot={bslot} n_calls={bcalls} cost=${bcost:.6f}")
    profile["baseline"] = {
        "method": "sembench bigquery/q9.sql verbatim",
        "sql": _q9_baseline_sql().strip(),
        "n_returned": len(bres_pair_ids),
        "score": {"precision": bp, "recall": br, "f1_score": b_f1},
        "latency_breakdown": {"wall_s": bwall, "slot_ms": bslot},
        "cost_breakdown": {"n_llm_calls": bcalls,
                           "n_llm_calls_method": "n_scope * (n_scope - 1)",
                           "per_row_cost_usd": per_row, "total_cost_usd": bcost},
    }

    # Cascade
    print(f"\n=== Cascade Stage 1: CTAS {STAGING_TABLE} from {n_unc} uncertain pairs ===")
    s1_df, s1_wall, s1_slot, s1_sql = _create_pairs_table(client, uncertain_pairs_idlist)
    print(f"  wall={s1_wall:.2f}s slot={s1_slot}")

    print(f"\n=== Cascade Stage 2: staging-driver JOIN × product_selection × 2 with AI.IF ===")
    if n_unc == 0:
        bq_pair_ids = set(); s2_wall, s2_slot = 0.0, 0
    else:
        s2_df, s2_wall, s2_slot, _ = run_query(client, _stage2_sql())
        bq_pair_ids = set(str(x) for x in s2_df["id"] if x is not None)
    s2_calls = n_unc
    cascade_cost = per_row * s2_calls
    print(f"  BQ returned {len(bq_pair_ids)} positive pairs")
    print(f"  wall={s2_wall:.2f}s slot={s2_slot} n_calls={s2_calls} cost=${cascade_cost:.6f}")

    cascade_pair_ids = confident_pos_pair_ids | bq_pair_ids
    cp, cr, c_f1 = f1_set(cascade_pair_ids, gt_pair_ids)
    print(f"  cascade {len(cascade_pair_ids)} pairs; P={cp:.4f} R={cr:.4f} F1={c_f1:.4f}")

    cascade_total_wall = t_dase + s1_wall + s2_wall
    profile["cascade"] = {
        "method": ("J-cascade: PairCosineSignal 28×28 image-cap → 2-thresh partition → "
                   "Stage1 staging pairs → Stage2 verbatim AI.IF"),
        "stage1_ctas": {
            "sql": s1_sql.strip()[:1500] + ("..." if len(s1_sql.strip()) > 1500 else ""),
            "latency_breakdown": {"wall_s": s1_wall, "slot_ms": s1_slot}, "cost_usd": 0.0,
        },
        "stage2_run": {
            "sql": _stage2_sql().strip(), "n_returned": len(bq_pair_ids),
            "latency_breakdown": {"wall_s": s2_wall, "slot_ms": s2_slot},
            "cost_breakdown": {"n_llm_calls": s2_calls, "n_llm_calls_method": "|uncertain pairs|",
                               "per_row_cost_usd": per_row, "total_cost_usd": cascade_cost},
        },
        "score": {"precision": cp, "recall": cr, "f1_score": c_f1},
        "totals": {
            "wall_s": cascade_total_wall,
            "wall_breakdown_s": {"dase": t_dase, "bq_stage1_ctas": s1_wall, "bq_stage2_aiif": s2_wall},
            "slot_ms_bq_total": s1_slot + s2_slot,
            "cost_usd": cascade_cost, "n_llm_calls": s2_calls,
        },
    }

    paper_n_calls = round(PAPER_BQ_Q9["cost_usd"] / per_row) if per_row else None
    profile["comparison"] = {
        "score": {"paper_BQ": PAPER_BQ_Q9["score_f1"], "paper_DASE_NN": PAPER_DASE_NN_Q9["score_f1"],
                  "ours_BQ": b_f1, "ours_cascade": c_f1},
        "wall_s": {"paper_BQ": PAPER_BQ_Q9["latency_s"], "paper_DASE_NN": PAPER_DASE_NN_Q9["latency_s"],
                   "ours_BQ": bwall, "ours_cascade": cascade_total_wall},
        "cost_usd": {"paper_BQ": PAPER_BQ_Q9["cost_usd"], "paper_DASE_NN": PAPER_DASE_NN_Q9["cost_usd"],
                     "ours_BQ": bcost, "ours_cascade": cascade_cost},
        "n_llm_calls": {"paper_BQ": paper_n_calls, "paper_DASE_NN": 0,
                        "ours_BQ": bcalls, "ours_cascade": s2_calls},
    }

    write_profile(profile, PROFILE_PATH)

    print_summary(
        f"Ecomm Q9",
        columns=["paper BQ", "DASE+NN", "ours BQ", "ours cascade"],
        rows=[
            ("F1",         [PAPER_BQ_Q9["score_f1"], PAPER_DASE_NN_Q9["score_f1"], b_f1, c_f1], ".2f"),
            ("wall (s)",   [PAPER_BQ_Q9["latency_s"], PAPER_DASE_NN_Q9["latency_s"], bwall, cascade_total_wall], ".2f"),
            ("cost ($)",   [PAPER_BQ_Q9["cost_usd"], PAPER_DASE_NN_Q9["cost_usd"], bcost, cascade_cost], ".4f"),
            ("#LLM calls", [paper_n_calls, 0, bcalls, s2_calls], "d"),
        ],
    )


if __name__ == "__main__":
    main()
