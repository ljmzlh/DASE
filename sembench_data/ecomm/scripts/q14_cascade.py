#!/usr/bin/env -S python -u
"""
Ecomm Q14 cascade — F + J + R (sem_filter + sem_join + sem_rank).

NL: For each product price<130, find top-1 image that BEST matches the product
    description AND that image depicts white socks.
GT: 2 product ids.
Eval: F1 over output id set.

Refactored to use dase_cascade. Operator (paper Table 3): F + J (and R inside BQ).

Two-pronged dase prefilter:
  (a) F image-side: MarginSignal(white-socks anchors, drop-only via TAU_LOW).
  (b) J pair-side : PairCosineSignal product text-emb × image-cap emb (drop-only via TAU_LOW).

PAIR_TAU_LOW=-1.0 disables pair-sim drop (lost a GT). Equivalent to original.

Stage 1 CTAS staging (product_id, image_id) pairs; Stage 2 verbatim BQ Q14 SQL
with staging-driven JOIN.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import (
    MarginSignal, PairCosineSignal, AbsoluteBand,
    bq_client, run_query,
    f1_set, build_profile, write_profile, print_summary,
)
from dase_cascade.calibration import _sum_tokens, _to_cost
from google.cloud import bigquery

ECOMM_DIR = os.path.abspath(os.path.join(_HERE, ".."))
PRODUCTS_TEXT_PARQUET = os.path.join(ECOMM_DIR, "data", "products_text.parquet")
PRODUCTS_IMAGE_PARQUET = os.path.join(ECOMM_DIR, "data", "products_image.parquet")
STYLES_PARQUET = os.path.join(ECOMM_DIR, "cache", "sf_500", "styles_details.parquet")
PROFILE_PATH = os.path.join(ECOMM_DIR, "outputs", "Q14.json")
BASELINE_CACHE = os.path.join(ECOMM_DIR, "outputs", "Q14_baseline_cache.json")

PROJECT = os.environ.get("GCP_PROJECT", "")
DATASET = "fashion_product_images"
GCS_BUCKET = f"{PROJECT}-mmb-fashion-product-images-bucket"
STAGING_TABLE = f"{DATASET}.q14_uncertain_pairs"

PRICE_LIMIT = 130
IMAGE_TAU_LOW = -0.02
PAIR_TAU_LOW = -1.0  # disabled

PAPER_BQ_Q14 = {"score_f1": 0.37, "latency_s": 73.6, "cost_usd": 4.26}
PAPER_DASE_NN_Q14 = {"score_f1": 0.0, "latency_s": 0.7, "cost_usd": 5e-6}

POS_WHITE_SOCKS = [
    "a pair of white socks",
    "white socks for feet, athletic or dress socks",
    "white sock product, footwear accessory",
]
NEG_WHITE_SOCKS = [
    "a non-socks fashion product, like a shirt, pants, dress, or accessory",
    "colored socks (black, gray, navy, etc.) — not white",
    "shoes, sandals, sneakers, or other non-sock footwear",
]


def _q14_sql_baseline():
    return f"""
SELECT
  ARRAY_AGG(
    ARRAY_FIRST(SPLIT(ARRAY_LAST(SPLIT(images.uri, '/')), '.'))
    ORDER BY AI.SCORE(
      ('The image ', images.ref, ' fits the description: ',
       styles_details.productDisplayName, ' ',
       styles_details.productDescriptors.description.value),
      connection_id => 'us.connection',
      endpoint => 'gemini-2.5-flash'
    ) ASC
    LIMIT 1
  )[0] AS id
FROM {DATASET}.STYLES_DETAILS as styles_details
JOIN EXTERNAL_OBJECT_TRANSFORM(TABLE `{DATASET}.IMAGES`, ['SIGNED_URL']) as images
  ON AI.IF(
    ('The image ', images.ref, ' fits the description: ',
     styles_details.productDisplayName, ' ',
     styles_details.productDescriptors.description.value),
    connection_id => 'us.connection',
    endpoint => 'gemini-2.5-flash'
  )
WHERE styles_details.price < {PRICE_LIMIT}
  AND AI.IF(
    ('The image ', images.ref, ' depicts white socks'),
    connection_id => 'us.connection',
    endpoint => 'gemini-2.5-flash'
  )
GROUP BY
  styles_details.id,
  styles_details.productDisplayName,
  styles_details.productDescriptors.description
"""


def _q14_sql_stage2():
    return f"""
SELECT
  ARRAY_AGG(
    ARRAY_FIRST(SPLIT(ARRAY_LAST(SPLIT(i.uri, '/')), '.'))
    ORDER BY AI.SCORE(
      ('The image ', i.ref, ' fits the description: ',
       p.productDisplayName, ' ',
       p.productDescriptors.description.value),
      connection_id => 'us.connection',
      endpoint => 'gemini-2.5-flash'
    ) ASC
    LIMIT 1
  )[0] AS id
FROM {STAGING_TABLE} sp
JOIN {DATASET}.STYLES_DETAILS p ON p.id = sp.product_id
JOIN EXTERNAL_OBJECT_TRANSFORM(TABLE `{DATASET}.IMAGES`, ['SIGNED_URL']) i
  ON CAST(ARRAY_FIRST(SPLIT(ARRAY_LAST(SPLIT(i.uri, '/')), '.')) AS INT64) = sp.image_id
WHERE p.price < {PRICE_LIMIT}
  AND AI.IF(
    ('The image ', i.ref, ' depicts white socks'),
    connection_id => 'us.connection',
    endpoint => 'gemini-2.5-flash'
  )
  AND AI.IF(
    ('The image ', i.ref, ' fits the description: ',
     p.productDisplayName, ' ',
     p.productDescriptors.description.value),
    connection_id => 'us.connection',
    endpoint => 'gemini-2.5-flash'
  )
GROUP BY
  p.id,
  p.productDisplayName,
  p.productDescriptors.description
"""


def _create_pairs_table(client, pair_list):
    if not pair_list:
        return run_query(client, f"CREATE OR REPLACE TABLE {STAGING_TABLE} (product_id INT64, image_id INT64)")
    structs = ",".join(f"STRUCT({int(p)} AS product_id, {int(i)} AS image_id)" for p, i in pair_list)
    return run_query(client, f"CREATE OR REPLACE TABLE {STAGING_TABLE} AS SELECT product_id, image_id FROM UNNEST([{structs}])")


def per_row_cost_q14(client, sample_uris, sample_titles, sample_descrs, k=10):
    """Q14 has 3 AI calls per pair; we calibrate one fit-prompt AI.GENERATE_BOOL as
    a representative per-call rate."""
    THINKING = "JSON '{\"generation_config\":{\"thinking_config\":{\"thinking_budget\":0}}}'"
    selects, params = [], []
    for i in range(k):
        selects.append(f"""
        SELECT AI.GENERATE_BOOL(
          ('The image ', img.ref, ' fits the description: ', @t_{i}, ' ', @d_{i}),
          connection_id => 'us.connection',
          endpoint => 'gemini-2.5-flash',
          model_params => {THINKING}
        ) AS verdict
        FROM EXTERNAL_OBJECT_TRANSFORM(TABLE {DATASET}.IMAGES, ['SIGNED_URL']) AS img
        WHERE img.uri = @uri_{i}""")
        params += [
            bigquery.ScalarQueryParameter(f"uri_{i}", "STRING", sample_uris[i]),
            bigquery.ScalarQueryParameter(f"t_{i}", "STRING", sample_titles[i]),
            bigquery.ScalarQueryParameter(f"d_{i}", "STRING", sample_descrs[i]),
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
        "method": "AI.GENERATE_BOOL on Q14 fit prompt + thinking_budget=0",
        "n_sample": n,
        "tokens_total": {"prompt_other": p_other, "prompt_audio": p_audio,
                         "output": out, "thoughts": thoughts},
        "sample_cost_usd": cost,
        "per_row_cost_usd": cost / n if n else 0.0,
        "elapsed_s": elapsed,
    }


def main():
    profile = build_profile(
        scenario="ecomm", query_id=14, scale_factor=500,
        params={"price_limit": PRICE_LIMIT, "image_tau_low": IMAGE_TAU_LOW,
                "pair_tau_low": PAIR_TAU_LOW},
        cascade_form=(
            "F+J+R cascade: (a) image-side MarginSignal(white-socks, drop-only via TAU_LOW); "
            "(b) pair-side PairCosineSignal text×image (drop-only); Stage1 CTAS uncertain pairs; "
            "Stage2 verbatim Q14 SQL with staging-driven JOIN."
        ),
        extra={"dase_prompts": {"pos_white_socks": POS_WHITE_SOCKS,
                                "neg_white_socks": NEG_WHITE_SOCKS}},
    )

    print("Loading + computing dase signals ...")
    sdf = pd.read_parquet(STYLES_PARQUET)
    sdf["id"] = sdf["id"].astype(np.int64)
    products_priced = sdf[sdf["price"] < PRICE_LIMIT].copy()
    product_ids = products_priced["id"].tolist()
    print(f"  in-scope products (price<{PRICE_LIMIT}): {len(product_ids)} ids = {product_ids}")

    # Image-side: white-socks margin (drop-only via TAU_LOW)
    pdf_img = pd.read_parquet(PRODUCTS_IMAGE_PARQUET)
    pdf_img["Id"] = pdf_img["Id"].astype(np.int64)
    img_emb = np.stack(pdf_img["embedding"].tolist()).astype(np.float32)
    image_ids = pdf_img["Id"].to_numpy()
    n_img = len(image_ids)

    import time as _t
    t0 = _t.time()
    white_socks_margin = MarginSignal(POS_WHITE_SOCKS, NEG_WHITE_SOCKS).compute(img_emb)
    image_keep_mask = white_socks_margin > IMAGE_TAU_LOW
    image_keep_ids = [int(image_ids[i]) for i in range(n_img) if image_keep_mask[i]]
    print(f"  image white-socks margin range: [{white_socks_margin.min():+.3f}, "
          f"{white_socks_margin.max():+.3f}]")
    print(f"  image-side: kept {len(image_keep_ids)}/{n_img} (tau_low={IMAGE_TAU_LOW})")

    # Pair-side: text emb × image-cap emb cosine
    pdf_text = pd.read_parquet(PRODUCTS_TEXT_PARQUET)
    pdf_text["Id"] = pdf_text["Id"].astype(np.int64)
    pdf_text_idx = pdf_text.set_index("Id")
    pair_sig = PairCosineSignal(embeddings_left=img_emb)  # placeholder; we use sim directly
    img_emb_norm = pair_sig._left
    image_id_to_idx = {int(image_ids[i]): i for i in range(n_img)}

    uncertain_pairs = []
    n_pair_drop_sim = 0
    for pid in product_ids:
        if pid not in pdf_text_idx.index:
            continue
        text_emb_p = np.array(pdf_text_idx.loc[pid, "embedding"], dtype=np.float32)
        text_emb_p = text_emb_p / (np.linalg.norm(text_emb_p) + 1e-12)
        for iid in image_keep_ids:
            idx = image_id_to_idx[iid]
            sim = float(img_emb_norm[idx] @ text_emb_p)
            if sim <= PAIR_TAU_LOW:
                n_pair_drop_sim += 1
                continue
            uncertain_pairs.append((pid, iid))
    n_uncertain = len(uncertain_pairs)
    n_total_pairs = len(product_ids) * n_img
    n_pair_drop_image = len(product_ids) * (n_img - len(image_keep_ids))
    print(f"  pair-side: total {n_total_pairs}, drop_image_side={n_pair_drop_image}, "
          f"drop_pair_sim={n_pair_drop_sim}, uncertain (→BQ)={n_uncertain}")
    t_dase = _t.time() - t0

    profile["dase_breakdown"] = {"dase_compute_s": t_dase, "total_s": t_dase}
    profile["data"] = {
        "n_products_in_scope": len(product_ids), "product_ids": product_ids,
        "n_images_total": n_img, "n_total_pairs": n_total_pairs,
    }
    profile["dase_partition"] = {
        "n_image_keep": len(image_keep_ids), "n_pair_uncertain": n_uncertain,
        "n_pair_drop_image_side": n_pair_drop_image,
        "n_pair_drop_pair_sim": n_pair_drop_sim,
        "image_tau_low": IMAGE_TAU_LOW, "pair_tau_low": PAIR_TAU_LOW,
    }

    # GT
    def gm(x): return x.get("typeName") if isinstance(x, dict) else None
    sdf["at"] = sdf["articleType"].apply(gm)
    gt_ids = set(int(x) for x in
                 sdf[(sdf["at"] == "Socks") & (sdf["baseColour"] == "White")
                     & (sdf["price"] < PRICE_LIMIT)]["id"])
    print(f"  GT product ids: {sorted(gt_ids)}")
    profile["data"]["n_gt"] = len(gt_ids)
    profile["data"]["gt_ids"] = sorted(list(gt_ids))

    client = bq_client(PROJECT)

    print("\n=== Per-row cost calibration ===")
    sdf_indexed = sdf.set_index("id")
    sample_uris, sample_titles, sample_descrs = [], [], []
    for i in range(min(10, n_img)):
        iid = int(image_ids[i])
        sample_uris.append(f"gs://{GCS_BUCKET}/{iid}.jpg")
        if iid in sdf_indexed.index:
            row = sdf_indexed.loc[iid]
            sample_titles.append(str(row["productDisplayName"] or ""))
            try:
                d = (row["productDescriptors"] or {}).get("description", {}).get("value", "") or ""
            except Exception:
                d = ""
            sample_descrs.append(d[:500])
        else:
            sample_titles.append(""); sample_descrs.append("")
    cal = per_row_cost_q14(client, sample_uris, sample_titles, sample_descrs, k=10)
    per_row = cal["per_row_cost_usd"]
    print(f"  per_row=${per_row:.6f} (single AI call); per pair has 3 AI calls")
    profile["calibration"] = cal

    if os.path.exists(BASELINE_CACHE):
        print(f"\n=== Baseline (cached from {BASELINE_CACHE}) ===")
        with open(BASELINE_CACHE) as f:
            cache = json.load(f)
        bres_ids = set(int(x) for x in cache["bres_ids"])
        bwall = cache["wall_s"]; bslot = cache.get("slot_ms")
    else:
        print("\n=== Baseline (sembench q14.sql verbatim) ===")
        bdf, bwall, bslot, _ = run_query(client, _q14_sql_baseline())
        bres_ids = set(int(x) for x in bdf["id"] if x is not None)
        with open(BASELINE_CACHE, "w") as f:
            json.dump({"bres_ids": sorted(list(bres_ids)),
                      "wall_s": bwall, "slot_ms": bslot}, f, indent=2)
        print(f"  cached to {BASELINE_CACHE}")
    bp, br, b_f1 = f1_set(bres_ids, gt_ids)
    bcalls = n_total_pairs * 3
    bcost = per_row * bcalls
    print(f"  returned {len(bres_ids)} ids; P={bp:.4f} R={br:.4f} F1={b_f1:.4f}")
    print(f"  wall={bwall:.2f}s slot={bslot} n_calls={bcalls} cost=${bcost:.6f}")
    profile["baseline"] = {
        "method": "sembench bigquery/q14.sql verbatim",
        "sql": _q14_sql_baseline().strip(),
        "result_ids": sorted(list(bres_ids)),
        "score": {"precision": bp, "recall": br, "f1_score": b_f1},
        "latency_breakdown": {"wall_s": bwall, "slot_ms": bslot},
        "cost_breakdown": {"n_llm_calls": bcalls,
                           "n_llm_calls_method": "n_total_pairs × 3 AI calls per pair",
                           "per_row_cost_usd": per_row, "total_cost_usd": bcost},
    }

    # Cascade Stage 1 + 2
    print(f"\n=== Cascade Stage 1: CTAS {STAGING_TABLE} from {n_uncertain} uncertain pairs ===")
    s1_df, s1_wall, s1_slot, s1_sql = _create_pairs_table(client, uncertain_pairs)
    print(f"  wall={s1_wall:.2f}s slot={s1_slot}")

    print(f"\n=== Cascade Stage 2: q14.sql staging-driven on {n_uncertain} uncertain pairs ===")
    if n_uncertain == 0:
        cascade_ids = set(); s2_wall, s2_slot = 0.0, 0
    else:
        s2_df, s2_wall, s2_slot, _ = run_query(client, _q14_sql_stage2())
        cascade_ids = set(int(x) for x in s2_df["id"] if x is not None)
    s2_calls = n_uncertain * 3
    cascade_cost = per_row * s2_calls
    print(f"  cascade returned {len(cascade_ids)} ids")
    print(f"  wall={s2_wall:.2f}s slot={s2_slot} n_calls={s2_calls} cost=${cascade_cost:.6f}")

    cp, cr, c_f1 = f1_set(cascade_ids, gt_ids)
    print(f"  P={cp:.4f} R={cr:.4f} F1={c_f1:.4f}")

    cascade_total_wall = t_dase + s1_wall + s2_wall
    profile["cascade"] = {
        "method": ("F+J+R cascade with double dase prefilter (image-side white-socks F drop + "
                   "pair-side cos-sim J drop); staging pair table drives Stage 2 verbatim BQ Q14."),
        "stage1_ctas": {
            "sql": s1_sql.strip()[:1500] + ("..." if len(s1_sql.strip()) > 1500 else ""),
            "latency_breakdown": {"wall_s": s1_wall, "slot_ms": s1_slot}, "cost_usd": 0.0,
        },
        "stage2_run": {
            "sql": _q14_sql_stage2().strip(), "result_ids": sorted(list(cascade_ids)),
            "latency_breakdown": {"wall_s": s2_wall, "slot_ms": s2_slot},
            "cost_breakdown": {"n_llm_calls": s2_calls,
                               "n_llm_calls_method": "n_uncertain_pairs × 3 AI calls per pair",
                               "per_row_cost_usd": per_row, "total_cost_usd": cascade_cost},
        },
        "cascade_ids": sorted(list(cascade_ids)),
        "score": {"precision": cp, "recall": cr, "f1_score": c_f1},
        "totals": {
            "wall_s": cascade_total_wall,
            "wall_breakdown_s": {"dase": t_dase, "bq_stage1_ctas": s1_wall, "bq_stage2_aiif": s2_wall},
            "slot_ms_bq_total": s1_slot + s2_slot,
            "cost_usd": cascade_cost, "n_llm_calls": s2_calls,
        },
    }

    paper_n_calls = round(PAPER_BQ_Q14["cost_usd"] / per_row) if per_row else None
    profile["comparison"] = {
        "score": {"paper_BQ": PAPER_BQ_Q14["score_f1"], "paper_DASE_NN": PAPER_DASE_NN_Q14["score_f1"],
                  "ours_BQ": b_f1, "ours_cascade": c_f1},
        "wall_s": {"paper_BQ": PAPER_BQ_Q14["latency_s"], "paper_DASE_NN": PAPER_DASE_NN_Q14["latency_s"],
                   "ours_BQ": bwall, "ours_cascade": cascade_total_wall},
        "cost_usd": {"paper_BQ": PAPER_BQ_Q14["cost_usd"], "paper_DASE_NN": PAPER_DASE_NN_Q14["cost_usd"],
                     "ours_BQ": bcost, "ours_cascade": cascade_cost},
        "n_llm_calls": {"paper_BQ": paper_n_calls, "paper_DASE_NN": 0,
                        "ours_BQ": bcalls, "ours_cascade": s2_calls},
    }

    write_profile(profile, PROFILE_PATH)

    print_summary(
        f"Ecomm Q14",
        columns=["paper BQ", "DASE+NN", "ours BQ", "ours cascade"],
        rows=[
            ("F1",         [PAPER_BQ_Q14["score_f1"], PAPER_DASE_NN_Q14["score_f1"], b_f1, c_f1], ".2f"),
            ("wall (s)",   [PAPER_BQ_Q14["latency_s"], PAPER_DASE_NN_Q14["latency_s"], bwall, cascade_total_wall], ".2f"),
            ("cost ($)",   [PAPER_BQ_Q14["cost_usd"], PAPER_DASE_NN_Q14["cost_usd"], bcost, cascade_cost], ".4f"),
            ("#LLM calls", [paper_n_calls, 0, bcalls, s2_calls], "d"),
        ],
    )


if __name__ == "__main__":
    main()
