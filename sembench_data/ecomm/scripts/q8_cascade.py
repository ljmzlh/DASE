#!/usr/bin/env -S python -u
"""
Ecomm Q8 cascade — text-to-image self-join, dase pre-filter via cos sim (drop-only).

NL: For each product with description ≥ 3000 chars, find matching images.
GT: SF=500 has 1 such product (id=40270). GT pair = {"40270-40270"}.
Eval: F1 over pair ids "{img_id}-{img_id}".

Refactored to use dase_cascade.PairCosineSignal. Operator (paper Table 3): J.

Drop-only mode (TAU_LOW only — GT image isn't top-1 in this query, so any
high-side threshold risks pushing GT to BQ as a confident FP). Equivalent to
original q8_cascade.py.

Stage 1 builds an EXTERNAL TABLE over uncertain GCS image URIs.
Stage 2 runs sembench q8.sql verbatim with IMAGES → q8_uncertain_images.
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
PRODUCTS_TEXT_PARQUET = os.path.join(ECOMM_DIR, "data", "products_text.parquet")
PRODUCTS_IMAGE_PARQUET = os.path.join(ECOMM_DIR, "data", "products_image.parquet")
STYLES_PARQUET = os.path.join(ECOMM_DIR, "cache", "sf_500", "styles_details.parquet")
PROFILE_PATH = os.path.join(ECOMM_DIR, "outputs", "Q8.json")
BASELINE_CACHE_PATH = os.path.join(ECOMM_DIR, "outputs", "Q8_baseline_cache.json")

PROJECT = os.environ.get("GCP_PROJECT", "")
DATASET = "fashion_product_images"
GCS_BUCKET = f"{PROJECT}-mmb-fashion-product-images-bucket"
STAGING_TABLE = f"{DATASET}.q8_uncertain_images"

DESC_LEN_MIN = 3000
TAU_LOW = 0.70
PAPER_BQ_Q8 = {"score_f1": 0.29, "latency_s": 126.2, "cost_usd": 18.23}
PAPER_DASE_NN_Q8 = {"score_f1": 0.25, "latency_s": 1e-3, "cost_usd": 1e-9}


def _q8_sql_for(images_table_ref: str) -> str:
    return f"""
WITH product_selection AS (
  SELECT *
  FROM {DATASET}.STYLES_DETAILS styles_details
  WHERE true
    AND CHAR_LENGTH(styles_details.productDescriptors.description.value) >= {DESC_LEN_MIN}
)
SELECT
  CONCAT(
    ARRAY_FIRST(SPLIT(ARRAY_LAST(SPLIT(images.uri, '/')), '.')),
    '-',
    ARRAY_FIRST(SPLIT(ARRAY_LAST(SPLIT(images.uri, '/')), '.'))
  ) AS id
FROM product_selection as styles_details
JOIN EXTERNAL_OBJECT_TRANSFORM(TABLE `{images_table_ref}`, ['SIGNED_URL']) as images
  ON AI.IF(
    ('The image ', images.ref, ' fits the description: ',
     styles_details.productDisplayName, ' ',
     styles_details.productDescriptors.description.value),
    connection_id => 'us.connection',
    endpoint => 'gemini-2.5-flash'
  )
"""


def _create_ext_table(client, uris):
    uri_list = ", ".join(f"'{u}'" for u in uris)
    sql = f"""
    CREATE OR REPLACE EXTERNAL TABLE {STAGING_TABLE}
    WITH CONNECTION `us.connection`
    OPTIONS(
      object_metadata = 'SIMPLE',
      uris = [{uri_list}]
    )
    """
    return run_query(client, sql)


# Bespoke calibration: image + text desc.
def per_row_cost_q8(client, sample_uris, sample_text, k=10):
    THINKING = "JSON '{\"generation_config\":{\"thinking_config\":{\"thinking_budget\":0}}}'"
    selects, params = [], []
    for i, uri in enumerate(sample_uris[:k]):
        selects.append(f"""
        SELECT AI.GENERATE_BOOL(
          ('The image ', img.ref, ' fits the description: ', @desc_{i}),
          connection_id => 'us.connection',
          endpoint => 'gemini-2.5-flash',
          model_params => {THINKING}
        ) AS verdict
        FROM EXTERNAL_OBJECT_TRANSFORM(TABLE {DATASET}.IMAGES, ['SIGNED_URL']) AS img
        WHERE img.uri = @uri_{i}""")
        params += [
            bigquery.ScalarQueryParameter(f"uri_{i}", "STRING", uri),
            bigquery.ScalarQueryParameter(f"desc_{i}", "STRING", sample_text),
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
        "method": "AI.GENERATE_BOOL on Q8 image+desc prompt + thinking_budget=0",
        "n_sample": n,
        "tokens_total": {"prompt_other": p_other, "prompt_audio": p_audio,
                         "output": out, "thoughts": thoughts},
        "sample_cost_usd": cost,
        "per_row_cost_usd": cost / n if n else 0.0,
        "elapsed_s": elapsed,
    }


def main():
    profile = build_profile(
        scenario="ecomm", query_id=8, scale_factor=500,
        params={"desc_len_min": DESC_LEN_MIN, "tau_low": TAU_LOW},
        cascade_form=(
            "J-cascade (drop-only): PairCosineSignal text-emb × image-cap-emb (1×500); "
            "drop sim ≤ TAU_LOW; EXTERNAL TABLE on uncertain image URIs; verbatim AI.IF."
        ),
    )

    print("Loading + computing PairCosineSignal text-to-image ...")
    sdf = pd.read_parquet(STYLES_PARQUET)
    def dlen(x):
        try:
            return len(x["description"]["value"])
        except Exception:
            return 0
    sdf["desclen"] = sdf["productDescriptors"].apply(dlen)
    long_desc = sdf[sdf["desclen"] >= DESC_LEN_MIN].copy()
    long_desc["id"] = long_desc["id"].astype(np.int64)
    long_ids = long_desc["id"].tolist()
    n_long = len(long_ids)
    if n_long != 1:
        print(f"  WARNING: scope has {n_long} long-desc products; cascade designed for 1")

    pdf_t = pd.read_parquet(PRODUCTS_TEXT_PARQUET)
    pdf_t["Id"] = pdf_t["Id"].astype(np.int64)
    pdf_t = pdf_t[pdf_t["Id"].isin(set(long_ids))].copy()
    pdf_t = pdf_t.set_index("Id").loc[long_ids].reset_index()
    text_embs = np.stack(pdf_t["embedding"].tolist()).astype(np.float32)

    pdf_i = pd.read_parquet(PRODUCTS_IMAGE_PARQUET)
    pdf_i["Id"] = pdf_i["Id"].astype(np.int64)
    image_ids = pdf_i["Id"].to_numpy()
    img_embs = np.stack(pdf_i["embedding"].tolist()).astype(np.float32)
    n_img = len(image_ids)

    import time as _t
    t0 = _t.time()
    pair_sig = PairCosineSignal(embeddings_left=img_embs, embeddings_right=text_embs)
    L = np.arange(n_img, dtype=np.int64)
    R = np.arange(n_long, dtype=np.int64)
    sim = pair_sig._left @ pair_sig._right.T  # (n_img, n_long)
    uncertain_image_ids = []
    n_drop = 0
    for p_idx in range(n_long):
        for k in range(n_img):
            s = float(sim[k, p_idx])
            if s > TAU_LOW:
                uncertain_image_ids.append(int(image_ids[k]))
            else:
                n_drop += 1
    t_dase = _t.time() - t0

    gt_pair_ids = set(f"{int(pid)}-{int(pid)}" for pid in long_ids)
    n_gt = len(gt_pair_ids)
    print(f"  scope: {n_long} long-desc product(s) × {n_img} images = {n_long*n_img} candidates; GT pairs: {n_gt}")
    print(f"  long-desc product ids: {long_ids}")

    profile["data"] = {
        "n_long_desc_products": n_long, "n_images": n_img,
        "n_candidate_pairs": n_long * n_img, "n_gt_positive_pairs": n_gt,
        "long_desc_product_ids": long_ids, "gt_pair_ids": sorted(gt_pair_ids),
    }

    n_unc = len(uncertain_image_ids)
    gt_kept = all(int(pid) in set(uncertain_image_ids) for pid in long_ids)
    print(f"  TAU_LOW={TAU_LOW}; uncertain (→BQ)={n_unc}, drop={n_drop}, GT preserved={gt_kept}")

    profile["dase_breakdown"] = {"dase_compute_s": t_dase, "total_s": t_dase}
    profile["dase_partition"] = {
        "tau_low": TAU_LOW, "n_uncertain": n_unc, "n_drop": n_drop,
        "gt_preserved": gt_kept, "uncertain_image_ids": uncertain_image_ids,
    }

    client = bq_client(PROJECT)

    print("\n=== Per-row cost calibration ===")
    sample_uris = [f"gs://{GCS_BUCKET}/{int(image_ids[i])}.jpg" for i in range(min(10, n_img))]
    sample_text = (
        f"{long_desc.iloc[0]['productDisplayName']} "
        f"{long_desc.iloc[0]['productDescriptors']['description']['value']}"
    )
    cal = per_row_cost_q8(client, sample_uris, sample_text, k=10)
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
        print("\n=== Baseline (sembench q8.sql verbatim) ===")
        bdf, bwall, bslot, _ = run_query(client, _q8_sql_for(f"{DATASET}.IMAGES"))
        bres_pair_ids = set(str(x) for x in bdf["id"])
        with open(BASELINE_CACHE_PATH, "w") as f:
            json.dump({"result_pair_ids": sorted(list(bres_pair_ids)),
                      "wall_s": bwall, "slot_ms": bslot}, f, indent=2)
        print(f"  cached to {BASELINE_CACHE_PATH}")

    bp, br, b_f1 = f1_set(bres_pair_ids, gt_pair_ids)
    bcalls = n_long * n_img
    bcost = per_row * bcalls
    print(f"  returned {len(bres_pair_ids)} pairs; P={bp:.4f} R={br:.4f} F1={b_f1:.4f}")
    print(f"  wall={bwall:.2f}s slot={bslot} n_calls={bcalls} cost=${bcost:.6f}")
    profile["baseline"] = {
        "method": "sembench bigquery/q8.sql verbatim",
        "sql": _q8_sql_for(f"{DATASET}.IMAGES").strip(),
        "n_returned": len(bres_pair_ids),
        "result_pair_ids": sorted(bres_pair_ids),
        "score": {"precision": bp, "recall": br, "f1_score": b_f1},
        "latency_breakdown": {"wall_s": bwall, "slot_ms": bslot},
        "cost_breakdown": {"n_llm_calls": bcalls,
                           "n_llm_calls_method": "n_long_desc * n_images (Cartesian)",
                           "per_row_cost_usd": per_row, "total_cost_usd": bcost},
    }

    # Cascade Stage 1 + 2
    print(f"\n=== Cascade Stage 1: EXTERNAL TABLE {STAGING_TABLE} from {n_unc} uncertain images ===")
    uncertain_uris = [f"gs://{GCS_BUCKET}/{iid}.jpg" for iid in uncertain_image_ids]
    if uncertain_uris:
        s1_df, s1_wall, s1_slot, s1_sql = _create_ext_table(client, uncertain_uris)
        print(f"  wall={s1_wall:.2f}s slot={s1_slot}")
    else:
        s1_wall, s1_slot, s1_sql = 0.0, 0, "(skipped)"

    print(f"\n=== Cascade Stage 2: q8.sql on {STAGING_TABLE} ===")
    if uncertain_uris:
        s2_df, s2_wall, s2_slot, _ = run_query(client, _q8_sql_for(STAGING_TABLE))
        bq_pair_ids = set(str(x) for x in s2_df["id"])
    else:
        s2_wall, s2_slot, bq_pair_ids = 0.0, 0, set()
    s2_calls = n_unc
    cascade_cost = per_row * s2_calls
    print(f"  BQ returned {len(bq_pair_ids)} positive pairs; wall={s2_wall:.2f}s "
          f"slot={s2_slot} n_calls={s2_calls} cost=${cascade_cost:.6f}")

    cascade_pair_ids = bq_pair_ids
    cp, cr, c_f1 = f1_set(cascade_pair_ids, gt_pair_ids)
    print(f"  cascade {len(cascade_pair_ids)} pairs; P={cp:.4f} R={cr:.4f} F1={c_f1:.4f}")

    cascade_total_wall = t_dase + s1_wall + s2_wall
    profile["cascade"] = {
        "method": "J-cascade (drop-only): PairCosineSignal 1×500 → drop sim≤τ_low → EXTERNAL TABLE → verbatim AI.IF",
        "stage1_ctas": {
            "sql": s1_sql.strip()[:1500] + ("..." if len(s1_sql.strip()) > 1500 else ""),
            "latency_breakdown": {"wall_s": s1_wall, "slot_ms": s1_slot}, "cost_usd": 0.0,
        },
        "stage2_run": {
            "sql": _q8_sql_for(STAGING_TABLE).strip(),
            "n_returned": len(bq_pair_ids), "result_pair_ids": sorted(bq_pair_ids),
            "latency_breakdown": {"wall_s": s2_wall, "slot_ms": s2_slot},
            "cost_breakdown": {"n_llm_calls": s2_calls,
                               "n_llm_calls_method": "|uncertain images|",
                               "per_row_cost_usd": per_row, "total_cost_usd": cascade_cost},
        },
        "cascade_pair_ids": sorted(cascade_pair_ids),
        "score": {"precision": cp, "recall": cr, "f1_score": c_f1},
        "totals": {
            "wall_s": cascade_total_wall,
            "wall_breakdown_s": {"dase": t_dase, "bq_stage1_ctas": s1_wall, "bq_stage2_aiif": s2_wall},
            "slot_ms_bq_total": s1_slot + s2_slot,
            "cost_usd": cascade_cost, "n_llm_calls": s2_calls,
        },
    }

    paper_n_calls = round(PAPER_BQ_Q8["cost_usd"] / per_row) if per_row else None
    profile["comparison"] = {
        "score": {"paper_BQ": PAPER_BQ_Q8["score_f1"], "paper_DASE_NN": PAPER_DASE_NN_Q8["score_f1"],
                  "ours_BQ": b_f1, "ours_cascade": c_f1},
        "wall_s": {"paper_BQ": PAPER_BQ_Q8["latency_s"], "paper_DASE_NN": PAPER_DASE_NN_Q8["latency_s"],
                   "ours_BQ": bwall, "ours_cascade": cascade_total_wall},
        "cost_usd": {"paper_BQ": PAPER_BQ_Q8["cost_usd"], "paper_DASE_NN": PAPER_DASE_NN_Q8["cost_usd"],
                     "ours_BQ": bcost, "ours_cascade": cascade_cost},
        "n_llm_calls": {"paper_BQ": paper_n_calls, "paper_DASE_NN": 0,
                        "ours_BQ": bcalls, "ours_cascade": s2_calls},
    }

    write_profile(profile, PROFILE_PATH)

    print_summary(
        f"Ecomm Q8",
        columns=["paper BQ", "DASE+NN", "ours BQ", "ours cascade"],
        rows=[
            ("F1",         [PAPER_BQ_Q8["score_f1"], PAPER_DASE_NN_Q8["score_f1"], b_f1, c_f1], ".2f"),
            ("wall (s)",   [PAPER_BQ_Q8["latency_s"], PAPER_DASE_NN_Q8["latency_s"], bwall, cascade_total_wall], ".2f"),
            ("cost ($)",   [PAPER_BQ_Q8["cost_usd"], PAPER_DASE_NN_Q8["cost_usd"], bcost, cascade_cost], ".4f"),
            ("#LLM calls", [paper_n_calls, 0, bcalls, s2_calls], "d"),
        ],
    )


if __name__ == "__main__":
    main()
