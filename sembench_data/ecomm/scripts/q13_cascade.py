#!/usr/bin/env -S python -u
"""
Ecomm Q13 cascade — complex sem_filter (image+text) on full 500 products.

NL: men's running t-shirt with round neck, short sleeves, blue/black, striped,
    suitable for outdoor warm-weather running.
GT: 12 products (8-condition structural AND).
Eval: F1 over id sets.

Refactored to use dase_cascade. Operator (paper Table 3): F.
F-cascade: MarginSignal(image-cap emb, 3-pos/3-neg) + AbsoluteBand(TAU_LOW, TAU_HIGH=1.0
drop-only) + AiIfVerifier with CTAS staging.

NOTE: TAU_HIGH=1.0 disables confident-pos (image-text emb confident-pos too noisy
for sem_filter); cascade is effectively drop-only via TAU_LOW. Equivalent to original.
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import (
    Cascade, MarginSignal, AbsoluteBand, AiIfVerifier,
    bq_client, per_row_cost, run_query,
    f1_set, build_profile, write_profile, print_summary,
)
from dase_cascade.calibration import _sum_tokens, _to_cost
from google.cloud import bigquery

ECOMM_DIR = os.path.abspath(os.path.join(_HERE, ".."))
PRODUCTS_IMAGE_PARQUET = os.path.join(ECOMM_DIR, "data", "products_image.parquet")
STYLES_PARQUET = os.path.join(ECOMM_DIR, "cache", "sf_500", "styles_details.parquet")
PROFILE_PATH = os.path.join(ECOMM_DIR, "outputs", "Q13.json")
BASELINE_CACHE = os.path.join(ECOMM_DIR, "outputs", "Q13_baseline_cache.json")

PROJECT = os.environ.get("GCP_PROJECT", "")
DATASET = "fashion_product_images"
GCS_BUCKET = f"{PROJECT}-mmb-fashion-product-images-bucket"
STAGING_TABLE = f"{DATASET}.q13_uncertain"

TAU_HIGH = 1.0   # disabled (drop-only mode)
TAU_LOW = -0.02

PAPER_BQ_Q13 = {"score_f1": 0.70, "latency_s": 22.4, "cost_usd": 0.38}
PAPER_DASE_NN_Q13 = {"score_f1": 0.58, "latency_s": 1.0, "cost_usd": 2e-5}

POSITIVE_PROMPTS = [
    "a men's running t-shirt with round neck and short sleeves, in blue or black, with a striped design, suitable for outdoor running in warm weather",
    "a blue or black short-sleeve striped sports tshirt for men, athletic running apparel",
    "a men's athletic round-neck short-sleeve striped running shirt for warm weather",
]
NEGATIVE_PROMPTS = [
    "women's clothing or non-tshirt apparel like trousers, shoes, accessories, dresses",
    "a white or green or other bright-colored t-shirt, or a non-striped solid-color shirt",
    "a long-sleeve or v-neck or hoodie or jacket, not a short-sleeve round-neck t-shirt",
]

JOIN_PROMPT_TEMPLATE = """
    You will receive a description of what a customer is looking for together with an image and a textual description of the product.
    Determine if they both match.

    I am looking for a running shirt for men with a round neck and short sleeves,
    preferably in blue or black, but not bright colors like white.
    Also definitely not green.
    It should be suitable for outdoor running in warm weather.
    If the t-shirt is not green, it should at least feature a striped design."""


def _q13_sql_for(table: str) -> str:
    return f"""
SELECT styles_details.id
FROM {table} AS styles_details
JOIN {DATASET}.IMAGE_MAPPING
  ON image_mapping.link = styles_details.styleImages.default.imageURL
JOIN EXTERNAL_OBJECT_TRANSFORM(TABLE `{DATASET}.IMAGES`, ['SIGNED_URL']) as images
  ON ARRAY_LAST(SPLIT(images.uri, '/')) = image_mapping.filename
WHERE true
  AND AI.IF(('''{JOIN_PROMPT_TEMPLATE}''',
    images.ref, ' and textual description ',
    styles_details.productDisplayName, ' ',
    styles_details.productDescriptors.description.value
  ),
  connection_id => 'us.connection',
  endpoint => 'gemini-2.5-flash')
"""


def make_q13_verifier():
    def make_staging(ids):
        id_list = ",".join(str(int(i)) for i in ids)
        return f"""
        CREATE OR REPLACE TABLE {STAGING_TABLE} AS
        SELECT * FROM {DATASET}.STYLES_DETAILS WHERE id IN ({id_list})
        """
    return AiIfVerifier(
        verify_sql=_q13_sql_for(STAGING_TABLE),
        make_staging_sql=make_staging,
        id_column="id", coerce_id=int,
    )


# Q13 calibration is bespoke (image + 2 text params) — inline.
def per_row_cost_q13(client, sample_uris, sample_titles, sample_descrs, k=10):
    THINKING = "JSON '{\"generation_config\":{\"thinking_config\":{\"thinking_budget\":0}}}'"
    selects, params = [], []
    for i in range(min(k, len(sample_uris))):
        selects.append(f"""
        SELECT AI.GENERATE_BOOL(
          ('''{JOIN_PROMPT_TEMPLATE}''', img.ref, ' and textual description ', @t_{i}, ' ', @d_{i}),
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
        "method": "AI.GENERATE_BOOL with Q13 prompt + thinking_budget=0",
        "n_sample": n,
        "tokens_total": {"prompt_other": p_other, "prompt_audio": p_audio,
                         "output": out, "thoughts": thoughts},
        "sample_cost_usd": cost,
        "per_row_cost_usd": cost / n if n else 0.0,
        "elapsed_s": elapsed,
    }


def attr_get(aa, key):
    if aa is None:
        return None
    try:
        for kv in aa:
            if kv[0] == key:
                return kv[1]
    except Exception:
        pass
    return None


def main():
    profile = build_profile(
        scenario="ecomm", query_id=13, scale_factor=500,
        params={"tau_high": TAU_HIGH, "tau_low": TAU_LOW},
        cascade_form=(
            "F-cascade: Cascade(MarginSignal(image-cap, 3pos/3neg), AbsoluteBand(TAU_LOW, TAU_HIGH=1.0 disabled), "
            "AiIfVerifier with CTAS staging). cascade_ids = confident_yes ∪ bq_yes."
        ),
        extra={"dase_prompts": {"positive": POSITIVE_PROMPTS, "negative": NEGATIVE_PROMPTS}},
    )

    print("Loading + computing dase margin on 500 products...")
    pdf = pd.read_parquet(PRODUCTS_IMAGE_PARQUET)
    sdf = pd.read_parquet(STYLES_PARQUET)
    n_total = len(pdf)
    pdf["Id"] = pdf["Id"].astype(np.int64)
    sdf["id"] = sdf["id"].astype(np.int64)
    image_emb = np.stack(pdf["embedding"].tolist()).astype(np.float32)
    ids = pdf["Id"].astype(int).tolist()

    # GT
    sdf["at"] = sdf["articleType"].apply(lambda x: x.get("typeName") if isinstance(x, dict) else None)
    sdf["sl_v"] = sdf["articleAttributes"].apply(lambda x: attr_get(x, "Sleeve Length"))
    sdf["nk_v"] = sdf["articleAttributes"].apply(lambda x: attr_get(x, "Neck"))
    sdf["pt_v"] = sdf["articleAttributes"].apply(lambda x: attr_get(x, "Pattern"))
    gt_mask = (
        (sdf["gender"] == "Men")
        & (sdf["usage"] == "Sports")
        & (sdf["at"] == "Tshirts")
        & (sdf["baseColour"].isin(["Blue", "Black"]))
        & (sdf["sl_v"] == "Short Sleeves")
        & (sdf["nk_v"] == "Round Neck")
        & (sdf["pt_v"] == "Striped")
        & (sdf["season"] != "Winter")
    )
    gt_ids = set(int(x) for x in sdf.loc[gt_mask, "id"])
    n_gt = len(gt_ids)
    print(f"  {n_total} products, GT positive: {n_gt} ids = {sorted(gt_ids)}")
    profile["data"] = {"n_products": n_total, "n_gt_positive": n_gt,
                       "gt_ids": sorted(list(gt_ids))}

    client = bq_client(PROJECT)

    print("\n=== Per-row cost calibration ===")
    sample_uris, sample_titles, sample_descrs = [], [], []
    sdf_indexed = sdf.set_index("id")
    for i in range(min(10, n_total)):
        pid = int(pdf.iloc[i]["Id"])
        sample_uris.append(f"gs://{GCS_BUCKET}/{pid}.jpg")
        row = sdf_indexed.loc[pid]
        sample_titles.append(str(row["productDisplayName"] or ""))
        try:
            d = (row["productDescriptors"] or {}).get("description", {}).get("value", "") or ""
        except Exception:
            d = ""
        sample_descrs.append(d[:500])
    cal = per_row_cost_q13(client, sample_uris, sample_titles, sample_descrs, k=10)
    per_row = cal["per_row_cost_usd"]
    print(f"  per_row=${per_row:.6f}, elapsed={cal['elapsed_s']:.1f}s")
    profile["calibration"] = cal

    # ── Cascade ──
    print("\n=== Cascade (MarginSignal → AbsoluteBand → AiIfVerifier) ===")
    cascade = Cascade(
        embeddings=image_emb,
        ids=ids,
        signal=MarginSignal(positive_prompts=POSITIVE_PROMPTS, negative_prompts=NEGATIVE_PROMPTS),
        band=AbsoluteBand(tau_low=TAU_LOW, tau_high=TAU_HIGH),
        verifier=make_q13_verifier(),
    )
    cres = cascade.run(client, per_row)

    confident_pos_ids = set(int(x) for x in cres.confident_pos_ids)
    bq_pos_ids = set(int(x) for x in cres.bq_yes_ids)
    cascade_ids = confident_pos_ids | bq_pos_ids
    cp, cr, c_f1 = f1_set(cascade_ids, gt_ids)
    cascade_total_wall = cres.total_wall_s
    cascade_total_slot = cres.verifier_result.ctas_slot_ms + cres.verifier_result.slot_ms
    print(f"  TAU_HIGH={TAU_HIGH}, TAU_LOW={TAU_LOW}")
    print(f"  confident_pos={len(confident_pos_ids)}, "
          f"uncertain={len(cres.uncertain_ids)}, bq_yes={len(bq_pos_ids)}")
    print(f"  cascade {len(cascade_ids)} ids; P={cp:.4f} R={cr:.4f} F1={c_f1:.4f}")

    profile["dase_breakdown"] = {
        "signal_compute_s": cres.timings_s.get("signal_compute", 0.0),
        "band_partition_s": cres.timings_s.get("band_partition", 0.0),
        "total_s": cres.timings_s.get("signal_compute", 0.0) + cres.timings_s.get("band_partition", 0.0),
    }
    profile["dase_partition"] = cres.partition.to_dict() | {
        "uncertain_ids": [int(x) for x in cres.uncertain_ids],
    }

    # ── Baseline (cached) ──
    import json
    if os.path.exists(BASELINE_CACHE):
        print(f"\n=== Baseline (cached from {BASELINE_CACHE}) ===")
        with open(BASELINE_CACHE) as f:
            cache = json.load(f)
        bres_ids = set(int(x) for x in cache["bres_ids"])
        bwall = cache["wall_s"]; bslot = cache.get("slot_ms")
    else:
        print("\n=== Baseline (sembench q13.sql verbatim on STYLES_DETAILS) ===")
        bdf, bwall, bslot, _ = run_query(client, _q13_sql_for(f"{DATASET}.STYLES_DETAILS"))
        bres_ids = set(int(x) for x in bdf["id"])
        with open(BASELINE_CACHE, "w") as f:
            json.dump({"bres_ids": sorted(list(bres_ids)), "wall_s": bwall, "slot_ms": bslot}, f, indent=2)
        print(f"  cached to {BASELINE_CACHE}")
    bp, br, b_f1 = f1_set(bres_ids, gt_ids)
    bcalls = n_total
    bcost = per_row * bcalls
    print(f"  returned {len(bres_ids)} ids; P={bp:.4f} R={br:.4f} F1={b_f1:.4f}")
    print(f"  wall={bwall:.2f}s slot={bslot} cost=${bcost:.6f}")
    profile["baseline"] = {
        "method": "sembench bigquery/q13.sql verbatim on STYLES_DETAILS",
        "sql": _q13_sql_for(f"{DATASET}.STYLES_DETAILS").strip(),
        "result_ids": sorted(list(bres_ids)),
        "score": {"precision": bp, "recall": br, "f1_score": b_f1},
        "latency_breakdown": {"wall_s": bwall, "slot_ms": bslot},
        "cost_breakdown": {"n_llm_calls": bcalls, "n_llm_calls_method": "scope size",
                           "per_row_cost_usd": per_row, "total_cost_usd": bcost},
    }

    profile["cascade"] = {
        "method": "F-cascade Cascade(MarginSignal, AbsoluteBand, AiIfVerifier).run() w/ CTAS staging",
        "verifier": cres.verifier_result.to_dict(),
        "cascade_ids": sorted(list(cascade_ids)),
        "score": {"precision": cp, "recall": cr, "f1_score": c_f1},
        "totals": {
            "wall_s": cascade_total_wall,
            "slot_ms_bq_total": cascade_total_slot,
            "cost_usd": cres.verifier_result.cost_usd,
            "n_llm_calls": cres.verifier_result.n_calls,
        },
    }
    paper_n_calls = round(PAPER_BQ_Q13["cost_usd"] / per_row) if per_row else None
    profile["comparison"] = {
        "score": {"paper_BQ": PAPER_BQ_Q13["score_f1"], "paper_DASE_NN": PAPER_DASE_NN_Q13["score_f1"],
                  "ours_BQ": b_f1, "ours_cascade": c_f1},
        "wall_s": {"paper_BQ": PAPER_BQ_Q13["latency_s"], "paper_DASE_NN": PAPER_DASE_NN_Q13["latency_s"],
                   "ours_BQ": bwall, "ours_cascade": cascade_total_wall},
        "cost_usd": {"paper_BQ": PAPER_BQ_Q13["cost_usd"], "paper_DASE_NN": PAPER_DASE_NN_Q13["cost_usd"],
                     "ours_BQ": bcost, "ours_cascade": cres.verifier_result.cost_usd},
        "n_llm_calls": {"paper_BQ": paper_n_calls, "paper_DASE_NN": 0,
                        "ours_BQ": bcalls, "ours_cascade": cres.verifier_result.n_calls},
    }

    write_profile(profile, PROFILE_PATH)

    print_summary(
        f"Ecomm Q13 (TAU_LOW={TAU_LOW}, TAU_HIGH={TAU_HIGH})",
        columns=["paper BQ", "DASE+NN", "ours BQ", "ours cascade"],
        rows=[
            ("F1",         [PAPER_BQ_Q13["score_f1"], PAPER_DASE_NN_Q13["score_f1"], b_f1, c_f1], ".2f"),
            ("wall (s)",   [PAPER_BQ_Q13["latency_s"], PAPER_DASE_NN_Q13["latency_s"], bwall, cascade_total_wall], ".2f"),
            ("cost ($)",   [PAPER_BQ_Q13["cost_usd"], PAPER_DASE_NN_Q13["cost_usd"], bcost, cres.verifier_result.cost_usd], ".4f"),
            ("#LLM calls", [paper_n_calls, 0, bcalls, cres.verifier_result.n_calls], "d"),
        ],
    )


if __name__ == "__main__":
    main()
