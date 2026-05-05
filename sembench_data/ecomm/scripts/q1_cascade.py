#!/usr/bin/env -S python -u
"""
Ecomm Q1 cascade — Reebok backpacks (text F operator).

NL: find product ids of Reebok backpacks (text-only).
GT: 7 ids = [1623, 1624, 5299, 5300, 5301, 5303, 5314].
Eval: F1 over id sets.

Refactored to use dase_cascade. Operator (paper Table 3): F.
F-cascade: MarginSignal(text) + AlphaBand + AiIfVerifier; client F1 over ids.
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import (
    Cascade, MarginSignal, AlphaBand, AiIfVerifier,
    bq_client, per_row_cost, run_query,
    f1_set, build_profile, write_profile, print_summary,
)

ECOMM_DIR = os.path.abspath(os.path.join(_HERE, ".."))
PRODUCTS_PARQUET = os.path.join(ECOMM_DIR, "data", "products_text.parquet")
STYLES_PARQUET   = os.path.join(ECOMM_DIR, "cache", "sf_500", "styles_details.parquet")
PROFILE_PATH     = os.path.join(ECOMM_DIR, "outputs", "Q1.json")

PROJECT = os.environ.get("GCP_PROJECT", "")
DATASET = "fashion_product_images"
STAGING = f"{DATASET}.q1_uncertain"

POSITIVE_PROMPTS = [
    "a Reebok-branded backpack",
    "a backpack from the Reebok brand",
    "a Reebok backpack product",
]
NEGATIVE_PROMPTS = [
    "a non-backpack product such as shoes, clothing, or accessories",
    "a backpack from a different brand, not Reebok",
    "a Reebok product that is not a backpack",
]

ALPHA = 0.2
PAPER_BQ_Q1 = {"score_f1": 0.59, "latency_s": 21.2, "cost_usd": 0.04}
PAPER_DASE_NN_Q1 = {"score_f1": 0.94, "latency_s": 0.7, "cost_usd": 5e-6}
SKIP_BASELINE = False

Q1_PROMPT_FRAG = (
    "('The product is a backpack from Reebok: ',\n"
    "     styles_details.productDisplayName, ' ',\n"
    "     styles_details.productDescriptors.description.value)"
)


def make_q1_verifier():
    """CTAS staging from uncertain ids; AI.IF on staging returns positive ids."""
    def make_staging(ids):
        id_list = ",".join(str(int(i)) for i in ids)
        return f"""
        CREATE OR REPLACE TABLE {STAGING} AS
        SELECT * FROM {DATASET}.STYLES_DETAILS WHERE id IN ({id_list})
        """
    verify_sql = f"""
SELECT id
FROM {STAGING} AS styles_details
WHERE TRUE
  AND AI.IF(
    {Q1_PROMPT_FRAG},
    connection_id => 'us.connection',
    endpoint => 'gemini-2.5-flash'
  )
"""
    return AiIfVerifier(
        verify_sql=verify_sql, make_staging_sql=make_staging,
        id_column="id", coerce_id=int,
    )


def run_baseline(client):
    sql = f"""
SELECT id
FROM {DATASET}.STYLES_DETAILS AS styles_details
WHERE TRUE
  AND AI.IF(
    {Q1_PROMPT_FRAG},
    connection_id => 'us.connection',
    endpoint => 'gemini-2.5-flash'
  )
"""
    return run_query(client, sql)


def main():
    profile = build_profile(
        scenario="ecomm", query_id=1, scale_factor=500,
        params={"alpha": ALPHA},
        cascade_form=(
            "F-cascade: Cascade(MarginSignal(text), AlphaBand, AiIfVerifier with CTAS staging); "
            "cascade_ids = dase_confident_pos ∪ bq_pos_in_uncertain."
        ),
        extra={"dase_prompts": {"positive": POSITIVE_PROMPTS, "negative": NEGATIVE_PROMPTS}},
    )

    print("Loading products + GT...")
    pdf = pd.read_parquet(PRODUCTS_PARQUET)
    sdf = pd.read_parquet(STYLES_PARQUET)
    n_total = len(pdf)
    gt_ids = set()
    for _, row in sdf.iterrows():
        at = row.get("articleType") or {}
        bn = row.get("brandName")
        tn = at.get("typeName") if isinstance(at, dict) else None
        if tn == "Backpacks" and bn == "Reebok":
            gt_ids.add(int(row["id"]))
    n_gt = len(gt_ids)
    print(f"  {n_total} products, GT positive: {n_gt} ids = {sorted(gt_ids)}")
    profile["data"] = {"n_products": n_total, "n_gt_positive": n_gt,
                       "gt_ids": sorted(list(gt_ids))}

    embeddings = np.stack(pdf["embedding"].tolist()).astype(np.float32)
    ids = pdf["Id"].astype(int).tolist()

    client = bq_client(PROJECT)

    # ── Per-row cost calibration ──
    print("\n=== Per-row cost calibration ===")
    sample_texts = [str(pdf.iloc[i]["text"]) for i in range(min(10, n_total))]
    cal = per_row_cost(
        client,
        prompt="The product is a backpack from Reebok: ",
        sample_texts=sample_texts,
        method_label="AI.GENERATE_BOOL with Q1 prompt + thinking_budget=0",
        k=10,
    )
    per_row = cal.per_row_cost_usd
    print(f"  per_row=${per_row:.6f}, sample_cost=${cal.sample_cost_usd:.6f}, elapsed={cal.elapsed_s:.1f}s")
    profile["calibration"] = cal.to_dict()

    # ── Cascade ──
    print("\n=== Cascade (MarginSignal → AlphaBand → AiIfVerifier) ===")
    cascade = Cascade(
        embeddings=embeddings,
        ids=ids,
        signal=MarginSignal(positive_prompts=POSITIVE_PROMPTS, negative_prompts=NEGATIVE_PROMPTS),
        band=AlphaBand(alpha=ALPHA),
        verifier=make_q1_verifier(),
    )
    cres = cascade.run(client, per_row)

    confident_pos_ids = set(cres.confident_pos_ids)
    bq_pos_ids = set(int(x) for x in cres.bq_yes_ids)
    cascade_ids = confident_pos_ids | bq_pos_ids
    cp, cr, c_f1 = f1_set(cascade_ids, gt_ids)
    n_uncertain = len(cres.uncertain_ids)
    print(f"  alpha={ALPHA}, n_uncertain={n_uncertain}")
    print(f"  dase confident_pos={len(confident_pos_ids)}, bq_yes={len(bq_pos_ids)}")
    print(f"  cascade {len(cascade_ids)} ids; P={cp:.4f} R={cr:.4f} F1={c_f1:.4f}")
    cascade_total_wall = cres.total_wall_s
    cascade_total_slot = cres.verifier_result.ctas_slot_ms + cres.verifier_result.slot_ms

    profile["dase_breakdown"] = {
        "signal_compute_s": cres.timings_s.get("signal_compute", 0.0),
        "band_partition_s": cres.timings_s.get("band_partition", 0.0),
        "total_s": cres.timings_s.get("signal_compute", 0.0) + cres.timings_s.get("band_partition", 0.0),
    }
    profile["dase_partition"] = cres.partition.to_dict() | {
        "uncertain_ids": [int(x) for x in cres.uncertain_ids],
    }

    # ── Baseline ──
    if SKIP_BASELINE:
        b_f1 = PAPER_BQ_Q1["score_f1"]; bwall = PAPER_BQ_Q1["latency_s"]; bslot = None
        bcost = PAPER_BQ_Q1["cost_usd"]; bcalls = n_total
        profile["baseline"] = {
            "_status": "aborted",
            "score": {"f1_score": b_f1, "_source": "paper Table 4(b)"},
            "latency_breakdown": {"wall_s": bwall, "slot_ms": None, "_source": "paper"},
            "cost_breakdown": {"n_llm_calls": bcalls, "per_row_cost_usd": per_row,
                               "total_cost_usd": bcost, "_source": "paper"},
            "method": "sembench bigquery/q1.sql verbatim — NOT EXECUTED",
        }
    else:
        print("\n=== Baseline (sembench q1.sql verbatim on STYLES_DETAILS) ===")
        bdf, bwall, bslot, bsql = run_baseline(client)
        bres_ids = set(int(x) for x in bdf["id"])
        bp, br, b_f1 = f1_set(bres_ids, gt_ids)
        bcalls = n_total
        bcost = per_row * bcalls
        print(f"  returned {len(bres_ids)} ids; P={bp:.4f} R={br:.4f} F1={b_f1:.4f}")
        print(f"  wall={bwall:.2f}s slot={bslot} cost=${bcost:.6f}")
        profile["baseline"] = {
            "method": "sembench bigquery/q1.sql verbatim on STYLES_DETAILS", "sql": bsql,
            "result_ids": sorted(list(bres_ids)),
            "score": {"precision": bp, "recall": br, "f1_score": b_f1},
            "latency_breakdown": {"wall_s": bwall, "slot_ms": bslot},
            "cost_breakdown": {"n_llm_calls": bcalls, "per_row_cost_usd": per_row,
                               "total_cost_usd": bcost},
        }

    profile["cascade"] = {
        "method": "F-cascade Cascade(MarginSignal, AlphaBand, AiIfVerifier).run() with CTAS staging",
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
    paper_n_calls = round(PAPER_BQ_Q1["cost_usd"] / per_row) if per_row else None
    profile["comparison"] = {
        "score": {"paper_BQ": PAPER_BQ_Q1["score_f1"], "paper_DASE_NN": PAPER_DASE_NN_Q1["score_f1"],
                  "ours_BQ": b_f1, "ours_cascade": c_f1},
        "wall_s": {"paper_BQ": PAPER_BQ_Q1["latency_s"], "paper_DASE_NN": PAPER_DASE_NN_Q1["latency_s"],
                   "ours_BQ": bwall, "ours_cascade": cascade_total_wall},
        "cost_usd": {"paper_BQ": PAPER_BQ_Q1["cost_usd"], "paper_DASE_NN": PAPER_DASE_NN_Q1["cost_usd"],
                     "ours_BQ": bcost, "ours_cascade": cres.verifier_result.cost_usd},
        "n_llm_calls": {"paper_BQ": paper_n_calls, "paper_DASE_NN": 0,
                        "ours_BQ": bcalls, "ours_cascade": cres.verifier_result.n_calls},
    }

    write_profile(profile, PROFILE_PATH)

    print_summary(
        f"Ecomm Q1 (alpha={ALPHA})",
        columns=["paper BQ", "DASE+NN", "ours BQ", "ours cascade"],
        rows=[
            ("F1",         [PAPER_BQ_Q1["score_f1"], PAPER_DASE_NN_Q1["score_f1"], b_f1, c_f1], ".2f"),
            ("wall (s)",   [PAPER_BQ_Q1["latency_s"], PAPER_DASE_NN_Q1["latency_s"], bwall, cascade_total_wall], ".2f"),
            ("cost ($)",   [PAPER_BQ_Q1["cost_usd"], PAPER_DASE_NN_Q1["cost_usd"], bcost, cres.verifier_result.cost_usd], ".4f"),
            ("#LLM calls", [paper_n_calls, 0, bcalls, cres.verifier_result.n_calls], "d"),
        ],
    )


if __name__ == "__main__":
    main()
