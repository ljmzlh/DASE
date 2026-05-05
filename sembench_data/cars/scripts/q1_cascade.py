#!/usr/bin/env -S python -u
"""
Cars Q1 cascade (v2) — text-unary F via dase_cascade.

NL: Find cars that were in a crash/accident/collision.
GT: SELECT DISTINCT car_id FROM text_complaints WHERE crash=TRUE  → 952 car_ids.
Eval: precision/recall/F1 on car_id set.

Refactored to use dase_cascade unified solver. Operator (paper Table 3): F.
The Cascade primitive (MarginSignal + AlphaBand + AiIfVerifier) drives the
prefilter+BQ stage. Final answer = DISTINCT(dase_confident_pos_car_ids ∪
bq_pos_on_uncertain_car_ids), computed client-side because we project from
complaint_id space → car_id space.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from google.cloud import bigquery

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import (
    Cascade, MarginSignal, AlphaBand, AiIfVerifier,
    bq_client, per_row_cost, run_query,
    f1_set, build_profile, write_profile, print_summary,
)

# ─── Paths / scenario constants ──────────────────────────────────────────
CARS_DIR = os.path.abspath(os.path.join(_HERE, ".."))
TEXT_PARQUET = os.path.join(CARS_DIR, "data", "text_complaints.parquet")
GT_CSV = os.path.join(CARS_DIR, "ground_truth", "Q1.csv")
PROFILE_PATH = os.path.join(CARS_DIR, "outputs", "Q1.json")

PROJECT = os.environ.get("GCP_PROJECT", "")
DATASET = "cars_dataset"
STAGING_TABLE = f"{DATASET}.q1_uncertain_complaints"

PROMPT = "You are be given a textual complaint entailing that the car was in a crash/accident/collision. Complaint: %s."

POSITIVE_PROMPTS = [
    "car was in a crash, accident, or collision",
    "vehicle crashed or collided with another object",
    "car was involved in a traffic accident or crash",
]
NEGATIVE_PROMPTS = [
    "car had a mechanical or maintenance issue",
    "vehicle had engine, brake, or electrical problems",
    "car needed repair due to wear and tear or defect",
]

ALPHA = 0.2
PAPER_BQ_Q1 = {"score_f1": 0.71, "latency_s": 61.7, "cost_usd": 1.44}
PAPER_DASE_NN_Q1 = {"score_f1": 0.66, "latency_s": 0.9, "cost_usd": 5e-6}
SKIP_BASELINE = False

Q1_BASELINE_SQL = f"""
SELECT DISTINCT c.car_id
FROM {DATASET}.complaints AS c
WHERE AI.IF(
    FORMAT('You are be given a textual complaint entailing that the car was in a crash/accident/collision. Complaint: %s.', c.summary),
    connection_id => 'us.connection',
    endpoint => 'gemini-2.5-flash'
)
"""

Q1_CASCADE_STAGE2_SQL = f"""
SELECT DISTINCT c.car_id AS id
FROM {STAGING_TABLE} AS c
WHERE AI.IF(
    FORMAT('You are be given a textual complaint entailing that the car was in a crash/accident/collision. Complaint: %s.', c.summary),
    connection_id => 'us.connection',
    endpoint => 'gemini-2.5-flash'
)
"""


def make_q1_verifier():
    """CTAS staging from uncertain complaint_ids, then AI.IF returning car_ids."""
    def make_staging(cids):
        items = ",".join(str(int(c)) for c in cids)
        return f"""
        CREATE OR REPLACE TABLE {STAGING_TABLE} AS
        SELECT * FROM {DATASET}.complaints
        WHERE complaint_id IN UNNEST([{items}])
        """
    return AiIfVerifier(
        verify_sql=Q1_CASCADE_STAGE2_SQL, make_staging_sql=make_staging,
        id_column="id", coerce_id=int,
    )


def per_row_cost_q1(client, sample_summaries, k=10):
    """Calibration uses the verbatim FORMAT('...', @s_i) shape — pass as raw SQL."""
    THINKING = "JSON '{\"generation_config\":{\"thinking_config\":{\"thinking_budget\":0}}}'"
    selects, params = [], []
    for i, s in enumerate(sample_summaries[:k]):
        selects.append(f"""
        SELECT AI.GENERATE_BOOL(
          FORMAT('{PROMPT}', @s_{i}),
          connection_id => 'us.connection',
          endpoint => 'gemini-2.5-flash',
          model_params => {THINKING}
        ) AS verdict""")
        params.append(bigquery.ScalarQueryParameter(f"s_{i}", "STRING", s))
    sql = " UNION ALL ".join(selects)
    cfg = bigquery.QueryJobConfig(query_parameters=params, use_query_cache=False)
    # Use dase_cascade's calibration helpers manually since per_row_cost only
    # supports its own SQL templates and we need verbatim FORMAT shape.
    from dase_cascade.calibration import _sum_tokens, _to_cost, CalibrationResult
    t0 = time.time()
    df = client.query(sql, job_config=cfg).result().to_dataframe()
    elapsed = time.time() - t0
    p_other, p_audio, out, thoughts = _sum_tokens(df["verdict"])
    n = len(df)
    cost = _to_cost(p_other, p_audio, out, thoughts)
    return CalibrationResult(
        method="AI.GENERATE_BOOL with Q1 FORMAT prompt + thinking_budget=0",
        n_sample=n,
        tokens_total={"prompt_other": p_other, "prompt_audio": p_audio, "output": out, "thoughts": thoughts},
        sample_cost_usd=cost,
        per_row_cost_usd=cost / n if n else 0.0,
        elapsed_s=elapsed,
    )


def main():
    profile = build_profile(
        scenario="cars", query_id=1, scale_factor=19672,
        prompt=PROMPT, params={"alpha": ALPHA},
        cascade_form=(
            f"F-cascade (staging table {STAGING_TABLE}); "
            "cascade_car_ids = DISTINCT(dase_confident_pos_car_ids ∪ bq_pos_on_uncertain_car_ids)"
        ),
        extra={
            "operator": "F",
            "structural_filter": "",
            "dase_prompts": {"positive": POSITIVE_PROMPTS, "negative": NEGATIVE_PROMPTS},
        },
    )

    print("Loading text_complaints + GT...")
    t = time.time()
    df = pd.read_parquet(TEXT_PARQUET)
    n_total = len(df)
    gt_ids = set(int(x) for x in pd.read_csv(GT_CSV)["car_id"])
    n_gt = len(gt_ids)
    t_load = time.time() - t

    print(f"  {n_total} complaints, GT positive cars: {n_gt}")
    profile["data"] = {"n_complaints": n_total, "n_gt_positive_cars": n_gt}

    text_emb = np.stack(df["embedding"].tolist()).astype(np.float32)
    complaint_ids = df["complaint_id"].astype(int).tolist()
    car_by_complaint = dict(zip(complaint_ids, df["car_id"].astype(int)))

    client = bq_client(PROJECT)

    # ── Per-row cost calibration ──
    print("\n=== Per-row cost calibration (k=10 on real summaries) ===")
    sample_summaries = [str(df.iloc[i]["summary"]) for i in range(min(10, n_total))]
    cal = per_row_cost_q1(client, sample_summaries, k=10)
    per_row = cal.per_row_cost_usd
    print(f"  per_row=${per_row:.6f}, sample_cost=${cal.sample_cost_usd:.6f}, elapsed={cal.elapsed_s:.1f}s")
    profile["calibration"] = cal.to_dict()

    # ── Cascade: Signal+Band over complaints, Verifier returns car_ids ──
    cascade = Cascade(
        embeddings=text_emb,
        ids=complaint_ids,
        signal=MarginSignal(positive_prompts=POSITIVE_PROMPTS, negative_prompts=NEGATIVE_PROMPTS),
        band=AlphaBand(alpha=ALPHA),
        verifier=make_q1_verifier(),
    )
    print("\n=== Cascade (MarginSignal → AlphaBand → AiIfVerifier) ===")
    cres = cascade.run(client, per_row)

    # Project complaint-level confident_pos → car_id; verifier already returns car_ids
    confident_pos_car_ids = {car_by_complaint[c] for c in cres.confident_pos_ids}
    bq_pos_in_uncertain_cars = set(cres.verifier_result.positive_ids)
    n_uncertain = len(cres.uncertain_ids)
    n_confident_neg = len(cres.confident_neg_ids)
    s2_calls = n_uncertain
    cascade_cost = cres.verifier_result.cost_usd
    cascade_car_ids = confident_pos_car_ids | bq_pos_in_uncertain_cars
    cp, cr, c_f1 = f1_set(cascade_car_ids, gt_ids)
    print(f"  alpha={ALPHA}, n_uncertain={n_uncertain}, confident_pos_cars={len(confident_pos_car_ids)}, confident_neg={n_confident_neg}")
    print(f"  BQ yes car_ids on uncertain: {len(bq_pos_in_uncertain_cars)}")
    print(f"  cascade {len(cascade_car_ids)} car_ids; P={cp:.4f} R={cr:.4f} F1={c_f1:.4f}")

    cascade_total_wall = t_load + cres.total_wall_s
    cascade_total_slot = cres.verifier_result.ctas_slot_ms + cres.verifier_result.slot_ms
    s1_wall = cres.verifier_result.ctas_wall_s
    s1_slot = cres.verifier_result.ctas_slot_ms
    s2_wall = cres.verifier_result.wall_s
    s2_slot = cres.verifier_result.slot_ms

    profile["dase_breakdown_s"] = {
        "data_load": t_load,
        "embed_prompts": cres.timings_s.get("signal_compute", 0.0),
        "margin_compute": 0.0,
        "rank_topk_or_partition": cres.timings_s.get("band_partition", 0.0),
        "total": t_load + cres.timings_s.get("signal_compute", 0.0) + cres.timings_s.get("band_partition", 0.0),
    }
    profile["dase_partition"] = {
        "n_uncertain_complaints": n_uncertain,
        "n_confident_pos_cars": len(confident_pos_car_ids),
        "n_confident_neg_complaints": n_confident_neg,
    }

    # ── Baseline ──
    if SKIP_BASELINE:
        print(f"\n=== Baseline ABORTED — using paper Table 4(e) numbers ===")
        b_p = b_r = None
        b_f1 = PAPER_BQ_Q1["score_f1"]; bwall = PAPER_BQ_Q1["latency_s"]; bslot = None
        bcost = PAPER_BQ_Q1["cost_usd"]; bcalls = round(bcost / per_row) if per_row else n_total
        bres_ids = set()
        profile["baseline"] = {
            "_status": "aborted",
            "score": {"f1_score": b_f1, "_source": "paper Table 4(e)"},
            "latency_breakdown": {"wall_s": bwall, "slot_ms": None, "_source": "paper"},
            "cost_breakdown": {"n_llm_calls": bcalls, "per_row_cost_usd": per_row,
                               "total_cost_usd": bcost, "_source": "paper"},
            "method": "sembench bigquery/Q1.sql verbatim — NOT EXECUTED",
            "sql": Q1_BASELINE_SQL.strip(),
        }
    else:
        print("\n=== Baseline (sembench Q1.sql verbatim on cars_dataset.complaints) ===")
        bdf, bwall, bslot, _ = run_query(client, Q1_BASELINE_SQL)
        bres_ids = set(int(x) for x in bdf["car_id"])
        bcalls = n_total
        bcost = per_row * bcalls
        b_p, b_r, b_f1 = f1_set(bres_ids, gt_ids)
        print(f"  returned {len(bres_ids)} car_ids; P={b_p:.4f} R={b_r:.4f} F1={b_f1:.4f}")
        print(f"  wall={bwall:.2f}s, slot_ms={bslot}, n_calls={bcalls}, cost=${bcost:.6f}")
        profile["baseline"] = {
            "method": "sembench bigquery/Q1.sql verbatim on cars_dataset.complaints",
            "sql": Q1_BASELINE_SQL.strip(),
            "result_ids": sorted(list(bres_ids)),
            "score": {"precision": b_p, "recall": b_r, "f1_score": b_f1},
            "latency_breakdown": {"wall_s": bwall, "slot_ms": bslot},
            "cost_breakdown": {
                "n_llm_calls": bcalls,
                "n_llm_calls_method": "scope size (Q1 no LIMIT, all complaints)",
                "per_row_cost_usd": per_row, "total_cost_usd": bcost,
            },
        }

    profile["cascade"] = {
        "method": "F-cascade via dase_cascade: Cascade(MarginSignal, AlphaBand, AiIfVerifier(CTAS staging)).run(); merge dase_confident_pos ∪ bq_pos_on_uncertain (DISTINCT car_id sets)",
        "stage1_ctas": {"sql": cres.verifier_result.ctas_sql,
                        "latency_breakdown": {"wall_s": s1_wall, "slot_ms": s1_slot}, "cost_usd": 0.0},
        "stage2_run": {
            "sql": Q1_CASCADE_STAGE2_SQL.strip(),
            "result_bq_pos_cars_in_uncertain": sorted(list(bq_pos_in_uncertain_cars)),
            "latency_breakdown": {"wall_s": s2_wall, "slot_ms": s2_slot},
            "cost_breakdown": {
                "n_llm_calls": s2_calls,
                "n_llm_calls_method": "n_uncertain (staging size)",
                "per_row_cost_usd": per_row, "total_cost_usd": cascade_cost,
            },
        },
        "result_ids": sorted(list(cascade_car_ids)),
        "score": {"precision": cp, "recall": cr, "f1_score": c_f1},
        "totals": {
            "wall_s": cascade_total_wall,
            "wall_breakdown_s": {
                "dase": profile["dase_breakdown_s"]["total"],
                "bq_stage1_ctas": s1_wall,
                "bq_stage2_aiif": s2_wall,
            },
            "slot_ms_bq_total": cascade_total_slot,
            "cost_usd": cascade_cost,
            "n_llm_calls": s2_calls,
        },
    }

    paper_n_calls = round(PAPER_BQ_Q1["cost_usd"] / per_row) if per_row else None
    profile["comparison"] = {
        "score": {"paper_BQ": PAPER_BQ_Q1["score_f1"], "paper_DASE_NN": PAPER_DASE_NN_Q1["score_f1"],
                  "ours_BQ": b_f1, "ours_cascade": c_f1,
                  "_baseline_source": "paper (aborted)" if SKIP_BASELINE else "ours"},
        "wall_s": {"paper_BQ": PAPER_BQ_Q1["latency_s"], "paper_DASE_NN": PAPER_DASE_NN_Q1["latency_s"],
                   "ours_BQ": bwall, "ours_cascade": cascade_total_wall},
        "slot_ms_bq": {"ours_BQ": bslot, "ours_cascade": cascade_total_slot, "cascade_stage2": s2_slot},
        "cost_usd": {"paper_BQ": PAPER_BQ_Q1["cost_usd"], "paper_DASE_NN": PAPER_DASE_NN_Q1["cost_usd"],
                     "ours_BQ": bcost, "ours_cascade": cascade_cost},
        "n_llm_calls": {"paper_BQ": paper_n_calls, "paper_DASE_NN": 0,
                        "ours_BQ": bcalls, "ours_cascade": s2_calls},
    }

    write_profile(profile, PROFILE_PATH)

    print_summary(
        f"Cars Q1 (alpha={ALPHA})",
        columns=["paper BQ", "DASE+NN", "ours BQ", "ours cascade"],
        rows=[
            ("score (F1)", [PAPER_BQ_Q1["score_f1"], PAPER_DASE_NN_Q1["score_f1"], b_f1, c_f1], ".2f"),
            ("wall (s)",   [PAPER_BQ_Q1["latency_s"], PAPER_DASE_NN_Q1["latency_s"], bwall, cascade_total_wall], ".2f"),
            ("cost ($)",   [PAPER_BQ_Q1["cost_usd"], PAPER_DASE_NN_Q1["cost_usd"], bcost, cascade_cost], ".4f"),
            ("#LLM calls", [None, 0, bcalls, s2_calls], "d"),
        ],
    )


if __name__ == "__main__":
    main()
