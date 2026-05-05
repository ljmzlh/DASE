#!/usr/bin/env -S python -u
"""
Cars Q4 cascade (v2) — text-unary F + scalar AVG aggregation via dase_cascade.

NL: What is the average age of cars with engine problems?
GT: avg_age = 13.487 @ sf_19672.
Eval: aggregation_single (relative_error → score=1/(1+rel_err)).

Refactored to use dase_cascade. Operator (paper Table 3): F.
Cascade(MarginSignal + AlphaBand + AiIfVerifier(CTAS staging)) → set of
yes car_ids (dase_confident_pos ∪ bq_yes_on_uncertain) → client-side
2026 - mean(year) aggregation.
"""
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
    relative_error_score, build_profile, write_profile, print_summary,
)

# ─── Paths / scenario constants ──────────────────────────────────────────
CARS_DIR = os.path.abspath(os.path.join(_HERE, ".."))
CARS_PARQUET = os.path.join(CARS_DIR, "data", "cars.parquet")
TEXT_PARQUET = os.path.join(CARS_DIR, "data", "text_complaints.parquet")
GT_CSV = os.path.join(CARS_DIR, "ground_truth", "Q4.csv")
PROFILE_PATH = os.path.join(CARS_DIR, "outputs", "Q4.json")

PROJECT = os.environ.get("GCP_PROJECT", "")
DATASET = "cars_dataset"
STAGING_TABLE = f"{DATASET}.q4_uncertain_complaints"

PROMPT = "In the complaint, the car has some problems with engine / connected to engine. Complaint: %s."

POSITIVE_PROMPTS = [
    "complaint about car engine problem",
    "the car has issues with the engine or engine-connected parts",
    "engine malfunction or failure described in the complaint",
]
NEGATIVE_PROMPTS = [
    "complaint about brakes, electrical, or non-engine issue",
    "car problem unrelated to the engine",
    "issues with steering, suspension, or other non-engine components",
]

ALPHA = 0.2
PAPER_BQ_Q4 = {"score": 0.99, "latency_s": 68.7, "cost_usd": 1.41}
PAPER_DASE_NN_Q4 = {"score": 0.99, "latency_s": 0.9, "cost_usd": 5e-6}
SKIP_BASELINE = False

Q4_BASELINE_SQL = f"""
SELECT 2026 - AVG(c.year) AS average_age
FROM {DATASET}.cars AS c
JOIN {DATASET}.complaints AS s ON c.car_id = s.car_id
WHERE AI.IF(
    FORMAT('In the complaint, the car has some problems with engine / connected to engine. Complaint: %s.', s.summary),
    connection_id => 'us.connection',
    endpoint => 'gemini-2.5-flash'
)
"""

Q4_CASCADE_STAGE2_SQL = f"""
SELECT DISTINCT s.car_id AS id
FROM {STAGING_TABLE} AS s
WHERE AI.IF(
    FORMAT('In the complaint, the car has some problems with engine / connected to engine. Complaint: %s.', s.summary),
    connection_id => 'us.connection',
    endpoint => 'gemini-2.5-flash'
)
"""


def make_q4_verifier():
    """CTAS staging from uncertain complaint_ids, then AI.IF returning car_ids."""
    def make_staging(cids):
        items = ",".join(str(int(c)) for c in cids)
        return f"""
        CREATE OR REPLACE TABLE {STAGING_TABLE} AS
        SELECT * FROM {DATASET}.complaints
        WHERE complaint_id IN UNNEST([{items}])
        """
    return AiIfVerifier(
        verify_sql=Q4_CASCADE_STAGE2_SQL, make_staging_sql=make_staging,
        id_column="id", coerce_id=int,
    )


def per_row_cost_q4(client, sample_summaries, k=10):
    """Q4 calibration uses verbatim FORMAT('...', @s_i) — pass as raw SQL."""
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
    from dase_cascade.calibration import _sum_tokens, _to_cost, CalibrationResult
    t0 = time.time()
    df = client.query(sql, job_config=cfg).result().to_dataframe()
    elapsed = time.time() - t0
    p_other, p_audio, out, thoughts = _sum_tokens(df["verdict"])
    n = len(df)
    cost = _to_cost(p_other, p_audio, out, thoughts)
    return CalibrationResult(
        method="AI.GENERATE_BOOL with Q4 FORMAT prompt + thinking_budget=0",
        n_sample=n,
        tokens_total={"prompt_other": p_other, "prompt_audio": p_audio, "output": out, "thoughts": thoughts},
        sample_cost_usd=cost,
        per_row_cost_usd=cost / n if n else 0.0,
        elapsed_s=elapsed,
    )


def main():
    profile = build_profile(
        scenario="cars", query_id=4, scale_factor=19672,
        prompt=PROMPT, params={"alpha": ALPHA},
        cascade_form=(
            f"F-cascade (staging table {STAGING_TABLE}); F + scalar AVG aggregation; "
            "cascade_avg_age = 2026 - cars.year[union(dase_confident_pos_cars, bq_yes_in_uncertain)].mean()"
        ),
        extra={
            "operator": "F",
            "structural_filter": "",
            "dase_prompts": {"positive": POSITIVE_PROMPTS, "negative": NEGATIVE_PROMPTS},
        },
    )

    print("Loading text_complaints + cars + GT...")
    t = time.time()
    df = pd.read_parquet(TEXT_PARQUET)
    cars = pd.read_parquet(CARS_PARQUET)[["car_id", "year"]]
    car_year = dict(zip(cars["car_id"], cars["year"]))
    n_total = len(df)
    gt_avg = float(pd.read_csv(GT_CSV)["average_age"].iloc[0])
    t_load = time.time() - t

    print(f"  {n_total} complaints; GT average_age = {gt_avg:.4f}")
    profile["data"] = {"n_complaints": n_total, "gt_average_age": gt_avg}

    text_emb = np.stack(df["embedding"].tolist()).astype(np.float32)
    complaint_ids = df["complaint_id"].astype(int).tolist()
    car_by_complaint = dict(zip(complaint_ids, df["car_id"].astype(int)))

    client = bq_client(PROJECT)

    print("\n=== Per-row cost calibration (k=10 on real summaries) ===")
    sample_summaries = [str(df.iloc[i]["summary"]) for i in range(min(10, n_total))]
    cal = per_row_cost_q4(client, sample_summaries, k=10)
    per_row = cal.per_row_cost_usd
    print(f"  per_row=${per_row:.6f}, sample_cost=${cal.sample_cost_usd:.6f}, elapsed={cal.elapsed_s:.1f}s")
    profile["calibration"] = cal.to_dict()

    # ── Cascade ──
    cascade = Cascade(
        embeddings=text_emb,
        ids=complaint_ids,
        signal=MarginSignal(positive_prompts=POSITIVE_PROMPTS, negative_prompts=NEGATIVE_PROMPTS),
        band=AlphaBand(alpha=ALPHA),
        verifier=make_q4_verifier(),
    )
    print("\n=== Cascade (MarginSignal → AlphaBand → AiIfVerifier) ===")
    cres = cascade.run(client, per_row)

    confident_pos_car_ids = {car_by_complaint[c] for c in cres.confident_pos_ids}
    bq_yes_uncertain_cars = set(cres.verifier_result.positive_ids)
    n_uncertain = len(cres.uncertain_ids)
    n_confident_neg = len(cres.confident_neg_ids)
    s2_calls = n_uncertain
    cascade_cost = cres.verifier_result.cost_usd

    union_yes_cars = confident_pos_car_ids | bq_yes_uncertain_cars
    years = [car_year[c] for c in union_yes_cars if c in car_year]
    cascade_avg = 2026 - float(np.mean(years)) if years else 0.0
    c_score = relative_error_score(cascade_avg, gt_avg)
    c_rel = abs(cascade_avg - gt_avg) / abs(gt_avg) if gt_avg else 0.0
    print(f"  alpha={ALPHA}, n_uncertain={n_uncertain}, confident_pos_cars={len(confident_pos_car_ids)}, confident_neg={n_confident_neg}")
    print(f"  cascade union_yes_cars: {len(union_yes_cars)} ({len(confident_pos_car_ids)} dase + {len(bq_yes_uncertain_cars)} bq)")
    print(f"  cascade avg_age={cascade_avg:.4f}, GT={gt_avg:.4f}, rel_err={c_rel:.4f}, score={c_score:.4f}")

    cascade_total_wall = t_load + cres.total_wall_s
    s1_wall = cres.verifier_result.ctas_wall_s
    s1_slot = cres.verifier_result.ctas_slot_ms
    s2_wall = cres.verifier_result.wall_s
    s2_slot = cres.verifier_result.slot_ms
    cascade_total_slot = s1_slot + s2_slot

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

    if SKIP_BASELINE:
        print(f"\n=== Baseline ABORTED — using paper Table 4(e) numbers ===")
        b_avg = None; b_rel = None
        b_score = PAPER_BQ_Q4["score"]; bwall = PAPER_BQ_Q4["latency_s"]; bslot = None
        bcost = PAPER_BQ_Q4["cost_usd"]; bcalls = round(bcost / per_row) if per_row else n_total
        profile["baseline"] = {
            "_status": "aborted",
            "score": {"score": b_score, "_source": "paper Table 4(e)"},
            "latency_breakdown": {"wall_s": bwall, "slot_ms": None, "_source": "paper"},
            "cost_breakdown": {"n_llm_calls": bcalls, "per_row_cost_usd": per_row,
                               "total_cost_usd": bcost, "_source": "paper"},
            "method": "sembench bigquery/Q4.sql verbatim — NOT EXECUTED",
            "sql": Q4_BASELINE_SQL.strip(),
        }
    else:
        print("\n=== Baseline (sembench Q4.sql verbatim, scalar AVG output) ===")
        bdf, bwall, bslot, _ = run_query(client, Q4_BASELINE_SQL)
        b_avg = float(bdf.iloc[0]["average_age"])
        b_score = relative_error_score(b_avg, gt_avg)
        b_rel = abs(b_avg - gt_avg) / abs(gt_avg) if gt_avg else 0.0
        bcalls = n_total
        bcost = per_row * bcalls
        print(f"  baseline avg_age={b_avg:.4f}, GT={gt_avg:.4f}, rel_err={b_rel:.4f}, score={b_score:.4f}")
        print(f"  wall={bwall:.2f}s, slot_ms={bslot}, n_calls={bcalls}, cost=${bcost:.6f}")
        profile["baseline"] = {
            "method": "sembench bigquery/Q4.sql verbatim on cars ⨝ complaints",
            "sql": Q4_BASELINE_SQL.strip(),
            "result_avg_age": b_avg,
            "score": {"relative_error": b_rel, "score": b_score},
            "latency_breakdown": {"wall_s": bwall, "slot_ms": bslot},
            "cost_breakdown": {
                "n_llm_calls": bcalls,
                "n_llm_calls_method": "scope size (no LIMIT)",
                "per_row_cost_usd": per_row,
                "total_cost_usd": bcost,
            },
        }

    profile["cascade"] = {
        "method": "F-cascade via dase_cascade: Cascade(MarginSignal, AlphaBand, AiIfVerifier(CTAS staging)).run() returns yes car_ids; locally union with dase_confident_pos and compute AVG year",
        "stage1_ctas": {"sql": cres.verifier_result.ctas_sql,
                        "latency_breakdown": {"wall_s": s1_wall, "slot_ms": s1_slot}, "cost_usd": 0.0},
        "stage2_run": {
            "sql": Q4_CASCADE_STAGE2_SQL.strip(),
            "result_bq_yes_cars_in_uncertain": sorted(list(bq_yes_uncertain_cars)),
            "latency_breakdown": {"wall_s": s2_wall, "slot_ms": s2_slot},
            "cost_breakdown": {
                "n_llm_calls": s2_calls,
                "n_llm_calls_method": "n_uncertain (staging size)",
                "per_row_cost_usd": per_row,
                "total_cost_usd": cascade_cost,
            },
        },
        "result_avg_age": cascade_avg,
        "n_union_yes_cars": len(union_yes_cars),
        "score": {"relative_error": c_rel, "score": c_score},
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

    paper_n_calls = round(PAPER_BQ_Q4["cost_usd"] / per_row) if per_row else None
    profile["comparison"] = {
        "score": {"paper_BQ": PAPER_BQ_Q4["score"], "paper_DASE_NN": PAPER_DASE_NN_Q4["score"],
                  "ours_BQ": b_score, "ours_cascade": c_score,
                  "_baseline_source": "paper (aborted)" if SKIP_BASELINE else "ours"},
        "wall_s": {"paper_BQ": PAPER_BQ_Q4["latency_s"], "paper_DASE_NN": PAPER_DASE_NN_Q4["latency_s"],
                   "ours_BQ": bwall, "ours_cascade": cascade_total_wall},
        "slot_ms_bq": {"ours_BQ": bslot, "ours_cascade": cascade_total_slot, "cascade_stage2": s2_slot},
        "cost_usd": {"paper_BQ": PAPER_BQ_Q4["cost_usd"], "paper_DASE_NN": PAPER_DASE_NN_Q4["cost_usd"],
                     "ours_BQ": bcost, "ours_cascade": cascade_cost},
        "n_llm_calls": {"paper_BQ": paper_n_calls, "paper_DASE_NN": 0,
                        "ours_BQ": bcalls, "ours_cascade": s2_calls},
    }

    write_profile(profile, PROFILE_PATH)

    print_summary(
        f"Cars Q4 (alpha={ALPHA})",
        columns=["paper BQ", "DASE+NN", "ours BQ", "ours cascade"],
        rows=[
            ("score",      [PAPER_BQ_Q4["score"], PAPER_DASE_NN_Q4["score"], b_score, c_score], ".2f"),
            ("wall (s)",   [PAPER_BQ_Q4["latency_s"], PAPER_DASE_NN_Q4["latency_s"], bwall, cascade_total_wall], ".2f"),
            ("cost ($)",   [PAPER_BQ_Q4["cost_usd"], PAPER_DASE_NN_Q4["cost_usd"], bcost, cascade_cost], ".4f"),
            ("#LLM calls", [None, 0, bcalls, s2_calls], "d"),
        ],
    )


if __name__ == "__main__":
    main()
