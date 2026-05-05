#!/usr/bin/env -S python -u
"""
Cars Q2 cascade (v2) — audio multimodal F + structural prefilter via dase_cascade.

NL: Find electric cars with available audio recordings that show a dead battery.
GT: 1 car_id @ sf_19672 ({98676}).
Eval: F1 on car_id set.

Refactored to use dase_cascade. Operator (paper Table 3): F.
Cascade(MarginSignal + AlphaBand + AiIfVerifier(CTAS staging)) on the
electric ⨝ audio scope; verifier returns car_ids; client unions with
dase_confident_pos_car_ids.
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
    f1_set, build_profile, write_profile, print_summary,
)

# ─── Paths / scenario constants ──────────────────────────────────────────
CARS_DIR = os.path.abspath(os.path.join(_HERE, ".."))
CARS_PARQUET = os.path.join(CARS_DIR, "data", "cars.parquet")
AUDIO_PARQUET = os.path.join(CARS_DIR, "data", "audio_cars.parquet")
GT_CSV = os.path.join(CARS_DIR, "ground_truth", "Q2.csv")
PROFILE_PATH = os.path.join(CARS_DIR, "outputs", "Q2.json")

PROJECT = os.environ.get("GCP_PROJECT", "")
BUCKET = f"{PROJECT}-cars_dataset"
GCS_FOLDER = "car_audios"
DATASET = "cars_dataset"
STAGING_TABLE = f"{DATASET}.q2_uncertain_audio_mm"

PROMPT = "You are given an audio recording of car diagnostics. Return true if the car from the recording has a dead battery, false otherwise."

POSITIVE_PROMPTS = [
    "audio recording of a car with a dead battery",
    "engine fails to start due to dead battery",
    "clicking sound from a car ignition with no battery power",
]
NEGATIVE_PROMPTS = [
    "audio of a car engine running normally",
    "audio of mechanical issues unrelated to battery",
    "engine sounds healthy with no electrical issue",
]

ALPHA = 0.5
PAPER_BQ_Q2 = {"score_f1": 0.08, "latency_s": 14.1, "cost_usd": 0.01}
PAPER_DASE_NN_Q2 = {"score_f1": 0.00, "latency_s": 0.7, "cost_usd": 5e-6}
SKIP_BASELINE = False

Q2_BASELINE_SQL = f"""
SELECT DISTINCT c.car_id
FROM {DATASET}.cars AS c
JOIN {DATASET}.audio_mm AS a ON c.car_id = a.car_id
WHERE c.fuel_type = 'Electric'
  AND AI.IF(
    ('{PROMPT}', a.image),
    connection_id => 'us.connection',
    endpoint => 'gemini-2.5-flash'
  )
"""

Q2_CASCADE_STAGE2_SQL = f"""
SELECT DISTINCT c.car_id AS id
FROM {DATASET}.cars AS c
JOIN {STAGING_TABLE} AS a ON c.car_id = a.car_id
WHERE c.fuel_type = 'Electric'
  AND AI.IF(
    ('{PROMPT}', a.image),
    connection_id => 'us.connection',
    endpoint => 'gemini-2.5-flash'
  )
"""


def make_q2_verifier():
    """CTAS staging from uncertain GCS uris (audio_path), then AI.IF returning car_ids."""
    def make_staging(uris):
        items = ",".join(f"'{u}'" for u in uris)
        return f"""
        CREATE OR REPLACE TABLE {STAGING_TABLE} AS
        SELECT m.audio_id, m.car_id, ot.ref AS image
        FROM {DATASET}.car_audio m
        JOIN {DATASET}.cars_audios ot ON ot.uri = m.audio_path
        WHERE m.audio_path IN UNNEST([{items}])
        """
    return AiIfVerifier(
        verify_sql=Q2_CASCADE_STAGE2_SQL, make_staging_sql=make_staging,
        id_column="id", coerce_id=int,
    )


def main():
    profile = build_profile(
        scenario="cars", query_id=2, scale_factor=19672,
        prompt=PROMPT, params={"alpha": ALPHA},
        cascade_form=(
            f"F-cascade (staging table {STAGING_TABLE}); audio multimodal; "
            "scope = electric cars ⨝ audio; "
            "cascade_car_ids = DISTINCT(dase_confident_pos_car_ids ∪ bq_pos_in_uncertain_car_ids)"
        ),
        extra={
            "operator": "F",
            "structural_filter": "c.fuel_type = 'Electric'",
            "dase_prompts": {"positive": POSITIVE_PROMPTS, "negative": NEGATIVE_PROMPTS},
        },
    )

    print("Loading cars + audio + GT, applying structural prefilter (Electric ⨝ audio)...")
    t = time.time()
    cars = pd.read_parquet(CARS_PARQUET)
    audio = pd.read_parquet(AUDIO_PARQUET)
    electric_ids = set(cars[cars["fuel_type"] == "Electric"]["car_id"])
    scope = audio[audio["car_id"].isin(electric_ids)].reset_index(drop=True)
    n_total = len(scope)
    gt_ids = set(int(x) for x in pd.read_csv(GT_CSV)["car_id"])
    n_gt = len(gt_ids)
    t_load = time.time() - t

    print(f"  scope: {n_total} (electric ⨝ audio); GT positive cars: {n_gt}")
    profile["data"] = {
        "n_cars_total": len(cars),
        "n_audio_total": len(audio),
        "n_electric_cars": len(electric_ids),
        "n_rows_in_scope": n_total,
        "n_gt_positive_cars": n_gt,
    }

    audio_emb = np.stack(scope["embedding"].tolist()).astype(np.float32)
    audio_uris = [f"gs://{BUCKET}/{GCS_FOLDER}/{os.path.basename(p)}" for p in scope["audio_path"]]
    car_by_uri = dict(zip(audio_uris, scope["car_id"].astype(int)))

    client = bq_client(PROJECT)

    print("\n=== Per-row cost calibration (k=5 multimodal audio) ===")
    sample_uris = audio_uris[:5] if len(audio_uris) >= 5 else audio_uris
    cal = per_row_cost(
        client, PROMPT,
        sample_uris=sample_uris,
        ext_table=f"{DATASET}.cars_audios",
        method_label="AI.GENERATE_BOOL multimodal (audio ref, Q2 prompt) + thinking_budget=0",
        k=5,
    )
    per_row = cal.per_row_cost_usd
    print(f"  per_row=${per_row:.6f}, sample_cost=${cal.sample_cost_usd:.6f}, elapsed={cal.elapsed_s:.1f}s")
    profile["calibration"] = cal.to_dict()

    # ── Cascade: Signal+Band on audio embeddings, Verifier on uncertain URIs ──
    cascade = Cascade(
        embeddings=audio_emb,
        ids=audio_uris,
        signal=MarginSignal(positive_prompts=POSITIVE_PROMPTS, negative_prompts=NEGATIVE_PROMPTS),
        band=AlphaBand(alpha=ALPHA),
        verifier=make_q2_verifier(),
    )
    print("\n=== Cascade (MarginSignal → AlphaBand → AiIfVerifier) ===")
    cres = cascade.run(client, per_row)

    confident_pos_car_ids = {car_by_uri[u] for u in cres.confident_pos_ids}
    bq_pos_in_uncertain_cars = set(cres.verifier_result.positive_ids)
    n_uncertain = len(cres.uncertain_ids)
    n_confident_neg = len(cres.confident_neg_ids)
    s2_calls = n_uncertain
    cascade_cost = cres.verifier_result.cost_usd
    cascade_car_ids = confident_pos_car_ids | bq_pos_in_uncertain_cars
    cp, cr, c_f1 = f1_set(cascade_car_ids, gt_ids)
    print(f"  alpha={ALPHA}, n_uncertain={n_uncertain}, confident_pos_cars={len(confident_pos_car_ids)}, confident_neg={n_confident_neg}")
    print(f"  BQ yes car_ids: {len(bq_pos_in_uncertain_cars)}")
    print(f"  cascade {len(cascade_car_ids)} car_ids; P={cp:.4f} R={cr:.4f} F1={c_f1:.4f}")

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
        "n_uncertain_rows": n_uncertain,
        "n_confident_pos_cars": len(confident_pos_car_ids),
        "n_confident_neg_rows": n_confident_neg,
    }

    if SKIP_BASELINE:
        print(f"\n=== Baseline ABORTED — using paper Table 4(e) numbers ===")
        b_p = b_r = None
        b_f1 = PAPER_BQ_Q2["score_f1"]; bwall = PAPER_BQ_Q2["latency_s"]; bslot = None
        bcost = PAPER_BQ_Q2["cost_usd"]; bcalls = round(bcost / per_row) if per_row else n_total
        bres_ids = set()
        profile["baseline"] = {
            "_status": "aborted",
            "score": {"f1_score": b_f1, "_source": "paper Table 4(e)"},
            "latency_breakdown": {"wall_s": bwall, "slot_ms": None, "_source": "paper"},
            "cost_breakdown": {"n_llm_calls": bcalls, "per_row_cost_usd": per_row,
                               "total_cost_usd": bcost, "_source": "paper"},
            "method": "sembench bigquery/Q2.sql verbatim — NOT EXECUTED",
            "sql": Q2_BASELINE_SQL.strip(),
        }
    else:
        print("\n=== Baseline (sembench Q2.sql verbatim) ===")
        bdf, bwall, bslot, _ = run_query(client, Q2_BASELINE_SQL)
        bres_ids = set(int(x) for x in bdf["car_id"])
        bcalls = n_total
        bcost = per_row * bcalls
        b_p, b_r, b_f1 = f1_set(bres_ids, gt_ids)
        print(f"  returned {len(bres_ids)} car_ids; P={b_p:.4f} R={b_r:.4f} F1={b_f1:.4f}")
        print(f"  wall={bwall:.2f}s, slot_ms={bslot}, n_calls={bcalls}, cost=${bcost:.6f}")
        profile["baseline"] = {
            "method": "sembench bigquery/Q2.sql verbatim on cars_dataset.cars ⨝ audio_mm",
            "sql": Q2_BASELINE_SQL.strip(),
            "result_ids": sorted(list(bres_ids)),
            "score": {"precision": b_p, "recall": b_r, "f1_score": b_f1},
            "latency_breakdown": {"wall_s": bwall, "slot_ms": bslot},
            "cost_breakdown": {
                "n_llm_calls": bcalls,
                "n_llm_calls_method": "scope size (electric ⨝ audio)",
                "per_row_cost_usd": per_row,
                "total_cost_usd": bcost,
            },
        }

    profile["cascade"] = {
        "method": "F-cascade via dase_cascade: Cascade(MarginSignal, AlphaBand, AiIfVerifier(CTAS staging from uncertain audio uris)).run(); merge dase_confident_pos ∪ bq_pos_on_uncertain (DISTINCT car_id sets)",
        "stage1_ctas": {"sql": cres.verifier_result.ctas_sql,
                        "latency_breakdown": {"wall_s": s1_wall, "slot_ms": s1_slot}, "cost_usd": 0.0},
        "stage2_run": {
            "sql": Q2_CASCADE_STAGE2_SQL.strip(),
            "result_bq_pos_cars_in_uncertain": sorted(list(bq_pos_in_uncertain_cars)),
            "latency_breakdown": {"wall_s": s2_wall, "slot_ms": s2_slot},
            "cost_breakdown": {
                "n_llm_calls": s2_calls,
                "n_llm_calls_method": "n_uncertain (staging size)",
                "per_row_cost_usd": per_row,
                "total_cost_usd": cascade_cost,
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

    paper_n_calls = round(PAPER_BQ_Q2["cost_usd"] / per_row) if per_row else None
    profile["comparison"] = {
        "score": {"paper_BQ": PAPER_BQ_Q2["score_f1"], "paper_DASE_NN": PAPER_DASE_NN_Q2["score_f1"],
                  "ours_BQ": b_f1, "ours_cascade": c_f1,
                  "_baseline_source": "paper (aborted)" if SKIP_BASELINE else "ours"},
        "wall_s": {"paper_BQ": PAPER_BQ_Q2["latency_s"], "paper_DASE_NN": PAPER_DASE_NN_Q2["latency_s"],
                   "ours_BQ": bwall, "ours_cascade": cascade_total_wall},
        "slot_ms_bq": {"ours_BQ": bslot, "ours_cascade": cascade_total_slot, "cascade_stage2": s2_slot},
        "cost_usd": {"paper_BQ": PAPER_BQ_Q2["cost_usd"], "paper_DASE_NN": PAPER_DASE_NN_Q2["cost_usd"],
                     "ours_BQ": bcost, "ours_cascade": cascade_cost},
        "n_llm_calls": {"paper_BQ": paper_n_calls, "paper_DASE_NN": 0,
                        "ours_BQ": bcalls, "ours_cascade": s2_calls},
    }

    write_profile(profile, PROFILE_PATH)

    print_summary(
        f"Cars Q2 (alpha={ALPHA})",
        columns=["paper BQ", "DASE+NN", "ours BQ", "ours cascade"],
        rows=[
            ("score (F1)", [PAPER_BQ_Q2["score_f1"], PAPER_DASE_NN_Q2["score_f1"], b_f1, c_f1], ".2f"),
            ("wall (s)",   [PAPER_BQ_Q2["latency_s"], PAPER_DASE_NN_Q2["latency_s"], bwall, cascade_total_wall], ".2f"),
            ("cost ($)",   [PAPER_BQ_Q2["cost_usd"], PAPER_DASE_NN_Q2["cost_usd"], bcost, cascade_cost], ".4f"),
            ("#LLM calls", [None, 0, bcalls, s2_calls], "d"),
        ],
    )


if __name__ == "__main__":
    main()
