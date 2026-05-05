#!/usr/bin/env -S python -u
"""
Cars Q3 cascade (v2) — F+L (top-K margin → BQ AI.IF + LIMIT 10) via dase_cascade.

NL: Find ten cars with manual transmission that are not damaged according to
    images. Return the VIN.
GT: 670 vins.
Eval: retrieval_limit (return ≤10 vins; F1 over 10-row gt sample).

Refactored to use dase_cascade. Operator (paper Table 3): F + L.
Cascade(MarginSignal + TopKBand + AiIfVerifier) — single BQ AI.IF + LIMIT
short-circuits at TARGET=10. The verifier returns vins; everything before is
client-side image_id IN (top-K) inlining.
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
    Cascade, MarginSignal, TopKBand, AiIfVerifier,
    bq_client, per_row_cost, run_query,
    build_profile, write_profile, print_summary,
)

# ─── Paths / scenario constants ──────────────────────────────────────────
CARS_DIR = os.path.abspath(os.path.join(_HERE, ".."))
CARS_PARQUET = os.path.join(CARS_DIR, "data", "cars.parquet")
IMAGE_PARQUET = os.path.join(CARS_DIR, "data", "image_cars.parquet")
GT_CSV = os.path.join(CARS_DIR, "ground_truth", "Q3.csv")
PROFILE_PATH = os.path.join(CARS_DIR, "outputs", "Q3.json")

PROJECT = os.environ.get("GCP_PROJECT", "")
BUCKET = f"{PROJECT}-cars_dataset"
GCS_FOLDER = "car_images"
DATASET = "cars_dataset"

PROMPT = "You are given an image of a vehicle or its parts. Return true if car is not damaged."

POSITIVE_PROMPTS = [
    "an undamaged car in pristine condition",
    "a clean intact vehicle without dents or scratches",
    "a car in good condition with no visible damage",
]
NEGATIVE_PROMPTS = [
    "a damaged or wrecked car with visible damage",
    "a car with dents, scratches, or broken parts",
    "a vehicle showing signs of accident damage",
]

TARGET = 10
K_CANDIDATES = 20
PAPER_BQ_Q3 = {"score_f1": 1.00, "latency_s": 36.4, "cost_usd": 1.66}
PAPER_DASE_NN_Q3 = {"score_f1": 1.00, "latency_s": 0.7, "cost_usd": 5e-6}
SKIP_BASELINE = False

Q3_BASELINE_SQL = f"""
SELECT p.vin
FROM {DATASET}.cars AS p
JOIN {DATASET}.car_mm AS x ON p.car_id = x.car_id
WHERE p.transmission = "Manual"
  AND AI.IF(
    ('{PROMPT}', x.image),
    connection_id => 'us.connection',
    endpoint => 'gemini-2.5-flash'
  )
LIMIT {TARGET}
"""


def make_q3_verifier():
    """Single AI.IF query: image_id IN (top-K) AND transmission='Manual' LIMIT TARGET."""
    def verify_sql_template(image_ids):
        id_list = ",".join(str(int(i)) for i in image_ids)
        return f"""
        SELECT p.vin AS id
        FROM {DATASET}.cars AS p
        JOIN {DATASET}.car_mm AS x ON p.car_id = x.car_id
        WHERE p.transmission = "Manual"
          AND x.image_id IN ({id_list})
          AND AI.IF(
            ('{PROMPT}', x.image),
            connection_id => 'us.connection',
            endpoint => 'gemini-2.5-flash'
          )
        LIMIT {TARGET}
        """
    return AiIfVerifier(
        verify_sql_template=verify_sql_template,
        id_column="id", coerce_id=str,
    )


def f1_against_gt_sample(sys_vins, gt_vins, target=10, seed=42):
    """Mirror evaluator._evaluate_q3 retrieval_limit logic."""
    sys_set = set(sys_vins); gt_set = set(gt_vins)
    correct = sys_set & gt_set
    n_correct = len(correct)
    rng = np.random.default_rng(seed)
    if n_correct == target:
        gt_sample = correct
    elif n_correct < target:
        false_cases = list(gt_set - sys_set)
        n_to_sample = min(target - n_correct, len(false_cases))
        sampled = list(rng.choice(false_cases, size=n_to_sample, replace=False)) if n_to_sample else []
        gt_sample = correct | set(sampled)
        if n_correct == 0:
            gt_sample = set(rng.choice(list(gt_set), size=min(target, len(gt_set)), replace=False))
    else:
        raise ValueError(f"n_correct={n_correct} > target={target}")
    tp = len(sys_set & gt_sample)
    p = tp / len(sys_set) if sys_set else 0.0
    r = tp / len(gt_sample) if gt_sample else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1, tp, len(sys_set), len(gt_sample)


def main():
    profile = build_profile(
        scenario="cars", query_id=3, scale_factor=19672,
        prompt=PROMPT, params={"K_candidates": K_CANDIDATES, "target": TARGET},
        cascade_form=(
            "F+L canonical (top-K by margin → BQ AI.IF on IN(K) + LIMIT TARGET); "
            f"K={K_CANDIDATES}, TARGET={TARGET}; BQ short-circuits at LIMIT"
        ),
        extra={
            "operator": "F+L",
            "structural_filter": "p.transmission = 'Manual'",
            "dase_prompts": {"positive": POSITIVE_PROMPTS, "negative": NEGATIVE_PROMPTS},
        },
    )

    print("Loading cars + images, applying structural prefilter (Manual ⨝ images)...")
    t = time.time()
    cars = pd.read_parquet(CARS_PARQUET)
    img = pd.read_parquet(IMAGE_PARQUET)
    manual = cars[cars["transmission"] == "Manual"][["car_id", "vin"]]
    scope = img.merge(manual, on="car_id", how="inner").reset_index(drop=True)
    n_total = len(scope)
    gt_vins = set(pd.read_csv(GT_CSV)["vin"].astype(str))
    n_gt = len(gt_vins)
    t_load = time.time() - t

    print(f"  scope: {n_total} (Manual ⨝ images); GT positive vins: {n_gt}")
    profile["data"] = {
        "n_cars_total": len(cars), "n_images_total": len(img),
        "n_manual_cars": int((cars["transmission"] == "Manual").sum()),
        "n_rows_in_scope": n_total, "n_gt_positive_vins": n_gt,
    }

    image_emb = np.stack(scope["embedding"].tolist()).astype(np.float32)
    image_ids = scope["image_id"].astype(int).tolist()

    client = bq_client(PROJECT)

    # ── Cascade: MarginSignal + TopKBand + Verifier ──
    cascade = Cascade(
        embeddings=image_emb,
        ids=image_ids,
        signal=MarginSignal(positive_prompts=POSITIVE_PROMPTS, negative_prompts=NEGATIVE_PROMPTS),
        band=TopKBand(k=K_CANDIDATES),
        verifier=make_q3_verifier(),
    )
    # Calibrate per-row cost on a stable sample of in-scope image uris (top-K
    # may not be known yet at calibration time; just pick first 5 in-scope).
    print("\n=== Per-row cost calibration (k=5 multimodal image) ===")
    sample_uris = [
        f"gs://{BUCKET}/{GCS_FOLDER}/{os.path.basename(scope.iloc[i]['image_path'])}"
        for i in range(min(5, n_total))
    ]
    cal = per_row_cost(
        client, PROMPT,
        sample_uris=sample_uris,
        ext_table=f"{DATASET}.cars_images",
        method_label="AI.GENERATE_BOOL multimodal (image ref, Q3 prompt) + thinking_budget=0",
        k=5,
    )
    per_row = cal.per_row_cost_usd
    print(f"  per_row=${per_row:.6f}, sample_cost=${cal.sample_cost_usd:.6f}, elapsed={cal.elapsed_s:.1f}s")
    profile["calibration"] = cal.to_dict()

    print(f"\n=== Cascade (MarginSignal → TopKBand({K_CANDIDATES}) → AiIfVerifier+LIMIT {TARGET}) ===")
    cres = cascade.run(client, per_row)

    # TopKBand puts everything in either uncertain (top-K) or confident_neg.
    top_K_image_ids = list(cres.uncertain_ids)
    # Map image_id → vin / margin / car_id for profile dump
    img_lookup = scope.set_index("image_id")
    top_K_vins = [str(img_lookup.loc[iid]["vin"]) for iid in top_K_image_ids]
    n_top_in_gt = sum(1 for v in top_K_vins if v in gt_vins)
    # margins are cres.scores
    top_K_margins = [float(cres.scores[scope.index[scope["image_id"] == iid][0]]) for iid in top_K_image_ids]

    cascade_vins = sorted(cres.verifier_result.positive_ids)
    cwall = cres.total_wall_s
    cslot = cres.verifier_result.slot_ms
    # n_calls heuristic mirrors original: BQ short-circuits in K; bound by K
    s2_calls = max(round(cslot / 2500), len(cascade_vins))
    s2_calls = min(s2_calls, K_CANDIDATES)
    cascade_cost = per_row * s2_calls
    cp, cr, c_f1, ctp, c_n_sys, c_n_gt_sample = f1_against_gt_sample(cascade_vins, gt_vins, TARGET)
    print(f"  top-K={K_CANDIDATES} margins: min={min(top_K_margins):.4f}, max={max(top_K_margins):.4f}")
    print(f"  top-K vins ∩ GT = {n_top_in_gt}/{K_CANDIDATES}")
    print(f"  cascade returned {len(cascade_vins)} vins; sample TP={ctp} (P={cp:.4f} R={cr:.4f} F1={c_f1:.4f})")
    print(f"  wall={cwall:.2f}s, slot_ms={cslot}, n_calls~{s2_calls}, cost=${cascade_cost:.6f}")

    cascade_total_wall = t_load + cwall

    profile["dase_breakdown_s"] = {
        "data_load": t_load,
        "embed_prompts": cres.timings_s.get("signal_compute", 0.0),
        "margin_compute": 0.0,
        "rank_topk_or_partition": cres.timings_s.get("band_partition", 0.0),
        "total": t_load + cres.timings_s.get("signal_compute", 0.0) + cres.timings_s.get("band_partition", 0.0),
    }
    profile["dase_top_K"] = {
        "image_ids": top_K_image_ids,
        "vins": top_K_vins,
        "margins": top_K_margins,
        "n_top_K_in_GT": n_top_in_gt,
    }

    # ── Baseline ──
    if SKIP_BASELINE:
        print(f"\n=== Baseline ABORTED — using paper Table 4(e) numbers ===")
        b_p = b_r = None
        b_f1 = PAPER_BQ_Q3["score_f1"]; bwall = PAPER_BQ_Q3["latency_s"]; bslot = None
        bcost = PAPER_BQ_Q3["cost_usd"]; bcalls = round(bcost / per_row) if per_row else n_total
        bres_vins = []
        profile["baseline"] = {
            "_status": "aborted",
            "score": {"f1_score": b_f1, "_source": "paper Table 4(e)"},
            "latency_breakdown": {"wall_s": bwall, "slot_ms": None, "_source": "paper"},
            "cost_breakdown": {"n_llm_calls": bcalls, "per_row_cost_usd": per_row,
                               "total_cost_usd": bcost, "_source": "paper"},
            "method": "sembench bigquery/Q3.sql verbatim — NOT EXECUTED",
            "sql": Q3_BASELINE_SQL.strip(),
        }
    else:
        print("\n=== Baseline (sembench Q3.sql verbatim, LIMIT 10 short-circuit) ===")
        bdf, bwall, bslot, _ = run_query(client, Q3_BASELINE_SQL)
        bres_vins = [str(v) for v in bdf["vin"]]
        b_p, b_r, b_f1, btp, b_n_sys, b_n_gt_sample = f1_against_gt_sample(bres_vins, gt_vins, TARGET)
        bcalls = max(round((bslot or 0) / 2500), len(bres_vins))
        bcalls = min(bcalls, n_total)
        bcost = per_row * bcalls
        print(f"  returned {len(bres_vins)} vins; sample TP={btp} (P={b_p:.4f} R={b_r:.4f} F1={b_f1:.4f})")
        print(f"  wall={bwall:.2f}s, slot_ms={bslot}, n_calls~{bcalls} (slot/2500), cost=${bcost:.6f}")
        profile["baseline"] = {
            "method": "sembench bigquery/Q3.sql verbatim on cars ⨝ car_mm (Manual + LIMIT 10)",
            "sql": Q3_BASELINE_SQL.strip(),
            "result_vins": bres_vins,
            "score": {"precision": b_p, "recall": b_r, "f1_score": b_f1},
            "latency_breakdown": {"wall_s": bwall, "slot_ms": bslot},
            "cost_breakdown": {
                "n_llm_calls": bcalls,
                "n_llm_calls_method": "max(round(slot_ms/2500), |returned|), capped at scope",
                "per_row_cost_usd": per_row,
                "total_cost_usd": bcost,
            },
        }

    profile["cascade"] = {
        "method": (
            f"F+L canonical via dase_cascade: Cascade(MarginSignal, TopKBand({K_CANDIDATES}), "
            f"AiIfVerifier).run() — IN(K) AI.IF + LIMIT {TARGET}"
        ),
        "stage1_ctas": {"latency_breakdown": {"wall_s": 0.0, "slot_ms": 0}, "cost_usd": 0.0,
                        "_note": "no staging; IN-list passed directly"},
        "stage2_run": {
            "sql": cres.verifier_result.sql,
            "result_vins": cascade_vins,
            "latency_breakdown": {"wall_s": cwall, "slot_ms": cslot},
            "cost_breakdown": {
                "n_llm_calls": s2_calls,
                "n_llm_calls_method": "max(round(slot/2500), |returned|), capped K",
                "per_row_cost_usd": per_row,
                "total_cost_usd": cascade_cost,
            },
        },
        "result_vins": cascade_vins,
        "score": {"precision": cp, "recall": cr, "f1_score": c_f1},
        "totals": {
            "wall_s": cascade_total_wall,
            "wall_breakdown_s": {
                "dase": profile["dase_breakdown_s"]["total"],
                "bq_stage1_ctas": 0.0,
                "bq_stage2_aiif": cwall,
            },
            "slot_ms_bq_total": cslot,
            "cost_usd": cascade_cost,
            "n_llm_calls": s2_calls,
        },
    }

    paper_n_calls = round(PAPER_BQ_Q3["cost_usd"] / per_row) if per_row else None
    profile["comparison"] = {
        "score": {"paper_BQ": PAPER_BQ_Q3["score_f1"], "paper_DASE_NN": PAPER_DASE_NN_Q3["score_f1"],
                  "ours_BQ": b_f1, "ours_cascade": c_f1,
                  "_baseline_source": "paper (aborted)" if SKIP_BASELINE else "ours"},
        "wall_s": {"paper_BQ": PAPER_BQ_Q3["latency_s"], "paper_DASE_NN": PAPER_DASE_NN_Q3["latency_s"],
                   "ours_BQ": bwall, "ours_cascade": cascade_total_wall},
        "slot_ms_bq": {"ours_BQ": bslot, "ours_cascade": cslot, "cascade_stage2": cslot},
        "cost_usd": {"paper_BQ": PAPER_BQ_Q3["cost_usd"], "paper_DASE_NN": PAPER_DASE_NN_Q3["cost_usd"],
                     "ours_BQ": bcost, "ours_cascade": cascade_cost},
        "n_llm_calls": {"paper_BQ": paper_n_calls, "paper_DASE_NN": 0,
                        "ours_BQ": bcalls, "ours_cascade": s2_calls},
    }

    write_profile(profile, PROFILE_PATH)

    print_summary(
        f"Cars Q3 (K={K_CANDIDATES})",
        columns=["paper BQ", "DASE+NN", "ours BQ", "ours cascade"],
        rows=[
            ("score (F1)", [PAPER_BQ_Q3["score_f1"], PAPER_DASE_NN_Q3["score_f1"], b_f1, c_f1], ".2f"),
            ("wall (s)",   [PAPER_BQ_Q3["latency_s"], PAPER_DASE_NN_Q3["latency_s"], bwall, cascade_total_wall], ".2f"),
            ("cost ($)",   [PAPER_BQ_Q3["cost_usd"], PAPER_DASE_NN_Q3["cost_usd"], bcost, cascade_cost], ".4f"),
            ("#LLM calls", [None, 0, bcalls, s2_calls], "d"),
        ],
    )


if __name__ == "__main__":
    main()
