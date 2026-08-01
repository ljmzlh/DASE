#!/usr/bin/env -S python -u
"""
Cars Q8 cascade (v2) — F+L (top-K margin → BQ AI.IF + LIMIT 100) via dase_cascade.

NL: Find a hundred cars with punctures and paint scratches on images.
GT: 163 cars whose damage_status contains BOTH 'paint_scratches' AND 'puncture'.
Eval: retrieval_limit (≤100 car_ids; F1 over 100-row gt sample).

Refactored to use dase_cascade. Operator (paper Table 3): F + L.
Cascade(MarginSignal + TopKBand(200) + AiIfVerifier) — single BQ AI.IF +
LIMIT 100 short-circuits at TARGET. Verifier returns car_ids.
"""
import json
import math
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
    build_profile, write_profile,
)

CARS_DIR = os.path.abspath(os.path.join(_HERE, ".."))
IMAGE_PARQUET = os.path.join(CARS_DIR, "data", "image_cars.parquet")
GT_CSV = os.path.join(CARS_DIR, "ground_truth", "Q8.csv")
PROFILE_PATH = os.path.join(CARS_DIR, "outputs", "Q8.json")

PROJECT = os.environ.get("GCP_PROJECT", "")
BUCKET = f"{PROJECT}-cars_dataset"
GCS_FOLDER = "car_images"
DATASET = "cars_dataset"

PROMPT = "You are given an image of a vehicle or its parts. Return true if car has both, puncture and paint scratches."

POSITIVE_PROMPTS = [
    "a car with both paint scratches and a puncture",
    "vehicle showing surface scratches and a hole or puncture",
    "image with paint scratches and a puncture mark",
]
NEGATIVE_PROMPTS = [
    "an undamaged car with no scratches or punctures",
    "a car with damage but no scratches or punctures",
    "a vehicle without surface scratches or punctures",
]

TARGET = 100
K_CANDIDATES = 200


def trunc2(x):
    return f"{math.floor(x * 100) / 100:.2f}"




def make_q8_verifier():
    """Single AI.IF: image_id IN (top-K) + LIMIT TARGET."""
    def verify_sql_template(image_ids):
        id_list = ",".join(str(int(i)) for i in image_ids)
        return f"""
        SELECT p.car_id AS id
        FROM {DATASET}.car_mm AS x
        JOIN {DATASET}.cars AS p ON p.car_id = x.car_id
        WHERE x.image_id IN ({id_list})
          AND AI.IF(
            ('{PROMPT}', x.image),
            connection_id => 'us.connection',
            endpoint => 'gemini-2.5-flash')
        LIMIT {TARGET}
        """
    return AiIfVerifier(verify_sql_template=verify_sql_template,
                        id_column="id", coerce_id=int)


def f1_against_gt_sample(sys_ids, gt_ids, target=100, seed=42):
    sys_set = set(sys_ids); gt_set = set(gt_ids)
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
        scenario="cars", query_id=8, scale_factor=19672,
        prompt=PROMPT, params={"K_candidates": K_CANDIDATES, "target": TARGET},
        cascade_form=(
            f"F+L canonical via dase_cascade: top-{K_CANDIDATES} by margin → "
            f"BQ AI.IF on image_id IN({K_CANDIDATES}) + LIMIT {TARGET}"
        ),
        extra={
            "operator": "F+L",
            "structural_filter": "",
            "dase_prompts": {"positive": POSITIVE_PROMPTS, "negative": NEGATIVE_PROMPTS},
        },
    )

    print("Loading image_cars + GT...")
    t = time.time()
    img = pd.read_parquet(IMAGE_PARQUET).reset_index(drop=True)
    n_total = len(img)
    gt_cars = set(int(x) for x in pd.read_csv(GT_CSV)["car_id"])
    n_gt = len(gt_cars)
    t_load = time.time() - t

    print(f"  scope: {n_total} images; GT positive cars: {n_gt}")
    profile["data"] = {"n_images_total": n_total, "n_gt_positive_cars": n_gt}

    image_emb = np.stack(img["embedding"].tolist()).astype(np.float32)
    image_ids = img["image_id"].astype(int).tolist()

    client = bq_client(PROJECT)

    print("\n=== Per-row cost calibration (k=5 multimodal image) ===")
    sample_uris = [
        f"gs://{BUCKET}/{GCS_FOLDER}/{os.path.basename(img.iloc[i]['image_path'])}"
        for i in range(min(5, n_total))
    ]
    cal = per_row_cost(
        client, PROMPT,
        sample_uris=sample_uris,
        ext_table=f"{DATASET}.cars_images",
        method_label="AI.GENERATE_BOOL multimodal (image, Q8 prompt) + thinking_budget=0",
        k=5,
    )
    per_row = cal.per_row_cost_usd
    print(f"  per_row=${per_row:.6f}, sample_cost=${cal.sample_cost_usd:.6f}, elapsed={cal.elapsed_s:.1f}s")
    profile["calibration"] = cal.to_dict()

    # ── Cascade ──
    cascade = Cascade(
        embeddings=image_emb,
        ids=image_ids,
        signal=MarginSignal(positive_prompts=POSITIVE_PROMPTS, negative_prompts=NEGATIVE_PROMPTS),
        band=TopKBand(k=K_CANDIDATES),
        verifier=make_q8_verifier(),
    )
    print(f"\n=== Cascade (MarginSignal → TopKBand({K_CANDIDATES}) → AiIfVerifier+LIMIT {TARGET}) ===")
    cres = cascade.run(client, per_row)

    top_K_image_ids = list(cres.uncertain_ids)
    img_lookup = img.set_index("image_id")
    top_K_car_ids = [int(img_lookup.loc[iid]["car_id"]) for iid in top_K_image_ids]
    top_K_margins = [float(cres.scores[img.index[img["image_id"] == iid][0]]) for iid in top_K_image_ids]
    n_top_in_gt = sum(1 for c in top_K_car_ids if c in gt_cars)

    cascade_cars = sorted(cres.verifier_result.positive_ids)
    cwall = cres.total_wall_s
    cslot = cres.verifier_result.slot_ms
    s2_calls = max(round(cslot / 2500), len(cascade_cars))
    s2_calls = min(s2_calls, K_CANDIDATES)
    cascade_cost = per_row * s2_calls
    cp, cr, c_f1, ctp, c_n_sys, c_n_gt_sample = f1_against_gt_sample(cascade_cars, gt_cars, TARGET)
    print(f"  top-K={K_CANDIDATES} margins: min={min(top_K_margins):.4f}, max={max(top_K_margins):.4f}")
    print(f"  top-K cars ∩ GT = {n_top_in_gt}/{K_CANDIDATES}")
    print(f"  cascade returned {len(cascade_cars)} cars; sample TP={ctp} (P={cp:.4f} R={cr:.4f} F1={c_f1:.4f})")
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
        "image_ids": top_K_image_ids, "car_ids": top_K_car_ids,
        "margins": top_K_margins, "n_top_K_in_GT": n_top_in_gt,
    }

    profile["cascade"] = {
        "method": (
            f"F+L canonical via dase_cascade: Cascade(MarginSignal, TopKBand({K_CANDIDATES}), "
            f"AiIfVerifier).run() — IN(K) AI.IF + LIMIT {TARGET}"
        ),
        "stage1_ctas": {"latency_breakdown": {"wall_s": 0.0, "slot_ms": 0}, "cost_usd": 0.0,
                        "_note": "no staging; IN-list passed directly"},
        "stage2_run": {
            "sql": cres.verifier_result.sql, "result_car_ids": cascade_cars,
            "latency_breakdown": {"wall_s": cwall, "slot_ms": cslot},
            "cost_breakdown": {
                "n_llm_calls": s2_calls,
                "n_llm_calls_method": "max(round(slot/2500), |returned|), capped K",
                "per_row_cost_usd": per_row, "total_cost_usd": cascade_cost,
            },
        },
        "result_ids": cascade_cars,
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


    write_profile(profile, PROFILE_PATH)



if __name__ == "__main__":
    main()
