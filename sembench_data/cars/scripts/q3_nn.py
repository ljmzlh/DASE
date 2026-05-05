"""
Cars Q3 — DASE-only (no BigQuery): top-10 manual-transmission cars whose
images show no damage.

NL: SELECT vin FROM cars WHERE transmission='Manual' AND not damaged LIMIT 10.
GT: cars/ground_truth/Q3.csv (670 vins).
Eval: _retrieval_limit (precision/recall/F1, limit=10).

Aligns with paper §5.1: counterfactual anchors. Per-image margin via
MarginSignal; per-car aggregate = MIN margin across that car's images
(a car is "not damaged" only if all of its images look not-damaged); then
take top-10 cars by margin descending. Same prompts as the Q3 cascade —
`cars/scripts/q3_cascade.py`.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import MarginSignal
from evaluator import _retrieval_limit

CARS_DIR              = os.path.abspath(os.path.join(_HERE, ".."))
CARS_PARQUET          = os.path.join(CARS_DIR, "data", "cars.parquet")
IMAGE_PARQUET         = os.path.join(CARS_DIR, "data", "image_cars.parquet")
GT_CSV                = os.path.join(CARS_DIR, "ground_truth", "Q3.csv")
EMBED_USAGE           = os.path.join(CARS_DIR, "cache", "embed_checkpoints", "embed_usage.json")
CAPTION_USAGE_DIR     = os.path.join(CARS_DIR, "cache", "embed_checkpoints")

POSITIVE = [
    "an undamaged car in pristine condition",
    "a clean intact vehicle without dents or scratches",
    "a car in good condition with no visible damage",
]
NEGATIVE = [
    "a damaged or wrecked car with visible damage",
    "a car with dents, scratches, or broken parts",
    "a vehicle showing signs of accident damage",
]


def main():
    cars_df = pd.read_parquet(CARS_PARQUET)
    image_df = pd.read_parquet(IMAGE_PARQUET)
    gt_df = pd.read_csv(GT_CSV)

    manual_cars = cars_df[cars_df["transmission"] == "Manual"]
    manual_car_ids = set(manual_cars["car_id"])
    image_manual = image_df[image_df["car_id"].isin(manual_car_ids)].copy()

    print(f"Manual cars: {len(manual_cars)}")
    print(f"Images for manual cars: {len(image_manual)}")
    print(f"Ground truth VINs: {len(gt_df)}")

    emb = np.array(image_manual["embedding"].tolist(), dtype=np.float32)
    margins = MarginSignal(positive_prompts=POSITIVE, negative_prompts=NEGATIVE).compute(emb)

    # Per car_id: take MIN margin across that car's images.
    # A car is "not damaged" only if ALL its images look not-damaged; ranking
    # by min-margin descending puts cars whose worst image is still confidently
    # not-damaged at the top.
    image_manual_with_margin = image_manual[["car_id"]].copy()
    image_manual_with_margin["margin"] = margins
    car_scores = image_manual_with_margin.groupby("car_id")["margin"].min().reset_index()

    ranked = car_scores.sort_values("margin", ascending=False)
    ranked_with_vin = ranked.merge(manual_cars[["car_id", "vin"]], on="car_id", how="left")
    sys_df = ranked_with_vin.head(10)[["vin"]].reset_index(drop=True)

    result = _retrieval_limit(sys_df, gt_df, limit=10)
    p, r, f1 = result.precision, result.recall, result.f1_score

    print(f"Manual cars ranked: {len(car_scores)}")
    print(f"Returned VINs (top 10 by margin): {len(sys_df)}")
    print(f"[SemBench] Precision={p:.4f}  Recall={r:.4f}  F1={f1:.4f}")

    # ── Cost ─────────────────────────────────────────────────────────
    if os.path.exists(EMBED_USAGE):
        with open(EMBED_USAGE) as f:
            embed_usage = json.load(f)
        image_embed_cost = embed_usage.get("image_caption", {}).get("est_cost_usd", 0.0)
        image_caption_cost = 0.0
        cap_file = os.path.join(CAPTION_USAGE_DIR, "image_caption_usage.json")
        if os.path.exists(cap_file):
            with open(cap_file) as f:
                image_caption_cost = json.load(f).get("est_caption_cost_usd", 0.0)
        total_cost = image_embed_cost + image_caption_cost
        print(f"\n=== Cost ===")
        print(f"  Columns used: image_cars (caption embedding) + cars (transmission filter)")
        print(f"  Image caption cost: ${image_caption_cost:.4f}")
        print(f"  Image embed cost:   ${image_embed_cost:.4f}")
        print(f"  Total cost:         ${total_cost:.4f}")


if __name__ == "__main__":
    main()
