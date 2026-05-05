"""
Cars Q8 — DASE-only (no BigQuery): top-100 cars whose images show BOTH
puncture and paint scratches.

NL: SELECT car_id FROM images WHERE both puncture and paint scratches LIMIT 100.
GT: cars/ground_truth/Q8.csv (163 car_ids).
Eval: _retrieval_limit (precision/recall/F1, limit=100).

Aligns with paper §5.1: counterfactual anchors. Per-image margin via
MarginSignal; per-car aggregate = MAX margin across that car's images
(any one strong image is enough); take top-100 cars by margin descending.
Same prompts as the Q8 cascade — `cars/scripts/q8_cascade.py`.
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
IMAGE_PARQUET         = os.path.join(CARS_DIR, "data", "image_cars.parquet")
GT_CSV                = os.path.join(CARS_DIR, "ground_truth", "Q8.csv")
EMBED_USAGE           = os.path.join(CARS_DIR, "cache", "embed_checkpoints", "embed_usage.json")
CAPTION_USAGE_DIR     = os.path.join(CARS_DIR, "cache", "embed_checkpoints")

POSITIVE = [
    "a car with both paint scratches and a puncture",
    "vehicle showing surface scratches and a hole or puncture",
    "image with paint scratches and a puncture mark",
]
NEGATIVE = [
    "an undamaged car with no scratches or punctures",
    "a car with damage but no scratches or punctures",
    "a vehicle without surface scratches or punctures",
]


def main():
    image_df = pd.read_parquet(IMAGE_PARQUET)
    gt_df = pd.read_csv(GT_CSV)

    print(f"Total images: {len(image_df)}")
    print(f"Ground truth car_ids: {len(gt_df)}")

    emb = np.array(image_df["embedding"].tolist(), dtype=np.float32)
    margins = MarginSignal(positive_prompts=POSITIVE, negative_prompts=NEGATIVE).compute(emb)

    # Per car_id: take MAX margin across that car's images (any single image
    # strongly matching is enough to flag the car).
    image_df_with_margin = image_df[["car_id"]].copy()
    image_df_with_margin["margin"] = margins
    car_scores = image_df_with_margin.groupby("car_id")["margin"].max().reset_index()

    ranked = car_scores.sort_values("margin", ascending=False)
    sys_df = ranked.head(100)[["car_id"]].reset_index(drop=True)

    result = _retrieval_limit(sys_df, gt_df, limit=100)
    p, r, f1 = result.precision, result.recall, result.f1_score

    print(f"Cars ranked by confidence: {len(car_scores)}")
    print(f"Returned car_ids (top 100 by margin): {len(sys_df)}")
    print(f"[SemBench] Precision={p:.4f}  Recall={r:.4f}  F1={f1:.4f}")

    # ── Cost ─────────────────────────────────────────────────────────
    if os.path.exists(EMBED_USAGE):
        with open(EMBED_USAGE) as f:
            embed_usage = json.load(f)
        image_embed = embed_usage.get("image_caption", {}).get("est_cost_usd", 0.0)
        image_caption = 0.0
        cap_file = os.path.join(CAPTION_USAGE_DIR, "image_caption_usage.json")
        if os.path.exists(cap_file):
            with open(cap_file) as f:
                image_caption = json.load(f).get("est_caption_cost_usd", 0.0)
        total_cost = image_embed + image_caption
        print(f"\n=== Cost ===")
        print(f"  Columns used: image_cars (caption embedding)")
        print(f"  Image caption cost: ${image_caption:.4f}")
        print(f"  Image embed cost:   ${image_embed:.4f}")
        print(f"  Total cost:         ${total_cost:.4f}")


if __name__ == "__main__":
    main()
