"""
Cars Q9 — DASE-only (no BigQuery): cars torn (image) AND bad ignition (audio).

NL: SELECT DISTINCT car_id FROM cars JOIN image JOIN audio
    WHERE car is torn AND ignition is bad.
GT: cars/ground_truth/Q9.csv (0 car_ids — empty in sf_19672 sample).
Eval: precision/recall/F1 (set retrieval over car_id) via GenericEvaluator.
      If GT is empty, evaluator returns F1=1.0 iff prediction is also empty.

Aligns with paper §5.1: counterfactual anchors per modality. Two MarginSignal
predictors (image: torn / audio: bad ignition); intersect predicted-positive
car_ids. Same prompts as the Q9 cascade — `cars/scripts/q9_cascade.py`.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import MarginSignal
from generic_evaluator import GenericEvaluator

CARS_DIR              = os.path.abspath(os.path.join(_HERE, ".."))
IMAGE_PARQUET         = os.path.join(CARS_DIR, "data", "image_cars.parquet")
AUDIO_PARQUET         = os.path.join(CARS_DIR, "data", "audio_cars.parquet")
GT_CSV                = os.path.join(CARS_DIR, "ground_truth", "Q9.csv")
EMBED_USAGE           = os.path.join(CARS_DIR, "cache", "embed_checkpoints", "embed_usage.json")
CAPTION_USAGE_DIR     = os.path.join(CARS_DIR, "cache", "embed_checkpoints")

POS_IMAGE = [
    "a torn car",
    "vehicle with torn or ripped material",
    "image of a car with tear damage",
]
NEG_IMAGE = [
    "an undamaged car",
    "a vehicle without tears or rips",
    "intact car body without damage",
]
POS_AUDIO = [
    "audio of a car with bad ignition",
    "ignition starting problem in a car",
    "engine fails to ignite",
]
NEG_AUDIO = [
    "audio of normal car ignition",
    "engine starts normally",
    "healthy engine startup",
]


def predict_per_row(df, pos_prompts, neg_prompts):
    if len(df) == 0:
        return np.array([], dtype=bool)
    emb = np.array(df["embedding"].tolist(), dtype=np.float32)
    margins = MarginSignal(positive_prompts=pos_prompts, negative_prompts=neg_prompts).compute(emb)
    return margins > 0


def main():
    image_df = pd.read_parquet(IMAGE_PARQUET)
    audio_df = pd.read_parquet(AUDIO_PARQUET)
    gt_df = pd.read_csv(GT_CSV)

    print(f"Images: {len(image_df)}, Audio: {len(audio_df)}")
    print(f"Ground truth car_ids: {len(gt_df)}")

    image_pred = predict_per_row(image_df, POS_IMAGE, NEG_IMAGE)  # True = torn
    audio_pred = predict_per_row(audio_df, POS_AUDIO, NEG_AUDIO)  # True = bad ignition

    torn_cars = set(image_df.loc[image_pred, "car_id"].unique())
    bad_ignition_cars = set(audio_df.loc[audio_pred, "car_id"].unique())

    intersect_cars = torn_cars & bad_ignition_cars
    sys_df = pd.DataFrame({"car_id": sorted(intersect_cars)})

    score = GenericEvaluator.compute_accuracy_score(
        "precision", gt_df, sys_df, id_column="car_id"
    )
    p, r, f1 = score.precision, score.recall, score.f1_score

    print(f"Image (torn): {len(torn_cars)}")
    print(f"Audio (bad ignition): {len(bad_ignition_cars)}")
    print(f"Intersection: {len(intersect_cars)}")
    print(f"[SemBench] Precision={p:.4f}  Recall={r:.4f}  F1={f1:.4f}")

    # ── Cost ─────────────────────────────────────────────────────────
    if os.path.exists(EMBED_USAGE):
        with open(EMBED_USAGE) as f:
            embed_usage = json.load(f)
        image_embed = embed_usage.get("image_caption", {}).get("est_cost_usd", 0.0)
        audio_embed = embed_usage.get("audio_caption", {}).get("est_cost_usd", 0.0)
        image_caption = audio_caption = 0.0
        for fname in ("image_caption_usage.json", "audio_caption_usage.json"):
            p_ = os.path.join(CAPTION_USAGE_DIR, fname)
            if os.path.exists(p_):
                with open(p_) as f:
                    cost = json.load(f).get("est_caption_cost_usd", 0.0)
                if "image" in fname:
                    image_caption = cost
                else:
                    audio_caption = cost
        total_cost = image_embed + audio_embed + image_caption + audio_caption
        print(f"\n=== Cost ===")
        print(f"  Columns used: image_cars + audio_cars")
        print(f"  Image caption: ${image_caption:.4f}")
        print(f"  Image embed:   ${image_embed:.4f}")
        print(f"  Audio caption: ${audio_caption:.4f}")
        print(f"  Audio embed:   ${audio_embed:.4f}")
        print(f"  Total cost:    ${total_cost:.4f}")


if __name__ == "__main__":
    main()
