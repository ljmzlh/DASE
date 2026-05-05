"""
Cars Q5 — DASE-only (no BigQuery): COUNT automatic cars damaged in BOTH
images and audio.

NL: SELECT COUNT(*) FROM cars WHERE transmission='Automatic' AND damaged
    according to image AND damaged according to audio.
GT: cars/ground_truth/Q5.csv (Automatic, count=5).
Eval: _aggregation_single (relative_error → score = 1/(1+rel_err)).

Aligns with paper §5.1: counterfactual anchors (per modality). Two
MarginSignal predictors (image + audio) on automatic-car rows; intersect
predicted-damaged car_ids; return COUNT. Same prompts as the Q5 cascade —
`cars/scripts/q5_cascade.py`.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import MarginSignal, relative_error_score
from evaluator import _aggregation_single

CARS_DIR              = os.path.abspath(os.path.join(_HERE, ".."))
CARS_PARQUET          = os.path.join(CARS_DIR, "data", "cars.parquet")
IMAGE_PARQUET         = os.path.join(CARS_DIR, "data", "image_cars.parquet")
AUDIO_PARQUET         = os.path.join(CARS_DIR, "data", "audio_cars.parquet")
GT_CSV                = os.path.join(CARS_DIR, "ground_truth", "Q5.csv")
EMBED_USAGE           = os.path.join(CARS_DIR, "cache", "embed_checkpoints", "embed_usage.json")
CAPTION_USAGE_DIR     = os.path.join(CARS_DIR, "cache", "embed_checkpoints")

POS_AUDIO = [
    "audio recording of a damaged or malfunctioning car",
    "engine, brake, or mechanical fault sounds",
    "audio of a car with abnormal mechanical problems",
]
NEG_AUDIO = [
    "audio of a healthy car running normally",
    "engine sounds with no fault",
    "normal car operation audio",
]
POS_IMAGE = [
    "an image of a damaged or wrecked car",
    "a car with dents, scratches, or broken parts",
    "a vehicle showing signs of accident damage",
]
NEG_IMAGE = [
    "an image of an undamaged intact car",
    "a clean car in good condition",
    "a vehicle with no visible damage",
]


def predict_damaged_cars(df, pos_prompts, neg_prompts):
    if len(df) == 0:
        return set()
    emb = np.array(df["embedding"].tolist(), dtype=np.float32)
    margins = MarginSignal(positive_prompts=pos_prompts, negative_prompts=neg_prompts).compute(emb)
    return set(df.loc[margins > 0, "car_id"].unique())


def main():
    cars_df = pd.read_parquet(CARS_PARQUET)
    image_df = pd.read_parquet(IMAGE_PARQUET)
    audio_df = pd.read_parquet(AUDIO_PARQUET)
    gt_df = pd.read_csv(GT_CSV)

    automatic_car_ids = set(cars_df.loc[cars_df["transmission"] == "Automatic", "car_id"])
    image_auto = image_df[image_df["car_id"].isin(automatic_car_ids)].copy()
    audio_auto = audio_df[audio_df["car_id"].isin(automatic_car_ids)].copy()

    print(f"Automatic cars: {len(automatic_car_ids)}")
    print(f"Images for automatic cars: {len(image_auto)}")
    print(f"Audio for automatic cars: {len(audio_auto)}")
    gt_count = int(gt_df.iloc[0]["count"])
    print(f"Ground truth count: {gt_count}")

    image_damaged_ids = predict_damaged_cars(image_auto, POS_IMAGE, NEG_IMAGE)
    audio_damaged_ids = predict_damaged_cars(audio_auto, POS_AUDIO, NEG_AUDIO)

    result_car_ids = automatic_car_ids & image_damaged_ids & audio_damaged_ids
    sys_df = pd.DataFrame({"count": [len(result_car_ids)]})

    result = _aggregation_single(sys_df, gt_df)
    score = relative_error_score(len(result_car_ids), gt_count)

    print(f"Predicted image-damaged automatic cars: {len(image_damaged_ids)}")
    print(f"Predicted audio-damaged automatic cars: {len(audio_damaged_ids)}")
    print(f"Predicted intersection count: {len(result_car_ids)}")
    print(f"Ground truth count: {gt_count}")
    print(
        f"[SemBench] RelError={result.relative_error:.4f}  "
        f"AbsError={result.absolute_error:.4f}  Score={score:.4f}"
    )

    # ── Cost ─────────────────────────────────────────────────────────
    if os.path.exists(EMBED_USAGE):
        with open(EMBED_USAGE) as f:
            embed_usage = json.load(f)
        image_embed_cost = embed_usage.get("image_caption", {}).get("est_cost_usd", 0.0)
        audio_embed_cost = embed_usage.get("audio_caption", {}).get("est_cost_usd", 0.0)
        image_caption_cost = audio_caption_cost = 0.0
        for fname, var in [("image_caption_usage.json", "image"), ("audio_caption_usage.json", "audio")]:
            p = os.path.join(CAPTION_USAGE_DIR, fname)
            if os.path.exists(p):
                with open(p) as f:
                    cost = json.load(f).get("est_caption_cost_usd", 0.0)
                if var == "image":
                    image_caption_cost = cost
                else:
                    audio_caption_cost = cost
        total_cost = image_embed_cost + audio_embed_cost + image_caption_cost + audio_caption_cost
        print(f"\n=== Cost ===")
        print(
            "  Columns used: cars (transmission filter) + "
            "image_cars (caption embedding) + audio_cars (caption embedding)"
        )
        print(f"  Image caption cost: ${image_caption_cost:.4f}")
        print(f"  Image embed cost:   ${image_embed_cost:.4f}")
        print(f"  Audio caption cost: ${audio_caption_cost:.4f}")
        print(f"  Audio embed cost:   ${audio_embed_cost:.4f}")
        print(f"  Total cost:         ${total_cost:.4f}")


if __name__ == "__main__":
    main()
