"""
Cars Q2 — DASE-only (no BigQuery): electric cars whose audio shows a dead battery.

NL: SELECT DISTINCT car_id FROM cars JOIN audio WHERE fuel_type='Electric'
    AND audio entails dead battery.
GT: cars/ground_truth/Q2.csv (1 car_id: 98676).
Eval: precision/recall/F1 (set retrieval over car_id) via GenericEvaluator.

Aligns with paper §5.1: counterfactual anchors. Predict dead battery if
mean_sim(positive_prompts) > mean_sim(negative_prompts) on the audio caption
embeddings of the structurally pre-filtered (Electric) car set. Same prompts
as the Q2 cascade — `cars/scripts/q2_cascade.py`.
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
CARS_PARQUET          = os.path.join(CARS_DIR, "data", "cars.parquet")
AUDIO_PARQUET         = os.path.join(CARS_DIR, "data", "audio_cars.parquet")
GT_CSV                = os.path.join(CARS_DIR, "ground_truth", "Q2.csv")
EMBED_USAGE           = os.path.join(CARS_DIR, "cache", "embed_checkpoints", "embed_usage.json")
CAPTION_USAGE_DIR     = os.path.join(CARS_DIR, "cache", "embed_checkpoints")

POSITIVE = [
    "audio recording of a car with a dead battery",
    "engine fails to start due to dead battery",
    "clicking sound from a car ignition with no battery power",
]
NEGATIVE = [
    "audio of a car engine running normally",
    "audio of mechanical issues unrelated to battery",
    "engine sounds healthy with no electrical issue",
]


def main():
    cars_df = pd.read_parquet(CARS_PARQUET)
    audio_df = pd.read_parquet(AUDIO_PARQUET)
    gt_df = pd.read_csv(GT_CSV)

    electric_car_ids = set(cars_df.loc[cars_df["fuel_type"] == "Electric", "car_id"])
    audio_electric = audio_df[audio_df["car_id"].isin(electric_car_ids)].copy()

    print(f"Electric cars: {len(electric_car_ids)}")
    print(f"Audio recordings for electric cars: {len(audio_electric)}")
    print(f"Ground truth car_ids: {len(gt_df)}")

    if len(audio_electric) == 0:
        print("No audio recordings for electric cars — predicting empty set.")
        sys_df = pd.DataFrame({"car_id": pd.Series(dtype=int)})
    else:
        emb = np.array(audio_electric["embedding"].tolist(), dtype=np.float32)
        margins = MarginSignal(positive_prompts=POSITIVE, negative_prompts=NEGATIVE).compute(emb)
        pred_mask = margins > 0
        pred_car_ids = audio_electric.loc[pred_mask, "car_id"].unique()
        sys_df = pd.DataFrame({"car_id": pred_car_ids})

    score = GenericEvaluator.compute_accuracy_score(
        "precision", gt_df, sys_df, id_column="car_id"
    )
    p, r, f1 = score.precision, score.recall, score.f1_score
    print(f"Predicted dead-battery electric cars: {len(sys_df)}")
    print(f"[SemBench] Precision={p:.4f}  Recall={r:.4f}  F1={f1:.4f}")

    # ── Cost ─────────────────────────────────────────────────────────
    if os.path.exists(EMBED_USAGE):
        with open(EMBED_USAGE) as f:
            embed_usage = json.load(f)
        audio_embed_cost = embed_usage.get("audio_caption", {}).get("est_cost_usd", 0.0)
        audio_caption_cost = 0.0
        audio_caption_usage_file = os.path.join(CAPTION_USAGE_DIR, "audio_caption_usage.json")
        if os.path.exists(audio_caption_usage_file):
            with open(audio_caption_usage_file) as f:
                audio_caption_cost = json.load(f).get("est_caption_cost_usd", 0.0)
        total_cost = audio_embed_cost + audio_caption_cost
        print(f"\n=== Cost ===")
        print(f"  Columns used: audio_cars (caption embedding) + cars (fuel_type filter)")
        print(f"  Audio caption cost: ${audio_caption_cost:.4f}")
        print(f"  Audio embed cost:   ${audio_embed_cost:.4f}")
        print(f"  Total cost:         ${total_cost:.4f}")


if __name__ == "__main__":
    main()
