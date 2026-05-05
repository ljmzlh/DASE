"""
Cars Q7 — DASE-only (no BigQuery): cars dented (image), worn-out brakes
(audio), or electrical-system problems (text complaint).

NL: UNION of three predicates over three modalities.
GT: cars/ground_truth/Q7.csv (4034 car_ids).
Eval: precision/recall/F1 (set retrieval over car_id) via GenericEvaluator.

Aligns with paper §5.1: counterfactual anchors per modality. Three
MarginSignal predictors then UNION the predicted-positive car_ids. Same
prompts as the Q7 cascade — `cars/scripts/q7_cascade.py`.
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
TEXT_PARQUET          = os.path.join(CARS_DIR, "data", "text_complaints.parquet")
GT_CSV                = os.path.join(CARS_DIR, "ground_truth", "Q7.csv")
EMBED_USAGE           = os.path.join(CARS_DIR, "cache", "embed_checkpoints", "embed_usage.json")
CAPTION_USAGE_DIR     = os.path.join(CARS_DIR, "cache", "embed_checkpoints")

POS_AUDIO = [
    "audio of worn out brakes",
    "squealing or grinding brake noise from a car",
    "audio of degraded or failing brake pads",
]
NEG_AUDIO = [
    "audio of healthy brakes with no noise",
    "engine sounds unrelated to brakes",
    "normal car operation audio",
]
POS_IMAGE = [
    "an image of a dented car",
    "a vehicle with body dents or panel damage",
    "a car with visible dents on its body",
]
NEG_IMAGE = [
    "an image of an undamaged car without dents",
    "a clean car with smooth body panels",
    "a vehicle in pristine condition",
]
POS_TEXT_ELEC = [
    "complaint about electrical system problem",
    "issue with electrical components or wiring",
    "complaint about car electrical malfunction",
]
NEG_TEXT_ELEC = [
    "complaint about mechanical issue not electrical",
    "engine, brake, or non-electrical problem",
    "issues unrelated to car electrical system",
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
    text_df = pd.read_parquet(TEXT_PARQUET)
    gt_df = pd.read_csv(GT_CSV)

    print(f"Images: {len(image_df)}, Audio: {len(audio_df)}, Text: {len(text_df)}")
    print(f"Ground truth car_ids: {len(gt_df)}")

    image_pred = predict_per_row(image_df, POS_IMAGE, NEG_IMAGE)  # True = dented
    audio_pred = predict_per_row(audio_df, POS_AUDIO, NEG_AUDIO)  # True = worn brakes
    text_pred = predict_per_row(text_df, POS_TEXT_ELEC, NEG_TEXT_ELEC)  # True = electrical

    image_cars = set(image_df.loc[image_pred, "car_id"].unique())
    audio_cars = set(audio_df.loc[audio_pred, "car_id"].unique())
    text_cars = set(text_df.loc[text_pred, "car_id"].unique())

    union_cars = image_cars | audio_cars | text_cars
    sys_df = pd.DataFrame({"car_id": sorted(union_cars)})

    score = GenericEvaluator.compute_accuracy_score(
        "precision", gt_df, sys_df, id_column="car_id"
    )
    p, r, f1 = score.precision, score.recall, score.f1_score

    print(f"Image (dented): {len(image_cars)}")
    print(f"Audio (worn brakes): {len(audio_cars)}")
    print(f"Text  (electrical): {len(text_cars)}")
    print(f"Union: {len(union_cars)}")
    print(f"[SemBench] Precision={p:.4f}  Recall={r:.4f}  F1={f1:.4f}")

    # ── Cost ─────────────────────────────────────────────────────────
    if os.path.exists(EMBED_USAGE):
        with open(EMBED_USAGE) as f:
            embed_usage = json.load(f)
        text_embed = embed_usage.get("text_complaint", {}).get("est_cost_usd", 0.0)
        image_embed = embed_usage.get("image_caption", {}).get("est_cost_usd", 0.0)
        audio_embed = embed_usage.get("audio_caption", {}).get("est_cost_usd", 0.0)
        image_caption = audio_caption = 0.0
        for fname in ("image_caption_usage.json", "audio_caption_usage.json"):
            p = os.path.join(CAPTION_USAGE_DIR, fname)
            if os.path.exists(p):
                with open(p) as f:
                    cost = json.load(f).get("est_caption_cost_usd", 0.0)
                if "image" in fname:
                    image_caption = cost
                else:
                    audio_caption = cost
        total_cost = text_embed + image_embed + audio_embed + image_caption + audio_caption
        print(f"\n=== Cost ===")
        print(f"  Columns used: text_complaints + image_cars + audio_cars")
        print(f"  Text embed:    ${text_embed:.4f}")
        print(f"  Image caption: ${image_caption:.4f}")
        print(f"  Image embed:   ${image_embed:.4f}")
        print(f"  Audio caption: ${audio_caption:.4f}")
        print(f"  Audio embed:   ${audio_embed:.4f}")
        print(f"  Total cost:    ${total_cost:.4f}")


if __name__ == "__main__":
    main()
