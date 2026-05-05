"""
Cars Q6 — DASE-only (no BigQuery): cars damaged according to one modality
but not another (text checks fire).

NL: per car, find evidence of "damaged" in one modality and "no_damage" in
    another. For text complaints, "damaged" := car was on fire.
GT: cars/ground_truth/Q6.csv (1906 unique car_ids whose denormalized status
    contains BOTH "damaged" AND "no_damage").
Eval: precision/recall/F1 (set retrieval over car_id) via GenericEvaluator.

Equivalent characterization (proven): for each car_id, let
  D = {modalities m ∈ {image, audio, text} : at least one row predicts "damaged"}
  N = {modalities m : at least one row predicts "no_damage"}
Car ∈ result iff D ≠ ∅ AND N ≠ ∅ AND |D ∪ N| ≥ 2.

Aligns with paper §5.1: counterfactual anchors per modality. Same prompts
as the Q6 cascade — `cars/scripts/q6_cascade.py` (image / audio / text-fire).
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
GT_CSV                = os.path.join(CARS_DIR, "ground_truth", "Q6.csv")
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
POS_TEXT_FIRE = [
    "complaint about car on fire or burned",
    "vehicle caught fire or had a fire incident",
    "fire damage to the car",
]
NEG_TEXT_FIRE = [
    "complaint about mechanical failure not involving fire",
    "non-fire damage or issue with the car",
    "issues unrelated to vehicle fire",
]


def predict_per_row(df, pos_prompts, neg_prompts):
    """Return boolean array: True = positive class predicted."""
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
    gt_car_ids = set(gt_df["car_id"].unique())

    print(f"Images: {len(image_df)}, Audio: {len(audio_df)}, Text: {len(text_df)}")
    print(f"Ground truth unique car_ids: {len(gt_car_ids)}")

    # Per-row predictions via MarginSignal (counterfactual anchors).
    image_pred = predict_per_row(image_df, POS_IMAGE, NEG_IMAGE)  # True = damaged
    audio_pred = predict_per_row(audio_df, POS_AUDIO, NEG_AUDIO)  # True = damaged
    text_pred = predict_per_row(text_df, POS_TEXT_FIRE, NEG_TEXT_FIRE)  # True = fire

    # Per car_id, accumulate which modalities have any damaged / any no_damage.
    D = {}
    N = {}

    def update(car_ids, pred_mask, modality):
        for cid, pred in zip(car_ids, pred_mask):
            if pred:
                D.setdefault(cid, set()).add(modality)
            else:
                N.setdefault(cid, set()).add(modality)

    update(image_df["car_id"].values, image_pred, "image")
    update(audio_df["car_id"].values, audio_pred, "audio")
    update(text_df["car_id"].values, text_pred, "text")

    pred_car_ids = set()
    for cid in set(D.keys()) | set(N.keys()):
        d = D.get(cid, set())
        n = N.get(cid, set())
        if d and n and len(d | n) >= 2:
            pred_car_ids.add(cid)

    sys_df = pd.DataFrame({"car_id": sorted(pred_car_ids)})

    score = GenericEvaluator.compute_accuracy_score(
        "precision", gt_df.drop_duplicates(subset=["car_id"]), sys_df, id_column="car_id"
    )
    p, r, f1 = score.precision, score.recall, score.f1_score

    print(f"Image positive (damaged) preds: {int(image_pred.sum())} / {len(image_pred)}")
    print(f"Audio positive (damaged) preds: {int(audio_pred.sum())} / {len(audio_pred)}")
    print(f"Text  positive (fire) preds:    {int(text_pred.sum())} / {len(text_pred)}")
    print(f"Predicted cars (D≠∅, N≠∅, |D∪N|≥2): {len(sys_df)}")
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
