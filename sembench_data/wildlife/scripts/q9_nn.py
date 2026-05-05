"""
Wildlife Q9 — DASE-only (no BigQuery): cities with both monkey IMAGES and monkey AUDIO.

NL: image_monkey_cities ∩ audio_monkey_cities.
Eval: set retrieval F1 (matches original).

Aligns with paper §5.1: counterfactual anchors per modality. Anchors verbatim
from q9_cascade.py (image + audio). Two MarginSignal passes; client-side
intersect over City.
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import MarginSignal

WILDLIFE_DIR = os.path.abspath(os.path.join(_HERE, ".."))
IMAGE_CSV    = os.path.join(WILDLIFE_DIR, "cache", "image_data.csv")
AUDIO_CSV    = os.path.join(WILDLIFE_DIR, "cache", "audio_data.csv")
IMG_EMB_PATH = os.path.join(WILDLIFE_DIR, "data", "image_embeddings.npz")
AUD_EMB_PATH = os.path.join(WILDLIFE_DIR, "data", "audio_embeddings.npz")

IMG_POSITIVE = [
    "a photograph of a monkey",
    "a wildlife camera trap image showing a monkey",
    "a monkey captured in the photo",
]
IMG_NEGATIVE = [
    "a photograph that does not contain a monkey",
    "a wildlife camera trap image of a non-monkey animal",
    "an animal photo without any monkey",
]
AUD_POSITIVE = [
    "a sound recording of a monkey",
    "audio of monkey vocalizations or calls",
    "monkey howling or chittering sound clip",
]
AUD_NEGATIVE = [
    "a sound recording of an animal that is not a monkey",
    "audio of a non-monkey animal vocalization",
    "animal sound clip without any monkey",
]


def main():
    df_i = pd.read_csv(IMAGE_CSV)
    df_a = pd.read_csv(AUDIO_CSV)
    img_emb = np.load(IMG_EMB_PATH)["caption_emb"]
    aud_emb = np.load(AUD_EMB_PATH)["caption_emb"]

    img_margins = MarginSignal(positive_prompts=IMG_POSITIVE, negative_prompts=IMG_NEGATIVE).compute(img_emb)
    aud_margins = MarginSignal(positive_prompts=AUD_POSITIVE, negative_prompts=AUD_NEGATIVE).compute(aud_emb)

    img_cities = set(df_i.loc[img_margins > 0, "City"])
    aud_cities = set(df_a.loc[aud_margins > 0, "City"])
    pred = sorted(img_cities & aud_cities)

    gt_img = set(df_i.loc[df_i["Species"].str.contains("MONKEY", case=False, na=False), "City"])
    gt_aud = set(df_a.loc[df_a["Animal"] == "Monkey", "City"])
    gt = sorted(gt_img & gt_aud)

    pred_set, gt_set = set(pred), set(gt)
    tp = len(pred_set & gt_set)
    prec = tp / len(pred_set) if pred_set else (1.0 if not gt_set else 0.0)
    rec = tp / len(gt_set) if gt_set else (1.0 if not pred_set else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print(f"Predicted: {pred}")
    print(f"Ground truth: {gt}")
    print(f"[SemBench] F1={f1:.4f}")


if __name__ == "__main__":
    main()
