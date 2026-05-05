"""
Wildlife Q5 — DASE-only (no BigQuery): cities with elephant images OR elephant audio.

NL: image_elephant_cities ∪ audio_elephant_cities.
Eval: set retrieval F1 (matches original).

Aligns with paper §5.1: counterfactual anchors per modality. Anchors verbatim
from q5_cascade.py (image + audio). Two MarginSignal passes; client-side union.
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
    "a photograph of an elephant",
    "a wildlife camera trap image showing an elephant",
    "an elephant captured in the photo",
]
IMG_NEGATIVE = [
    "a photograph that does not contain an elephant",
    "a wildlife camera trap image of a non-elephant animal",
    "an animal photo without any elephant",
]
AUD_POSITIVE = [
    "a sound recording of an elephant",
    "audio of an elephant trumpeting or vocalizing",
    "elephant call sound clip",
]
AUD_NEGATIVE = [
    "a sound recording of an animal that is not an elephant",
    "audio of a non-elephant animal vocalization",
    "animal sound clip without any elephant",
]


def main():
    df_i = pd.read_csv(IMAGE_CSV)
    df_a = pd.read_csv(AUDIO_CSV)
    img_emb = np.load(IMG_EMB_PATH)["caption_emb"]
    aud_emb = np.load(AUD_EMB_PATH)["caption_emb"]

    img_margins = MarginSignal(positive_prompts=IMG_POSITIVE, negative_prompts=IMG_NEGATIVE).compute(img_emb)
    aud_margins = MarginSignal(positive_prompts=AUD_POSITIVE, negative_prompts=AUD_NEGATIVE).compute(aud_emb)

    pred_cities = set(df_i.loc[img_margins > 0, "City"]) | set(df_a.loc[aud_margins > 0, "City"])

    gt_cities = (
        set(df_i.loc[df_i["Species"].str.contains("ELEPHANT", case=False, na=False), "City"])
        | set(df_a.loc[df_a["Animal"] == "Elephant", "City"])
    )

    tp = len(pred_cities & gt_cities)
    prec = tp / len(pred_cities) if pred_cities else (1.0 if not gt_cities else 0.0)
    rec = tp / len(gt_cities) if gt_cities else (1.0 if not pred_cities else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print(f"Predicted: {sorted(pred_cities)}")
    print(f"Ground truth: {sorted(gt_cities)}")
    print(f"[SemBench] F1={f1:.4f}")


if __name__ == "__main__":
    main()
