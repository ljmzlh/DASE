"""
Wildlife Q10 — DASE-only (no BigQuery): (City, StationID) with most zebra pictures.

NL: SELECT City, StationID FROM ImageData WHERE Species LIKE '%ZEBRA%'
    GROUP BY City, StationID ORDER BY COUNT(*) DESC LIMIT 1.
Eval: F1 = 1.0 if predicted (City, StationID) == GT top else 0.0 (matches original).

Aligns with paper §5.1: counterfactual anchors. Anchors verbatim from
q10_cascade.py (zebra). Single MarginSignal pass; client-side GROUP BY argmax.
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
EMB_PATH     = os.path.join(WILDLIFE_DIR, "data", "image_embeddings.npz")

POSITIVE = [
    "a photograph of a zebra",
    "a wildlife camera trap image showing a zebra",
    "an animal with black and white stripes, a zebra",
]
NEGATIVE = [
    "a photograph that does not contain a zebra",
    "a wildlife camera trap image of an animal that is not a zebra",
    "an animal scene with no zebra in it",
]


def main():
    df = pd.read_csv(IMAGE_CSV)
    image_emb = np.load(EMB_PATH)["caption_emb"]

    margins = MarginSignal(positive_prompts=POSITIVE, negative_prompts=NEGATIVE).compute(image_emb)
    pred_z = df.loc[margins > 0]
    pred_counts = pred_z.groupby(["City", "StationID"]).size()
    pred_top = pred_counts.idxmax()

    gt_z = df[df["Species"].str.contains("ZEBRA", case=False, na=False)]
    gt_counts = gt_z.groupby(["City", "StationID"]).size()
    gt_top = gt_counts.idxmax()

    f1 = 1.0 if pred_top == gt_top else 0.0
    print(f"Predicted: {pred_top}")
    print(f"Ground truth: {gt_top}")
    print(f"[SemBench] F1={f1:.4f}")


if __name__ == "__main__":
    main()
