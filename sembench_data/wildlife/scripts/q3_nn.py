"""
Wildlife Q3 — DASE-only (no BigQuery): city with most zebra pictures.

NL: SELECT City FROM ImageData WHERE Species LIKE '%ZEBRA%' GROUP BY City
    ORDER BY COUNT(*) DESC LIMIT 1.
Eval: F1 = 1.0 if predicted city == GT city else 0.0 (matches original).

Aligns with paper §5.1: counterfactual anchors (this is X vs this is not X).
Anchors are verbatim from q3_cascade.py.
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
    pred_mask = margins > 0
    pred_zebra = df.loc[pred_mask]
    pred_city = pred_zebra.groupby("City").size().idxmax()

    gt_zebra = df[df["Species"].str.contains("ZEBRA", case=False, na=False)]
    gt_city = gt_zebra.groupby("City").size().idxmax()

    f1 = 1.0 if pred_city == gt_city else 0.0
    print(f"Predicted: {pred_city}")
    print(f"Ground truth: {gt_city}")
    print(f"[SemBench] F1={f1:.4f}")


if __name__ == "__main__":
    main()
