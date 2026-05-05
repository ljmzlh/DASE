"""
Wildlife Q7 — DASE-only (no BigQuery): cities where zebras AND impala co-occur in images.

NL: image_zebra_cities ∩ image_impala_cities.
Eval: set retrieval F1 (matches original).

Aligns with paper §5.1: counterfactual anchors per concept. Anchors verbatim
from q7_cascade.py (zebra + impala). Two MarginSignal passes on the same image
embeddings; client-side intersect over City.
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

ZEBRA_POSITIVE = [
    "a photograph of a zebra",
    "a wildlife camera trap image showing a zebra",
    "an animal with black and white stripes, a zebra",
]
ZEBRA_NEGATIVE = [
    "a photograph that does not contain a zebra",
    "a wildlife camera trap image of an animal that is not a zebra",
    "an animal scene with no zebra in it",
]
IMPALA_POSITIVE = [
    "a photograph of an impala",
    "a wildlife camera trap image showing an impala antelope",
    "an impala animal in the picture",
]
IMPALA_NEGATIVE = [
    "a photograph that does not contain an impala",
    "a wildlife camera trap image of a non-impala animal",
    "an animal scene without any impala",
]


def main():
    df = pd.read_csv(IMAGE_CSV)
    image_emb = np.load(EMB_PATH)["caption_emb"]

    z_margins = MarginSignal(positive_prompts=ZEBRA_POSITIVE, negative_prompts=ZEBRA_NEGATIVE).compute(image_emb)
    i_margins = MarginSignal(positive_prompts=IMPALA_POSITIVE, negative_prompts=IMPALA_NEGATIVE).compute(image_emb)

    zebra_cities = set(df.loc[z_margins > 0, "City"])
    impala_cities = set(df.loc[i_margins > 0, "City"])
    pred = sorted(zebra_cities & impala_cities)

    gt_zebra = set(df.loc[df["Species"].str.contains("ZEBRA", case=False, na=False), "City"])
    gt_impala = set(df.loc[df["Species"].str.contains("IMPALA", case=False, na=False), "City"])
    gt = sorted(gt_zebra & gt_impala)

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
