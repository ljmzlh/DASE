"""
Wildlife Q1 — DASE-only (no BigQuery): count zebra images.

NL: SELECT COUNT(*) FROM ImageData WHERE Species LIKE '%ZEBRA%'.
GT: 11 zebra images (8 'ZEBRA' + 3 'IMPALA, ZEBRA').
Eval: relative_error → score = 1 / (1 + |pred − gt| / gt).

Aligns with paper §5.1: "DASE runs ranked DASE alone, using
embedding-distance-based filtering. Semantic filters are resolved via
counterfactual anchors (e.g., 'this is X' versus 'this is not X')."

Same MarginSignal (3 pos / 3 neg) the cascade uses; here we just take
sign(margin) > 0 as the prediction (no BQ verification on uncertain rows).
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import MarginSignal, relative_error_score

WILDLIFE_DIR = os.path.abspath(os.path.join(_HERE, ".."))
IMAGE_CSV    = os.path.join(WILDLIFE_DIR, "cache", "image_data.csv")
EMB_PATH     = os.path.join(WILDLIFE_DIR, "data", "image_embeddings.npz")

# Counterfactual anchors — verbatim from the cascade for consistency.
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
    n_total = len(df)
    n_gt = int(df["Species"].str.contains("ZEBRA", case=False, na=False).sum())

    margins = MarginSignal(positive_prompts=POSITIVE, negative_prompts=NEGATIVE).compute(image_emb)
    pred_count = int((margins > 0).sum())
    score = relative_error_score(pred_count, n_gt)

    print(f"Total images: {n_total}")
    print(f"Predicted zebra count: {pred_count}")
    print(f"Ground truth: {n_gt}")
    print(f"[SemBench] RelativeError={abs(pred_count - n_gt) / n_gt:.4f}  Score={score:.4f}")


if __name__ == "__main__":
    main()
