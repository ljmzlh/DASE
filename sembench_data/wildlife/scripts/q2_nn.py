"""
Wildlife Q2 — DASE-only (no BigQuery): count elephant audio recordings.

NL: SELECT COUNT(*) FROM AudioData WHERE Animal = 'Elephant'.
Eval: relative_error → reported as RelativeError (matches original).

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

from dase_cascade import MarginSignal

WILDLIFE_DIR = os.path.abspath(os.path.join(_HERE, ".."))
AUDIO_CSV    = os.path.join(WILDLIFE_DIR, "cache", "audio_data.csv")
EMB_PATH     = os.path.join(WILDLIFE_DIR, "data", "audio_embeddings.npz")

# Counterfactual anchors — verbatim from q2_cascade.py for paper consistency.
POSITIVE = [
    "a sound recording of an elephant",
    "audio of an elephant trumpeting or vocalizing",
    "elephant call sound clip",
]
NEGATIVE = [
    "a sound recording of an animal that is not an elephant",
    "audio of a non-elephant animal vocalization",
    "animal sound clip without any elephant",
]


def main():
    df = pd.read_csv(AUDIO_CSV)
    audio_emb = np.load(EMB_PATH)["caption_emb"]
    n_total = len(df)
    gt_count = int((df["Animal"] == "Elephant").sum())

    margins = MarginSignal(positive_prompts=POSITIVE, negative_prompts=NEGATIVE).compute(audio_emb)
    pred_count = int((margins > 0).sum())

    rel_err = abs(pred_count - gt_count) / gt_count if gt_count else (0.0 if pred_count == 0 else 1.0)
    print(f"Total audios: {n_total}")
    print(f"Predicted: {pred_count}")
    print(f"Ground truth: {gt_count}")
    print(f"[SemBench] RelativeError={rel_err:.4f}")


if __name__ == "__main__":
    main()
