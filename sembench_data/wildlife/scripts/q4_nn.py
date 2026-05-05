"""
Wildlife Q4 — DASE-only (no BigQuery): city with most elephant audio recordings.

NL: SELECT City FROM AudioData WHERE Animal='Elephant' GROUP BY City
    ORDER BY COUNT(*) DESC LIMIT 1.
Eval: F1 = 1.0 if predicted city == GT city else 0.0 (matches original).

Aligns with paper §5.1: counterfactual anchors (this is X vs this is not X).
Anchors are verbatim from q4_cascade.py.
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

    margins = MarginSignal(positive_prompts=POSITIVE, negative_prompts=NEGATIVE).compute(audio_emb)
    pred_mask = margins > 0
    pred_el = df.loc[pred_mask]
    pred_city = pred_el.groupby("City").size().idxmax()

    gt_el = df[df["Animal"] == "Elephant"]
    gt_city = gt_el.groupby("City").size().idxmax()

    f1 = 1.0 if pred_city == gt_city else 0.0
    print(f"Predicted: {pred_city}")
    print(f"Ground truth: {gt_city}")
    print(f"[SemBench] F1={f1:.4f}")


if __name__ == "__main__":
    main()
