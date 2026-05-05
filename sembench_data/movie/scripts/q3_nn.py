"""
Movie Q3 — DASE-only (no BigQuery): count positive reviews for `taken_3`.

NL: COUNT positive reviews for movie 'taken_3'. Return positive_review_cnt.
GT: SELECT COUNT(*) FROM Reviews WHERE id='taken_3' AND scoreSentiment='POSITIVE'.
Eval: relative_error → score = 1 / (1 + |pred − gt| / gt) (sembench evaluate_q3).

Aligns with paper §5.1: counterfactual anchors. Same MarginSignal (3 pos / 3 neg)
the Q3 cascade uses on the taken_3 scope; here we just take sign(margin) > 0 as
the per-row prediction (no BQ verification on uncertain rows).
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import MarginSignal, relative_error_score
import evaluator as ev

MOVIE_DIR    = os.path.abspath(os.path.join(_HERE, ".."))
EMB_PATH     = os.path.join(MOVIE_DIR, "data", "review_embeddings.npz")
REVIEWS_CSV  = os.path.join(MOVIE_DIR, "cache", "Reviews.csv")

MOVIE_ID = "taken_3"

# Counterfactual anchors — verbatim from q3_cascade.py for paper consistency.
POSITIVE = [
    "this is a clearly positive movie review",
    "the reviewer praises the film and recommends it",
    "an enthusiastic, favorable review of the movie",
]
NEGATIVE = [
    "this is a clearly negative movie review",
    "the reviewer criticizes the film and dislikes it",
    "an unfavorable, dismissive review of the movie",
]


def main():
    df = pd.read_csv(REVIEWS_CSV)
    review_emb = np.load(EMB_PATH)["reviewText_emb"]
    keep = ~df["reviewId"].duplicated()
    df = df[keep].reset_index(drop=True)
    review_emb = review_emb[keep.values]

    # ── filter to taken_3 scope ──
    sub = (df["id"] == MOVIE_ID).values
    sub_df = df[sub].reset_index(drop=True)
    sub_emb = review_emb[sub]
    print(f"Reviews for '{MOVIE_ID}': {len(sub_df)}")

    margins = MarginSignal(positive_prompts=POSITIVE, negative_prompts=NEGATIVE).compute(sub_emb)
    pred_count = int((margins > 0).sum())

    # ── ground truth count via sembench evaluator ──
    gt_df = ev.get_ground_truth(3)
    n_gt_pos = int(float(gt_df.iloc[0, 0]))
    score = relative_error_score(pred_count, n_gt_pos)

    print(f"Predicted positive_review_cnt : {pred_count}")
    print(f"Ground truth positive_review_cnt: {n_gt_pos}")
    rel_err = abs(pred_count - n_gt_pos) / n_gt_pos if n_gt_pos else 0.0
    print(f"[SemBench] RelativeError={rel_err:.4f}  Score={score:.4f}")


if __name__ == "__main__":
    main()
