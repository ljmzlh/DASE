"""
Movie Q8 — DASE-only (no BigQuery): positive AND negative review counts for `taken_3`.

NL: COUNT positive and negative reviews for movie 'taken_3' (GROUP BY scoreSentiment).
GT: SELECT scoreSentiment, COUNT(*) FROM Reviews WHERE id='taken_3' GROUP BY scoreSentiment.
Eval: relative_error over (POSITIVE_count, NEGATIVE_count) (sembench evaluate_q8).

Aligns with paper §5.1: counterfactual anchors. Same MarginSignal as Q3 (and the
Q3/Q8 cascades) on the taken_3 scope; sign(margin) > 0 → POSITIVE prediction,
otherwise NEGATIVE. Pure Python aggregation — no BQ verification.
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import MarginSignal
import evaluator as ev

MOVIE_DIR    = os.path.abspath(os.path.join(_HERE, ".."))
EMB_PATH     = os.path.join(MOVIE_DIR, "data", "review_embeddings.npz")
REVIEWS_CSV  = os.path.join(MOVIE_DIR, "cache", "Reviews.csv")

MOVIE_ID = "taken_3"

# Same prompts as q3_cascade.py / q3_nn.py for paper consistency (Q3 ≡ Q8 sem_filter).
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
    pos_pred = margins > 0
    pred_pos_count = int(pos_pred.sum())
    pred_neg_count = int((~pos_pred).sum())

    print(f"\n{'scoreSentiment':<16} {'predicted':>10}")
    print("-" * 28)
    print(f"{'POSITIVE':<16} {pred_pos_count:>10}")
    print(f"{'NEGATIVE':<16} {pred_neg_count:>10}")

    # ── SemBench-aligned metric ──
    sys_df = pd.DataFrame({
        "scoreSentiment": ["POSITIVE", "NEGATIVE"],
        "count": [pred_pos_count, pred_neg_count],
    })
    metric = ev.evaluate_q8(sys_df)
    print(f"\n[SemBench] RelativeError={metric.relative_error:.4f}  "
          f"MAPE={metric.mean_absolute_percentage_error:.2f}%")


if __name__ == "__main__":
    main()
