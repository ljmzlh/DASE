"""
Movie Q9 — DASE-only (no BigQuery): score reviews 1-5 for `ant_man`.

NL: Score 1-5 how much the reviewer liked the movie, for ant_man scope.
GT: SPLIT_PART(originalScore, '/', 1) -> float per review.
Eval: Spearman + Kendall_tau (sembench evaluate_q9).

Aligns with paper §5.1: anchor argmax via embedding distance — pick the
nearest sentiment-rubric anchor for each review (5 classes). Same anchors
the Q9 cascade uses for the prefilter signal; here we just use argmax score
without LLM refinement on uncertain rows.
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import ConfidenceMarginSignal
import evaluator as ev

MOVIE_DIR    = os.path.abspath(os.path.join(_HERE, ".."))
EMB_PATH     = os.path.join(MOVIE_DIR, "data", "review_embeddings.npz")
REVIEWS_CSV  = os.path.join(MOVIE_DIR, "cache", "Reviews.csv")

MOVIE_ID = "ant_man_and_the_wasp_quantumania"

# Verbatim from q9_cascade.py for paper consistency.
ANCHORS = {
    1: "Very negative. Strong negative sentiment, indicating high dissatisfaction, frustration, or anger.",
    2: "Negative. Noticeably negative sentiment, indicating some level of dissatisfaction but without strong anger or frustration.",
    3: "Neutral. Expresses no clear positive or negative sentiment. May be factual or descriptive without emotional language.",
    4: "Positive. Noticeably positive sentiment, indicating general satisfaction.",
    5: "Very positive. Strong positive sentiment, indicating high satisfaction.",
}
SCORE_KEYS = [1, 2, 3, 4, 5]
ANCHOR_PROMPTS = [ANCHORS[k] for k in SCORE_KEYS]


def main():
    df = pd.read_csv(REVIEWS_CSV)
    review_emb = np.load(EMB_PATH)["reviewText_emb"]
    keep = ~df["reviewId"].duplicated()
    df = df[keep].reset_index(drop=True)
    review_emb = review_emb[keep.values]

    # ── filter to ant_man scope ──
    sub = (df["id"] == MOVIE_ID).values
    sub_df = df[sub].reset_index(drop=True)
    sub_emb = review_emb[sub]
    print(f"Reviews for '{MOVIE_ID}': {len(sub_df)}")

    signal = ConfidenceMarginSignal(anchors=ANCHOR_PROMPTS)
    _ = signal.compute(sub_emb)
    pred_scores = np.array([SCORE_KEYS[i] for i in signal.last_argmax], dtype=float)

    sys_df = pd.DataFrame({
        "reviewId": sub_df["reviewId"].values,
        "reviewScore": pred_scores,
    })
    metric = ev.evaluate_q9(sys_df)
    print(f"Predicted score distribution: {pd.Series(pred_scores).value_counts().sort_index().to_dict()}")
    print(f"[SemBench] Spearman={metric.spearman_correlation:.4f}  "
          f"KendallTau={metric.kendall_tau:.4f}")


if __name__ == "__main__":
    main()
