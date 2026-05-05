"""
Movie Q10 — DASE-only (no BigQuery): rank movies by avg review score (1-5).

NL: Rank movies by audience preference based on review sentiment.
GT: SELECT M.id AS movieId, M.audienceScore AS movieScore FROM Movies AS M.
Eval: Spearman correlation over per-movie scores (sembench evaluate_q10).

Aligns with paper §5.1: anchor argmax via embedding distance on EVERY review
(5 sentiment-rubric anchors), then client-side groupby movieId mean for
per-movie ranking. Same anchors as the Q9/Q10 cascade's prefilter signal;
no LLM verification.
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

# Verbatim from q10_cascade.py for paper consistency.
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
    print(f"Total unique reviews: {len(df)};  movies: {df['id'].nunique()}")

    signal = ConfidenceMarginSignal(anchors=ANCHOR_PROMPTS)
    _ = signal.compute(review_emb)
    pred_scores = np.array([SCORE_KEYS[i] for i in signal.last_argmax], dtype=float)

    df = df.copy()
    df["predicted_score"] = pred_scores

    movie_avg = (
        df.groupby("id")["predicted_score"].mean().reset_index()
        .rename(columns={"id": "movieId", "predicted_score": "movieScore"})
        .sort_values("movieScore", ascending=False).reset_index(drop=True)
    )

    metric = ev.evaluate_q10(movie_avg[["movieId", "movieScore"]])
    print(f"Aggregated to {len(movie_avg)} movies (groupby mean).")
    print(f"[SemBench] Spearman={metric.spearman_correlation:.4f}  "
          f"KendallTau={metric.kendall_tau:.4f}")


if __name__ == "__main__":
    main()
