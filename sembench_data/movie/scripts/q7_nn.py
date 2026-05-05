"""
Movie Q7 — DASE-only (no BigQuery): ALL opposite-sentiment review pairs for `ant_man`.

NL: ALL pairs of reviews with OPPOSITE sentiment for ant_man_and_the_wasp_quantumania.
GT: pairs (r1, r2) where r1.scoreSentiment != r2.scoreSentiment (no LIMIT).
Eval: precision/recall/F1 over returned pairs (sembench evaluate_q7).

Aligns with paper §5.1: pair cosine signal for J via a calibrated distance
threshold. PairCosineSignal scores all ordered self-pairs (i, j) i!=j;
predict OPPOSITE when sim < tau (low sim ≈ opposite sentiment). Same primitive
as the Q7 cascade prefilter (top-K_neg). No LLM verification.
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import PairCosineSignal
import evaluator as ev

MOVIE_DIR    = os.path.abspath(os.path.join(_HERE, ".."))
EMB_PATH     = os.path.join(MOVIE_DIR, "data", "review_embeddings.npz")
REVIEWS_CSV  = os.path.join(MOVIE_DIR, "cache", "Reviews.csv")

MOVIE_ID = "ant_man_and_the_wasp_quantumania"
# Predict opposite-sentiment when pair cosine sim < TAU.
# Calibrated to roughly match the Q7 cascade's K_neg=1000 cutoff on ant_man (~128 reviews).
SIM_TAU = 0.55


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
    n_total = len(sub_df)
    print(f"Reviews for '{MOVIE_ID}': {n_total}")
    print(f"Pair sim threshold tau={SIM_TAU} (predict OPPOSITE when sim < tau)")

    pair_signal = PairCosineSignal(embeddings_left=sub_emb)
    # Compute full sim matrix (small ant_man scope ≈ 128 rows → ~16k pairs).
    Lm = pair_signal._left
    S = Lm @ Lm.T
    np.fill_diagonal(S, np.inf)  # exclude i==j
    ii, jj = np.where(S < SIM_TAU)

    rows = []
    rids = sub_df["reviewId"].values
    for a, b in zip(ii, jj):
        rows.append({
            "id": MOVIE_ID,
            "reviewId1": str(rids[a]),
            "reviewId2": str(rids[b]),
        })
    sys_df = pd.DataFrame(rows)
    print(f"Retrieved pairs: {len(sys_df)}")

    metric = ev.evaluate_q7(sys_df)
    print(f"[SemBench] Precision={metric.precision:.4f}  Recall={metric.recall:.4f}  F1={metric.f1_score:.4f}")


if __name__ == "__main__":
    main()
