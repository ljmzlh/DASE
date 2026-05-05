"""
Movie Q6 — DASE-only (no BigQuery): opposite-sentiment review pairs for `ant_man`.

NL: 10 pairs of reviews with OPPOSITE sentiment for ant_man_and_the_wasp_quantumania.
GT: pairs (r1, r2) where r1.scoreSentiment != r2.scoreSentiment within ant_man scope.
Eval: precision/recall/F1 over up to 10 returned pairs (sembench evaluate_q6).

Aligns with paper §5.1: pair cosine signal for J. PairCosineSignal scores all
ordered self-pairs (i, j) i!=j; LOW-sim pairs likely have opposite sentiment.
We return the TOP_PAIRS lowest-sim pairs as the prediction (no LLM verification).
Same primitive as the Q6 cascade (top-K_neg pool).
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
TOP_PAIRS = 10


def select_bottom_sim_pairs(pair_signal: PairCosineSignal, n: int, k: int):
    """Top-k lowest-sim ordered self-pairs (i != j) — i.e. likely opposite sentiment."""
    L = np.arange(n, dtype=np.int64)
    Lm = pair_signal._left[L]
    S = Lm @ Lm.T  # (n, n)
    np.fill_diagonal(S, np.inf)  # exclude i==j from "lowest"
    flat = S.flatten()
    k = min(k, np.isfinite(flat).sum())
    bot_idx = np.argpartition(flat, k - 1)[:k]
    pairs = [(int(idx // n), int(idx % n), float(flat[idx])) for idx in bot_idx]
    pairs.sort(key=lambda t: t[2])
    return pairs


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

    pair_signal = PairCosineSignal(embeddings_left=sub_emb)
    bot_pairs = select_bottom_sim_pairs(pair_signal, n_total, TOP_PAIRS)

    rows = []
    for i, j, _s in bot_pairs:
        rows.append({
            "id": str(sub_df.iloc[i]["id"]),
            "reviewId1": str(sub_df.iloc[i]["reviewId"]),
            "reviewId2": str(sub_df.iloc[j]["reviewId"]),
        })
    sys_df = pd.DataFrame(rows)
    metric = ev.evaluate_q6(sys_df)
    print(f"Returned {len(sys_df)} bottom-similarity pairs")
    print(f"[SemBench] Precision={metric.precision:.4f}  Recall={metric.recall:.4f}  F1={metric.f1_score:.4f}")


if __name__ == "__main__":
    main()
