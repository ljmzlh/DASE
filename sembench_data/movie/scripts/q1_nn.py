"""
Movie Q1 — DASE-only (no BigQuery): top-5 most positive reviews.

NL: Find five clearly positive movie reviews; return reviewId.
GT: Reviews.scoreSentiment == 'POSITIVE'.
Eval: precision/recall over the 5 returned ids (sembench evaluate_q1).

Aligns with paper §5.1: top-K retrieval by embedding distance to a single
"positive review" query. The Q1 cascade uses the same prompt for the
(prefilter top-K → BQ AI.IF) sequence; here we just return the top-K
without LLM verification.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import embed_query, cosine_sim_batch
import evaluator as ev

MOVIE_DIR    = os.path.abspath(os.path.join(_HERE, ".."))
EMB_PATH     = os.path.join(MOVIE_DIR, "data", "review_embeddings.npz")
REVIEWS_CSV  = os.path.join(MOVIE_DIR, "cache", "Reviews.csv")

DEFAULT_QUERY = "this is a clearly positive movie review"
TOP_K = 5
DIM = 3072


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=str, default=DEFAULT_QUERY)
    ap.add_argument("--top",   type=int, default=TOP_K)
    args = ap.parse_args()

    df = pd.read_csv(REVIEWS_CSV)
    review_emb = np.load(EMB_PATH)["reviewText_emb"]
    keep = ~df["reviewId"].duplicated()
    df = df[keep].reset_index(drop=True)
    review_emb = review_emb[keep.values]

    print(f"Query: {args.query!r}")
    qe = embed_query([args.query], dim=DIM)[0]
    sims = cosine_sim_batch(qe, review_emb)
    top_idx = np.argsort(-sims)[: args.top]
    top_rids = [str(df.iloc[i]["reviewId"]) for i in top_idx]

    sys_df = pd.DataFrame({"reviewId": top_rids})
    metric = ev.evaluate_q1(sys_df)
    print(f"Top-{args.top} reviewIds: {top_rids}")
    print(f"[SemBench] Precision={metric.precision:.4f}  Recall={metric.recall:.4f}  F1={metric.f1_score:.4f}")


if __name__ == "__main__":
    main()
