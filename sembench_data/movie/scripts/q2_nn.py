"""
Movie Q2 — DASE-only (no BigQuery): top-5 most positive reviews for `taken_3`.

NL: Five clearly positive reviews for movie 'taken_3'. Return reviewId.
GT: Reviews where id='taken_3' AND scoreSentiment='POSITIVE'.
Eval: precision/recall/F1 over the 5 returned ids (sembench evaluate_q2).

Aligns with paper §5.1: top-K retrieval by embedding distance to a single
"positive review" query within the 'taken_3' scope. Q2 cascade is F+L
(text filter id='taken_3' + AI.IF + LIMIT); here we mirror the same
prefilter prompt and just return top-K without LLM verification.
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

MOVIE_ID = "taken_3"
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

    # ── filter to taken_3 scope (structural filter, mirrors cascade) ──
    sub = (df["id"] == MOVIE_ID).values
    sub_df = df[sub].reset_index(drop=True)
    sub_emb = review_emb[sub]
    print(f"Reviews for '{MOVIE_ID}': {len(sub_df)}")

    print(f"Query: {args.query!r}")
    qe = embed_query([args.query], dim=DIM)[0]
    sims = cosine_sim_batch(qe, sub_emb)
    top_idx = np.argsort(-sims)[: args.top]
    top_rids = [str(sub_df.iloc[i]["reviewId"]) for i in top_idx]

    sys_df = pd.DataFrame({"reviewId": top_rids})
    metric = ev.evaluate_q2(sys_df)
    print(f"Top-{args.top} reviewIds: {top_rids}")
    print(f"[SemBench] Precision={metric.precision:.4f}  Recall={metric.recall:.4f}  F1={metric.f1_score:.4f}")


if __name__ == "__main__":
    main()
