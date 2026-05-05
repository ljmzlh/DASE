"""Shared utilities for review embedding queries."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from llm_tool import embed_batch  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
CACHE_DIR = os.path.join(SCRIPT_DIR, "cache")

DEFAULT_QUERY = "positive review"
DIM = 3072
TOP_K = 5


def cosine_distance(a, b):
    """Cosine distance = 1 - cosine_similarity.  a: (dim,), b: (N, dim)."""
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return 1.0 - b_norm @ a_norm


def load_data():
    """Load review embeddings and CSV. Returns (df, review_emb)."""
    emb_path = os.path.join(DATA_DIR, "review_embeddings.npz")
    review_emb = np.load(emb_path)["reviewText_emb"]  # (N, dim)
    df = pd.read_csv(os.path.join(CACHE_DIR, "Reviews.csv"))
    assert len(df) == review_emb.shape[0], "Row count mismatch between CSV and embeddings"
    return df, review_emb


def embed_query(query, dim):
    """Embed a query string. Returns (dim,) float32 array."""
    return np.array(
        embed_batch([query], task_type="RETRIEVAL_QUERY", output_dimensionality=dim)[0],
        dtype=np.float32,
    )


def print_results(ranked_idx, dists, sub_df, gt_ids, top_k):
    """Display ranked results and print F1 metrics."""
    print(f"\nTop {top_k} closest reviews (smallest cosine distance):\n")
    for rank, idx in enumerate(ranked_idx, 1):
        row = sub_df.iloc[idx]
        rid = str(row["reviewId"])
        hit = "HIT" if rid in gt_ids else "MISS"
        print(f"  #{rank}  dist={dists[idx]:.4f}  sentiment={row['scoreSentiment']}  [{hit}]")
        print(f"       movie={row['id']}  critic={row.get('criticName', 'N/A')}  reviewId={rid}")
        print(f"       \"{row['reviewText'][:120]}{'...' if len(str(row['reviewText'])) > 120 else ''}\"")
        print()

    retrieved_list = [str(sub_df.iloc[idx]["reviewId"]) for idx in ranked_idx]
    k = len(retrieved_list)
    tp = sum(1 for rid in retrieved_list if rid in gt_ids)
    precision = tp / k
    recall = tp / k
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    print(f"Among top {k}: TP={tp}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
