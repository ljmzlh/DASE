"""Shared contrastive filter for MMQA Q6a–c (tampa_international_airport Destinations)."""
from __future__ import annotations

import os
import sys
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from tools.llm_tool import embed_batch  # noqa: E402


def cosine_similarity(query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
    q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-12)
    d_norm = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-12)
    return d_norm @ q_norm


def embed_queries(texts: List[str]) -> np.ndarray:
    return np.array(embed_batch(texts, task_type="RETRIEVAL_QUERY"), dtype=np.float32)


def predict_airlines(
    df: pd.DataFrame,
    positive_prompts: List[str],
    negative_prompts: List[str],
) -> List[str]:
    """Contrastive: mean sim to explicit YES prompts > mean sim to explicit NO prompts."""
    emb = np.array(df["embedding"].tolist(), dtype=np.float32)
    pos_embs = embed_queries(positive_prompts)
    neg_embs = embed_queries(negative_prompts)
    pos_sims = np.mean(
        [cosine_similarity(e, emb) for e in pos_embs], axis=0
    )
    neg_sims = np.mean(
        [cosine_similarity(e, emb) for e in neg_embs], axis=0
    )
    mask = pos_sims > neg_sims
    out = df.loc[mask, "Airlines"].astype(str).str.strip()
    return [a for a in out.tolist() if a and a.lower() != "nan"]
