"""Runtime helpers shared across cascade primitives.

Wraps BQ client setup, embedding helper, query execution, and the per-token
price table. All cascade modules go through this so each script doesn't
re-invent the same boilerplate.
"""
import os
import time
from typing import Iterable, List, Optional, Tuple

import numpy as np

from google.cloud import bigquery

# Gemini 2.5 Flash on BQ; per-token USD rates as used across the existing scripts.
PRICES = {
    "input_other": 0.30 / 1e6,
    "input_audio": 1.00 / 1e6,
    "output":      2.50 / 1e6,
}

# Default embedding dim used by every existing script (Gemini-embedding-001).
DEFAULT_EMBED_DIM = 3072


def embed_query(prompts: Iterable[str], dim: int = DEFAULT_EMBED_DIM) -> np.ndarray:
    """Embed a list of anchor prompts as RETRIEVAL_QUERY vectors → (n, dim) float32."""
    from tools.llm_tool import embed_batch
    out = embed_batch(list(prompts), task_type="RETRIEVAL_QUERY", output_dimensionality=dim)
    return np.asarray(out, dtype=np.float32)


def bq_client(project: Optional[str] = None) -> bigquery.Client:
    """Return a BQ Client; defaults to $GCP_PROJECT."""
    proj = project or os.environ.get("GCP_PROJECT", "")
    if not proj:
        raise RuntimeError("set $GCP_PROJECT or pass project= explicitly")
    return bigquery.Client(project=proj)


def run_query(
    client: bigquery.Client,
    sql: str,
    params: Optional[List[bigquery.ScalarQueryParameter]] = None,
    use_cache: bool = False,
) -> Tuple["object", float, int, str]:
    """Execute SQL → (DataFrame, wall_s, slot_ms, sql).

    All cascade scripts share this exact return shape; replaces ~5 verbatim
    copies across q*_cascade.py.
    """
    cfg = bigquery.QueryJobConfig(
        query_parameters=params or [],
        use_query_cache=use_cache,
    )
    t0 = time.time()
    job = client.query(sql, job_config=cfg)
    df = job.result().to_dataframe()
    wall = time.time() - t0
    job.reload()
    slot = int(job.slot_millis or 0)
    return df, wall, slot, sql


def cosine_sim_batch(query: np.ndarray, batch: np.ndarray) -> np.ndarray:
    """Cosine similarity between a single query vector (d,) and a batch (n,d)
    → (n,) float array. Both inputs may be unnormalized."""
    q = query / (np.linalg.norm(query) + 1e-12)
    b = batch / (np.linalg.norm(batch, axis=1, keepdims=True) + 1e-12)
    return b @ q
