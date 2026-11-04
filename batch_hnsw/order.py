"""
Locality order for batched HNSW probes: k-means groups of ~group_size left rows,
groups chained by centroid, rows chained nearest-neighbour inside a group.
"""

import os
from typing import List, Tuple

# OpenBLAS can segfault once more threads call into it than it was built for:
# its own pool (OPENBLAS_NUM_THREADS) and k-means' OpenMP threads, one per core
# by default, which each call BLAS. k-means does not need that many.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("OMP_NUM_THREADS", "16")

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits


def _chain(points: np.ndarray, start: int) -> List[int]:
    """Greedy nearest-neighbour chain through `points`, starting at `start`."""
    n = len(points)
    sq = (points * points).sum(1)
    d = sq[:, None] + sq[None, :] - 2.0 * points @ points.T
    order = [start]
    seen = np.zeros(n, dtype=bool)
    seen[start] = True
    cur = start
    for _ in range(n - 1):
        row = np.where(seen, np.inf, d[cur])
        cur = int(row.argmin())
        seen[cur] = True
        order.append(cur)
    return order


def locality_order(vecs: np.ndarray, group_size: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """k-means groups of ~group_size rows; groups chained by centroid, rows chained inside a group.

    Returns (row indices in processing order, group id per processed row).
    """
    k = max(1, len(vecs) // group_size)
    km = MiniBatchKMeans(n_clusters=k, batch_size=4096, n_init=1, max_iter=50, random_state=seed)
    with threadpool_limits(limits=16):  # in case numpy/sklearn were loaded before the env above
        labels = km.fit_predict(vecs)
    centers = km.cluster_centers_.astype(np.float64)
    group_order = _chain(centers, 0) if k > 1 else [0]
    order: List[int] = []
    groups: List[int] = []
    for g in group_order:
        members = np.flatnonzero(labels == g)
        if len(members) == 0:
            continue
        pts = vecs[members].astype(np.float64)
        start = int(((pts - centers[g]) ** 2).sum(1).argmin())
        order.extend(members[_chain(pts, start)].tolist())
        groups.extend([g] * len(members))
    return np.array(order), np.array(groups)
