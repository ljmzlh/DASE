"""ClusterCascade — cluster-based prefilter for SEM_MAP queries.

Used by ecomm Q3 (brand extraction over 500 products), ecomm Q12 (color
labeling), mmqa Q4 (genre clustering). Structurally distinct from the
Signal+Band+Verifier cascade because the prefilter doesn't bucket rows into
yes/uncertain/no; it groups them, picks one rep per group, sends reps to BQ,
then propagates each rep's BQ-generated label back to its group members.

Pipeline:
  1. Cluster all `n` rows using a sklearn-style clusterer (KMeans /
     AgglomerativeClustering / etc.) on the embeddings.
  2. For each cluster, pick one representative (default: closest to mean
     embedding, i.e. medoid-of-centroid).
  3. Run the verifier (typically AiGenerateVerifier with AI.GENERATE) on just
     the K representatives.
  4. Propagate each rep's generated value to all members of its cluster.

Caller uses `result.predicted` ({id: generated_value}) to compute their score
(typically ARI vs GT cluster labels).
"""
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from dase_cascade.verifier import BqVerifier, VerifierResult


@dataclass
class ClusterCascadeResult:
    labels: np.ndarray                # per-row cluster id (sklearn .labels_)
    n_clusters: int
    rep_indices: List[int]            # row index of representative per cluster
    rep_ids: List[Any]                # ids[rep_indices]
    verifier_result: VerifierResult
    predicted: Dict[Any, Any]         # id → generated value (propagated from rep)
    timings_s: dict = field(default_factory=dict)

    def cluster_size_stats(self) -> dict:
        bins = np.bincount(self.labels)
        return {
            "min": int(bins.min()), "max": int(bins.max()),
            "mean": float(bins.mean()), "n_singletons": int((bins == 1).sum()),
        }

    def to_dict(self) -> dict:
        return {
            "n_clusters": self.n_clusters,
            "cluster_size_stats": self.cluster_size_stats(),
            "rep_ids_sample": self.rep_ids[:10],
            "verifier": self.verifier_result.to_dict(),
            "n_predicted": len(self.predicted),
            "timings_s": self.timings_s,
        }


@dataclass
class ClusterCascade:
    embeddings: np.ndarray             # (n, d)
    ids:        Sequence[Any]
    clusterer:  Any                    # sklearn clusterer with .fit(X) and .labels_
    verifier:   BqVerifier             # typically AiGenerateVerifier
    rep_strategy: str = "centroid_nearest"   # or "first"

    def __post_init__(self):
        if len(self.ids) != self.embeddings.shape[0]:
            raise ValueError(
                f"ids/embeddings mismatch: {len(self.ids)} vs {self.embeddings.shape[0]}"
            )

    def _pick_rep(self, cluster_emb: np.ndarray, cluster_idx: np.ndarray) -> int:
        if self.rep_strategy == "first":
            return int(cluster_idx[0])
        # centroid_nearest
        centroid = cluster_emb.mean(axis=0)
        dist = np.linalg.norm(cluster_emb - centroid, axis=1)
        return int(cluster_idx[dist.argmin()])

    def run(self, client, per_row_cost: float) -> ClusterCascadeResult:
        t = time.time()
        self.clusterer.fit(self.embeddings)
        labels = np.asarray(self.clusterer.labels_)
        t_cluster = time.time() - t

        n_clusters = int(labels.max() + 1)
        rep_indices: List[int] = []
        for c in range(n_clusters):
            mask = labels == c
            if not mask.any():
                continue
            cluster_idx = np.where(mask)[0]
            rep_indices.append(self._pick_rep(self.embeddings[cluster_idx], cluster_idx))
        rep_ids = [self.ids[i] for i in rep_indices]

        t = time.time()
        vres = self.verifier.verify(client, rep_ids, per_row_cost)
        t_verify = time.time() - t

        # Build cluster → label map from verifier output (id → value),
        # then propagate to every row in the cluster.
        rep_id_to_idx = dict(zip(rep_ids, rep_indices))
        cluster_to_value: Dict[int, Any] = {}
        for rid, v in vres.values.items():
            ridx = rep_id_to_idx.get(rid)
            if ridx is None:
                continue
            cluster_to_value[int(labels[ridx])] = v
        # Reps not returned by BQ get a sentinel
        for rid in rep_ids:
            ridx = rep_id_to_idx[rid]
            c = int(labels[ridx])
            cluster_to_value.setdefault(c, "UNKNOWN")

        ids_arr = list(self.ids)
        predicted = {ids_arr[i]: cluster_to_value[int(labels[i])] for i in range(len(ids_arr))}

        return ClusterCascadeResult(
            labels=labels, n_clusters=n_clusters,
            rep_indices=rep_indices, rep_ids=rep_ids,
            verifier_result=vres, predicted=predicted,
            timings_s={
                "cluster_fit":  t_cluster,
                "verify_total": t_verify,
                "verify_ctas":  vres.ctas_wall_s,
                "verify_bq":    vres.wall_s,
            },
        )
