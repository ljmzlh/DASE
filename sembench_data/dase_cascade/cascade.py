"""Cascade — the primary unified solver.

Composition: Signal → Band → Verifier.

  cascade = Cascade(
      embeddings = (n, d) row embedding matrix,
      ids        = (n,)   row ids (any hashable; must align with embeddings),
      signal     = a Signal (e.g. MarginSignal),
      band       = a Band   (e.g. AlphaBand(0.2)),
      verifier   = a BqVerifier (e.g. AiIfVerifier(...)),
  )
  result = cascade.run(client, per_row_cost)

The result has three buckets — confident_pos / uncertain / confident_neg —
plus a VerifierResult for the BQ stage. The caller assembles the final answer
(union, intersection, count, GROUP BY, …) — that's intentionally not absorbed,
because it varies too much per Q (paper §5.1: F vs J vs M differ here).
"""
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

import numpy as np

from dase_cascade.band import Band, Partition
from dase_cascade.signal import Signal
from dase_cascade.verifier import BqVerifier, VerifierResult


@dataclass
class CascadeResult:
    scores: np.ndarray
    partition: Partition
    verifier_result: VerifierResult
    confident_pos_ids: List[Any] = field(default_factory=list)   # rows[idx] for idx in partition.confident_pos
    uncertain_ids:     List[Any] = field(default_factory=list)
    confident_neg_ids: List[Any] = field(default_factory=list)
    bq_yes_ids:        List[Any] = field(default_factory=list)   # subset of uncertain that BQ said TRUE
    timings_s: dict = field(default_factory=dict)                # signal_compute / partition / verify

    def positive_ids(self) -> set:
        """The cascade's final positive set: confident_pos ∪ bq_yes."""
        return set(self.confident_pos_ids) | set(self.bq_yes_ids)

    @property
    def total_wall_s(self) -> float:
        """End-to-end wall (signal + band + verify); skips overlapping breakdowns."""
        return (self.timings_s.get("signal_compute", 0.0)
              + self.timings_s.get("band_partition", 0.0)
              + self.timings_s.get("verify_total",   0.0))

    def to_dict(self) -> dict:
        return {
            "partition": self.partition.to_dict(),
            "verifier": self.verifier_result.to_dict(),
            "confident_pos_ids": list(self.confident_pos_ids),
            "uncertain_ids": list(self.uncertain_ids),
            "bq_yes_ids": list(self.bq_yes_ids),
            "timings_s": self.timings_s,
        }


@dataclass
class Cascade:
    embeddings: np.ndarray             # (n, d)
    ids:        Sequence[Any]          # length n
    signal:     Signal
    band:       Band
    verifier:   BqVerifier

    def __post_init__(self):
        if len(self.ids) != self.embeddings.shape[0]:
            raise ValueError(
                f"ids/embeddings mismatch: {len(self.ids)} vs {self.embeddings.shape[0]}"
            )

    def run(self, client, per_row_cost: float) -> CascadeResult:
        t = time.time()
        scores = self.signal.compute(self.embeddings)
        t_signal = time.time() - t

        t = time.time()
        part = self.band.partition(scores)
        t_band = time.time() - t

        ids_arr = list(self.ids)
        confident_pos_ids = [ids_arr[i] for i in part.confident_pos.tolist()]
        uncertain_ids     = [ids_arr[i] for i in part.uncertain.tolist()]
        confident_neg_ids = [ids_arr[i] for i in part.confident_neg.tolist()]

        t = time.time()
        vres = self.verifier.verify(client, uncertain_ids, per_row_cost)
        t_verify = time.time() - t

        # Don't intersect with uncertain_ids — verifier may return a transformed
        # namespace (e.g. wildlife Q5 returns Cities, not input URIs). Caller
        # decides what to do with the BQ result set.
        bq_yes = list(vres.positive_ids)

        return CascadeResult(
            scores=scores, partition=part, verifier_result=vres,
            confident_pos_ids=confident_pos_ids,
            uncertain_ids=uncertain_ids,
            confident_neg_ids=confident_neg_ids,
            bq_yes_ids=bq_yes,
            timings_s={
                "signal_compute": t_signal,
                "band_partition": t_band,
                "verify_total":   t_verify,
                "verify_ctas":    vres.ctas_wall_s,
                "verify_bq":      vres.wall_s,
            },
        )
