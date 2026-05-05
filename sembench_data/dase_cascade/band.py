"""Band — partition a score array into {confident_pos, uncertain, confident_neg}.

Three rules cover all existing cascade scripts:

  AlphaBand(α)         bottom α-fraction of |score| → uncertain;  rest split by sign
                       (used by wildlife / cars / movie q3-4 / mmqa q3a-…  — the dominant)
  AbsoluteBand(lo, hi) score < lo → neg, score > hi → pos, else → uncertain
                       (used by ecomm q2 / q4 / q5 / q13 / q14)
  TopKBand(k)          top-K by score → uncertain; everything else → neg
                       (used by movie q1 / q5-7 — semantic retrieval)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import List

import numpy as np


@dataclass(frozen=True)
class Partition:
    """Three disjoint index sets of a score array. Indices into the input rows."""
    confident_pos: np.ndarray   # int64
    uncertain:     np.ndarray
    confident_neg: np.ndarray

    def to_dict(self) -> dict:
        return {
            "n_confident_pos": int(self.confident_pos.size),
            "n_uncertain":     int(self.uncertain.size),
            "n_confident_neg": int(self.confident_neg.size),
        }


class Band(ABC):
    @abstractmethod
    def partition(self, scores: np.ndarray) -> Partition:
        ...

    @abstractmethod
    def to_dict(self) -> dict:
        ...


@dataclass
class AlphaBand(Band):
    """Bottom-α by |score| → uncertain; remaining: score>0 → pos, score<0 → neg."""
    alpha: float

    def partition(self, scores: np.ndarray) -> Partition:
        n = scores.size
        n_uncertain = int(round(self.alpha * n))
        order = np.argsort(np.abs(scores))
        uncertain = np.sort(order[:n_uncertain])
        confident_mask = np.ones(n, dtype=bool)
        confident_mask[uncertain] = False
        pos = np.where(confident_mask & (scores > 0))[0]
        neg = np.where(confident_mask & ~(scores > 0))[0]
        return Partition(confident_pos=pos, uncertain=uncertain, confident_neg=neg)

    def to_dict(self) -> dict:
        return {"type": "alpha", "alpha": self.alpha}


@dataclass
class AbsoluteBand(Band):
    """score > tau_high → pos, score < tau_low → neg, else uncertain."""
    tau_low: float
    tau_high: float

    def partition(self, scores: np.ndarray) -> Partition:
        pos = np.where(scores > self.tau_high)[0]
        neg = np.where(scores < self.tau_low)[0]
        unc_mask = np.ones(scores.size, dtype=bool)
        unc_mask[pos] = False
        unc_mask[neg] = False
        uncertain = np.where(unc_mask)[0]
        return Partition(confident_pos=pos, uncertain=uncertain, confident_neg=neg)

    def to_dict(self) -> dict:
        return {"type": "absolute", "tau_low": self.tau_low, "tau_high": self.tau_high}


@dataclass
class TopKBand(Band):
    """Top-K by score → uncertain (sent to BQ for verification);
    all other rows → confident_neg. Used for `R` (semantic retrieval) Qs.
    No confident_pos because retrieval explicitly defers verdict to LLM."""
    k: int

    def partition(self, scores: np.ndarray) -> Partition:
        n = scores.size
        order = np.argsort(-scores)
        uncertain = np.sort(order[:self.k])
        all_idx = np.arange(n)
        neg = np.setdiff1d(all_idx, uncertain, assume_unique=True)
        return Partition(
            confident_pos=np.array([], dtype=np.int64),
            uncertain=uncertain,
            confident_neg=neg,
        )

    def to_dict(self) -> dict:
        return {"type": "topk", "k": self.k}
