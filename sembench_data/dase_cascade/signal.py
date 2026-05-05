"""Signal — cheap proxy score over rows, used to decide what to send to BQ.

  MarginSignal       row-level: mean(pos_sim) − mean(neg_sim).  Used by F.
  RoleMarginSignal   row-level multi-class: per-role margin = sim(this) − mean(sim(other roles)).
  PairCosineSignal   pair-level cosine sim (lazy: caller asks for sim(i, j)).  Used by J.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from dase_cascade.runtime import embed_query, cosine_sim_batch


class Signal(ABC):
    """A per-row score: rows × embeddings → np.ndarray of shape (n,)."""

    @abstractmethod
    def compute(self, embeddings: np.ndarray) -> np.ndarray:
        ...


@dataclass
class MarginSignal(Signal):
    """Standard contrastive margin: mean(pos cos sim) − mean(neg cos sim).

    Used by all `F` Qs (zebra/elephant/comedy/crash/...).
    """
    positive_prompts: Sequence[str]
    negative_prompts: Sequence[str]
    embed_dim: int = 3072
    # Internal: cached after first compute
    _pos_embs: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _neg_embs: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def compute(self, embeddings: np.ndarray) -> np.ndarray:
        if self._pos_embs is None:
            self._pos_embs = embed_query(self.positive_prompts, dim=self.embed_dim)
            self._neg_embs = embed_query(self.negative_prompts, dim=self.embed_dim)
        pos = np.mean([cosine_sim_batch(p, embeddings) for p in self._pos_embs], axis=0)
        neg = np.mean([cosine_sim_batch(n, embeddings) for n in self._neg_embs], axis=0)
        return pos - neg

    def to_dict(self) -> dict:
        return {"type": "margin", "positive": list(self.positive_prompts),
                "negative": list(self.negative_prompts)}


@dataclass
class RoleMarginSignal(Signal):
    """Per-role contrastive margin used in multi-role classification (ecomm Q10/Q11).

    For target role r: margin[r] = sim(role_prompts[r]) − mean(sim(role_prompts[other]))
    Caller picks one target_role to compute(); use multiple instances for multi-role.
    """
    role_prompts: Dict[str, str]
    target_role: str
    embed_dim: int = 3072
    _embs: Optional[Dict[str, np.ndarray]] = field(default=None, init=False, repr=False)

    def compute(self, embeddings: np.ndarray) -> np.ndarray:
        if self.target_role not in self.role_prompts:
            raise ValueError(f"target_role={self.target_role!r} not in role_prompts keys")
        if self._embs is None:
            self._embs = {}
            for r, pr in self.role_prompts.items():
                e = embed_query([pr], dim=self.embed_dim)[0]
                e /= np.linalg.norm(e) + 1e-12
                self._embs[r] = e
        # caller normalizes embeddings? we don't assume; cosine_sim_batch handles it
        sims = {r: cosine_sim_batch(self._embs[r], embeddings) for r in self.role_prompts}
        others = [r for r in self.role_prompts if r != self.target_role]
        neg = np.mean([sims[o] for o in others], axis=0)
        return sims[self.target_role] - neg

    def to_dict(self) -> dict:
        return {"type": "role_margin", "target_role": self.target_role,
                "role_prompts": dict(self.role_prompts)}


@dataclass
class ConfidenceMarginSignal(Signal):
    """Multi-class anchor argmax + top1−top2 confidence.

    For each row, predicts class = argmax_k cos(anchors[k], emb_row), and
    returns the confidence margin (top1 − top2). Used by `C`-style queries
    (e.g. cars Q10 24-class car-component classification) and by every
    "anchor-argmax" nn script (mmqa Q2a/Q2b, ecomm Q2/Q5/Q6/Q12, movie Q9/Q10,
    wildlife multi-prompt Qs).

    Side-channel: `last_argmax` exposes the per-row predicted class index
    after compute(). Caller maps idx → label.
    """
    anchors: Sequence[str]
    embed_dim: int = 3072
    last_argmax: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _anchor_embs: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def compute(self, embeddings: np.ndarray) -> np.ndarray:
        if self._anchor_embs is None:
            self._anchor_embs = embed_query(self.anchors, dim=self.embed_dim)
        # (K, n) sim matrix → (n, K)
        q = self._anchor_embs / (np.linalg.norm(self._anchor_embs, axis=1, keepdims=True) + 1e-12)
        d = embeddings        / (np.linalg.norm(embeddings,        axis=1, keepdims=True) + 1e-12)
        sims = (q @ d.T).T
        sorted_sims = np.sort(sims, axis=1)
        self.last_argmax = np.argmax(sims, axis=1)
        return sorted_sims[:, -1] - sorted_sims[:, -2]

    def to_dict(self) -> dict:
        return {"type": "confidence_margin", "anchors": list(self.anchors)}


@dataclass
class PairCosineSignal:
    """Pair-level cosine sim — for `J` (semantic join). Not a row-level Signal:
    caller asks for sim(i, j) explicitly. Embeddings are L2-normalized lazily.
    """
    embeddings_left: np.ndarray
    embeddings_right: Optional[np.ndarray] = None  # None ⇒ self-join

    def __post_init__(self):
        self._left = self._normalize(self.embeddings_left)
        self._right = self._normalize(self.embeddings_right) if self.embeddings_right is not None else self._left

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    def sim(self, i: int, j: int) -> float:
        return float(self._left[i] @ self._right[j])

    def all_pairs_above(self, indices_left, indices_right, threshold: float):
        """Yield (i, j, sim) for all pairs with sim > threshold. Eager computation;
        caller should keep |left| × |right| reasonable (cascade uses this on the
        already-pruned pool, not the full Cartesian)."""
        L = self._left[indices_left]
        R = self._right[indices_right]
        S = L @ R.T  # (|left|, |right|)
        ii, jj = np.where(S > threshold)
        return [(int(indices_left[a]), int(indices_right[b]), float(S[a, b])) for a, b in zip(ii, jj)]

    def to_dict(self) -> dict:
        return {"type": "pair_cosine"}
