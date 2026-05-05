"""
dase_cascade — unified primitives for sembench cascade scripts.

Aligns with paper §2.1 / §5.1:
  cascade = cheap proxy prefilter (DASE)  →  LLM verifier on uncertain (BQ)

Public API:
  Signals:    MarginSignal, RoleMarginSignal, PairCosineSignal
  Bands:      AlphaBand, AbsoluteBand, TopKBand   (→ Partition)
  Verifiers:  AiIfVerifier, AiGenerateVerifier     (BQ stage)
  Solvers:    Cascade, ClusterCascade
  Scoring:    f1_set, relative_error_score, ari_score
  Output:     build_profile, write_profile, print_summary
  Runtime:    embed_query, bq_client, run_query, PRICES, per_row_cost
"""
import os as _os
import sys as _sys

# Patch sys.path so cascade scripts can `from dase_cascade import ...`
# and downstream code can find `tools.llm_tool` at the dase_clean root.
_HERE = _os.path.dirname(_os.path.abspath(__file__))           # sembench_data/dase_cascade/
_SEMBENCH_MY = _os.path.dirname(_HERE)                          # sembench_data/
_DASE_ROOT = _os.path.dirname(_SEMBENCH_MY)                     # dase_clean/
for _p in (_DASE_ROOT, _SEMBENCH_MY):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from dase_cascade.runtime import (
    PRICES, embed_query, bq_client, run_query, cosine_sim_batch,
)
from dase_cascade.calibration import per_row_cost
from dase_cascade.signal import (
    Signal, MarginSignal, RoleMarginSignal, ConfidenceMarginSignal, PairCosineSignal,
)
from dase_cascade.band import Band, AlphaBand, AbsoluteBand, TopKBand, Partition
from dase_cascade.verifier import (
    BqVerifier, AiIfVerifier, AiGenerateVerifier, VerifierResult,
)
from dase_cascade.cascade import Cascade, CascadeResult
from dase_cascade.cluster import ClusterCascade, ClusterCascadeResult
from dase_cascade.score import f1_set, relative_error_score, ari_score
from dase_cascade.profile import build_profile, write_profile, print_summary

__all__ = [
    "PRICES", "embed_query", "bq_client", "run_query", "cosine_sim_batch",
    "per_row_cost",
    "Signal", "MarginSignal", "RoleMarginSignal", "ConfidenceMarginSignal", "PairCosineSignal",
    "Band", "AlphaBand", "AbsoluteBand", "TopKBand", "Partition",
    "BqVerifier", "AiIfVerifier", "AiGenerateVerifier", "VerifierResult",
    "Cascade", "CascadeResult",
    "ClusterCascade", "ClusterCascadeResult",
    "f1_set", "relative_error_score", "ari_score",
    "build_profile", "write_profile", "print_summary",
]
