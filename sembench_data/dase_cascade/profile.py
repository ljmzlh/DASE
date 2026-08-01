"""Build and write measured DASE cascade profiles.

This module provides the shared profile metadata and JSON serialization used
by the SemBench cascade scripts:

  prof = build_profile(scenario, query_id, scale_factor, ...)
  prof["calibration"] = cal.to_dict()
  prof["cascade"]     = ...
  write_profile(prof, "outputs/Q5.json")
"""
import json
import os
from typing import Any, Dict, Optional


def build_profile(
    scenario: str, query_id, scale_factor: int,
    *, model: str = "gemini-2.5-flash",
    prompt: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    cascade_form: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Initialize the standard profile dict shape used by all sembench cascades."""
    prof: Dict[str, Any] = {
        "scenario": scenario,
        "query_id": query_id,
        "scale_factor": scale_factor,
        "model": model,
        "thinking_budget_for_calibration": 0,
    }
    if prompt is not None:
        prof["prompt"] = prompt
    if params is not None:
        prof["params"] = params
    if cascade_form is not None:
        prof["cascade_form"] = cascade_form
    if extra:
        prof.update(extra)
    return prof


def write_profile(profile: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(profile, f, indent=2, default=_json_default)
    print(f"\nProfile saved to {path}")


def _json_default(o):
    """Coerce numpy / set / non-JSON types."""
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")
