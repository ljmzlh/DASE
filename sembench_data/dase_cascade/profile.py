"""Profile builder + write + 4-column comparison printer.

Existing scripts hand-roll a JSON with ~10 standard fields and a print-summary
block. This module gives:

  prof = build_profile(scenario, query_id, scale_factor, ...)
  prof["calibration"] = cal.to_dict()
  prof["baseline"]    = ...
  prof["cascade"]     = ...
  prof["comparison"]  = ...
  write_profile(prof, "outputs/Q5.json")
  print_summary("Wildlife Q5", paper={...}, baseline={...}, cascade={...})

Schema is identical to existing outputs/QN.json files so eval notebooks /
build_cascade_summary.py keep working unchanged.
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


def print_summary(
    title: str,
    *, columns,                     # list[str], e.g. ["paper BQ", "ours BQ", "ours cascade"]
    rows,                           # list[(label, [v1, v2, ...]) or (label, [v1, ...], fmt)]
) -> None:
    """Render a small comparison table — intentionally minimal vs the
    bespoke string-formatting blocks in each script."""
    print(f"\n=== Summary — {title} ===")
    headers = ["metric"] + list(columns)
    print("  ".join(f"{h:>14}" for h in headers))
    for entry in rows:
        if len(entry) == 2:
            label, vals = entry
            fmt = "{}"
        else:
            label, vals, fmt = entry
        cells = [label] + [_fmt_cell(v, fmt) for v in vals]
        print("  ".join(f"{c:>14}" for c in cells))


def _fmt_cell(v, fmt: str) -> str:
    if v is None:
        return "-"
    try:
        return format(v, fmt) if fmt and fmt != "{}" else format(v)
    except (ValueError, TypeError):
        return str(v)
