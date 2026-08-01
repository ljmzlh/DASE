#!/usr/bin/env -S python -u
"""Derive Movie Q4's positivity ratio from the DASE+BigQuery Q3 profile."""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import write_profile

MOVIE_DIR = os.path.abspath(os.path.join(_HERE, ".."))
PROFILE_DIR = os.path.join(MOVIE_DIR, "outputs")
Q3_PATH = os.path.join(PROFILE_DIR, "Q3.json")
Q4_PATH = os.path.join(PROFILE_DIR, "Q4.json")
PROMPT = "Determine if the following movie review is clearly positive, review: "
MOVIE_ID = "taken_3"


def main() -> None:
    with open(Q3_PATH, encoding="utf-8") as stream:
        q3 = json.load(stream)

    total = q3["data"]["n_reviews_in_scope"]
    ground_truth_positive = q3["data"]["n_gt_positive_in_scope"]
    cascade_count = q3["cascade"]["cascade_count"]
    ground_truth_ratio = ground_truth_positive / total
    cascade_ratio = cascade_count / total
    relative_error = abs(cascade_ratio - ground_truth_ratio) / ground_truth_ratio
    score = 1.0 / (1.0 + relative_error)

    profile = {
        "scenario": "movie",
        "query_id": 4,
        "scale_factor": q3.get("scale_factor", 2000),
        "model": q3.get("model", "gemini-2.5-flash"),
        "prompt": PROMPT,
        "structural_filter": f"id = '{MOVIE_ID}'",
        "params": q3.get("params", {}),
        "dase_prompts": q3.get("dase_prompts", {}),
        "cascade_form": (
            "DASE F-cascade derived from Q3; positivity ratio equals "
            "Q3 cascade_count divided by the shared scope size."
        ),
        "data": q3["data"],
        "dase_breakdown": q3.get("dase_breakdown", {}),
        "dase_partition": q3.get("dase_partition", {}),
        "calibration": q3["calibration"],
        "cascade": {
            "method": "Q3 DASE cascade with ratio aggregation",
            "verifier": q3["cascade"].get("verifier", {}),
            "cascade_ratio": cascade_ratio,
            "cascade_count": cascade_count,
            "cascade_count_breakdown": q3["cascade"].get(
                "cascade_count_breakdown", {}
            ),
            "score": {"relative_error": relative_error, "score": score},
            "totals": q3["cascade"]["totals"],
        },
    }
    write_profile(profile, Q4_PATH)
    print(f"Q4 DASE cascade derived from Q3: score={score:.4f}")


if __name__ == "__main__":
    main()
