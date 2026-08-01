#!/usr/bin/env -S python -u
"""Derive Movie Q8 sentiment counts from the DASE+BigQuery Q3 profile."""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import write_profile
import evaluator as ev

MOVIE_DIR = os.path.abspath(os.path.join(_HERE, ".."))
PROFILE_DIR = os.path.join(MOVIE_DIR, "outputs")
Q3_PATH = os.path.join(PROFILE_DIR, "Q3.json")
Q8_PATH = os.path.join(PROFILE_DIR, "Q8.json")
PROMPT = "Determine if the following movie review is clearly positive, review: "
MOVIE_ID = "taken_3"


def main() -> None:
    with open(Q3_PATH, encoding="utf-8") as stream:
        q3 = json.load(stream)

    total = q3["data"]["n_reviews_in_scope"]
    positive = q3["cascade"]["cascade_count"]
    negative = total - positive
    ground_truth = ev.get_ground_truth(8)
    system_result = pd.DataFrame(
        {"scoreSentiment": ["POSITIVE", "NEGATIVE"], "count": [positive, negative]}
    )
    metric = ev._sentiment_counts(system_result, ground_truth)
    score = 1.0 / (1.0 + metric.relative_error)

    profile = {
        "scenario": "movie",
        "query_id": 8,
        "scale_factor": q3.get("scale_factor", 2000),
        "model": q3.get("model", "gemini-2.5-flash"),
        "prompt": PROMPT,
        "structural_filter": f"id = '{MOVIE_ID}'",
        "params": q3.get("params", {}),
        "dase_prompts": q3.get("dase_prompts", {}),
        "cascade_form": (
            "DASE F-cascade derived from Q3; POSITIVE is Q3 cascade_count "
            "and NEGATIVE is the remaining shared scope."
        ),
        "data": q3["data"],
        "dase_breakdown": q3.get("dase_breakdown", {}),
        "dase_partition": q3.get("dase_partition", {}),
        "calibration": q3["calibration"],
        "cascade": {
            "method": "Q3 DASE cascade with sentiment-count aggregation",
            "verifier": q3["cascade"].get("verifier", {}),
            "result_counts": {"POSITIVE": positive, "NEGATIVE": negative},
            "score": {"relative_error": metric.relative_error, "score": score},
            "totals": q3["cascade"]["totals"],
        },
    }
    write_profile(profile, Q8_PATH)
    print(f"Q8 DASE cascade derived from Q3: score={score:.4f}")


if __name__ == "__main__":
    main()
