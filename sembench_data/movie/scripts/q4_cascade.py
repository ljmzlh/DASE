#!/usr/bin/env -S python -u
"""
Movie Q4 cascade v2 — DERIVED FROM Q3 (text F + ratio aggregation).

Q3 and Q4 share the same sem_filter (id='taken_3' AND
AI.IF('clearly positive', reviewText)) over the same scope (120 rows).
The only difference is the aggregation:
  Q3 returns COUNT(*)            -> integer count
  Q4 returns SUM/COUNT as ratio  -> count / 120

Because the BQ work (LLM evaluations) is identical, we DO NOT re-run BQ
for Q4. We load outputs/Q3.json (produced by q3_cascade_v2.py — same
schema as the original) and transform:
  - replace count with ratio (count / scope_size)
  - recompute relative_error and score against GT ratio
  - rewrite the SQL strings to Q4 forms (verbatim from sembench)

This v2 swaps the original ad-hoc paths for dase_cascade.profile + write_profile
but keeps the derived-from-Q3 logic intact (Q4 is `F (alpha-band → AI.IF)`
identical to Q3 except for the final aggregation form).
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import write_profile, print_summary

MOVIE_DIR    = os.path.abspath(os.path.join(_HERE, ".."))
PROFILE_DIR  = os.path.join(MOVIE_DIR, "outputs")
Q3_PATH      = os.path.join(PROFILE_DIR, "Q3.json")
Q4_PATH      = os.path.join(PROFILE_DIR, "Q4.json")

PAPER_BQ_Q4 = {"score": 0.64, "latency_s": 2.8, "cost_usd": 0.005}
PROMPT      = "Determine if the following movie review is clearly positive, review: "
MOVIE_ID    = "taken_3"

Q4_BASELINE_SQL = f"""
SELECT CAST(SUM(CASE WHEN AI.IF(
  ('{PROMPT}', r.reviewText),
  connection_id => 'us.connection',
  endpoint => 'gemini-2.5-flash'
) THEN 1 ELSE 0 END) AS FLOAT64) / COUNT(*) AS positivity_ratio
FROM movie.reviews AS r
WHERE r.id = '{MOVIE_ID}'
"""

Q4_CASCADE_SQL = f"""
-- conceptual: dase confident_pos count + (BQ AI.IF on uncertain) count, divided by scope size.
-- Implemented in Python via Q3's cascade_count (same per-row LLM work).
"""


def main():
    with open(Q3_PATH) as f:
        q3 = json.load(f)

    n_total = q3["data"]["n_reviews_in_scope"]
    n_gt_pos = q3["data"]["n_gt_positive_in_scope"]
    gt_ratio = n_gt_pos / n_total

    bcount = q3["baseline"]["result_count"]
    ccount = q3["cascade"]["cascade_count"]
    per_row = q3["calibration"]["per_row_cost_usd"]

    bratio = bcount / n_total
    cratio = ccount / n_total
    brel = abs(bratio - gt_ratio) / gt_ratio
    crel = abs(cratio - gt_ratio) / gt_ratio
    bscore = 1.0 / (1.0 + brel)
    cscore = 1.0 / (1.0 + crel)

    # Cost / call inheritance — extract from whichever schema Q3 used (v2 vs original)
    bcalls = q3["baseline"]["cost_breakdown"]["n_llm_calls"]
    bcost  = q3["baseline"]["cost_breakdown"]["total_cost_usd"]
    ccalls = q3["cascade"]["totals"]["n_llm_calls"]
    ccost  = q3["cascade"]["totals"]["cost_usd"]

    q4 = {
        "scenario": "movie",
        "query_id": 4,
        "scale_factor": q3.get("scale_factor", 2000),
        "model": q3.get("model", "gemini-2.5-flash"),
        "thinking_budget_for_calibration": q3.get("thinking_budget_for_calibration", 0),
        "prompt": PROMPT,
        "structural_filter": f"id = '{MOVIE_ID}'",
        "params": q3.get("params", {}),
        "dase_prompts": q3.get("dase_prompts", {}),
        "cascade_form": (
            "F-cascade (derived from Q3): MarginSignal + AlphaBand + AiIfVerifier; "
            "cascade_ratio = (|dase confident_pos| + BQ_pos_on_uncertain) / scope_size"
        ),
        "_derivation_note": (
            "Q4 derived from Q3.json. Q3 and Q4 share the same sem_filter and scope; only "
            "aggregation form differs (Q3=COUNT, Q4=SUM/COUNT). BQ work is identical, so we "
            "transform Q3 results: ratio = count / scope_size; score recomputed against GT "
            "ratio. Baseline SQL string is Q4 verbatim form (model_params stripped). "
            "wall_s and slot_ms inherit from Q3."
        ),

        "data": q3["data"],
        "dase_breakdown": q3.get("dase_breakdown", {}),
        "dase_partition": q3.get("dase_partition", {}),
        "calibration": q3["calibration"],

        "baseline": {
            "method": "sembench bigquery/Q4.sql verbatim on movie.reviews (model_params stripped) — DERIVED FROM Q3 BASELINE",
            "sql": Q4_BASELINE_SQL.strip(),
            "result_ratio": bratio,
            "result_count": bcount,
            "score": {"relative_error": brel, "score": bscore},
            "latency_breakdown": q3["baseline"].get("latency_breakdown", {}),
            "cost_breakdown": {
                "n_llm_calls": bcalls,
                "n_llm_calls_method": "scope size (Q4 no LIMIT, all rows evaluated)",
                "per_row_cost_usd": per_row,
                "total_cost_usd": bcost,
            },
        },

        "cascade": {
            "method": (
                "F-cascade derived from Q3 (same Cascade(MarginSignal, AlphaBand, AiIfVerifier).run() output); "
                "ratio = cascade_count / scope_size"
            ),
            "verifier": q3["cascade"].get("verifier", {}),
            "cascade_ratio": cratio,
            "cascade_count": ccount,
            "cascade_count_breakdown": q3["cascade"].get("cascade_count_breakdown", {}),
            "score": {"relative_error": crel, "score": cscore},
            "totals": q3["cascade"]["totals"],
        },

        "comparison": {
            "score":       {"paper": PAPER_BQ_Q4["score"],   "baseline": bscore, "cascade": cscore},
            "wall_s":      q3.get("comparison", {}).get("wall_s", {}),
            "slot_ms_bq":  q3.get("comparison", {}).get("slot_ms_bq", {}),
            "cost_usd":    {"paper": PAPER_BQ_Q4["cost_usd"], "baseline": bcost, "cascade": ccost},
            "n_llm_calls": {
                "paper_implied": round(PAPER_BQ_Q4["cost_usd"] / per_row) if per_row else None,
                "baseline": bcalls,
                "cascade": ccalls,
            },
        },
    }

    write_profile(q4, Q4_PATH)
    print(f"Q4 derived from Q3.")

    print_summary(
        "Movie Q4 (derived from Q3)",
        columns=["paper", "baseline", "cascade"],
        rows=[
            ("score",      [PAPER_BQ_Q4["score"], bscore, cscore], ".2f"),
            ("ratio",      [None, bratio, cratio], ".4f"),
            ("cost ($)",   [PAPER_BQ_Q4["cost_usd"], bcost, ccost], ".4f"),
            ("#LLM calls", [None, bcalls, ccalls], "d"),
        ],
    )


if __name__ == "__main__":
    main()
