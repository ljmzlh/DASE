#!/usr/bin/env -S python -u
"""
Movie Q8 cascade v2 — DERIVED FROM Q3 (sentiment counts via CASE).

Q3 and Q8 share the same row-level sem_filter (id='taken_3' AND
AI.IF('clearly positive', reviewText)) over the same scope (120 rows).
They only differ in aggregation form:
  Q3 returns COUNT(True rows)               -> integer count
  Q8 returns CASE WHEN ... GROUP BY label   -> (POSITIVE_count, NEGATIVE_count)

Because the per-row LLM work and classifications are identical, we DO NOT
re-run BQ for Q8. We load outputs/Q3.json (produced by q3_cascade_v2.py — same
schema as the original) and transform:
  cascade_POSITIVE = Q3 cascade_count
  cascade_NEGATIVE = scope_size - Q3 cascade_count
  baseline likewise: baseline_POSITIVE = Q3 baseline_count;
                     baseline_NEGATIVE = scope_size - Q3 baseline_count

This v2 swaps the original ad-hoc paths for dase_cascade.profile + write_profile
and keeps the derived-from-Q3 logic intact.
"""
import json
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import write_profile, print_summary
import evaluator as ev

MOVIE_DIR    = os.path.abspath(os.path.join(_HERE, ".."))
PROFILE_DIR  = os.path.join(MOVIE_DIR, "outputs")
Q3_PATH      = os.path.join(PROFILE_DIR, "Q3.json")
Q8_PATH      = os.path.join(PROFILE_DIR, "Q8.json")

PROMPT      = "Determine if the following movie review is clearly positive, review: "
MOVIE_ID    = "taken_3"
PAPER_BQ_Q8 = {"score": 0.78, "latency_s": 13.3, "cost_usd": 0.02}

Q8_BASELINE_SQL = f"""
SELECT
  sentiment_label AS scoreSentiment,
  COUNT(*) AS count
FROM (
  SELECT
    CASE WHEN AI.IF(
      ('{PROMPT}', r.reviewText),
      connection_id => 'us.connection',
      endpoint => 'gemini-2.5-flash'
    ) THEN 'POSITIVE' ELSE 'NEGATIVE' END AS sentiment_label
  FROM movie.reviews AS r
  WHERE r.id = '{MOVIE_ID}'
) AS sentiment_results
GROUP BY sentiment_label
"""


def main():
    with open(Q3_PATH) as f:
        q3 = json.load(f)

    n_total = q3["data"]["n_reviews_in_scope"]
    per_row = q3["calibration"]["per_row_cost_usd"]

    gt_df = ev.get_ground_truth(8)
    gt_counts = {}
    for _, row in gt_df.iterrows():
        gt_counts[str(row.iloc[0]).strip().upper()] = int(row.iloc[1])
    n_gt_pos = gt_counts.get("POSITIVE", 0)
    n_gt_neg = gt_counts.get("NEGATIVE", 0)

    baseline_pos = q3["baseline"]["result_count"]
    baseline_neg = n_total - baseline_pos
    cascade_pos  = q3["cascade"]["cascade_count"]
    cascade_neg  = n_total - cascade_pos

    baseline_sys_df = pd.DataFrame({"scoreSentiment": ["POSITIVE", "NEGATIVE"], "count": [baseline_pos, baseline_neg]})
    cascade_sys_df  = pd.DataFrame({"scoreSentiment": ["POSITIVE", "NEGATIVE"], "count": [cascade_pos, cascade_neg]})
    bm = ev._sentiment_counts(baseline_sys_df, gt_df)
    cm = ev._sentiment_counts(cascade_sys_df, gt_df)
    bscore = 1.0 / (1.0 + bm.relative_error)
    cscore = 1.0 / (1.0 + cm.relative_error)

    bcalls = q3["baseline"]["cost_breakdown"]["n_llm_calls"]
    bcost  = q3["baseline"]["cost_breakdown"]["total_cost_usd"]
    ccalls = q3["cascade"]["totals"]["n_llm_calls"]
    ccost  = q3["cascade"]["totals"]["cost_usd"]

    q8 = {
        "scenario": "movie",
        "query_id": 8,
        "scale_factor": q3.get("scale_factor", 2000),
        "model": q3.get("model", "gemini-2.5-flash"),
        "thinking_budget_for_calibration": q3.get("thinking_budget_for_calibration", 0),
        "prompt": PROMPT,
        "structural_filter": f"id = '{MOVIE_ID}'",
        "params": q3.get("params", {}),
        "dase_prompts": q3.get("dase_prompts", {}),
        "cascade_form": (
            "F-cascade derived from Q3 (same Cascade(MarginSignal, AlphaBand, AiIfVerifier).run()): "
            "cascade_POSITIVE = Q3 cascade_count; cascade_NEGATIVE = scope_size - cascade_POSITIVE."
        ),
        "_derivation_note": (
            "Q8 derived from Q3.json. Q3 and Q8 share the same row-level sem_filter on "
            "(id='taken_3', 'clearly positive') over the same 120-row scope; they differ "
            "only in aggregation form (Q3 = COUNT(True rows); Q8 = CASE WHEN ... GROUP BY "
            "label → POSITIVE / NEGATIVE counts). Per-row LLM work is identical, so Q8 = Q3 "
            "+ (scope_size - Q3_count) for the NEGATIVE bucket. BQ was NOT re-run; "
            "latency/cost/slot_ms inherit from Q3."
        ),

        "data": dict(q3["data"], n_gt_negative_in_scope=n_gt_neg),
        "dase_breakdown": q3.get("dase_breakdown", {}),
        "dase_partition": q3.get("dase_partition", {}),
        "calibration": q3["calibration"],

        "baseline": {
            "method": "sembench bigquery/Q8.sql verbatim on movie.reviews — DERIVED FROM Q3 BASELINE",
            "sql": Q8_BASELINE_SQL.strip(),
            "result_counts": {"POSITIVE": baseline_pos, "NEGATIVE": baseline_neg},
            "score": {"relative_error": bm.relative_error, "score": bscore,
                      "_per_class_rel_err": {
                          "POSITIVE": abs(baseline_pos - n_gt_pos) / n_gt_pos if n_gt_pos else 0.0,
                          "NEGATIVE": abs(baseline_neg - n_gt_neg) / n_gt_neg if n_gt_neg else 0.0,
                      }},
            "latency_breakdown": q3["baseline"].get("latency_breakdown", {}),
            "cost_breakdown": {
                "n_llm_calls": bcalls,
                "n_llm_calls_method": "scope size (Q8 no LIMIT, all rows evaluated)",
                "per_row_cost_usd": per_row,
                "total_cost_usd": bcost,
            },
        },

        "cascade": {
            "method": (
                "F-cascade derived from Q3 (same Cascade run); cascade_POSITIVE = Q3 cascade_count; "
                "cascade_NEGATIVE = scope_size - cascade_POSITIVE."
            ),
            "verifier": q3["cascade"].get("verifier", {}),
            "result_counts": {"POSITIVE": cascade_pos, "NEGATIVE": cascade_neg},
            "result_breakdown": {
                "dase_confident_pos":   q3["cascade"]["cascade_count_breakdown"]["dase_confident_pos"],
                "bq_pos_on_uncertain":  q3["cascade"]["cascade_count_breakdown"]["bq_uncertain_pos"],
                "dase_confident_neg":   (n_total - q3["dase_partition"].get("n_uncertain", 0)
                                          - q3["cascade"]["cascade_count_breakdown"]["dase_confident_pos"]),
                "bq_neg_on_uncertain":  (q3["dase_partition"].get("n_uncertain", 0)
                                          - q3["cascade"]["cascade_count_breakdown"]["bq_uncertain_pos"]),
            },
            "score": {"relative_error": cm.relative_error, "score": cscore,
                      "_per_class_rel_err": {
                          "POSITIVE": abs(cascade_pos - n_gt_pos) / n_gt_pos if n_gt_pos else 0.0,
                          "NEGATIVE": abs(cascade_neg - n_gt_neg) / n_gt_neg if n_gt_neg else 0.0,
                      }},
            "totals": q3["cascade"]["totals"],
        },

        "comparison": {
            "score":      {"paper": PAPER_BQ_Q8["score"],   "baseline": bscore, "cascade": cscore},
            "wall_s":     q3.get("comparison", {}).get("wall_s", {}),
            "slot_ms_bq": q3.get("comparison", {}).get("slot_ms_bq", {}),
            "cost_usd":   {"paper": PAPER_BQ_Q8["cost_usd"], "baseline": bcost, "cascade": ccost,
                           "_paper_note": "Q8 reuses Q3 cascade (identical per-row LLM work); cost reported from our derivation."},
            "n_llm_calls": {
                "paper_implied": round(PAPER_BQ_Q8["cost_usd"] / per_row) if per_row else None,
                "baseline": bcalls,
                "cascade": ccalls,
            },
        },
    }

    write_profile(q8, Q8_PATH)
    print(f"Q8 derived from Q3.")

    print_summary(
        "Movie Q8 (derived from Q3)",
        columns=["paper", "baseline", "cascade"],
        rows=[
            ("score",      [PAPER_BQ_Q8["score"], bscore, cscore], ".2f"),
            ("POS",        [n_gt_pos, baseline_pos, cascade_pos], "d"),
            ("NEG",        [n_gt_neg, baseline_neg, cascade_neg], "d"),
            ("cost ($)",   [PAPER_BQ_Q8["cost_usd"], bcost, ccost], ".4f"),
            ("#LLM calls", [None, bcalls, ccalls], "d"),
        ],
    )


if __name__ == "__main__":
    main()
