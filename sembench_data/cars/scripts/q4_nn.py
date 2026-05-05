"""
Cars Q4 — DASE-only (no BigQuery): average age of cars with engine problems.

NL: SELECT 2026 - AVG(year) FROM cars WHERE complaint entails engine problem.
GT: cars/ground_truth/Q4.csv (average_age = 13.4868).
Eval: _aggregation_single (relative_error → score = 1/(1+rel_err)).

Aligns with paper §5.1: counterfactual anchors. Per-complaint margin via
MarginSignal; predict engine problem if margin > 0; aggregate to DISTINCT
car_ids and compute 2026 - AVG(year). Same prompts as the Q4 cascade —
`cars/scripts/q4_cascade.py`.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import MarginSignal, relative_error_score
from evaluator import _aggregation_single

CARS_DIR              = os.path.abspath(os.path.join(_HERE, ".."))
CARS_PARQUET          = os.path.join(CARS_DIR, "data", "cars.parquet")
TEXT_PARQUET          = os.path.join(CARS_DIR, "data", "text_complaints.parquet")
GT_CSV                = os.path.join(CARS_DIR, "ground_truth", "Q4.csv")
EMBED_USAGE           = os.path.join(CARS_DIR, "cache", "embed_checkpoints", "embed_usage.json")

POSITIVE = [
    "complaint about car engine problem",
    "the car has issues with the engine or engine-connected parts",
    "engine malfunction or failure described in the complaint",
]
NEGATIVE = [
    "complaint about brakes, electrical, or non-engine issue",
    "car problem unrelated to the engine",
    "issues with steering, suspension, or other non-engine components",
]


def main():
    text_df = pd.read_parquet(TEXT_PARQUET)
    cars_df = pd.read_parquet(CARS_PARQUET)
    gt_df = pd.read_csv(GT_CSV)

    print(f"Total complaints: {len(text_df)}")
    gt_avg = float(gt_df.iloc[0, 0])
    print(f"Ground truth average_age: {gt_avg:.4f}")

    emb = np.array(text_df["embedding"].tolist(), dtype=np.float32)
    margins = MarginSignal(positive_prompts=POSITIVE, negative_prompts=NEGATIVE).compute(emb)
    pred_mask = margins > 0
    pred_car_ids = text_df.loc[pred_mask, "car_id"].unique()

    pred_cars = cars_df[cars_df["car_id"].isin(pred_car_ids)]
    avg_age = 2026 - pred_cars["year"].mean()

    sys_df = pd.DataFrame({"average_age": [avg_age]})
    result = _aggregation_single(sys_df, gt_df)
    score = relative_error_score(float(avg_age), gt_avg)

    print(f"Predicted engine-problem cars: {len(pred_car_ids)}")
    print(f"Predicted average_age: {avg_age:.4f}")
    print(f"Ground truth average_age: {gt_avg:.4f}")
    print(
        f"[SemBench] RelError={result.relative_error:.4f}  "
        f"AbsError={result.absolute_error:.4f}  Score={score:.4f}"
    )

    # ── Cost ─────────────────────────────────────────────────────────
    if os.path.exists(EMBED_USAGE):
        with open(EMBED_USAGE) as f:
            embed_usage = json.load(f)
        embed_cost = embed_usage.get("text_complaint", {}).get("est_cost_usd", 0.0)
        print(f"\n=== Cost ===")
        print(f"  Columns used: text_complaints (summary) + cars (year)")
        print(f"  Embedding cost: ${embed_cost:.4f}")
        print(f"  Total cost:     ${embed_cost:.4f}")


if __name__ == "__main__":
    main()
