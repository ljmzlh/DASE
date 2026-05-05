"""
Cars Q1 — DASE-only (no BigQuery): find cars in a crash/accident/collision.

NL: SELECT DISTINCT car_id FROM text_complaints WHERE crash=TRUE.
GT: cars/ground_truth/Q1.csv.
Eval: precision/recall/F1 (set retrieval over car_id).

Aligns with paper §5.1: counterfactual anchors. Predict crash if
mean_sim(positive_prompts) > mean_sim(negative_prompts). Same prompts as
the Q1 cascade — `cars/scripts/q1_cascade.py`.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import MarginSignal, f1_set
from generic_evaluator import GenericEvaluator

CARS_DIR        = os.path.abspath(os.path.join(_HERE, ".."))
TEXT_PARQUET    = os.path.join(CARS_DIR, "data", "text_complaints.parquet")
GT_CSV          = os.path.join(CARS_DIR, "ground_truth", "Q1.csv")
EMBED_USAGE     = os.path.join(CARS_DIR, "cache", "embed_checkpoints", "embed_usage.json")

POSITIVE = [
    "car was in a crash, accident, or collision",
    "vehicle crashed or collided with another object",
    "car was involved in a traffic accident or crash",
]
NEGATIVE = [
    "car had a mechanical or maintenance issue",
    "vehicle had engine, brake, or electrical problems",
    "car needed repair due to wear and tear or defect",
]


def main():
    df = pd.read_parquet(TEXT_PARQUET)
    emb = np.array(df["embedding"].tolist(), dtype=np.float32)
    gt_df = pd.read_csv(GT_CSV)

    print(f"Total complaints: {len(df)};  GT crash cars: {len(gt_df)}")

    margins = MarginSignal(positive_prompts=POSITIVE, negative_prompts=NEGATIVE).compute(emb)
    pred_mask = margins > 0
    pred_car_ids = df.loc[pred_mask, "car_id"].unique()
    sys_df = pd.DataFrame({"car_id": pred_car_ids})

    # Use the SemBench evaluator for the official metric reported in paper Table 3
    score = GenericEvaluator.compute_accuracy_score("precision", gt_df, sys_df, id_column="car_id")
    p, r, f1 = score.precision, score.recall, score.f1_score
    p_set, r_set, f1_set_v = f1_set(set(pred_car_ids), set(gt_df["car_id"]))
    print(f"Predicted crash cars: {len(pred_car_ids)}")
    print(f"[SemBench]  P={p:.4f}  R={r:.4f}  F1={f1:.4f}")
    print(f"[set check] P={p_set:.4f}  R={r_set:.4f}  F1={f1_set_v:.4f}")

    if os.path.exists(EMBED_USAGE):
        with open(EMBED_USAGE) as f:
            embed_usage = json.load(f)
        embed_cost = embed_usage.get("text_complaint", {}).get("est_cost_usd", 0.0)
        print(f"\n=== Cost ===  embedding=${embed_cost:.4f}  caption=$0.0000  total=${embed_cost:.4f}")


if __name__ == "__main__":
    main()
