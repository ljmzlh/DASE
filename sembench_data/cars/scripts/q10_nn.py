"""
Cars Q10 — DASE-only (no BigQuery): 24-class component classification.

NL: For each complaint, predict the problematic car component (24 classes).
GT: cars/ground_truth/Q10.csv (19,657 (car_id, problem_category) rows).
Eval: macro F1 across 24 classes.

Aligns with paper §5.1: anchor argmax via embedding distance — pick the
nearest category prompt for each complaint. Same anchors and template as
the Q10 cascade (`cars/scripts/q10_cascade.py`).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import ConfidenceMarginSignal
from generic_evaluator import GenericEvaluator

CARS_DIR        = os.path.abspath(os.path.join(_HERE, ".."))
TEXT_PARQUET    = os.path.join(CARS_DIR, "data", "text_complaints.parquet")
GT_CSV          = os.path.join(CARS_DIR, "ground_truth", "Q10.csv")
EMBED_USAGE     = os.path.join(CARS_DIR, "cache", "embed_checkpoints", "embed_usage.json")

CATEGORIES = [
    "electrical system", "power train", "engine", "steering", "service brakes",
    "structure", "air bags", "engine and engine cooling", "vehicle speed control",
    "visibility/wiper", "fuel/propulsion system", "forward collision avoidance",
    "exterior lighting", "suspension", "fuel system", "visibility", "wheels",
    "seat belts", "back over prevention", "tires", "seats", "latches/locks/linkages",
    "lane departure", "equipment",
]
ANCHOR_PROMPTS = [f"car problem with {c}" for c in CATEGORIES]


def main():
    df = pd.read_parquet(TEXT_PARQUET)
    emb = np.array(df["embedding"].tolist(), dtype=np.float32)
    gt_df = pd.read_csv(GT_CSV)
    if "index" in gt_df.columns:
        gt_df = gt_df.drop(columns=["index"])

    print(f"Total complaints: {len(df)};  GT rows: {len(gt_df)};  K={len(CATEGORIES)} classes")

    signal = ConfidenceMarginSignal(anchors=ANCHOR_PROMPTS)
    _ = signal.compute(emb)
    pred_cats = [CATEGORIES[i] for i in signal.last_argmax]

    sys_df = pd.DataFrame({"car_id": df["car_id"].values, "problem_category": pred_cats})
    sys_df["problem_category"] = sys_df["problem_category"].astype(str).str.lower().str.replace("\n", "")
    gt_norm = gt_df.copy()
    gt_norm["problem_category"] = gt_norm["problem_category"].astype(str).str.lower().str.replace("\n", "")

    f1 = GenericEvaluator.compute_f1_score_classify(
        gt_norm, sys_df, result_column="problem_category", id_column="car_id",
    )
    print(f"\nPredicted top categories:\n{sys_df['problem_category'].value_counts().head(10).to_string()}")
    print(f"\n[SemBench] macro F1={f1:.4f}")

    if os.path.exists(EMBED_USAGE):
        with open(EMBED_USAGE) as f:
            embed_usage = json.load(f)
        embed_cost = embed_usage.get("text_complaint", {}).get("est_cost_usd", 0.0)
        print(f"\n=== Cost ===  embedding=${embed_cost:.4f}  total=${embed_cost:.4f}")


if __name__ == "__main__":
    main()
