"""
MMQA Q6b — DASE-only (no BigQuery): airlines with destinations in Germany.

Operator: F (text-only binary classification on tampa_international_airport).
GT: query/natural_language/q6b.json → list of airlines.
Eval: F1 over Airlines sets (GenericEvaluator + f1_set check).

Aligns with paper §5.1: counterfactual anchors. Same MarginSignal (mean(pos) −
mean(neg)) the q6b cascade uses — predict positive iff margin > 0, no LLM
verification on uncertain rows.
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

MMQA_DIR = os.path.abspath(os.path.join(_HERE, ".."))
DATA_DIR = os.path.join(MMQA_DIR, "data")
NL_PATH = os.path.join(MMQA_DIR, "query", "natural_language", "q6b.json")
EMBED_USAGE = os.path.join(MMQA_DIR, "cache", "embed_checkpoints", "embed_usage.json")

# Counterfactual anchors — verbatim from q6b_cascade.py.
POSITIVE_PROMPTS = [
    "The destination list includes at least one city or airport located in Germany.",
    "These destinations mention flying to Germany or a German city such as Berlin, Munich, or Frankfurt.",
    "The carrier serves a destination that is in Germany.",
]
NEGATIVE_PROMPTS = [
    "No destination in this list is in Germany; only non-German cities are listed.",
    "The text describes routes that do not include any German airport or city.",
    "Germany does not appear as a served country in these destinations.",
]


def main():
    df = pd.read_parquet(os.path.join(DATA_DIR, "tampa_international_airport.parquet"))
    with open(NL_PATH) as f:
        gt_airlines = json.load(f)["ground_truth"]
    gt_df = pd.DataFrame({"Airlines": gt_airlines})

    text_emb = np.array(df["embedding"].tolist(), dtype=np.float32)
    margins = MarginSignal(positive_prompts=POSITIVE_PROMPTS,
                           negative_prompts=NEGATIVE_PROMPTS).compute(text_emb)
    pred_mask = margins > 0
    raw = df.loc[pred_mask, "Airlines"].astype(str).str.strip()
    pred_airlines = [a for a in raw.tolist() if a and a.lower() != "nan"]
    pred_df = pd.DataFrame({"Airlines": pred_airlines})

    print(f"Rows: {len(df)}  |  GT airlines: {len(gt_df)}  |  Predicted: {len(pred_df)}")

    score = GenericEvaluator.compute_accuracy_score(
        "f1-score", gt_df, pred_df, id_column="Airlines",
    )
    p_set, r_set, f1_set_v = f1_set(pred_airlines, gt_airlines)
    print(f"[SemBench]  P={score.precision:.4f}  R={score.recall:.4f}  F1={score.f1_score:.4f}")
    print(f"[set check] P={p_set:.4f}  R={r_set:.4f}  F1={f1_set_v:.4f}")

    try:
        with open(EMBED_USAGE) as f:
            t = json.load(f).get("tasks", {}).get("tampa_destinations", {})
        print(f"\nRow embed cost (tampa_destinations): ~${t.get('est_cost_usd', 0):.4f}")
    except OSError:
        pass


if __name__ == "__main__":
    main()
