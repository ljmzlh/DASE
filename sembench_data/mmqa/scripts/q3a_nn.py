"""
MMQA Q3a — DASE-only (no BigQuery): which movies are comedies?

Operator: F (text-only binary classification on lizzy_caplan_text_data, 200 rows).
GT: query/natural_language/q3a.json → list of comedy titles.
Eval: F1 over title sets (GenericEvaluator + f1_set check).

Aligns with paper §5.1: counterfactual anchors. Same MarginSignal (mean(pos) −
mean(neg)) the q3a cascade uses — predict positive iff margin > 0, no LLM
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
NL_PATH = os.path.join(MMQA_DIR, "query", "natural_language", "q3a.json")
EMBED_USAGE = os.path.join(MMQA_DIR, "cache", "embed_checkpoints", "embed_usage.json")

# Counterfactual anchors — verbatim from q3a_cascade.py.
POSITIVE_PROMPTS = [
    "a comedy movie",
    "a humorous comedy film",
    "a movie in the comedy genre",
]
NEGATIVE_PROMPTS = [
    "not a comedy movie",
    "this film is not a comedy and not primarily humorous",
    "not a funny or joke-driven comedy",
]


def main():
    df = pd.read_parquet(os.path.join(DATA_DIR, "lizzy_caplan_text_data.parquet"))
    with open(NL_PATH) as f:
        gt_titles = json.load(f)["ground_truth"]
    gt_df = pd.DataFrame({"title": gt_titles})

    text_emb = np.array(df["embedding"].tolist(), dtype=np.float32)
    margins = MarginSignal(positive_prompts=POSITIVE_PROMPTS,
                           negative_prompts=NEGATIVE_PROMPTS).compute(text_emb)
    pred_mask = margins > 0
    pred_titles = df.loc[pred_mask, "title"].astype(str).tolist()
    pred_df = pd.DataFrame({"title": pred_titles})

    print(f"Rows: {len(df)}  |  GT titles: {len(gt_df)}  |  Predicted: {len(pred_df)}")

    score = GenericEvaluator.compute_accuracy_score(
        "f1-score", gt_df, pred_df, id_column="title",
    )
    p_set, r_set, f1_set_v = f1_set(pred_titles, gt_titles)
    print(f"[SemBench]  P={score.precision:.4f}  R={score.recall:.4f}  F1={score.f1_score:.4f}")
    print(f"[set check] P={p_set:.4f}  R={r_set:.4f}  F1={f1_set_v:.4f}")

    try:
        with open(EMBED_USAGE) as f:
            usage = json.load(f)
        t = usage.get("tasks", {}).get("lizzy_caplan_text", {})
        q_cost = t.get("est_cost_usd", 0.0)
        print("\n=== Cost (row embeddings in embed_usage) ===")
        print(f"  lizzy_caplan_text embed: ~${q_cost:.4f}")
        print("  (+ small on-the-fly QUERY embeds for contrastive prompts)")
    except OSError:
        pass


if __name__ == "__main__":
    main()
