"""
Ecomm Q1 — DASE-only (no BigQuery): Reebok backpacks (text F operator).

NL: find the product ids of products that are backpacks from Reebok.
GT: 7 ids (articleType.typeName='Backpacks' AND brandName='Reebok').
Eval: F1 over id sets.

Aligns with paper §5.1: counterfactual anchors (margin). Decomposed into
two sub-filters (is_backpack × is_reebok); intersection. Same prompts as
the Q1 cascade — `ecomm/scripts/q1_cascade.py`.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import MarginSignal
from generic_evaluator import GenericEvaluator

ECOMM_DIR              = os.path.abspath(os.path.join(_HERE, ".."))
DATA_DIR               = os.path.join(ECOMM_DIR, "data")
GT_DIR                 = os.path.join(ECOMM_DIR, "ground_truth")
STYLES_DETAILS_PARQUET = os.path.join(ECOMM_DIR, "cache", "sf_500", "styles_details.parquet")
EMBED_USAGE            = os.path.join(ECOMM_DIR, "cache", "embed_checkpoints", "embed_usage.json")

# Decomposed sub-filters: same prompts as the Q1 cascade decomposition.
POS_BACKPACK = [
    "a backpack",
    "a fashion backpack product",
    "a bag that is a backpack",
]
NEG_BACKPACK = [
    "a fashion product that is not a backpack",
    "a non-backpack clothing or footwear item",
    "a product that is not a bag",
]

POS_REEBOK = [
    "a Reebok product",
    "a fashion item made by the brand Reebok",
    "Reebok branded apparel or footwear",
]
NEG_REEBOK = [
    "a product not made by Reebok",
    "a non-Reebok fashion product",
    "a product from a brand other than Reebok",
]


def ensure_ground_truth() -> pd.DataFrame:
    gt_path = os.path.join(GT_DIR, "Q1.csv")
    if os.path.isfile(gt_path):
        return pd.read_csv(gt_path)
    os.makedirs(GT_DIR, exist_ok=True)
    src = pd.read_parquet(
        STYLES_DETAILS_PARQUET, columns=["id", "articleType", "brandName"]
    )
    article_type = src["articleType"].apply(
        lambda x: x.get("typeName") if isinstance(x, dict) else None
    )
    mask = (article_type == "Backpacks") & (src["brandName"] == "Reebok")
    gt_df = pd.DataFrame({"Id": src.loc[mask, "id"].astype(np.int64).values})
    gt_df.to_csv(gt_path, index=False)
    print(f"[GT] generated {gt_path}: {len(gt_df)} Reebok backpacks")
    return gt_df


def main():
    df = pd.read_parquet(os.path.join(DATA_DIR, "products_text.parquet"))
    emb_matrix = np.array(df["embedding"].tolist(), dtype=np.float32)
    print(f"Total products: {len(df)} (embedding dim={emb_matrix.shape[1]})")

    gt_df = ensure_ground_truth()
    print(f"Ground truth Reebok backpacks: {len(gt_df)}")
    print()

    # Two sub-filters via MarginSignal; intersect.
    backpack_margin = MarginSignal(POS_BACKPACK, NEG_BACKPACK).compute(emb_matrix)
    reebok_margin   = MarginSignal(POS_REEBOK,   NEG_REEBOK  ).compute(emb_matrix)
    mask_backpack = backpack_margin > 0
    mask_reebok   = reebok_margin > 0
    mask = mask_backpack & mask_reebok

    pred_ids = df.loc[mask, "Id"].astype(np.int64).unique()
    sys_df = pd.DataFrame({"Id": pred_ids})

    score = GenericEvaluator.compute_accuracy_score(
        "f1-score", gt_df, sys_df, id_column="Id"
    )

    print("POS_BACKPACK:", POS_BACKPACK)
    print("NEG_BACKPACK:", NEG_BACKPACK)
    print("POS_REEBOK:  ", POS_REEBOK)
    print("NEG_REEBOK:  ", NEG_REEBOK)
    print()
    print(f"is_backpack predictions: {int(mask_backpack.sum())}")
    print(f"is_reebok   predictions: {int(mask_reebok.sum())}")
    print(f"intersection (Reebok backpacks): {len(pred_ids)}")
    print(f"ground truth Reebok backpacks:   {len(gt_df)}")
    print(
        f"[SemBench] Precision={score.precision:.4f}  "
        f"Recall={score.recall:.4f}  F1={score.f1_score:.4f}"
    )

    # ── Cost ─────────────────────────────────────────────────────────
    if os.path.exists(EMBED_USAGE):
        with open(EMBED_USAGE) as f:
            embed_usage = json.load(f)
        embed_cost = embed_usage.get("tasks", {}).get("product_text", {}).get("est_cost_usd", 0.0)
        total_cost = embed_cost
        print(f"\n=== Cost ===")
        print(f"  Columns used:   Title, Description (products_text.parquet)")
        print(f"  Embedding cost: ${embed_cost:.4f}")
        print(f"  Caption cost:   $0.0000")
        print(f"  Total cost:     ${total_cost:.4f}")


if __name__ == "__main__":
    main()
