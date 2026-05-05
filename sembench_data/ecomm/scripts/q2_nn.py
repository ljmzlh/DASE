"""
Ecomm Q2 — DASE-only (no BigQuery): yellow+silver sports shoes (image F operator).

NL: find product ids of products where the image shows a pair of sports shoes
    that are predominantly yellow and silver.
GT: 5 ids.
Eval: F1 over id sets.

Aligns with paper §5.1: counterfactual anchors (margin). Decomposed into 3
independent sub-filters (sports-shoes × has-yellow × has-silver) and
AND-combined. Same prompts as the Q2 cascade — `ecomm/scripts/q2_cascade.py`.
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
CAPTION_USAGE          = os.path.join(ECOMM_DIR, "cache", "embed_checkpoints", "image_caption_usage.json")

# Three decomposed sub-filters; same prompts as the Q2 cascade.
POS_SHOES = [
    "a pair of sports shoes",
    "athletic sneakers designed for sports",
    "running or training sports footwear",
]
NEG_SHOES = [
    "a product that is not a pair of sports shoes",
    "a fashion item that is not athletic footwear",
    "an apparel product that is not sports shoes",
]
POS_YELLOW = [
    "a product that has yellow in its color",
    "a fashion item featuring the color yellow",
    "a product whose colors include yellow",
]
NEG_YELLOW = [
    "a product that does not have yellow",
    "a product without any yellow color",
    "a product whose colors do not include yellow",
]
POS_SILVER = [
    "a product that has silver in its color",
    "a fashion item featuring the color silver",
    "a product whose colors include silver",
]
NEG_SILVER = [
    "a product that does not have silver",
    "a product without any silver color",
    "a product whose colors do not include silver",
]


def ensure_ground_truth() -> pd.DataFrame:
    gt_path = os.path.join(GT_DIR, "Q2.csv")
    if os.path.isfile(gt_path):
        return pd.read_csv(gt_path)

    os.makedirs(GT_DIR, exist_ok=True)
    src = pd.read_parquet(
        STYLES_DETAILS_PARQUET,
        columns=["id", "articleType", "baseColour", "colour1", "colour2"],
    )
    article_type = src["articleType"].apply(
        lambda x: x.get("typeName") if isinstance(x, dict) else None
    )

    def _has_both(row) -> bool:
        colors = {
            c for c in (row["baseColour"], row["colour1"], row["colour2"])
            if isinstance(c, str)
        }
        return ("Yellow" in colors) and ("Silver" in colors)

    mask = (article_type == "Sports Shoes") & src.apply(_has_both, axis=1)
    gt_df = pd.DataFrame({"Id": src.loc[mask, "id"].astype(np.int64).values})
    gt_df.to_csv(gt_path, index=False)
    print(f"[GT] generated {gt_path}: {len(gt_df)} yellow+silver sports shoes")
    return gt_df


def main():
    df = pd.read_parquet(os.path.join(DATA_DIR, "products_image.parquet"))
    emb_matrix = np.array(df["embedding"].tolist(), dtype=np.float32)
    print(f"Total products: {len(df)} (embedding dim={emb_matrix.shape[1]})")

    gt_df = ensure_ground_truth()
    print(f"Ground truth yellow+silver sports shoes: {len(gt_df)}")
    print()

    # Three sub-filters via MarginSignal, intersect.
    shoes_margin  = MarginSignal(POS_SHOES,  NEG_SHOES ).compute(emb_matrix)
    yellow_margin = MarginSignal(POS_YELLOW, NEG_YELLOW).compute(emb_matrix)
    silver_margin = MarginSignal(POS_SILVER, NEG_SILVER).compute(emb_matrix)
    mask_shoes  = shoes_margin  > 0
    mask_yellow = yellow_margin > 0
    mask_silver = silver_margin > 0
    mask = mask_shoes & mask_yellow & mask_silver

    pred_ids = df.loc[mask, "Id"].astype(np.int64).unique()
    sys_df = pd.DataFrame({"Id": pred_ids})

    score = GenericEvaluator.compute_accuracy_score(
        "f1-score", gt_df, sys_df, id_column="Id"
    )

    print("POS_SHOES: ", POS_SHOES)
    print("NEG_SHOES: ", NEG_SHOES)
    print("POS_YELLOW:", POS_YELLOW)
    print("NEG_YELLOW:", NEG_YELLOW)
    print("POS_SILVER:", POS_SILVER)
    print("NEG_SILVER:", NEG_SILVER)
    print()
    print(f"is_sports_shoes predictions: {int(mask_shoes.sum())}")
    print(f"has_yellow       predictions: {int(mask_yellow.sum())}")
    print(f"has_silver       predictions: {int(mask_silver.sum())}")
    print(f"intersection (yellow+silver sports shoes): {len(pred_ids)}")
    print(f"ground truth yellow+silver sports shoes:   {len(gt_df)}")
    print(
        f"[SemBench] Precision={score.precision:.4f}  "
        f"Recall={score.recall:.4f}  F1={score.f1_score:.4f}"
    )

    # ── Cost ─────────────────────────────────────────────────────────
    if os.path.exists(EMBED_USAGE):
        with open(EMBED_USAGE) as f:
            embed_usage = json.load(f)
        embed_cost = embed_usage.get("tasks", {}).get("image_caption", {}).get("est_cost_usd", 0.0)

        caption_cost = 0.0
        if os.path.isfile(CAPTION_USAGE):
            with open(CAPTION_USAGE) as f:
                caption_usage = json.load(f)
            caption_cost = caption_usage.get("est_caption_cost_usd", 0.0)

        total_cost = embed_cost + caption_cost
        print(f"\n=== Cost ===")
        print(f"  Columns used:     Image (products_image.parquet via captions)")
        print(f"  Captioning cost:  ${caption_cost:.4f}")
        print(f"  Embedding cost:   ${embed_cost:.4f}")
        print(f"  Total cost:       ${total_cost:.4f}")


if __name__ == "__main__":
    main()
