"""
Ecomm Q14 — DASE-only (no BigQuery): F + J composite (white socks under $130).

NL: For each fashion product (price < 130), find the single image that best
    matches the product's description, but only if that image depicts white socks.
    Gold SQL collapses to: articleType='Socks' AND baseColour='White' AND price<130.
GT: 2 product ids.
Eval: F1 over output id set.

Aligns with paper §5.1: counterfactual anchors (margin) for the white-socks
F sub-filter (image-side) — same prompts as the Q14 cascade
(`ecomm/scripts/q14_cascade.py`). Tabular price filter is a hard predicate.
We add a pure-color "white" sub-filter on the image side to discriminate
white-vs-colored socks, parallel to the cascade's image-side pruning.
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

PRICE_LIMIT = 130

# Same socks-vs-non-socks anchors used by Q14 cascade's image-side filter
# (cascade fuses these with "white" anchors via MarginSignal).
POS_SOCKS = [
    "a pair of socks",
    "fashion socks worn on the feet",
    "hosiery socks",
]
NEG_SOCKS = [
    "a product that is not socks",
    "a fashion item that is not a pair of socks",
    "apparel or footwear that is not socks",
]

POS_WHITE = [
    "a product that has white in its color",
    "a fashion item featuring the color white",
    "a product whose colors include white",
]
NEG_WHITE = [
    "a product that does not have white",
    "a product without any white color",
    "a product whose colors do not include white",
]


def ensure_ground_truth() -> pd.DataFrame:
    gt_path = os.path.join(GT_DIR, "Q14.csv")
    if os.path.isfile(gt_path):
        return pd.read_csv(gt_path)

    os.makedirs(GT_DIR, exist_ok=True)
    src = pd.read_parquet(
        STYLES_DETAILS_PARQUET,
        columns=["id", "articleType", "baseColour", "price"],
    )
    article_type = src["articleType"].apply(
        lambda x: x.get("typeName") if isinstance(x, dict) else None
    )
    mask = (
        (article_type == "Socks")
        & (src["baseColour"] == "White")
        & (src["price"] < PRICE_LIMIT)
    )
    gt_df = pd.DataFrame({"Id": src.loc[mask, "id"].astype(np.int64).values})
    gt_df.to_csv(gt_path, index=False)
    print(f"[GT] generated {gt_path}: {len(gt_df)} white socks under ${PRICE_LIMIT}")
    return gt_df


def main():
    image_df = pd.read_parquet(os.path.join(DATA_DIR, "products_image.parquet"))
    prod_df = pd.read_parquet(os.path.join(DATA_DIR, "products.parquet"))

    image_df = image_df.sort_values("Id").reset_index(drop=True)
    prod_df = prod_df.sort_values("Id").reset_index(drop=True)
    assert (image_df["Id"].values == prod_df["Id"].values).all(), \
        "image and products parquets have mismatched Ids"

    emb_matrix = np.array(image_df["embedding"].tolist(), dtype=np.float32)
    prices = prod_df["Price"].astype(float).values
    ids = image_df["Id"].astype(np.int64).values
    print(f"Total products: {len(ids)} (image dim={emb_matrix.shape[1]})")

    gt_df = ensure_ground_truth()
    print(f"Ground truth white socks under ${PRICE_LIMIT}: {len(gt_df)}")
    print()

    # Two MarginSignal sub-filters on image embeddings + tabular price.
    socks_margin = MarginSignal(POS_SOCKS, NEG_SOCKS).compute(emb_matrix)
    white_margin = MarginSignal(POS_WHITE, NEG_WHITE).compute(emb_matrix)
    mask_socks = socks_margin > 0
    mask_white = white_margin > 0
    mask_cheap = prices < PRICE_LIMIT
    mask = mask_socks & mask_white & mask_cheap

    pred_ids = np.unique(ids[mask])
    sys_df = pd.DataFrame({"Id": pred_ids})

    score = GenericEvaluator.compute_accuracy_score(
        "f1-score", gt_df, sys_df, id_column="Id"
    )

    print("POS_SOCKS:", POS_SOCKS)
    print("NEG_SOCKS:", NEG_SOCKS)
    print("POS_WHITE:", POS_WHITE)
    print("NEG_WHITE:", NEG_WHITE)
    print()
    print(f"is_socks         predictions: {int(mask_socks.sum())}")
    print(f"is_white         predictions: {int(mask_white.sum())}")
    print(f"price < {PRICE_LIMIT:>3d}      predictions: {int(mask_cheap.sum())}")
    print(f"intersection (white socks < ${PRICE_LIMIT}): {len(pred_ids)}")
    print(f"ground truth:                       {len(gt_df)}")
    print(
        f"[SemBench] Precision={score.precision:.4f}  "
        f"Recall={score.recall:.4f}  F1={score.f1_score:.4f}"
    )

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
        print(f"  Columns used:     Image (caption embeddings) + Price (tabular)")
        print(f"  Captioning cost:  ${caption_cost:.4f}")
        print(f"  Embedding cost:   ${embed_cost:.4f}")
        print(f"  Total cost:       ${total_cost:.4f}")


if __name__ == "__main__":
    main()
