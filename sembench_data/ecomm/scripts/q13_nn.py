"""
Ecomm Q13 — DASE-only (no BigQuery): complex sem_filter on image embeddings.

NL: men's running t-shirt with round neck, short sleeves, blue/black, striped,
    suitable for outdoor warm-weather running.
GT: 12 products (8-condition structural AND).
Eval: F1 over id sets.

Aligns with paper §5.1: counterfactual anchors (margin). Same prompts as
the Q13 cascade (`ecomm/scripts/q13_cascade.py`): MarginSignal on image-cap
embeddings, predict positive when margin > 0.
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

# Same prompts as Q13 cascade.
POSITIVE_PROMPTS = [
    "a men's running t-shirt with round neck and short sleeves, in blue or black, with a striped design, suitable for outdoor running in warm weather",
    "a blue or black short-sleeve striped sports tshirt for men, athletic running apparel",
    "a men's athletic round-neck short-sleeve striped running shirt for warm weather",
]
NEGATIVE_PROMPTS = [
    "women's clothing or non-tshirt apparel like trousers, shoes, accessories, dresses",
    "a white or green or other bright-colored t-shirt, or a non-striped solid-color shirt",
    "a long-sleeve or v-neck or hoodie or jacket, not a short-sleeve round-neck t-shirt",
]


def _article_type_name(x):
    return x.get("typeName") if isinstance(x, dict) else None


def _has_attr(attrs, key: str, value: str) -> bool:
    if not isinstance(attrs, list):
        return False
    for kv in attrs:
        if isinstance(kv, (tuple, list)) and len(kv) >= 2:
            if str(kv[0]) == key and str(kv[1]) == value:
                return True
    return False


def ensure_ground_truth() -> pd.DataFrame:
    gt_path = os.path.join(GT_DIR, "Q13.csv")
    if os.path.isfile(gt_path):
        return pd.read_csv(gt_path)

    os.makedirs(GT_DIR, exist_ok=True)
    src = pd.read_parquet(
        STYLES_DETAILS_PARQUET,
        columns=["id", "gender", "usage", "articleType", "baseColour", "articleAttributes", "season"],
    )
    mask = (
        (src["gender"] == "Men")
        & (src["usage"] == "Sports")
        & (src["articleType"].apply(_article_type_name) == "Tshirts")
        & (src["baseColour"].isin(["Blue", "Black"]))
        & (src["articleAttributes"].apply(lambda a: _has_attr(a, "Sleeve Length", "Short Sleeves")))
        & (src["articleAttributes"].apply(lambda a: _has_attr(a, "Neck", "Round Neck")))
        & (src["articleAttributes"].apply(lambda a: _has_attr(a, "Pattern", "Striped")))
        & (src["season"] != "Winter")
    )
    gt_df = pd.DataFrame({"Id": src.loc[mask, "id"].astype(np.int64).values})
    gt_df.to_csv(gt_path, index=False)
    print(f"[GT] generated {gt_path}: {len(gt_df)} rows")
    return gt_df


def main():
    image_df = pd.read_parquet(os.path.join(DATA_DIR, "products_image.parquet"))
    image_df["Id"] = image_df["Id"].astype(np.int64)
    image_emb = np.array(image_df["embedding"].tolist(), dtype=np.float32)
    ids = image_df["Id"].values
    n_total = len(ids)
    print(f"Rows: {n_total} (image dim={image_emb.shape[1]})")

    gt_df = ensure_ground_truth()
    print(f"Ground truth rows: {len(gt_df)}")

    # MarginSignal on image-cap embeddings — same signal as the cascade.
    margins = MarginSignal(POSITIVE_PROMPTS, NEGATIVE_PROMPTS).compute(image_emb)
    pred_mask = margins > 0
    pred_ids = ids[pred_mask]
    sys_df = pd.DataFrame({"Id": pred_ids})

    print(f"\nmargin range: [{margins.min():+.3f}, {margins.max():+.3f}]")
    print(f"Predicted rows: {len(sys_df)}")

    score = GenericEvaluator.compute_accuracy_score(
        "f1-score", gt_df, sys_df, id_column="Id"
    )
    print(
        f"[SemBench] Precision={score.precision:.4f}  "
        f"Recall={score.recall:.4f}  F1={score.f1_score:.4f}"
    )

    if os.path.exists(EMBED_USAGE):
        with open(EMBED_USAGE) as f:
            embed_usage = json.load(f)
        image_embed_cost = embed_usage.get("tasks", {}).get("image_caption", {}).get("est_cost_usd", 0.0)

        caption_cost = 0.0
        if os.path.isfile(CAPTION_USAGE):
            with open(CAPTION_USAGE) as f:
                caption_usage = json.load(f)
            caption_cost = caption_usage.get("est_caption_cost_usd", 0.0)

        total_cost = image_embed_cost + caption_cost
        print("\n=== Cost ===")
        print("  Columns used:   Image (products_image.parquet via captions)")
        print(f"  Image embed:    ${image_embed_cost:.4f}")
        print(f"  Caption cost:   ${caption_cost:.4f}")
        print(f"  Total cost:     ${total_cost:.4f}")


if __name__ == "__main__":
    main()
