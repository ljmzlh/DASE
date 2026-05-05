"""
Ecomm Q5 — DASE-only (no BigQuery): 5-class text classification (sem_classify).

NL: Classify each Apparel product into Dress / Bottomwear / Socks / Topwear / Innerwear.
GT: 228 Apparel products (excluding Saree, Apparel Set, Loungewear and Nightwear).
Eval: Adjusted-Rand-Index.

Aligns with paper §5.1: anchor argmax via embedding distance — pick the
nearest category prompt for each product. Same anchors as the Q5 cascade
(`ecomm/scripts/q5_cascade.py`).
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

ECOMM_DIR              = os.path.abspath(os.path.join(_HERE, ".."))
DATA_DIR               = os.path.join(ECOMM_DIR, "data")
GT_DIR                 = os.path.join(ECOMM_DIR, "ground_truth")
STYLES_DETAILS_PARQUET = os.path.join(ECOMM_DIR, "cache", "sf_500", "styles_details.parquet")
EMBED_USAGE            = os.path.join(ECOMM_DIR, "cache", "embed_checkpoints", "embed_usage.json")

EXCLUDED_SUBCATS = {"Saree", "Apparel Set", "Loungewear and Nightwear"}

# Same anchors as the Q5 cascade.
CATEGORIES = ["Dress", "Bottomwear", "Socks", "Topwear", "Innerwear"]
CATEGORY_ANCHORS = {
    "Dress": "Dress: A dress is a one-piece outer garment that is worn on the torso, hangs down over the legs, and often consist of a bodice attached to a skirt.",
    "Bottomwear": "Bottomwear: Bottomwear refers to clothing worn on the lower part of the body, such as trousers, jeans, skirts, shorts, and leggings.",
    "Socks": "Socks: Socks are a type of clothing worn on the feet, typically made of soft fabric, designed to provide comfort and warmth.",
    "Topwear": "Topwear: Topwear refers to clothing worn on the upper part of the body, such as shirts, blouses, t-shirts, and jackets.",
    "Innerwear": "Innerwear: Innerwear refers to clothing worn beneath outer garments, typically close to the skin, such as underwear, bras, and undershirts.",
}


def prefilter_ids() -> np.ndarray:
    src = pd.read_parquet(
        STYLES_DETAILS_PARQUET, columns=["id", "masterCategory", "subCategory"]
    )
    master = src["masterCategory"].apply(
        lambda x: x.get("typeName") if isinstance(x, dict) else None
    )
    sub = src["subCategory"].apply(
        lambda x: x.get("typeName") if isinstance(x, dict) else None
    )
    mask = (master == "Apparel") & (~sub.isin(EXCLUDED_SUBCATS))
    return src.loc[mask, "id"].astype(np.int64).to_numpy()


def ensure_ground_truth() -> pd.DataFrame:
    gt_path = os.path.join(GT_DIR, "Q5.csv")
    if os.path.isfile(gt_path):
        return pd.read_csv(gt_path)

    os.makedirs(GT_DIR, exist_ok=True)
    src = pd.read_parquet(
        STYLES_DETAILS_PARQUET, columns=["id", "masterCategory", "subCategory"]
    )
    master = src["masterCategory"].apply(
        lambda x: x.get("typeName") if isinstance(x, dict) else None
    )
    sub = src["subCategory"].apply(
        lambda x: x.get("typeName") if isinstance(x, dict) else None
    )
    mask = (master == "Apparel") & (~sub.isin(EXCLUDED_SUBCATS))
    gt_df = pd.DataFrame(
        {"id": src.loc[mask, "id"].astype(np.int64).values, "category": sub[mask].values}
    )
    gt_df.to_csv(gt_path, index=False)
    print(f"[GT] generated {gt_path}: {len(gt_df)} rows")
    return gt_df


def main():
    df = pd.read_parquet(os.path.join(DATA_DIR, "products_text.parquet"))
    gt_df = ensure_ground_truth()

    keep_ids = set(prefilter_ids().tolist())
    df = df[df["Id"].astype(np.int64).isin(keep_ids)].copy()
    text_embs = np.array(df["embedding"].tolist(), dtype=np.float32)
    print(f"Filtered products: {len(df)} (embedding dim={text_embs.shape[1]})")

    # ConfidenceMarginSignal: anchor argmax + top1−top2 confidence.
    anchor_texts = [CATEGORY_ANCHORS[c] for c in CATEGORIES]
    signal = ConfidenceMarginSignal(anchors=anchor_texts)
    _ = signal.compute(text_embs)
    pred_labels = [CATEGORIES[i] for i in signal.last_argmax]

    sys_df = pd.DataFrame(
        {"id": df["Id"].astype(np.int64).values, "category": pred_labels}
    )

    score = GenericEvaluator.compute_accuracy_score(
        "adjusted-rand-index", gt_df, sys_df
    )
    ari = float(score.accuracy)

    print("Labels:", CATEGORIES)
    print("GT category distribution:")
    print(gt_df["category"].value_counts().to_string())
    print("\nPred category distribution:")
    print(sys_df["category"].value_counts().to_string())
    print(f"\n[SemBench] Adjusted-Rand-Index={ari:.4f}")

    if os.path.exists(EMBED_USAGE):
        with open(EMBED_USAGE) as f:
            embed_usage = json.load(f)
        embed_cost = embed_usage.get("tasks", {}).get("product_text", {}).get("est_cost_usd", 0.0)
        print("\n=== Cost ===")
        print("  Columns used:   Title, Description (products_text.parquet)")
        print(f"  Embedding cost: ${embed_cost:.4f}")
        print("  Caption cost:   $0.0000")
        print(f"  Total cost:     ${embed_cost:.4f}")


if __name__ == "__main__":
    main()
