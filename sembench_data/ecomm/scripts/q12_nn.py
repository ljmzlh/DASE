"""
Ecomm Q12 — DASE-only (no BigQuery): per-product JSON {id, brand, category}.

NL: For each Adidas or Puma product, emit JSON {id, brand, category} where
    category ∈ {accessories, apparel, footwear}.
GT: 58 GT JSON id strings (lower(brandName) ∈ {adidas, puma}).
Eval: F1 over JSON id strings.

Aligns with paper §5.1: counterfactual anchors (margin) for both brand
filter and category classification — F + sem_classify composed client-side.
Brand uses two MarginSignal filters (adidas, puma) with margin-magnitude
tie-break; category uses 3 MarginSignal filters with argmax tie-break.
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

# Brand prompts (text embeddings).
POS_ADIDAS = [
    "an Adidas product",
    "a fashion item made by the brand Adidas",
    "Adidas branded apparel or footwear",
]
NEG_ADIDAS = [
    "a product not made by Adidas",
    "a non-Adidas fashion product",
    "a product from a brand other than Adidas",
]

POS_PUMA = [
    "a Puma product",
    "a fashion item made by the brand Puma",
    "Puma branded apparel or footwear",
]
NEG_PUMA = [
    "a product not made by Puma",
    "a non-Puma fashion product",
    "a product from a brand other than Puma",
]

# Category prompts (image embeddings).
POS_ACCESSORIES = [
    "a fashion accessory product",
    "an accessory item like a watch, bag, or jewellery",
    "a fashion accessory such as a belt, wallet, or sunglasses",
]
NEG_ACCESSORIES = [
    "a clothing apparel item",
    "a pair of footwear or shoes",
    "a product that is not a fashion accessory",
]

POS_APPAREL = [
    "a clothing apparel item",
    "a garment worn on the body such as a shirt, pants, or dress",
    "a piece of apparel clothing",
]
NEG_APPAREL = [
    "a fashion accessory product",
    "a pair of footwear or shoes",
    "a product that is not clothing or apparel",
]

POS_FOOTWEAR = [
    "a pair of footwear",
    "shoes, sneakers, sandals, or boots",
    "a footwear product worn on the feet",
]
NEG_FOOTWEAR = [
    "a clothing apparel item",
    "a fashion accessory product",
    "a product that is not footwear or shoes",
]

CATEGORY_PROMPTS = [
    ("accessories", POS_ACCESSORIES, NEG_ACCESSORIES),
    ("apparel", POS_APPAREL, NEG_APPAREL),
    ("footwear", POS_FOOTWEAR, NEG_FOOTWEAR),
]


def ensure_ground_truth() -> pd.DataFrame:
    gt_path = os.path.join(GT_DIR, "Q12.csv")
    if os.path.isfile(gt_path):
        return pd.read_csv(gt_path)

    os.makedirs(GT_DIR, exist_ok=True)
    src = pd.read_parquet(
        STYLES_DETAILS_PARQUET,
        columns=["id", "brandName", "masterCategory"],
    )
    brand_lower = src["brandName"].astype(str).str.lower()
    master_cat = src["masterCategory"].apply(
        lambda x: x.get("typeName") if isinstance(x, dict) else None
    )
    mask = brand_lower.isin(["adidas", "puma"])
    sub = pd.DataFrame({
        "id": src.loc[mask, "id"].astype(np.int64).values,
        "brand": brand_lower[mask].values,
        "category": master_cat[mask].astype(str).str.lower().values,
    })

    def _to_json(row):
        return json.dumps(
            {"id": int(row["id"]), "brand": row["brand"], "category": row["category"]},
            separators=(", ", ": "),
        )

    gt_df = pd.DataFrame({"Id": sub.apply(_to_json, axis=1).values})
    gt_df.to_csv(gt_path, index=False)
    print(f"[GT] generated {gt_path}: {len(gt_df)} adidas/puma rows")
    return gt_df


def main():
    text_df = pd.read_parquet(os.path.join(DATA_DIR, "products_text.parquet"))
    image_df = pd.read_parquet(os.path.join(DATA_DIR, "products_image.parquet"))
    text_df = text_df.sort_values("Id").reset_index(drop=True)
    image_df = image_df.sort_values("Id").reset_index(drop=True)
    assert (text_df["Id"].values == image_df["Id"].values).all(), \
        "text and image parquets have mismatched Ids"

    text_emb = np.array(text_df["embedding"].tolist(), dtype=np.float32)
    image_emb = np.array(image_df["embedding"].tolist(), dtype=np.float32)
    ids = text_df["Id"].astype(np.int64).values
    print(f"Total products: {len(ids)} "
          f"(text dim={text_emb.shape[1]}, image dim={image_emb.shape[1]})")

    gt_df = ensure_ground_truth()
    print(f"Ground truth adidas/puma rows: {len(gt_df)}")
    print()

    # Brand: two MarginSignal sub-filters; union; per-row margin tie-break.
    adidas_margin = MarginSignal(POS_ADIDAS, NEG_ADIDAS).compute(text_emb)
    puma_margin   = MarginSignal(POS_PUMA,   NEG_PUMA  ).compute(text_emb)
    is_adidas = adidas_margin > 0
    is_puma = puma_margin > 0
    brand_mask = is_adidas | is_puma
    brand_label = np.where(adidas_margin >= puma_margin, "adidas", "puma")

    print(f"is_adidas predictions: {int(is_adidas.sum())}")
    print(f"is_puma   predictions: {int(is_puma.sum())}")
    print(f"brand union:           {int(brand_mask.sum())}")

    # Category: three MarginSignal filters; pass if margin > 0; argmax across passers.
    cat_margins = {}
    for label, pos, neg in CATEGORY_PROMPTS:
        cat_margins[label] = MarginSignal(pos, neg).compute(image_emb)

    n = len(ids)
    category_label = np.full(n, "", dtype=object)
    for i in range(n):
        passing = [(lbl, cat_margins[lbl][i]) for lbl, _, _ in CATEGORY_PROMPTS
                   if cat_margins[lbl][i] > 0]
        if not passing:
            category_label[i] = ""
        elif len(passing) == 1:
            category_label[i] = passing[0][0]
        else:
            category_label[i] = max(passing, key=lambda t: t[1])[0]

    dropped_no_cat = int(((category_label == "") & brand_mask).sum())
    print(f"dropped (no category pass, brand-matched): {dropped_no_cat}")
    for label in ("accessories", "apparel", "footwear"):
        cnt = int(((category_label == label) & brand_mask).sum())
        print(f"  category={label:12s} (brand-matched): {cnt}")

    # Assemble JSON output for surviving rows.
    final_mask = brand_mask & (category_label != "")
    out_rows = []
    for i in np.nonzero(final_mask)[0]:
        out_rows.append(json.dumps(
            {
                "id": int(ids[i]),
                "brand": str(brand_label[i]),
                "category": str(category_label[i]),
            },
            separators=(", ", ": "),
        ))
    sys_df = pd.DataFrame({"Id": out_rows})
    print(f"\nfinal predictions: {len(sys_df)}")
    print(f"ground truth:      {len(gt_df)}")

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
        text_cost = embed_usage.get("tasks", {}).get("product_text", {}).get("est_cost_usd", 0.0)
        image_cost = embed_usage.get("tasks", {}).get("image_caption", {}).get("est_cost_usd", 0.0)

        caption_cost = 0.0
        if os.path.isfile(CAPTION_USAGE):
            with open(CAPTION_USAGE) as f:
                caption_usage = json.load(f)
            caption_cost = caption_usage.get("est_caption_cost_usd", 0.0)

        total_cost = text_cost + image_cost + caption_cost
        print(f"\n=== Cost ===")
        print(f"  Columns used:     Title+Description (text), Image (caption)")
        print(f"  Captioning cost:  ${caption_cost:.4f}")
        print(f"  Text embed cost:  ${text_cost:.4f}")
        print(f"  Image embed cost: ${image_cost:.4f}")
        print(f"  Total cost:       ${total_cost:.4f}")


if __name__ == "__main__":
    main()
