"""
Ecomm Q10 — DASE-only (no BigQuery): outfit triple join (3-way SEMANTIC_JOIN).

NL: matching outfits (shoe + lower + upper) with shared brand + color, each
    ≤ 1000 INR, in 4 base colors.
GT: 8 GT triples at SF=500.
Eval: F1 over triple ids "{shoe}-{bottom}-{top}".

Aligns with paper §5.1: combine semantic-filter argmax (RoleMarginSignal
multi-class roles + ConfidenceMarginSignal-style color argmax) with
threshold-based join (PairCosineSignal). Same role prompts as the Q10
cascade — `ecomm/scripts/q10_cascade.py`.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import (
    RoleMarginSignal, ConfidenceMarginSignal, PairCosineSignal,
)
from generic_evaluator import GenericEvaluator

ECOMM_DIR              = os.path.abspath(os.path.join(_HERE, ".."))
DATA_DIR               = os.path.join(ECOMM_DIR, "data")
GT_DIR                 = os.path.join(ECOMM_DIR, "ground_truth")
STYLES_DETAILS_PARQUET = os.path.join(ECOMM_DIR, "cache", "sf_500", "styles_details.parquet")
EMBED_USAGE            = os.path.join(ECOMM_DIR, "cache", "embed_checkpoints", "embed_usage.json")
CAPTION_USAGE          = os.path.join(ECOMM_DIR, "cache", "embed_checkpoints", "image_caption_usage.json")

ALLOWED_COLORS = ["Black", "Blue", "Red", "White"]
PAIR_SIM_THRESHOLD = 0.85

# Verbatim sembench BQ template prompts — same as the Q10 cascade.
SHOE_PROMPT = (
    "The image depicts a (pair of) shoe(s), sandal(s), flip-flop(s). "
    "If there are multiple products in the picture, always refer to the most promiment one."
)
LOWER_PROMPT = (
    "The image depicts a piece of apparel that can be worn on the lower part of the body, "
    "like pants, shorts, skirts, ... "
    "If there are multiple products in the picture, always refer to the most promiment one."
)
UPPER_PROMPT = (
    "The image depicts a piece of apparel that can be worn on the upper part of the body, "
    "like t-shirts, shirts, pullovers, hoodies, but still require some sort of clothing on "
    "the lower body, which means, e.g., not a dress. "
    "If there are multiple products in the picture, always refer to the most promiment one."
)
ROLE_PROMPTS = {"shoe": SHOE_PROMPT, "lower": LOWER_PROMPT, "upper": UPPER_PROMPT}

# Per-color anchor for argmax color classification.
COLOR_ANCHORS = [f"a {c.lower()} colored fashion product" for c in ALLOWED_COLORS]


def prefilter_ids() -> np.ndarray:
    src = pd.read_parquet(STYLES_DETAILS_PARQUET, columns=["id", "price"])
    return src.loc[src["price"] <= 1000, "id"].astype(np.int64).to_numpy()


def ensure_ground_truth() -> pd.DataFrame:
    gt_path = os.path.join(GT_DIR, "Q10.csv")
    if os.path.isfile(gt_path):
        return pd.read_csv(gt_path)

    os.makedirs(GT_DIR, exist_ok=True)
    src = pd.read_parquet(
        STYLES_DETAILS_PARQUET,
        columns=["id", "baseColour", "price", "brandName", "masterCategory", "subCategory"],
    )
    src["master"] = src["masterCategory"].apply(
        lambda x: x.get("typeName") if isinstance(x, dict) else None
    )
    src["sub"] = src["subCategory"].apply(
        lambda x: x.get("typeName") if isinstance(x, dict) else None
    )
    src["id"] = src["id"].astype(np.int64)

    shoes = src[(src["master"] == "Footwear") & (src["price"] <= 1000)].copy()
    lower = src[
        (src["master"] == "Apparel")
        & (src["sub"] == "Bottomwear")
        & (src["price"] <= 1000)
    ].copy()
    upper = src[
        (src["master"] == "Apparel")
        & (src["sub"] == "Topwear")
        & (src["price"] <= 1000)
    ].copy()

    shoes = shoes[shoes["baseColour"].isin(ALLOWED_COLORS)]
    s_l = shoes.merge(
        lower,
        on=["baseColour", "brandName"],
        suffixes=("_shoe", "_bottom"),
        how="inner",
    )
    s_l_u = s_l.merge(
        upper,
        on=["baseColour", "brandName"],
        how="inner",
    )
    gt_ids = (
        s_l_u["id_shoe"].astype(str)
        + "-"
        + s_l_u["id_bottom"].astype(str)
        + "-"
        + s_l_u["id"].astype(str)
    )
    gt_df = pd.DataFrame({"id": gt_ids.unique()})
    gt_df.to_csv(gt_path, index=False)
    print(f"[GT] generated {gt_path}: {len(gt_df)} triples")
    return gt_df


def main():
    gt_df = ensure_ground_truth()
    keep_ids = set(prefilter_ids().tolist())

    df = pd.read_parquet(os.path.join(DATA_DIR, "products_image.parquet"))
    df = df[df["Id"].astype(np.int64).isin(keep_ids)].copy()
    df["Id"] = df["Id"].astype(np.int64)
    df = df.reset_index(drop=True)
    emb = np.array(df["embedding"].tolist(), dtype=np.float32)
    ids = df["Id"].to_numpy()

    print(f"Candidates (price<=1000): {len(df)} (embedding dim={emb.shape[1]})")
    print(f"Ground truth triples: {len(gt_df)}")
    print(f"Pair sim threshold (brand proxy): {PAIR_SIM_THRESHOLD:.2f}")

    # Role argmax via RoleMarginSignal (same prompts as Q10 cascade).
    role_scores = {}
    for role in ROLE_PROMPTS:
        signal = RoleMarginSignal(role_prompts=ROLE_PROMPTS, target_role=role)
        role_scores[role] = signal.compute(emb)
    role_names = list(ROLE_PROMPTS.keys())  # ["shoe", "lower", "upper"]
    score_mat = np.stack([role_scores[r] for r in role_names], axis=1)
    role_pred = np.array([role_names[i] for i in score_mat.argmax(axis=1)])

    # Color argmax via ConfidenceMarginSignal.
    color_signal = ConfidenceMarginSignal(anchors=COLOR_ANCHORS)
    _ = color_signal.compute(emb)
    color_pred = np.array([ALLOWED_COLORS[i] for i in color_signal.last_argmax])

    df["pred_role"] = role_pred
    df["pred_color"] = color_pred
    print("Role distribution:")
    print(df["pred_role"].value_counts().to_string())
    print("\nColor distribution:")
    print(df["pred_color"].value_counts().to_string())

    # Pair-cosine for "same brand" proxy.
    pair_signal = PairCosineSignal(embeddings_left=emb)
    sim = pair_signal._left @ pair_signal._left.T
    id_to_idx = {int(i): idx for idx, i in enumerate(ids.tolist())}

    shoes = df[df["pred_role"] == "shoe"]["Id"].tolist()
    bottoms = df[df["pred_role"] == "lower"]["Id"].tolist()
    tops = df[df["pred_role"] == "upper"]["Id"].tolist()
    color_map = dict(zip(df["Id"].tolist(), df["pred_color"].tolist()))

    triples: list[str] = []
    for s in shoes:
        for b in bottoms:
            if s == b:
                continue
            if color_map[s] != color_map[b]:
                continue
            if sim[id_to_idx[s], id_to_idx[b]] < PAIR_SIM_THRESHOLD:
                continue
            for t in tops:
                if t == s or t == b:
                    continue
                if color_map[t] != color_map[s]:
                    continue
                if sim[id_to_idx[s], id_to_idx[t]] < PAIR_SIM_THRESHOLD:
                    continue
                if sim[id_to_idx[b], id_to_idx[t]] < PAIR_SIM_THRESHOLD:
                    continue
                triples.append(f"{s}-{b}-{t}")

    sys_df = pd.DataFrame({"id": pd.unique(np.array(triples, dtype=object))})
    print(f"\nPredicted triples: {len(sys_df)}")

    score = GenericEvaluator.compute_accuracy_score(
        "f1-score", gt_df, sys_df, id_column="id"
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
        print(f"  Caption cost:   ${caption_cost:.4f}")
        print(f"  Embedding cost: ${image_embed_cost:.4f}")
        print(f"  Total cost:     ${total_cost:.4f}")


if __name__ == "__main__":
    main()
