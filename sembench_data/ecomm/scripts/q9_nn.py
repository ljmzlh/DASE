"""
Ecomm Q9 — DASE-only (no BigQuery): image-to-image self-join (SEMANTIC_JOIN).

NL: pairs of products price<800 in 6 base colours (mono colour1/colour2='')
    depicting the same category + same dominant surface color.
GT: F1 over directional pair ids "{a_id}-{b_id}" (excluding self-pairs).

Aligns with paper §5.1: "semantic joins via a calibrated distance threshold".
PairCosineSignal scores all image pairs in the candidate pool; predict
positive above threshold (no LLM verification).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import PairCosineSignal
from generic_evaluator import GenericEvaluator

ECOMM_DIR              = os.path.abspath(os.path.join(_HERE, ".."))
DATA_DIR               = os.path.join(ECOMM_DIR, "data")
GT_DIR                 = os.path.join(ECOMM_DIR, "ground_truth")
STYLES_DETAILS_PARQUET = os.path.join(ECOMM_DIR, "cache", "sf_500", "styles_details.parquet")
EMBED_USAGE            = os.path.join(ECOMM_DIR, "cache", "embed_checkpoints", "embed_usage.json")
CAPTION_USAGE          = os.path.join(ECOMM_DIR, "cache", "embed_checkpoints", "image_caption_usage.json")

ALLOWED_COLORS = {"Black", "Blue", "Red", "White", "Orange", "Green"}
SIMILARITY_THRESHOLD = 0.85


def prefilter_ids() -> np.ndarray:
    src = pd.read_parquet(
        STYLES_DETAILS_PARQUET, columns=["id", "baseColour", "colour1", "colour2", "price"]
    )
    mask = (
        src["baseColour"].isin(ALLOWED_COLORS)
        & (src["colour1"] == "")
        & (src["colour2"] == "")
        & (src["price"] < 800)
    )
    return src.loc[mask, "id"].astype(np.int64).to_numpy()


def ensure_ground_truth() -> pd.DataFrame:
    gt_path = os.path.join(GT_DIR, "Q9.csv")
    if os.path.isfile(gt_path):
        return pd.read_csv(gt_path)

    os.makedirs(GT_DIR, exist_ok=True)
    src = pd.read_parquet(
        STYLES_DETAILS_PARQUET,
        columns=["id", "baseColour", "colour1", "colour2", "price", "articleType"],
    )
    src["article_type_name"] = src["articleType"].apply(
        lambda x: x.get("typeName") if isinstance(x, dict) else None
    )
    mask = (
        src["baseColour"].isin(ALLOWED_COLORS)
        & (src["colour1"] == "")
        & (src["colour2"] == "")
        & (src["price"] < 800)
    )
    sel = src.loc[mask, ["id", "baseColour", "article_type_name"]].copy()
    sel["id"] = sel["id"].astype(np.int64)

    pair_ids: list[str] = []
    for _, group in sel.groupby(["baseColour", "article_type_name"], dropna=False):
        ids = group["id"].tolist()
        for left_id in ids:
            for right_id in ids:
                if left_id == right_id:
                    continue
                pair_ids.append(f"{left_id}-{right_id}")

    gt_df = pd.DataFrame({"id": pair_ids})
    gt_df.to_csv(gt_path, index=False)
    print(f"[GT] generated {gt_path}: {len(gt_df)} pairs")
    return gt_df


def main():
    gt_df = ensure_ground_truth()
    keep_ids = set(prefilter_ids().tolist())

    df = pd.read_parquet(os.path.join(DATA_DIR, "products_image.parquet"))
    df = df[df["Id"].astype(np.int64).isin(keep_ids)].copy()
    df["Id"] = df["Id"].astype(np.int64)

    emb = np.array(df["embedding"].tolist(), dtype=np.float32)
    ids = df["Id"].to_numpy()
    n = len(ids)

    print(f"Filtered products: {n} (embedding dim={emb.shape[1]})")
    print(f"Ground truth pairs: {len(gt_df)}")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD:.2f}")

    # PairCosineSignal self-join; drop self pairs.
    pair_signal = PairCosineSignal(embeddings_left=emb)
    all_idx = np.arange(n, dtype=np.int64)
    triples = pair_signal.all_pairs_above(all_idx, all_idx, SIMILARITY_THRESHOLD)
    pred_ids = [f"{ids[i]}-{ids[j]}" for i, j, _ in triples if i != j]
    sys_df = pd.DataFrame({"id": pred_ids})
    print(f"Predicted pairs: {len(sys_df)}")

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
