"""
Ecomm Q8 — DASE-only (no BigQuery): text-to-image semantic join (SEMANTIC_JOIN).

NL: For each product with description ≥ 3000 chars, find matching images.
GT: SF=500 has 1 such product (id=40270). GT pair = {"40270-40270"}.
Eval: F1 over directional pair ids "{l_id}-{r_id}".

Aligns with paper §5.1: "semantic joins via a calibrated distance threshold".
PairCosineSignal scores text × image pairs in the candidate pool; predict
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

SIMILARITY_THRESHOLD = 0.75


def _extract_desc_value(x) -> str:
    if isinstance(x, dict):
        d = x.get("description")
        if isinstance(d, dict):
            v = d.get("value")
            if isinstance(v, str):
                return v
    return ""


def prefilter_left_ids() -> np.ndarray:
    src = pd.read_parquet(STYLES_DETAILS_PARQUET, columns=["id", "productDescriptors"])
    desc = src["productDescriptors"].apply(_extract_desc_value)
    mask = desc.str.len() >= 3000
    return src.loc[mask, "id"].astype(np.int64).to_numpy()


def ensure_ground_truth() -> pd.DataFrame:
    gt_path = os.path.join(GT_DIR, "Q8.csv")
    if os.path.isfile(gt_path):
        return pd.read_csv(gt_path)

    os.makedirs(GT_DIR, exist_ok=True)
    left_ids = prefilter_left_ids()
    gt_df = pd.DataFrame({"id": [f"{i}-{i}" for i in left_ids]})
    gt_df.to_csv(gt_path, index=False)
    print(f"[GT] generated {gt_path}: {len(gt_df)} pairs")
    return gt_df


def main():
    gt_df = ensure_ground_truth()
    left_ids = set(prefilter_left_ids().tolist())

    text_df = pd.read_parquet(os.path.join(DATA_DIR, "products_text.parquet"))
    image_df = pd.read_parquet(os.path.join(DATA_DIR, "products_image.parquet"))

    text_df = text_df[text_df["Id"].astype(np.int64).isin(left_ids)].copy()
    text_df["Id"] = text_df["Id"].astype(np.int64)
    image_df["Id"] = image_df["Id"].astype(np.int64)

    text_emb = np.array(text_df["embedding"].tolist(), dtype=np.float32)
    image_emb = np.array(image_df["embedding"].tolist(), dtype=np.float32)

    print(
        f"Left products (desc>=3000): {len(text_df)}  "
        f"Right images: {len(image_df)}"
    )
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD:.2f}")
    print(f"Ground truth pairs: {len(gt_df)}")

    # PairCosineSignal: text-emb (left) × image-cap-emb (right).
    pair_signal = PairCosineSignal(
        embeddings_left=text_emb, embeddings_right=image_emb
    )
    n_left, n_right = len(text_df), len(image_df)
    L = np.arange(n_left, dtype=np.int64)
    R = np.arange(n_right, dtype=np.int64)
    triples = pair_signal.all_pairs_above(L, R, SIMILARITY_THRESHOLD)

    l_ids = text_df["Id"].to_numpy()
    r_ids = image_df["Id"].to_numpy()
    pred_ids = [f"{l_ids[i]}-{r_ids[j]}" for i, j, _ in triples]
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
        text_embed_cost = embed_usage.get("tasks", {}).get("product_text", {}).get("est_cost_usd", 0.0)
        image_embed_cost = embed_usage.get("tasks", {}).get("image_caption", {}).get("est_cost_usd", 0.0)

        caption_cost = 0.0
        if os.path.isfile(CAPTION_USAGE):
            with open(CAPTION_USAGE) as f:
                caption_usage = json.load(f)
            caption_cost = caption_usage.get("est_caption_cost_usd", 0.0)

        total_cost = text_embed_cost + image_embed_cost + caption_cost
        print("\n=== Cost ===")
        print("  Columns used:   Title, Description, Image")
        print(f"  Text embed:     ${text_embed_cost:.4f}")
        print(f"  Image embed:    ${image_embed_cost:.4f}")
        print(f"  Caption cost:   ${caption_cost:.4f}")
        print(f"  Total cost:     ${total_cost:.4f}")


if __name__ == "__main__":
    main()
