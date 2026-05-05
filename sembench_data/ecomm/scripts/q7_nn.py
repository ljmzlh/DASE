"""
Ecomm Q7 — DASE-only (no BigQuery): semantic-join over product text pairs.

NL: Find product pairs that match by article-type + brand (price <= 500).
GT: ecomm/ground_truth/Q7.csv  (generated from gold SQL on first run).
Eval: F1 over directional (id_a-id_b) pair strings.

Aligns with paper §5.1: "semantic joins via a calibrated distance threshold".
PairCosineSignal scores all pairs in the candidate pool; predict-positive
above threshold (no LLM verification).
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

SIMILARITY_THRESHOLD = 0.90
PRICE_LIMIT = 500


def prefilter_ids() -> np.ndarray:
    src = pd.read_parquet(STYLES_DETAILS_PARQUET, columns=["id", "price"])
    return src.loc[src["price"] <= PRICE_LIMIT, "id"].astype(np.int64).to_numpy()


def ensure_ground_truth() -> pd.DataFrame:
    gt_path = os.path.join(GT_DIR, "Q7.csv")
    if os.path.isfile(gt_path):
        return pd.read_csv(gt_path)
    os.makedirs(GT_DIR, exist_ok=True)
    src = pd.read_parquet(STYLES_DETAILS_PARQUET,
                          columns=["id", "price", "articleType", "brandName"])
    src = src[src["price"] <= PRICE_LIMIT].copy()
    src["article_type_name"] = src["articleType"].apply(
        lambda x: x.get("typeName") if isinstance(x, dict) else None)
    src["id"] = src["id"].astype(np.int64)
    pair_ids: list[str] = []
    for _, group in src.groupby(["article_type_name", "brandName"], dropna=False):
        ids = group["id"].to_list()
        for left_id in ids:
            for right_id in ids:
                pair_ids.append(f"{left_id}-{right_id}")
    gt = pd.DataFrame({"id": pair_ids})
    gt.to_csv(gt_path, index=False)
    print(f"[GT] generated {gt_path}: {len(gt)} pairs")
    return gt


def main():
    df = pd.read_parquet(os.path.join(DATA_DIR, "products_text.parquet"))
    keep_ids = set(prefilter_ids().tolist())
    df = df[df["Id"].astype(np.int64).isin(keep_ids)].copy()
    df["Id"] = df["Id"].astype(np.int64)

    emb = np.array(df["embedding"].tolist(), dtype=np.float32)
    ids = df["Id"].to_numpy()
    n = len(ids)
    print(f"Filtered products: {n};  threshold={SIMILARITY_THRESHOLD:.2f}")

    gt_df = ensure_ground_truth()
    print(f"GT pairs: {len(gt_df)}")

    pair_signal = PairCosineSignal(embeddings_left=emb)
    all_idx = np.arange(n, dtype=np.int64)
    triples = pair_signal.all_pairs_above(all_idx, all_idx, SIMILARITY_THRESHOLD)
    pred_pair_ids = [f"{ids[i]}-{ids[j]}" for i, j, _ in triples]
    sys_df = pd.DataFrame({"id": pred_pair_ids})
    print(f"Predicted pairs: {len(sys_df)}")

    score = GenericEvaluator.compute_accuracy_score("f1-score", gt_df, sys_df, id_column="id")
    print(f"[SemBench] P={score.precision:.4f}  R={score.recall:.4f}  F1={score.f1_score:.4f}")

    if os.path.exists(EMBED_USAGE):
        with open(EMBED_USAGE) as f:
            embed_usage = json.load(f)
        embed_cost = embed_usage.get("tasks", {}).get("product_text", {}).get("est_cost_usd", 0.0)
        print(f"\n=== Cost ===  embedding=${embed_cost:.4f}  total=${embed_cost:.4f}")


if __name__ == "__main__":
    main()
