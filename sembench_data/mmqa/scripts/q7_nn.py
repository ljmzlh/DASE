"""
MMQA Q7 — DASE-only (no BigQuery): cross-modal sem-join (airline name × image
logo).

Operator: J. GT: query/natural_language/q7.json → list of (Airlines,
image_filename) pairs.
Eval: F1 over (Airlines | image_filename) pair_id (GenericEvaluator + f1_set
check).

Aligns with paper §5.1: "semantic joins via a calibrated distance threshold".
PairCosineSignal scores all (airline phrase × image caption) pairs; the
per-airline top-1-GAP prefilter (verbatim from q7_cascade.py) keeps the
borderline pool, and we predict-positive on every surviving pair (no LLM
verification on uncertain pairs).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import PairCosineSignal, embed_query, f1_set
from generic_evaluator import GenericEvaluator

MMQA_DIR = os.path.abspath(os.path.join(_HERE, ".."))
DATA_DIR = os.path.join(MMQA_DIR, "data")
NL_PATH = os.path.join(MMQA_DIR, "query", "natural_language", "q7.json")
EMBED_USAGE = os.path.join(MMQA_DIR, "cache", "embed_checkpoints", "embed_usage.json")
CAPTION_USAGE = os.path.join(MMQA_DIR, "cache", "embed_checkpoints", "image_caption_usage.json")

# Same anchor template + per-airline top-1 gap as q7_cascade.py
AIRLINE_PROMPT_TEMPLATE = "the logo of {a}"
GAP = 0.05


def predict_pairs(tampa_df: pd.DataFrame, images_df: pd.DataFrame) -> pd.DataFrame:
    distinct_airlines = sorted(set(tampa_df["Airlines"].astype(str).str.strip().tolist()))
    distinct_airlines = [a for a in distinct_airlines if a and a.lower() != "nan"]

    phrases = [AIRLINE_PROMPT_TEMPLATE.format(a=a) for a in distinct_airlines]
    # embed_query handles BatchEmbedContents chunking under the hood.
    chunks = [embed_query(phrases[i:i + 100]) for i in range(0, len(phrases), 100)]
    a_emb = np.concatenate(chunks, axis=0)
    i_emb = np.array(images_df["embedding"].tolist(), dtype=np.float32)

    pair_signal = PairCosineSignal(embeddings_left=a_emb, embeddings_right=i_emb)
    # Per-airline top-1-GAP isn't expressible as a uniform threshold, so we
    # use the normalized internals to compute the full cross-sim matrix.
    S = pair_signal._left @ pair_signal._right.T  # (n_airlines, n_images)

    rows = []
    cands_per_airline = []
    for ai, a in enumerate(distinct_airlines):
        thr = S[ai].max() - GAP
        keep_iidx = np.where(S[ai] >= thr)[0]
        cands_per_airline.append(len(keep_iidx))
        for ii in keep_iidx:
            rows.append((a, str(images_df.iloc[int(ii)]["image_filename"])))
    print(f"  prefilter cands per airline: min={min(cands_per_airline)} "
          f"median={int(np.median(cands_per_airline))} "
          f"max={max(cands_per_airline)} total={len(rows)}")
    return pd.DataFrame(rows, columns=["Airlines", "image_filename"])


def main():
    tampa_df = pd.read_parquet(os.path.join(DATA_DIR, "tampa_international_airport.parquet"))
    images_df = pd.read_parquet(os.path.join(DATA_DIR, "images.parquet"))
    with open(NL_PATH) as f:
        gt_pairs = json.load(f)["ground_truth"]
    gt_df = pd.DataFrame(gt_pairs, columns=["Airlines", "image_filename"])

    print(f"  airlines (distinct, post-clean): from {tampa_df['Airlines'].nunique()} raw, "
          f"images: {len(images_df)}, GT pairs: {len(gt_df)}, GAP={GAP}")

    pred_df = predict_pairs(tampa_df, images_df)
    print(f"Predicted pairs: {len(pred_df)}")
    if len(pred_df) <= 15:
        print(pred_df.to_string(index=False))

    def add_pair_id(df):
        out = df.copy()
        out["pair_id"] = (
            out["Airlines"].astype(str) + "|" + out["image_filename"].astype(str)
        )
        return out

    score = GenericEvaluator.compute_accuracy_score(
        "f1-score", add_pair_id(gt_df), add_pair_id(pred_df), id_column="pair_id",
    )
    pred_set = {f"{a}|{fn}" for a, fn in zip(pred_df["Airlines"], pred_df["image_filename"])}
    gt_set = {f"{a}|{fn}" for a, fn in gt_pairs}
    p_set, r_set, f1_set_v = f1_set(pred_set, gt_set)
    print(f"[SemBench]  P={score.precision:.4f}  R={score.recall:.4f}  F1={score.f1_score:.4f}")
    print(f"[set check] P={p_set:.4f}  R={r_set:.4f}  F1={f1_set_v:.4f}")

    try:
        with open(CAPTION_USAGE) as f:
            caption_usd = float(json.load(f).get("est_caption_cost_usd", 0.0))
        with open(EMBED_USAGE) as f:
            u = json.load(f).get("tasks", {})
        img_emb_usd = float(u.get("image_caption", {}).get("est_cost_usd", 0.0))
        tampa_emb_usd = float(u.get("tampa_destinations", {}).get("est_cost_usd", 0.0))
        total = caption_usd + img_emb_usd + tampa_emb_usd
        print("\n=== Cost (USD, estimated) ===")
        print(f"  Image caption (Flash):     ${caption_usd:.4f}")
        print(f"  Image caption embedding:   ${img_emb_usd:.4f}")
        print(f"  Tampa Destinations embed:  ${tampa_emb_usd:.4f}")
        print(f"  Total cost:                ${total:.4f}")
    except OSError:
        pass


if __name__ == "__main__":
    main()
