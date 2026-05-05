"""
MMQA Q2b — DASE-only (no BigQuery): Q2a logo cross-join + per-image color label.

Operator: J + M (compose). NL: same as Q2a + "What is the color of each logo?"
GT: 5 (ID, image_filename, color="blue") triples (mmqa/ground_truth/Q2b.csv).
Eval: F1 over composite (ID|image_filename|color) triple_id (GenericEvaluator
+ f1_set check).

Aligns with paper §5.1: same cascading distance signals as Q2a (counterfactual
anchors for logo + calibrated distance threshold for sem-join), plus an
anchor-argmax (ConfidenceMarginSignal) over a fixed color palette for the
attribute-extract stage (no LLM verification).

Stage A: reuse q2a_nn.predict_pairs (MarginSignal logo filter +
PairCosineSignal track-sim per-image top-1-GAP) → (ID, image_filename) pairs.
Stage B: ConfidenceMarginSignal over the COLOR_PALETTE anchors per matched
image; argmax → predicted color.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
sys.path.insert(0, _HERE)

from dase_cascade import ConfidenceMarginSignal, f1_set
from generic_evaluator import GenericEvaluator

import q2a_nn  # noqa: E402  — reuse Stage A pipeline

MMQA_DIR = os.path.abspath(os.path.join(_HERE, ".."))
DATA_DIR = os.path.join(MMQA_DIR, "data")
GT_DIR = os.path.join(MMQA_DIR, "ground_truth")
EMBED_USAGE = os.path.join(MMQA_DIR, "cache", "embed_checkpoints", "embed_usage.json")
CAPTION_USAGE = os.path.join(MMQA_DIR, "cache", "embed_checkpoints", "image_caption_usage.json")

# Fixed color palette (no GT tuning) — anchors for ConfidenceMarginSignal.
COLOR_PALETTE = [
    "red", "orange", "yellow", "green", "blue", "purple",
    "black", "white", "gold", "silver", "brown", "pink",
]
COLOR_TEMPLATE = "a logo whose dominant color is {color}"


def classify_color(caption_embs: np.ndarray) -> list[str]:
    anchors = [COLOR_TEMPLATE.format(color=c) for c in COLOR_PALETTE]
    signal = ConfidenceMarginSignal(anchors=anchors)
    _ = signal.compute(caption_embs)
    return [COLOR_PALETTE[i] for i in signal.last_argmax]


def main():
    images_df = pd.read_parquet(os.path.join(DATA_DIR, "images.parquet"))
    ap_warrior_df = pd.read_parquet(os.path.join(DATA_DIR, "ap_warrior.parquet"))
    gt_df = pd.read_csv(os.path.join(GT_DIR, "Q2b.csv"))

    print(f"Images: {len(images_df)}  |  Races: {len(ap_warrior_df)}  "
          f"|  GT triples: {len(gt_df)}")

    # ── Stage A: reuse Q2a logo cross-join ─────────────────────────
    pred_pairs = q2a_nn.predict_pairs(images_df, ap_warrior_df)
    print(f"Matched pairs from Q2a: {len(pred_pairs)}")

    if len(pred_pairs) == 0:
        pred_df = pd.DataFrame(columns=["ID", "image_filename", "color"])
    else:
        # ── Stage B: per-image color anchor-argmax ─────────────────
        matched_images = (
            pred_pairs[["image_filename"]]
            .drop_duplicates()
            .merge(images_df[["image_filename", "embedding"]],
                   on="image_filename", how="left")
        )
        m_embs = np.array(matched_images["embedding"].tolist(), dtype=np.float32)
        colors = classify_color(m_embs)
        color_map = dict(zip(matched_images["image_filename"], colors))
        pred_df = pred_pairs.assign(
            color=pred_pairs["image_filename"].map(color_map),
        )

    print(f"Predicted triples: {len(pred_df)}")
    if len(pred_df) <= 20:
        print(pred_df.to_string(index=False))
    else:
        print(pred_df.head(10).to_string(index=False))

    def add_triple_id(df):
        out = df.copy()
        out["triple_id"] = (
            out["ID"].astype(str)
            + "|" + out["image_filename"].astype(str)
            + "|" + out["color"].astype(str).str.lower()
        )
        return out

    score = GenericEvaluator.compute_accuracy_score(
        "f1-score", add_triple_id(gt_df), add_triple_id(pred_df), id_column="triple_id",
    )
    pred_set = set(add_triple_id(pred_df)["triple_id"])
    gt_set = set(add_triple_id(gt_df)["triple_id"])
    p_set, r_set, f1_set_v = f1_set(pred_set, gt_set)
    print(f"[SemBench]  P={score.precision:.4f}  R={score.recall:.4f}  F1={score.f1_score:.4f}")
    print(f"[set check] P={p_set:.4f}  R={r_set:.4f}  F1={f1_set_v:.4f}")

    try:
        with open(CAPTION_USAGE) as f:
            caption_cost = json.load(f)["est_caption_cost_usd"]
        with open(EMBED_USAGE) as f:
            embed_cost = json.load(f)["tasks"]["image_caption"]["est_cost_usd"]
        total_cost = caption_cost + embed_cost
        print("\n=== Cost ===")
        print(f"  Columns used: images (caption + embedding), ap_warrior.Track")
        print(f"  Image caption cost: ${caption_cost:.4f}")
        print(f"  Image embed cost:   ${embed_cost:.4f}")
        print(f"  Total cost:         ${total_cost:.4f}")
    except (OSError, KeyError):
        pass


if __name__ == "__main__":
    main()
