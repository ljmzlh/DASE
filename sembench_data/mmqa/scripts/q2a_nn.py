"""
MMQA Q2a — DASE-only (no BigQuery): cross-modal sem-join (track × image) for
A.P. Warrior racetrack logos.

Operator: J. NL: "Identify the images containing logos, if available, for each
racetrack in which A.P. Warrior was a contender."
GT: 5 (ID, image_filename) pairs (mmqa/ground_truth/Q2a.csv).
Eval: F1 on composite (ID|image_filename) pair_id (GenericEvaluator + f1_set
check).

Aligns with paper §5.1: "semantic joins via a calibrated distance threshold"
plus "counterfactual anchors". Two cascading distance signals (verbatim from
q2a_cascade.py):

  1. MarginSignal (logo vs nonlogo) — drop captions with logo_margin < LOGO_LO.
     Note: the cascade uses max-of-pos − max-of-neg (asymmetric) here rather
     than the symmetric mean−mean, so we replicate that semantics on the
     embedded prompts directly.
  2. PairCosineSignal (image caption × track name) — per-surviving-image
     top-1-GAP prefilter on the track-sim distribution.

Predict-positive on every surviving (track, image) pair (no LLM verification),
then expand to (ID, image_filename) via ap_warrior.
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import PairCosineSignal, embed_query, f1_set
from dase_cascade.runtime import cosine_sim_batch
from generic_evaluator import GenericEvaluator

MMQA_DIR = os.path.abspath(os.path.join(_HERE, ".."))
DATA_DIR = os.path.join(MMQA_DIR, "data")
GT_DIR = os.path.join(MMQA_DIR, "ground_truth")
EMBED_USAGE = os.path.join(MMQA_DIR, "cache", "embed_checkpoints", "embed_usage.json")
CAPTION_USAGE = os.path.join(MMQA_DIR, "cache", "embed_checkpoints", "image_caption_usage.json")

# Counterfactual anchors — verbatim from q2a_cascade.py.
LOGO_PHRASES = [
    "a logo or wordmark as the main subject of the image",
    "a brand emblem or typographic logo as the primary visual focus",
]
NONLOGO_PHRASES = [
    "a photograph where the main subject is not a logo, such as people, animals, or action",
    "a candid or landscape image where no logo or wordmark is the primary focus",
]

LOGO_LO = 0.02      # logo_margin (max-pos − max-neg) threshold; below ⇒ drop.
TRACK_GAP = 0.10    # tracks within GAP of the per-image top-1 are kept.


def predict_pairs(images_df: pd.DataFrame, ap_warrior_df: pd.DataFrame) -> pd.DataFrame:
    cap_emb = np.array(images_df["embedding"].tolist(), dtype=np.float32)
    n_img = len(images_df)
    distinct_tracks = sorted(ap_warrior_df["Track"].unique().tolist())
    n_tracks = len(distinct_tracks)

    # ── Stage 1: logo MarginSignal (max-of-pos − max-of-neg, per cascade) ──
    pos_logo = embed_query(LOGO_PHRASES)
    neg_logo = embed_query(NONLOGO_PHRASES)
    pos_best = np.maximum.reduce([cosine_sim_batch(p, cap_emb) for p in pos_logo])
    neg_best = np.maximum.reduce([cosine_sim_batch(n, cap_emb) for n in neg_logo])
    logo_margin = pos_best - neg_best

    # ── Stage 2: PairCosineSignal — image caption × track-name ──
    track_emb = embed_query(distinct_tracks)
    pair_signal = PairCosineSignal(embeddings_left=cap_emb, embeddings_right=track_emb)
    pair_track_sim = pair_signal._left @ pair_signal._right.T  # (n_img, n_tracks)

    surviving_pairs = []  # (track, image_filename)
    n_dropped = 0
    cands_per_img = []
    for i in range(n_img):
        if logo_margin[i] < LOGO_LO:
            n_dropped += 1
            continue
        thr = pair_track_sim[i].max() - TRACK_GAP
        keep = [ti for ti in range(n_tracks) if pair_track_sim[i, ti] >= thr]
        cands_per_img.append(len(keep))
        fn = str(images_df.iloc[i]["image_filename"])
        for ti in keep:
            surviving_pairs.append((distinct_tracks[ti], fn))
    print(f"  Stage 1 dropped (not logo): {n_dropped} / {n_img} images")
    print(f"  Stage 2 surviving (track, image) pairs: {len(surviving_pairs)} "
          f"(per kept image: min={min(cands_per_img,default=0)}, "
          f"median={int(np.median(cands_per_img)) if cands_per_img else 0}, "
          f"max={max(cands_per_img,default=0)})")

    # Expand to (ID, image_filename) via ap_warrior
    track_to_files: dict[str, set] = {}
    for t, fn in surviving_pairs:
        track_to_files.setdefault(t, set()).add(fn)
    apw = ap_warrior_df.copy()
    apw["ID"] = apw["ID"].astype(int)
    pred_rows = []
    for _, r in apw.iterrows():
        for fn in track_to_files.get(r["Track"], set()):
            pred_rows.append({"ID": int(r["ID"]), "image_filename": fn})
    return (
        pd.DataFrame(pred_rows, columns=["ID", "image_filename"])
        .drop_duplicates()
        .reset_index(drop=True)
    )


def main():
    images_df = pd.read_parquet(os.path.join(DATA_DIR, "images.parquet"))
    ap_warrior_df = pd.read_parquet(os.path.join(DATA_DIR, "ap_warrior.parquet"))
    gt_df = pd.read_csv(os.path.join(GT_DIR, "Q2a.csv"))

    print(f"Images: {len(images_df)}  |  Races: {len(ap_warrior_df)}  "
          f"|  GT pairs: {len(gt_df)}")

    pred_df = predict_pairs(images_df, ap_warrior_df)
    print(f"Predicted pairs: {len(pred_df)}")
    if len(pred_df) <= 20:
        print(pred_df.to_string(index=False))

    def add_pair_id(df):
        out = df.copy()
        out["pair_id"] = out["ID"].astype(str) + "|" + out["image_filename"].astype(str)
        return out

    score = GenericEvaluator.compute_accuracy_score(
        "f1-score", add_pair_id(gt_df), add_pair_id(pred_df), id_column="pair_id",
    )
    pred_set = {f"{int(r.ID)}|{r.image_filename}" for r in pred_df.itertuples(index=False)}
    gt_set = {f"{int(r.ID)}|{r.image_filename}" for r in gt_df.itertuples(index=False)}
    p_set, r_set, f1_set_v = f1_set(pred_set, gt_set)
    print(f"[SemBench]  P={score.precision:.4f}  R={score.recall:.4f}  F1={score.f1_score:.4f}")
    print(f"[set check] P={p_set:.4f}  R={r_set:.4f}  F1={f1_set_v:.4f}")

    try:
        import json as _json
        with open(CAPTION_USAGE) as f:
            caption_cost = _json.load(f)["est_caption_cost_usd"]
        with open(EMBED_USAGE) as f:
            embed_cost = _json.load(f)["tasks"]["image_caption"]["est_cost_usd"]
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
