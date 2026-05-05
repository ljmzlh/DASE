#!/usr/bin/env -S python -u
"""
MMQA Q2a cascade — cross-modal sem-join (track × image) for A.P. Warrior logos.

NL: Identify images containing logos of racetracks where A.P. Warrior raced.
GT: 5 (ID, image_filename) pairs — all map to Santa Anita Park logo.

Operator: J (cross-modal semantic join). Refactored to use dase_cascade primitives:

  Stage 0: MarginSignal (logo vs nonlogo phrases) — drop images with
           logo_margin < LOGO_LO (confident "not a logo").
  Stage 1: PairCosineSignal (image caption × track name embeddings) —
           per-surviving-image top-1-GAP prefilter on track-sim distribution.
  Stage 2: AiIfVerifier (CTAS + AI.IF) on candidate (track, image_uri) pairs.
  Stage 3: client-side expand to (ID, image_filename) via ap_warrior table.

Per the v1 design, no DASE confident_pos is emitted directly; all surviving
candidate pairs are sent to BQ for verification.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DASE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
SEMBENCH_MY = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, DASE_ROOT)
sys.path.insert(0, SEMBENCH_MY)

from google.cloud import bigquery  # noqa: E402

from generic_evaluator import GenericEvaluator  # noqa: E402

from dase_cascade import (  # noqa: E402
    MarginSignal, PairCosineSignal, AiIfVerifier,
    embed_query, bq_client, per_row_cost, run_query,
    build_profile, write_profile, print_summary,
)

MMQA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(MMQA_DIR, "data")
GT_DIR = os.path.join(MMQA_DIR, "ground_truth")
PROFILE_DIR = os.path.join(MMQA_DIR, "outputs")
PROFILE_PATH = os.path.join(PROFILE_DIR, "Q2a.json")

PROJECT = os.environ.get("GCP_PROJECT", "")
DATASET = "mmqa"
STAGING_TABLE = f"{DATASET}.q2a_uncertain_pairs"

PROMPT_PREFIX = "You will be provided with a horse racetrack name and an image. Determine if the image shows the logo of the racetrack."

LOGO_PHRASES = [
    "a logo or wordmark as the main subject of the image",
    "a brand emblem or typographic logo as the primary visual focus",
]
NONLOGO_PHRASES = [
    "a photograph where the main subject is not a logo, such as people, animals, or action",
    "a candid or landscape image where no logo or wordmark is the primary focus",
]
NONE_PHRASE = "not the logo or name of any specific horse racing track"

ALPHA = 0.2  # filter default; small GT (5 pairs), bumping doesn't help much
LOGO_LO = 0.02         # logo_margin below this → not a clear logo, drop
TRACK_GAP = 0.10       # tracks within GAP of the top-1 are competitive
PAPER_BQ_Q2a = {"score": None, "latency_s": None, "cost_usd": None}  # TBD: paper Q2 mmqa row mapping unclear
PAPER_DASE_NN_Q2a = {"score": None, "latency_s": 1e-3, "cost_usd": 1e-9}  # placeholder
SKIP_BASELINE = True  # full-Cartesian baseline is long-running; reuse paper numbers


def make_q2a_verifier():
    """CTAS staging from (track, image_uri) tuples; AI.IF on staging."""
    def make_staging(ids):
        def _esc(s):
            return s.replace("\\", "\\\\").replace("'", "\\'")
        tracks = [t for t, _ in ids]
        uris = [u for _, u in ids]
        track_arr = ",".join(f"'{_esc(t)}'" for t in tracks)
        uri_arr = ",".join(f"'{_esc(u)}'" for u in uris)
        return f"""
        CREATE OR REPLACE TABLE {STAGING_TABLE} AS
        SELECT track, ot.uri AS uri, ot.ref AS image
        FROM UNNEST([{track_arr}]) AS track WITH OFFSET pos
        JOIN UNNEST([{uri_arr}]) AS u WITH OFFSET pos2 ON pos = pos2
        JOIN {DATASET}.images ot ON ot.uri = u
        """

    verify_sql = f"""
    SELECT CONCAT(track, '|', uri) AS pair_id
    FROM {STAGING_TABLE}
    WHERE AI.IF(
      (CONCAT('{PROMPT_PREFIX} Racetrack: ', track, '.'), image),
      connection_id => 'us.connection', endpoint => 'gemini-2.5-flash')
    """
    return AiIfVerifier(
        verify_sql=verify_sql, make_staging_sql=make_staging,
        id_column="pair_id", coerce_id=str,
    )


def per_row_cost_q2a(client, sample_uris, sample_track):
    """Bespoke calibration: image (ot.ref) + bound track string. Use the
    `sample_uris+ext_table` path of per_row_cost — its prompt syntax is
    `('<prompt>', ot.ref)` which lines up with the AI.IF binding shape."""
    return per_row_cost(
        client,
        prompt=f"{PROMPT_PREFIX} Racetrack: {sample_track}.",
        sample_uris=sample_uris,
        ext_table=f"{DATASET}.images",
        method_label="AI.GENERATE_BOOL on images.ref + thinking_budget=0 (track inlined in prompt)",
        k=len(sample_uris),
    )


def main():
    profile = build_profile(
        scenario="mmqa", query_id="2a", scale_factor=200,
        prompt=PROMPT_PREFIX,
        params={"alpha": ALPHA, "LOGO_LO": LOGO_LO, "TRACK_GAP": TRACK_GAP},
        cascade_form=(
            "J cascade: MarginSignal(logo vs nonlogo) drops confident_neg; "
            "PairCosineSignal(image × track) per-image top-1-GAP prefilter; "
            "AiIfVerifier (CTAS + AI.IF) on uncertain (track, image) pairs; "
            "client-side expand to (ID, image_filename)."
        ),
        extra={
            "dase_prompts": {
                "logo_positive": LOGO_PHRASES,
                "logo_negative": NONLOGO_PHRASES,
                "none_track": NONE_PHRASE,
            },
        },
    )

    print("Loading images + ap_warrior + GT...")
    images_df = pd.read_parquet(os.path.join(DATA_DIR, "images.parquet"))
    apw_df = pd.read_parquet(os.path.join(DATA_DIR, "ap_warrior.parquet"))
    gt_df = pd.read_csv(os.path.join(GT_DIR, "Q2a.csv"))
    n_img = len(images_df)
    distinct_tracks = sorted(apw_df["Track"].unique().tolist())
    n_tracks = len(distinct_tracks)
    n_pairs = n_img * n_tracks
    print(f"  {n_img} images × {n_tracks} tracks = {n_pairs} pairs; GT {len(gt_df)} pairs")
    profile["data"] = {"n_images": n_img, "n_tracks": n_tracks,
                        "tracks": distinct_tracks, "n_pairs": n_pairs,
                        "n_gt_pairs": len(gt_df)}
    images_df["GcsUri"] = images_df["image_filename"].apply(
        lambda f: f"gs://<YOUR_GCP_PROJECT>-mmqa-images/{f}")

    # ── Stage 0: MarginSignal — logo classification ──
    cap_emb = np.array(images_df["embedding"].tolist(), dtype=np.float32)
    logo_signal = MarginSignal(positive_prompts=LOGO_PHRASES, negative_prompts=NONLOGO_PHRASES)
    # Use max-of-pos − max-of-neg (asymmetric), matching v1's "max" semantics
    # rather than MarginSignal's mean — so we recompute with max here.
    pos_logo = embed_query(LOGO_PHRASES)
    neg_logo = embed_query(NONLOGO_PHRASES)

    def _cossim(q, batch):
        from dase_cascade.runtime import cosine_sim_batch
        return cosine_sim_batch(q, batch)

    pos_best = np.maximum.reduce([_cossim(p, cap_emb) for p in pos_logo])
    neg_best = np.maximum.reduce([_cossim(n, cap_emb) for n in neg_logo])
    logo_margin = pos_best - neg_best

    # ── Stage 1: PairCosineSignal — image × track per-image top-1-GAP ──
    track_emb = embed_query(distinct_tracks)
    pair_signal = PairCosineSignal(embeddings_left=cap_emb, embeddings_right=track_emb)
    pair_track_sim = pair_signal._left @ pair_signal._right.T  # (n_img, n_tracks)

    candidate_pairs_for_bq = []
    n_dropped_not_logo = 0
    n_logo_candidate = 0
    candidates_per_image = []
    for i in range(n_img):
        if logo_margin[i] < LOGO_LO:
            n_dropped_not_logo += 1
            continue
        n_logo_candidate += 1
        thr = pair_track_sim[i].max() - TRACK_GAP
        keep_track_idx = [ti for ti in range(n_tracks) if pair_track_sim[i, ti] >= thr]
        candidates_per_image.append(len(keep_track_idx))
        uri = images_df.iloc[i]["GcsUri"]
        for ti in keep_track_idx:
            candidate_pairs_for_bq.append((distinct_tracks[ti], uri))
    print(f"  Stage 0 dropped (not logo): {n_dropped_not_logo} images")
    print(f"  Stage 1 candidates: {n_logo_candidate} logo images, "
          f"{len(candidate_pairs_for_bq)} pairs sent to BQ "
          f"(per-image candidate counts: min={min(candidates_per_image,default=0)}, "
          f"median={int(np.median(candidates_per_image)) if candidates_per_image else 0}, "
          f"max={max(candidates_per_image,default=0)})")

    dase_pairs = []  # no dase-only confident matches in pure prefilter design
    uncertain_pairs_for_bq = candidate_pairs_for_bq
    print(f"  dase confident pairs: {len(dase_pairs)}; uncertain pairs sent to BQ: {len(uncertain_pairs_for_bq)}")

    profile["dase_partition"] = {
        "n_dropped_not_logo": n_dropped_not_logo,
        "n_logo_candidate_images": n_logo_candidate,
        "n_pairs_to_bq": len(uncertain_pairs_for_bq),
        "candidates_per_image_stats": {
            "min": int(min(candidates_per_image, default=0)),
            "median": float(np.median(candidates_per_image)) if candidates_per_image else 0,
            "max": int(max(candidates_per_image, default=0)),
        },
    }

    client = bq_client(PROJECT)

    print("\n=== Per-pair cost calibration ===")
    sample_uris = [images_df.iloc[i]["GcsUri"] for i in range(min(5, n_img))]
    cal = per_row_cost_q2a(client, sample_uris, distinct_tracks[0])
    per_row = cal.per_row_cost_usd
    print(f"  per_pair=${per_row:.6f}")
    profile["calibration"] = cal.to_dict()

    if SKIP_BASELINE:
        bcost = (PAPER_BQ_Q2a["cost_usd"] if PAPER_BQ_Q2a["cost_usd"] is not None
                 else per_row * n_pairs)
        bwall = PAPER_BQ_Q2a["latency_s"] if PAPER_BQ_Q2a["latency_s"] is not None else None
        bcalls = n_pairs
        bscore = PAPER_BQ_Q2a["score"]
        b_pairs = None
        profile["baseline"] = {
            "_status": "skipped (full-Cartesian baseline is long-running)",
            "score": {"f1": bscore, "_source": "paper Table 4 (TBD mapping)"},
            "latency_breakdown": {"wall_s": bwall, "_source": "paper"},
            "cost_breakdown": {"n_llm_calls": bcalls, "per_pair_cost_usd": per_row,
                                "total_cost_usd": bcost,
                                "_source": "estimated (per_pair × n_pairs)"},
        }
    else:
        raise NotImplementedError("baseline run not implemented; SKIP_BASELINE=True")

    # ── Stage 2: AiIfVerifier (CTAS + AI.IF) ──
    if uncertain_pairs_for_bq:
        print(f"\n=== AiIfVerifier on {len(uncertain_pairs_for_bq)} pairs ===")
        verifier = make_q2a_verifier()
        vres = verifier.verify(client, uncertain_pairs_for_bq, per_row)
        print(f"  bq verified pairs: {len(vres.positive_ids)}")
        bq_pairs = []
        for pid in vres.positive_ids:
            track, uri = pid.split("|", 1)
            fn = os.path.basename(uri)
            bq_pairs.append((track, None, fn))
        print(f"  CTAS wall={vres.ctas_wall_s:.2f}s, AI.IF wall={vres.wall_s:.2f}s, slot_ms={vres.slot_ms}")
    else:
        from dase_cascade.verifier import VerifierResult
        vres = VerifierResult(positive_ids=set())
        bq_pairs = []
        print("\n  No uncertain pairs; skipping verifier.")

    s2_calls = vres.n_calls
    cascade_cost = vres.cost_usd

    # Combine: cascade pairs = dase confident + bq verified (track, filename) pairs
    cascade_track_to_files = {}
    for t, _, fn in dase_pairs + bq_pairs:
        cascade_track_to_files.setdefault(t, set()).add(fn)
    print(f"  cascade_track_to_files: {{t: count}} = {{ {', '.join(f'{t!r}: {len(s)}' for t,s in cascade_track_to_files.items())} }}")

    # Expand to (ID, image_filename) via ap_warrior
    apw_df_int = apw_df.copy()
    apw_df_int["ID"] = apw_df_int["ID"].astype(int)
    pred_rows = []
    for _, r in apw_df_int.iterrows():
        track = r["Track"]
        files = cascade_track_to_files.get(track, set())
        for fn in files:
            pred_rows.append({"ID": int(r["ID"]), "image_filename": fn})
    pred_df = (pd.DataFrame(pred_rows, columns=["ID", "image_filename"])
               .drop_duplicates().reset_index(drop=True))
    print(f"  cascade pairs after expand: {len(pred_df)}")

    # Eval
    def add_pair_id(df):
        out = df.copy()
        out["pair_id"] = out["ID"].astype(str) + "|" + out["image_filename"].astype(str)
        return out
    cscore = GenericEvaluator.compute_accuracy_score(
        "f1-score", add_pair_id(gt_df), add_pair_id(pred_df), id_column="pair_id")
    print(f"  cascade F1={cscore.f1_score:.4f}  P={cscore.precision:.4f}  R={cscore.recall:.4f}")

    cascade_total_wall = vres.ctas_wall_s + vres.wall_s
    profile["cascade"] = {
        "method": "MarginSignal logo filter + PairCosineSignal track-sim prefilter + AiIfVerifier (CTAS+AI.IF) + client-side expand to (ID, filename)",
        "verifier": vres.to_dict(),
        "cascade_track_to_n_files": {t: len(s) for t, s in cascade_track_to_files.items()},
        "cascade_n_pred_pairs": len(pred_df),
        "score": {"precision": cscore.precision, "recall": cscore.recall, "f1": cscore.f1_score},
        "totals": {"wall_s": cascade_total_wall, "slot_ms_bq_total": vres.ctas_slot_ms + vres.slot_ms,
                   "cost_usd": cascade_cost, "n_llm_calls": s2_calls},
    }

    profile["comparison"] = {
        "score": {"paper_BQ": PAPER_BQ_Q2a["score"], "paper_DASE_NN": PAPER_DASE_NN_Q2a["score"],
                   "ours_BQ": bscore if SKIP_BASELINE else None, "ours_cascade": cscore.f1_score},
        "wall_s": {"paper_BQ": PAPER_BQ_Q2a["latency_s"], "paper_DASE_NN": PAPER_DASE_NN_Q2a["latency_s"],
                    "ours_BQ": bwall, "ours_cascade": cascade_total_wall},
        "cost_usd": {"paper_BQ": PAPER_BQ_Q2a["cost_usd"], "paper_DASE_NN": PAPER_DASE_NN_Q2a["cost_usd"],
                      "ours_BQ": bcost, "ours_cascade": cascade_cost},
        "n_llm_calls": {"paper_BQ": n_pairs, "paper_DASE_NN": 0,
                         "ours_BQ": bcalls, "ours_cascade": s2_calls},
    }

    write_profile(profile, PROFILE_PATH)

    print_summary(
        f"MMQA Q2a (alpha={ALPHA})",
        columns=["paper BQ", "DASE+NN", "ours BQ", "ours cascade"],
        rows=[
            ("score (F1)", [PAPER_BQ_Q2a["score"], PAPER_DASE_NN_Q2a["score"], bscore, cscore.f1_score], ".2f"),
            ("cost ($)",   [PAPER_BQ_Q2a["cost_usd"], PAPER_DASE_NN_Q2a["cost_usd"], bcost, cascade_cost], ".4f"),
            ("#LLM calls", [n_pairs, 0, bcalls, s2_calls], "d"),
        ],
    )


if __name__ == "__main__":
    main()
