#!/usr/bin/env -S python -u
"""
MMQA Q2b cascade — q2a logo cross-join (J) + color extraction (M).

NL: Same as Q2a + "What is the color of each logo?"
GT: 5 triples (ID, image_filename, color="blue")

Operator: J + M (compose). Refactored to use dase_cascade primitives:

  Stage A — sem-join (logo cross-join, identical to q2a_v2):
    MarginSignal(logo) drops not-logo images; PairCosineSignal(image × track)
    per-image top-1-GAP prefilter; AiIfVerifier on candidate (track, image) pairs.
    Expand verified pairs via ap_warrior.Track to (ID, image_filename).

  Stage B — sem-extract (color attribute):
    Cluster (ID, image) pairs by image_filename (trivial equality cluster:
    same image always same color). For each unique image_filename, run ONE
    AI.GENERATE("logo color?", image) call → propagate to all (ID, image)
    pairs sharing that image. The "cluster" reduces |pairs| → |unique images|
    AI.GENERATE calls; we use a hand-rolled SEM_MAP-on-uris query because the
    ClusterCascade primitive is built on row embeddings, not on a pre-known
    {pair → group} mapping.
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
    MarginSignal, PairCosineSignal, AiIfVerifier, AiGenerateVerifier,
    embed_query, bq_client, per_row_cost, run_query,
    build_profile, write_profile, print_summary,
)

MMQA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(MMQA_DIR, "data")
GT_DIR = os.path.join(MMQA_DIR, "ground_truth")
PROFILE_DIR = os.path.join(MMQA_DIR, "outputs")
PROFILE_PATH = os.path.join(PROFILE_DIR, "Q2b.json")

PROJECT = os.environ.get("GCP_PROJECT", "")
DATASET = "mmqa"
STAGING_TABLE_A = f"{DATASET}.q2b_uncertain_pairs"

# Stage A — same as q2a_cascade
LOGO_PROMPT_PREFIX = "You will be provided with a horse racetrack name and an image. Determine if the image shows the logo of the racetrack."
LOGO_PHRASES = [
    "a logo or wordmark as the main subject of the image",
    "a brand emblem or typographic logo as the primary visual focus",
]
NONLOGO_PHRASES = [
    "a photograph where the main subject is not a logo, such as people, animals, or action",
    "a candid or landscape image where no logo or wordmark is the primary focus",
]
LOGO_LO = 0.02
TRACK_GAP = 0.10

# Stage B — color extraction
COLOR_PROMPT = "What is the dominant color of the logo in this image? Only respond with the color name."

PAPER_BQ_Q2b = {"score": None, "latency_s": None, "cost_usd": None}  # TBD
PAPER_DASE_NN_Q2b = {"score": None, "latency_s": 1e-3, "cost_usd": 1e-9}
SKIP_BASELINE = True


def make_stage_a_verifier():
    """CTAS staging from candidate (track, uri) pairs; AI.IF on staging.
    Returns positive pair_ids in 'track|uri' format."""
    def make_staging(ids):
        def _esc(s):
            return s.replace("\\", "\\\\").replace("'", "\\'")
        tracks = [t for t, _ in ids]
        uris = [u for _, u in ids]
        track_arr = ",".join(f"'{_esc(t)}'" for t in tracks)
        uri_arr = ",".join(f"'{_esc(u)}'" for u in uris)
        return f"""
        CREATE OR REPLACE TABLE {STAGING_TABLE_A} AS
        SELECT track, ot.uri AS uri, ot.ref AS image
        FROM UNNEST([{track_arr}]) AS track WITH OFFSET pos
        JOIN UNNEST([{uri_arr}]) AS u WITH OFFSET pos2 ON pos = pos2
        JOIN {DATASET}.images ot ON ot.uri = u
        """
    verify_sql = f"""
    SELECT CONCAT(track, '|', uri) AS pair_id
    FROM {STAGING_TABLE_A}
    WHERE AI.IF(
      (CONCAT('{LOGO_PROMPT_PREFIX} Racetrack: ', track, '.'), image),
      connection_id => 'us.connection', endpoint => 'gemini-2.5-flash')
    """
    return AiIfVerifier(
        verify_sql=verify_sql, make_staging_sql=make_staging,
        id_column="pair_id", coerce_id=str,
    )


def make_color_verifier():
    """AI.GENERATE color extraction on a list of image_uris (one call per uri)."""
    def verify_sql_template(uris):
        if not uris:
            return "SELECT '' AS uri, '' AS color FROM UNNEST([]) WHERE FALSE"
        def _esc(s):
            return s.replace("\\", "\\\\").replace("'", "\\'")
        rows_sql = []
        for i, uri in enumerate(uris):
            rows_sql.append(f"""
            SELECT '{_esc(uri)}' AS uri, AI.GENERATE(
              ('{COLOR_PROMPT}', ot.ref),
              connection_id => 'us.connection',
              endpoint => 'gemini-2.5-flash'
            ).result AS color
            FROM {DATASET}.images ot
            WHERE ot.uri = '{_esc(uri)}'""")
        return " UNION ALL ".join(rows_sql)

    return AiGenerateVerifier(
        verify_sql_template=verify_sql_template,
        id_column="uri", value_column="color",
        coerce_id=str, coerce_value=lambda v: str(v).strip().lower(),
    )


def per_row_cost_q2b(client, sample_uris, sample_track):
    """Image (ot.ref) + track string in prompt, mirrors q2a calibration."""
    return per_row_cost(
        client,
        prompt=f"{LOGO_PROMPT_PREFIX} Racetrack: {sample_track}.",
        sample_uris=sample_uris,
        ext_table=f"{DATASET}.images",
        method_label="AI.GENERATE_BOOL on images.ref + thinking_budget=0 (track inlined in prompt)",
        k=len(sample_uris),
    )


def main():
    profile = build_profile(
        scenario="mmqa", query_id="2b", scale_factor=200,
        params={"LOGO_LO": LOGO_LO, "TRACK_GAP": TRACK_GAP},
        cascade_form=(
            "Stage A J cascade: MarginSignal(logo) + PairCosineSignal(image × track) "
            "+ AiIfVerifier on (track, image) pairs. Stage B M cascade: cluster by "
            "image_filename + AiGenerateVerifier (color) on unique images; propagate."
        ),
        extra={
            "logo_prompt_prefix": LOGO_PROMPT_PREFIX,
            "color_prompt": COLOR_PROMPT,
            "dase_prompts": {"logo_positive": LOGO_PHRASES, "logo_negative": NONLOGO_PHRASES},
        },
    )

    print("Loading images + ap_warrior + GT...")
    images_df = pd.read_parquet(os.path.join(DATA_DIR, "images.parquet"))
    apw_df = pd.read_parquet(os.path.join(DATA_DIR, "ap_warrior.parquet"))
    gt_df = pd.read_csv(os.path.join(GT_DIR, "Q2b.csv"))
    n_img = len(images_df)
    distinct_tracks = sorted(apw_df["Track"].unique().tolist())
    n_tracks = len(distinct_tracks)
    n_pairs = n_img * n_tracks
    print(f"  {n_img} images × {n_tracks} tracks = {n_pairs} pairs; GT {len(gt_df)} triples")
    profile["data"] = {"n_images": n_img, "n_tracks": n_tracks,
                        "tracks": distinct_tracks, "n_pairs": n_pairs,
                        "n_gt_triples": len(gt_df)}
    images_df["GcsUri"] = images_df["image_filename"].apply(
        lambda f: f"gs://<YOUR_GCP_PROJECT>-mmqa-images/{f}")

    # ── Stage 0: MarginSignal — logo classification (max-of-pos − max-of-neg) ──
    cap_emb = np.array(images_df["embedding"].tolist(), dtype=np.float32)
    pos_logo = embed_query(LOGO_PHRASES)
    neg_logo = embed_query(NONLOGO_PHRASES)
    from dase_cascade.runtime import cosine_sim_batch
    pos_best = np.maximum.reduce([cosine_sim_batch(p, cap_emb) for p in pos_logo])
    neg_best = np.maximum.reduce([cosine_sim_batch(n, cap_emb) for n in neg_logo])
    logo_margin = pos_best - neg_best

    # ── Stage 1: PairCosineSignal — image × track per-image top-1-GAP ──
    track_emb = embed_query(distinct_tracks)
    pair_signal = PairCosineSignal(embeddings_left=cap_emb, embeddings_right=track_emb)
    pair_track_sim = pair_signal._left @ pair_signal._right.T

    candidate_pairs = []
    n_dropped = 0
    n_logo_cand = 0
    cands_per_img = []
    for i in range(n_img):
        if logo_margin[i] < LOGO_LO:
            n_dropped += 1
            continue
        n_logo_cand += 1
        thr = pair_track_sim[i].max() - TRACK_GAP
        keep = [ti for ti in range(n_tracks) if pair_track_sim[i, ti] >= thr]
        cands_per_img.append(len(keep))
        uri = images_df.iloc[i]["GcsUri"]
        for ti in keep:
            candidate_pairs.append((distinct_tracks[ti], uri))
    print(f"  Stage 1 dropped (not logo): {n_dropped}")
    print(f"  Stage 2 candidates: {n_logo_cand} logo images, {len(candidate_pairs)} pairs to BQ "
          f"(per-image: min={min(cands_per_img,default=0)}, median={int(np.median(cands_per_img)) if cands_per_img else 0}, max={max(cands_per_img,default=0)})")
    profile["dase_partition"] = {
        "n_dropped_not_logo": n_dropped, "n_logo_candidate_images": n_logo_cand,
        "n_pairs_to_bq": len(candidate_pairs),
    }

    client = bq_client(PROJECT)

    print("\n=== Cost calibration ===")
    sample_uris = [images_df.iloc[i]["GcsUri"] for i in range(min(5, n_img))]
    cal = per_row_cost_q2b(client, sample_uris, distinct_tracks[0])
    per_pair_cost = cal.per_row_cost_usd
    print(f"  per_pair=${per_pair_cost:.6f}")
    profile["calibration"] = cal.to_dict()

    # ── Stage A — AiIfVerifier on candidate pairs ──
    print(f"\n=== Stage A: AiIfVerifier on {len(candidate_pairs)} pairs ===")
    verifier_a = make_stage_a_verifier()
    vres_a = verifier_a.verify(client, candidate_pairs, per_pair_cost)
    print(f"  CTAS wall={vres_a.ctas_wall_s:.2f}s, AI.IF wall={vres_a.wall_s:.2f}s, verified={len(vres_a.positive_ids)}")

    # Recover (track, uri, fn) triples from pair_ids
    sa_rows = []
    for pid in vres_a.positive_ids:
        track, uri = pid.split("|", 1)
        sa_rows.append({"track": track, "uri": uri, "image_filename": os.path.basename(uri)})
    sa2_df = pd.DataFrame(sa_rows, columns=["track", "uri", "image_filename"])

    # Expand to (ID, image_filename) via ap_warrior
    track_to_files = {}
    for _, r in sa2_df.iterrows():
        track_to_files.setdefault(r["track"], set()).add(r["image_filename"])
    apw_df_int = apw_df.copy()
    apw_df_int["ID"] = apw_df_int["ID"].astype(int)
    pred_pairs = []
    for _, r in apw_df_int.iterrows():
        for fn in track_to_files.get(r["Track"], set()):
            pred_pairs.append({"ID": int(r["ID"]), "image_filename": fn})
    pred_pairs_df = pd.DataFrame(pred_pairs, columns=["ID", "image_filename"]).drop_duplicates().reset_index(drop=True)
    print(f"  Stage A output: {len(pred_pairs_df)} (ID, image) pairs")

    # ── Stage B — AiGenerateVerifier (color) on unique image URIs ──
    unique_images = pred_pairs_df["image_filename"].unique().tolist() if len(pred_pairs_df) else []
    print(f"\n=== Stage B: AiGenerateVerifier(color) on {len(unique_images)} unique images ===")
    if unique_images:
        unique_uris = [f"gs://<YOUR_GCP_PROJECT>-mmqa-images/{fn}" for fn in unique_images]
        verifier_b = make_color_verifier()
        vres_b = verifier_b.verify(client, unique_uris, per_pair_cost)
        # color_map keyed by uri; convert to filename
        color_by_fn = {os.path.basename(u): c for u, c in vres_b.values.items()}
        print(f"  AI.GENERATE wall={vres_b.wall_s:.2f}s, slot_ms={vres_b.slot_ms}, colors={color_by_fn}")
    else:
        from dase_cascade.verifier import VerifierResult
        vres_b = VerifierResult(positive_ids=set())
        color_by_fn = {}
        print("  (no unique images; Stage B skipped)")

    # ── Build cascade triples ──
    cascade_rows = []
    for _, r in pred_pairs_df.iterrows():
        c = color_by_fn.get(r["image_filename"], "")
        cascade_rows.append({"ID": int(r["ID"]), "image_filename": r["image_filename"], "color": c})
    cascade_df = pd.DataFrame(cascade_rows, columns=["ID", "image_filename", "color"])
    print(f"  cascade triples: {len(cascade_df)}")
    if len(cascade_df) <= 10:
        print(cascade_df.to_string(index=False))

    # ── Eval — F1 on (ID, image, color) triple ──
    def add_triple_id(df):
        out = df.copy()
        out["triple_id"] = (out["ID"].astype(str) + "|" + out["image_filename"].astype(str)
                            + "|" + out["color"].astype(str).str.lower())
        return out
    cscore = GenericEvaluator.compute_accuracy_score(
        "f1-score", add_triple_id(gt_df), add_triple_id(cascade_df), id_column="triple_id")
    print(f"  cascade F1={cscore.f1_score:.4f}  P={cscore.precision:.4f}  R={cscore.recall:.4f}")

    # ── Cost / wall accounting ──
    n_calls_stage_a = vres_a.n_calls
    n_calls_stage_b = vres_b.n_calls
    cascade_cost = per_pair_cost * (n_calls_stage_a + n_calls_stage_b)
    cascade_wall = vres_a.ctas_wall_s + vres_a.wall_s + vres_b.wall_s
    cascade_slot = vres_a.ctas_slot_ms + vres_a.slot_ms + vres_b.slot_ms

    if SKIP_BASELINE:
        bcost = (PAPER_BQ_Q2b["cost_usd"] if PAPER_BQ_Q2b["cost_usd"] is not None
                 else per_pair_cost * (n_pairs + 5))  # 1200 AI.IF + 5 AI.GENERATE
        bwall = PAPER_BQ_Q2b["latency_s"]
        bscore = PAPER_BQ_Q2b["score"]
        bcalls = n_pairs + 5
        profile["baseline"] = {
            "_status": "skipped (1200 cross-join AI.IF + 5 AI.GENERATE)",
            "score": {"f1": bscore, "_source": "paper Table 4 (TBD mapping)"},
            "latency_breakdown": {"wall_s": bwall},
            "cost_breakdown": {"n_llm_calls": bcalls, "per_pair_cost_usd": per_pair_cost,
                                "total_cost_usd": bcost, "_source": "estimated"},
        }

    profile["cascade"] = {
        "method": "Stage A J cascade (MarginSignal+PairCosineSignal+AiIfVerifier) + Stage B M cascade (image-cluster + AiGenerateVerifier color)",
        "stage_a_verifier": vres_a.to_dict(),
        "stage_a_n_pairs_to_bq": n_calls_stage_a,
        "stage_a_n_verified": len(sa2_df),
        "stage_a_pred_pairs": len(pred_pairs_df),
        "stage_b_verifier": vres_b.to_dict(),
        "stage_b_n_unique_images": n_calls_stage_b,
        "stage_b_color_map": color_by_fn,
        "score": {"precision": cscore.precision, "recall": cscore.recall, "f1": cscore.f1_score},
        "totals": {"wall_s": cascade_wall, "slot_ms_bq_total": cascade_slot,
                   "cost_usd": cascade_cost,
                   "n_llm_calls": n_calls_stage_a + n_calls_stage_b,
                   "n_llm_calls_breakdown": {"stage_a_aiif": n_calls_stage_a,
                                              "stage_b_aigen": n_calls_stage_b}},
    }

    profile["comparison"] = {
        "score": {"paper_BQ": PAPER_BQ_Q2b["score"], "paper_DASE_NN": PAPER_DASE_NN_Q2b["score"],
                   "ours_BQ": bscore, "ours_cascade": cscore.f1_score},
        "wall_s": {"paper_BQ": PAPER_BQ_Q2b["latency_s"], "paper_DASE_NN": PAPER_DASE_NN_Q2b["latency_s"],
                    "ours_BQ": bwall, "ours_cascade": cascade_wall},
        "cost_usd": {"paper_BQ": PAPER_BQ_Q2b["cost_usd"], "paper_DASE_NN": PAPER_DASE_NN_Q2b["cost_usd"],
                      "ours_BQ": bcost, "ours_cascade": cascade_cost},
        "n_llm_calls": {"paper_BQ": n_pairs + 5, "paper_DASE_NN": 0,
                         "ours_BQ": bcalls, "ours_cascade": n_calls_stage_a + n_calls_stage_b},
    }

    write_profile(profile, PROFILE_PATH)

    print_summary(
        "MMQA Q2b",
        columns=["paper BQ", "DASE+NN", "ours BQ", "ours cascade"],
        rows=[
            ("score (F1)", [PAPER_BQ_Q2b["score"], PAPER_DASE_NN_Q2b["score"], bscore, cscore.f1_score], ".2f"),
            ("cost ($)",   [PAPER_BQ_Q2b["cost_usd"], PAPER_DASE_NN_Q2b["cost_usd"], bcost, cascade_cost], ".4f"),
            ("#LLM calls", [n_pairs + 5, 0, bcalls, n_calls_stage_a + n_calls_stage_b], "d"),
        ],
    )


if __name__ == "__main__":
    main()
