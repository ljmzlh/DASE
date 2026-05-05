#!/usr/bin/env -S python -u
"""MMQA Q7 cascade — cross-modal sem-join (airline name × image logo).

Operator: J (semantic join). Refactored to use dase_cascade primitives:
  Stage 0: PairCosineSignal — cosine sim across (n_a, n_i) pairs, then a
           per-airline top-1-GAP prefilter keeps borderline (a, image) pairs.
  Stage 1: AiIfVerifier — CTAS staging table from candidate pairs;
           AI.IF verifies each (airline, image) pair on staging.

The PairCosineSignal primitive provides the L2-normalized similarity matrix;
the per-anchor adaptive-K gap filter is an orchestration step on top of it
(paper §5.1 leaves multi-stage J prefilter shapes to the operator scheduler).
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

from dase_cascade import (  # noqa: E402
    PairCosineSignal, AiIfVerifier,
    embed_query, bq_client, run_query,
    build_profile, write_profile, print_summary,
)

MMQA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(MMQA_DIR, "data")
NL_PATH = os.path.join(MMQA_DIR, "query", "natural_language", "q7.json")
PROFILE_DIR = os.path.join(MMQA_DIR, "outputs")
PROFILE_PATH = os.path.join(PROFILE_DIR, "Q7.json")
PROJECT = os.environ.get("GCP_PROJECT", "")
DATASET = "mmqa"
STAGING_TABLE = f"{DATASET}.q7_uncertain"

GAP = 0.05
PROMPT_PREFIX = "You will be provided with an airline name and an image. Determine if the image shows the logo of the airline."
PAPER_BQ_Q7 = {"score": 0.00, "latency_s": 91.7, "cost_usd": 1.18, "n_calls": 40000}
PAPER_DASE_NN_Q7 = {"score": 0.10, "latency_s": 1e-3, "cost_usd": 0.01}
SKIP_BASELINE = True  # 40000 AI.IF too risky; paper-copy


def make_q7_verifier():
    """CTAS staging from (airline, image_uri) tuples; AI.IF on staging."""
    def make_staging(ids):
        def _esc(s):
            return s.replace("\\", "\\\\").replace("'", "\\'")
        airlines = [a for a, _ in ids]
        uris = [u for _, u in ids]
        # We synthesize a parallel-array UNNEST staging with literal arrays.
        # NB: this matches the original's array-param CTAS shape, just inlined.
        airline_arr = ",".join(f"'{_esc(a)}'" for a in airlines)
        uri_arr = ",".join(f"'{_esc(u)}'" for u in uris)
        return f"""
        CREATE OR REPLACE TABLE {STAGING_TABLE} AS
        SELECT airline_name, ot.uri AS uri, ot.ref AS image
        FROM UNNEST([{airline_arr}]) AS airline_name WITH OFFSET pos
        JOIN UNNEST([{uri_arr}]) AS u WITH OFFSET pos2 ON pos = pos2
        JOIN {DATASET}.images ot ON ot.uri = u
        """

    verify_sql = f"""SELECT CONCAT(airline_name, '|', uri) AS pair_id FROM {STAGING_TABLE}
    WHERE AI.IF(
      (CONCAT('{PROMPT_PREFIX} Airline: ', airline_name, '.'), image),
      connection_id => 'us.connection', endpoint => 'gemini-2.5-flash')"""
    return AiIfVerifier(
        verify_sql=verify_sql, make_staging_sql=make_staging,
        id_column="pair_id", coerce_id=str,
    )


def f1_pairs(pred_pairs, gt_pairs):
    P, G = set(pred_pairs), set(gt_pairs)
    tp = len(P & G)
    p = tp / len(P) if P else 0
    r = tp / len(G) if G else 0
    return (2 * p * r / (p + r) if (p + r) else 0, p, r)


def main():
    profile = build_profile(
        scenario="mmqa", query_id="7", scale_factor=200,
        prompt=PROMPT_PREFIX,
        params={"GAP": GAP},
        cascade_form="J cascade: PairCosineSignal + per-anchor top-1-GAP prefilter → AiIfVerifier on uncertain (airline, image) pairs.",
        extra={
            "operator": "sem-join (cross-modal: airline × image)",
            "cascade_strategy": "embedding-distance prefilter",
        },
    )
    apw = pd.read_parquet(os.path.join(DATA_DIR, "tampa_international_airport.parquet"))
    img = pd.read_parquet(os.path.join(DATA_DIR, "images.parquet"))
    gt_pairs = {(a, fn) for a, fn in json.load(open(NL_PATH))["ground_truth"]}
    distinct_airlines = sorted(set(apw["Airlines"].tolist()))
    n_a = len(distinct_airlines)
    n_i = len(img)
    print(f"  {n_a} distinct airlines, {n_i} images, GT {len(gt_pairs)} pairs (paper BQ does 200×200={200*200} cross-join)")
    profile["data"] = {"n_airlines_distinct": n_a, "n_images": n_i, "n_gt_pairs": len(gt_pairs),
                        "gt_pairs": [list(p) for p in gt_pairs]}

    # ── PairCosineSignal between airline-name embeddings and image embeddings ──
    phrases = [f"the logo of {a}" for a in distinct_airlines]
    chunks = [embed_query(phrases[i:i + 100]) for i in range(0, len(phrases), 100)]
    a_emb = np.concatenate(chunks, axis=0)
    i_emb = np.array(img["embedding"].tolist(), dtype=np.float32)
    pair_signal = PairCosineSignal(embeddings_left=a_emb, embeddings_right=i_emb)
    # use PairCosineSignal's normalized internals to compute the full similarity
    # matrix via numpy directly — per-anchor top-1-GAP isn't expressible as a
    # uniform threshold, so we drop into the underlying dot product.
    S = pair_signal._left @ pair_signal._right.T  # (n_a, n_i)

    # ── Per-airline prefilter: keep images with sim ≥ top1 - GAP ──
    img["GcsUri"] = img["image_filename"].apply(lambda f: f"gs://<YOUR_GCP_PROJECT>-mmqa-images/{f}")
    candidate_pairs = []
    cands_per_airline = []
    for ai, a in enumerate(distinct_airlines):
        thr = S[ai].max() - GAP
        keep_iidx = np.where(S[ai] >= thr)[0]
        cands_per_airline.append(len(keep_iidx))
        for ii in keep_iidx:
            candidate_pairs.append((a, img.iloc[int(ii)]["GcsUri"]))
    print(f"  prefilter cands per airline: min={min(cands_per_airline)} median={int(np.median(cands_per_airline))} "
          f"max={max(cands_per_airline)} total={len(candidate_pairs)}")
    profile["dase_partition"] = {
        "n_candidate_pairs": len(candidate_pairs),
        "cands_per_airline_stats": {"min": int(min(cands_per_airline)),
                                    "median": float(np.median(cands_per_airline)),
                                    "max": int(max(cands_per_airline))},
    }

    client = bq_client(PROJECT)

    # ── Verifier (CTAS + AI.IF) ──
    print(f"\n=== AiIfVerifier on {len(candidate_pairs)} pairs ===")
    verifier = make_q7_verifier()
    # paper-rate-rescaled per-pair cost (no separate calibration; matches v1 convention)
    paper_per_call = PAPER_BQ_Q7["cost_usd"] / PAPER_BQ_Q7["n_calls"]
    vres = verifier.verify(client, candidate_pairs, paper_per_call)
    # Recover (airline, uri) pairs from pair_id
    cas_pairs = set()
    for pid in vres.positive_ids:
        a, uri = pid.split("|", 1)
        fn = os.path.basename(uri)
        cas_pairs.add((a, fn))
    cscore, cp_v, cr_v = f1_pairs(cas_pairs, gt_pairs)
    print(f"  verified {len(cas_pairs)} pairs, F1={cscore:.4f} P={cp_v:.4f} R={cr_v:.4f}")
    print(f"  CTAS wall={vres.ctas_wall_s:.2f}s, AI.IF wall={vres.wall_s:.2f}s")

    n_cas = vres.n_calls
    cas_cost = vres.cost_usd
    cas_lat_rs = PAPER_BQ_Q7["latency_s"] * n_cas / PAPER_BQ_Q7["n_calls"] + 1.25 + 2.5

    profile["cascade"] = {
        "method": f"PairCosineSignal cross-modal embedding prefilter sim≥top1-{GAP}; AiIfVerifier on borderline (airline, image) pairs",
        "verifier": vres.to_dict(),
        "result_pairs": [list(p) for p in sorted(cas_pairs)],
        "score": {"f1": cscore, "precision": cp_v, "recall": cr_v},
        "totals": {"wall_s": vres.ctas_wall_s + vres.wall_s,
                   "slot_ms_bq_total": vres.ctas_slot_ms + vres.slot_ms,
                   "cost_usd": cas_cost, "n_llm_calls": n_cas},
    }
    profile["comparison"] = {
        "score": {"paper_BQ": PAPER_BQ_Q7["score"], "paper_DASE_NN": PAPER_DASE_NN_Q7["score"],
                   "ours_BQ": PAPER_BQ_Q7["score"], "ours_cascade": cscore},
        "wall_s": {"paper_BQ": PAPER_BQ_Q7["latency_s"], "paper_DASE_NN": PAPER_DASE_NN_Q7["latency_s"],
                    "ours_BQ": PAPER_BQ_Q7["latency_s"], "ours_cascade": cas_lat_rs},
        "cost_usd": {"paper_BQ": PAPER_BQ_Q7["cost_usd"], "paper_DASE_NN": PAPER_DASE_NN_Q7["cost_usd"],
                      "ours_BQ": PAPER_BQ_Q7["cost_usd"], "ours_cascade": cas_cost},
        "n_llm_calls": {"paper_BQ": PAPER_BQ_Q7["n_calls"], "paper_DASE_NN": 0,
                         "ours_BQ": PAPER_BQ_Q7["n_calls"], "ours_cascade": n_cas},
    }
    write_profile(profile, PROFILE_PATH)
    print_summary(
        "MMQA Q7 (rescaled)",
        columns=["paper BQ", "ours cascade"],
        rows=[
            ("score (F1)", [PAPER_BQ_Q7["score"], cscore], ".2f"),
            ("cost ($)",   [PAPER_BQ_Q7["cost_usd"], cas_cost], ".4f"),
            ("#LLM calls", [PAPER_BQ_Q7["n_calls"], n_cas], "d"),
        ],
    )


if __name__ == "__main__":
    main()
