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

from dase_cascade import (  # noqa: E402
    PairCosineSignal, AiIfVerifier,
    embed_query, bq_client, per_row_cost,
    build_profile, write_profile,
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


def per_row_cost_q7(client, sample_uris, sample_airline):
    return per_row_cost(
        client,
        prompt=f"{PROMPT_PREFIX} Airline: {sample_airline}.",
        sample_uris=sample_uris,
        ext_table=f"{DATASET}.images",
        method_label="AI.GENERATE_BOOL on images.ref + thinking_budget=0 (airline inlined in prompt)",
        k=len(sample_uris),
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
    dase_started = time.perf_counter()
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
    dase_wall = time.perf_counter() - dase_started

    client = bq_client(PROJECT)

    # ── Verifier (CTAS + AI.IF) ──
    print(f"\n=== AiIfVerifier on {len(candidate_pairs)} pairs ===")
    if not candidate_pairs:
        raise RuntimeError("DASE prefilter produced no candidate pairs")
    verifier = make_q7_verifier()
    sample_uris = list(dict.fromkeys(uri for _, uri in candidate_pairs))[:5]
    calibration = per_row_cost_q7(client, sample_uris, candidate_pairs[0][0])
    profile["calibration"] = calibration.to_dict()
    print(f"  calibrated verifier cost=${calibration.per_row_cost_usd:.6f}/pair "
          f"on {calibration.n_sample} images")
    verifier_started = time.perf_counter()
    vres = verifier.verify(client, candidate_pairs, calibration.per_row_cost_usd)
    verifier_wall = time.perf_counter() - verifier_started

    # Recover (airline, uri) pairs from pair_id
    cas_pairs = set()
    for pair_id in vres.positive_ids:
        airline, uri = pair_id.split("|", 1)
        cas_pairs.add((airline, os.path.basename(uri)))
    cscore, cp_v, cr_v = f1_pairs(cas_pairs, gt_pairs)
    print(f"  verified {len(cas_pairs)} pairs, F1={cscore:.4f} P={cp_v:.4f} R={cr_v:.4f}")
    print(f"  CTAS wall={vres.ctas_wall_s:.2f}s, AI.IF wall={vres.wall_s:.2f}s")


    profile["cascade"] = {
        "method": f"PairCosineSignal cross-modal embedding prefilter sim≥top1-{GAP}; AiIfVerifier on candidate pairs",
        "verifier": vres.to_dict(),
        "result_pairs": [list(pair) for pair in sorted(cas_pairs)],
        "score": {"f1": cscore, "precision": cp_v, "recall": cr_v},
        "totals": {
            "wall_s": dase_wall + verifier_wall,
            "wall_breakdown_s": {"dase": dase_wall, "bq_verifier": verifier_wall},
            "slot_ms_bq_total": vres.ctas_slot_ms + vres.slot_ms,
            "cost_usd": vres.cost_usd,
            "n_llm_calls": vres.n_calls,
        },
    }

    write_profile(profile, PROFILE_PATH)


if __name__ == "__main__":
    main()
