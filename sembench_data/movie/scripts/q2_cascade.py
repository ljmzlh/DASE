#!/usr/bin/env -S python -u
"""
Movie Q2 cascade v2 — top-K positive reviews for `taken_3` (text F + L).

NL: Five positive reviews for movie 'taken_3'. Return reviewId.
GT: Reviews where id='taken_3' AND scoreSentiment='POSITIVE'.
Eval: _retrieval_limit, limit=5.

Refactored to use dase_cascade. Operator (paper Table 3): F + L.
The TopKBand selects K_CANDIDATES candidates by margin desc on the taken_3
subset; the verifier runs a single BQ AI.IF with WHERE reviewId IN (top-K)
AND id='taken_3' LIMIT TARGET. BQ short-circuits at LIMIT.
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import (
    Cascade, MarginSignal, TopKBand, AiIfVerifier,
    bq_client, per_row_cost, run_query,
    build_profile, write_profile,
)
import evaluator as ev  # sembench's per-Q evaluator (movie Q2: _retrieval_limit)

MOVIE_DIR    = os.path.abspath(os.path.join(_HERE, ".."))
EMB_PATH     = os.path.join(MOVIE_DIR, "data", "review_embeddings.npz")
REVIEWS_CSV  = os.path.join(MOVIE_DIR, "cache", "Reviews.csv")
PROFILE_PATH = os.path.join(MOVIE_DIR, "outputs", "Q2.json")

PROJECT  = os.environ.get("GCP_PROJECT", "")
PROMPT   = "Determine if the following movie review is clearly positive, review: "
MOVIE_ID = "taken_3"

POSITIVE = [
    "this is a clearly positive movie review",
    "the reviewer praises the film and recommends it",
    "an enthusiastic, favorable review of the movie",
]
NEGATIVE = [
    "this is a clearly negative movie review",
    "the reviewer criticizes the film and dislikes it",
    "an unfavorable, dismissive review of the movie",
]

K_CANDIDATES = 10
TARGET = 5


def make_q2_verifier():
    """Single AI.IF query: WHERE reviewId IN (top-K) AND id='taken_3' LIMIT TARGET."""
    def verify_sql_template(rids):
        rid_list = ",".join(str(int(r)) for r in rids)
        return f"""
        SELECT reviewId AS id FROM movie.reviews AS r
        WHERE reviewId IN ({rid_list}) AND r.id = '{MOVIE_ID}'
          AND AI.IF(('{PROMPT}', r.reviewText),
                    connection_id => 'us.connection',
                    endpoint => 'gemini-2.5-flash')
        LIMIT {TARGET}
        """
    return AiIfVerifier(verify_sql_template=verify_sql_template, id_column="id", coerce_id=int)




def per_row_cost_movie_q2(client):
    """Calibration uses AI.GENERATE_BOOL on movie.reviews scoped to id='taken_3' LIMIT 10
    (matches the existing Q2 calibration). Inline as raw text_from_table_sql."""
    THINKING = "JSON '{\"generation_config\":{\"thinking_config\":{\"thinking_budget\":0}}}'"
    sql = f"""
    SELECT AI.GENERATE_BOOL(
      ('{PROMPT}', r.reviewText),
      connection_id => 'us.connection',
      endpoint => 'gemini-2.5-flash',
      model_params => {THINKING}
    ) AS verdict
    FROM movie.reviews AS r
    WHERE r.id = '{MOVIE_ID}'
    LIMIT 10
    """
    return per_row_cost(
        client, prompt=PROMPT, text_from_table_sql=sql,
        method_label=f"AI.GENERATE_BOOL on movie.reviews (id='{MOVIE_ID}') + thinking_budget=0", k=10,
    )


def main():
    profile = build_profile(
        scenario="movie", query_id=2, scale_factor=2000,
        prompt=PROMPT, params={"K_candidates": K_CANDIDATES, "target": TARGET},
        cascade_form="F+L cascade: MarginSignal + TopKBand + AiIfVerifier (single BQ AI.IF w/ IN(top-K) + id filter + LIMIT short-circuit)",
        extra={
            "structural_filter": f"id = '{MOVIE_ID}'",
            "dase_prompts": {"positive": POSITIVE, "negative": NEGATIVE},
        },
    )

    print(f"Loading data + filtering to {MOVIE_ID} subset...")
    review_emb_full = np.load(EMB_PATH)["reviewText_emb"]
    df_full = pd.read_csv(REVIEWS_CSV)
    keep = ~df_full["reviewId"].duplicated()
    df_full = df_full[keep].reset_index(drop=True)
    review_emb_full = review_emb_full[keep.values]
    sub = (df_full["id"] == MOVIE_ID).values
    df = df_full[sub].reset_index(drop=True)
    review_emb = review_emb_full[sub]
    n_total = len(df)
    n_gt_pos = int((df["scoreSentiment"] == "POSITIVE").sum())
    print(f"  {MOVIE_ID}: {n_total} reviews, GT positive: {n_gt_pos}")
    profile["data"] = {
        "n_reviews_total_dedup": len(df_full),
        "n_reviews_in_scope": n_total,
        "n_gt_positive_in_scope": n_gt_pos,
    }

    client = bq_client(PROJECT)

    print("\n=== Per-row cost calibration ===")
    cal = per_row_cost_movie_q2(client)
    per_row = cal.per_row_cost_usd
    print(f"  per_row=${per_row:.6f}")
    profile["calibration"] = cal.to_dict()

    # ── Cascade: MarginSignal + TopKBand + AiIfVerifier ──
    cascade = Cascade(
        embeddings=review_emb,
        ids=df["reviewId"].astype(int).tolist(),
        signal=MarginSignal(positive_prompts=POSITIVE, negative_prompts=NEGATIVE),
        band=TopKBand(k=K_CANDIDATES),
        verifier=make_q2_verifier(),
    )
    print("\n=== Cascade (MarginSignal → TopKBand → AiIfVerifier) ===")
    cres = cascade.run(client, per_row)

    accepted = sorted(cres.verifier_result.positive_ids)
    sys_df = pd.DataFrame({"reviewId": accepted})
    cmetric = ev.evaluate_q2(sys_df)
    cwall = cres.total_wall_s
    # n_calls: BQ counted candidates, but LIMIT short-circuits — keep parity with Q1 v2 + original Q2
    # original used: min(max(round(slot_ms/2500), |returned|), K_CANDIDATES)
    s2_slot = cres.verifier_result.slot_ms
    s2_calls_est = round(s2_slot / 2500) if s2_slot else len(accepted)
    ccalls = max(s2_calls_est, len(accepted))
    ccalls = min(ccalls, K_CANDIDATES)
    ccost = per_row * ccalls
    print(f"  top-K reviewIds: {cres.uncertain_ids}")
    print(f"  accepted (BQ AI.IF + LIMIT={TARGET}): {accepted}")
    print(f"  P={cmetric.precision:.4f} R={cmetric.recall:.4f} F1={cmetric.f1_score:.4f}")
    print(f"  wall={cwall:.2f}s  calls={ccalls}  cost=${ccost:.6f}")

    profile["dase_partition"] = cres.partition.to_dict()
    profile["dase_top_K_reviewIds"] = list(cres.uncertain_ids)


    profile["cascade"] = {
        "method": "Cascade(MarginSignal, TopKBand, AiIfVerifier).run() — IN(K) AI.IF + id filter + LIMIT TARGET",
        "verifier": cres.verifier_result.to_dict(),
        "result_ids": accepted,
        "score": {"precision": cmetric.precision, "recall": cmetric.recall, "f1": cmetric.f1_score},
        "totals": {
            "wall_s": cwall,
            "wall_breakdown_s": {
                "dase": cres.timings_s.get("signal_compute", 0) + cres.timings_s.get("band_partition", 0),
                "bq_stage2": cres.verifier_result.wall_s,
            },
            "slot_ms_bq_total": cres.verifier_result.slot_ms,
            "cost_usd": ccost,
            "per_row_cost_usd": per_row,
            "n_llm_calls": ccalls,
        },
    }

    write_profile(profile, PROFILE_PATH)



if __name__ == "__main__":
    main()
