#!/usr/bin/env -S python -u
"""
Movie Q1 cascade — top-K positive reviews (text F + R retrieval).

NL: Five clearly positive reviews (any movie). Return reviewId.
GT: Reviews.scoreSentiment == 'POSITIVE'.
Eval: precision/recall/F1 over returned ids (limit=5).

Refactored to use dase_cascade. Operator (paper Table 3): F + R.
The TopKBand selects K candidates by margin desc; the verifier runs a single
BQ AI.IF with WHERE reviewId IN (top-K) LIMIT TARGET. BQ short-circuits at LIMIT.
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
    f1_set, build_profile, write_profile,
)
import evaluator as ev  # sembench's per-Q evaluator (movie Q1: _retrieval_limit)

MOVIE_DIR    = os.path.abspath(os.path.join(_HERE, ".."))
EMB_PATH     = os.path.join(MOVIE_DIR, "data", "review_embeddings.npz")
REVIEWS_CSV  = os.path.join(MOVIE_DIR, "cache", "Reviews.csv")
PROFILE_PATH = os.path.join(MOVIE_DIR, "outputs", "Q1.json")

PROJECT = os.environ.get("GCP_PROJECT", "")
PROMPT  = "Determine if the following movie review is clearly positive, review: "

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


def make_movie_verifier():
    """Single AI.IF query: WHERE reviewId IN (top-K) LIMIT TARGET (BQ short-circuits)."""
    def verify_sql_template(rids):
        rid_list = ",".join(str(int(r)) for r in rids)
        return f"""
        SELECT reviewId AS id FROM movie.reviews
        WHERE reviewId IN ({rid_list})
          AND AI.IF(('{PROMPT}', reviewText),
                    connection_id => 'us.connection',
                    endpoint => 'gemini-2.5-flash')
        LIMIT {TARGET}
        """
    return AiIfVerifier(verify_sql_template=verify_sql_template, id_column="id", coerce_id=int)




def per_row_cost_movie(client):
    """Movie Q1's calibration uses a SELECT … FROM movie.reviews LIMIT k pattern,
    not URI/inline-text. Inline that as the raw text_from_table_sql."""
    THINKING = "JSON '{\"generation_config\":{\"thinking_config\":{\"thinking_budget\":0}}}'"
    sql = f"""
    SELECT AI.GENERATE_BOOL(
      ('{PROMPT}', r.reviewText),
      connection_id => 'us.connection',
      endpoint => 'gemini-2.5-flash',
      model_params => {THINKING}
    ) AS verdict
    FROM movie.reviews AS r LIMIT 10
    """
    return per_row_cost(
        client, prompt=PROMPT, text_from_table_sql=sql,
        method_label="AI.GENERATE_BOOL on movie.reviews + thinking_budget=0", k=10,
    )


def main():
    profile = build_profile(
        scenario="movie", query_id=1, scale_factor=2000,
        prompt=PROMPT, params={"K_candidates": K_CANDIDATES, "target": TARGET},
        cascade_form="F+R cascade: MarginSignal + TopKBand + AiIfVerifier (single BQ query w/ LIMIT short-circuit)",
        extra={"dase_prompts": {"positive": POSITIVE, "negative": NEGATIVE}},
    )

    print("Loading data + dedup...")
    review_emb = np.load(EMB_PATH)["reviewText_emb"]
    df = pd.read_csv(REVIEWS_CSV)
    keep = ~df["reviewId"].duplicated()
    df = df[keep].reset_index(drop=True)
    review_emb = review_emb[keep.values]
    n_total = len(df)
    n_gt_pos = int((df["scoreSentiment"] == "POSITIVE").sum())
    print(f"  {n_total} unique reviews; n_gt_positive={n_gt_pos}")
    profile["data"] = {"n_reviews_dedup": n_total, "n_gt_positive": n_gt_pos}

    client = bq_client(PROJECT)

    print("\n=== Per-row cost calibration ===")
    cal = per_row_cost_movie(client)
    per_row = cal.per_row_cost_usd
    print(f"  per_row=${per_row:.6f}")
    profile["calibration"] = cal.to_dict()

    # ── Cascade: MarginSignal + TopKBand + Verifier ──
    cascade = Cascade(
        embeddings=review_emb,
        ids=df["reviewId"].astype(int).tolist(),
        signal=MarginSignal(positive_prompts=POSITIVE, negative_prompts=NEGATIVE),
        band=TopKBand(k=K_CANDIDATES),
        verifier=make_movie_verifier(),
    )
    print("\n=== Cascade (MarginSignal → TopKBand → AiIfVerifier) ===")
    cres = cascade.run(client, per_row)

    # For TopK retrieval: confident_pos is empty by construction (band defers all
    # to verifier); the cascade "answer" is exactly the verifier's positive_ids.
    accepted = sorted(cres.verifier_result.positive_ids)
    sys_df = pd.DataFrame({"reviewId": accepted})
    cmetric = ev.evaluate_q1(sys_df)
    cwall = cres.total_wall_s
    ccalls = max(cres.verifier_result.n_calls, len(accepted))  # verifier counted candidates; LIMIT short-circuits
    ccost = cres.verifier_result.cost_usd
    print(f"  top-K reviewIds: {cres.uncertain_ids}")
    print(f"  accepted (BQ AI.IF + LIMIT={TARGET}): {accepted}")
    print(f"  P={cmetric.precision:.4f} R={cmetric.recall:.4f} F1={cmetric.f1_score:.4f}")
    print(f"  wall={cwall:.2f}s  calls={ccalls}  cost=${ccost:.6f}")

    profile["dase_partition"] = cres.partition.to_dict()
    profile["dase_top_K_reviewIds"] = list(cres.uncertain_ids)

    # n_calls estimate from slot-ms (matches existing script convention)

    profile["cascade"] = {
        "method": "Cascade(MarginSignal, TopKBand, AiIfVerifier).run() — IN(K) AI.IF + LIMIT TARGET",
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
