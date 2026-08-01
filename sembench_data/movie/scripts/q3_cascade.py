#!/usr/bin/env -S python -u
"""
Movie Q3 cascade v2 — count of positive reviews for `taken_3` (text F + COUNT).

NL: Count of positive reviews for movie 'taken_3'. Return positive_review_cnt.
GT: SELECT COUNT(*) FROM Reviews WHERE id='taken_3' AND scoreSentiment='POSITIVE' -> 14
Eval: _aggregation_single (relative_error → score = 1/(1+rel_err))

Refactored to use dase_cascade. Operator (paper Table 3): F.
Cascade: MarginSignal + AlphaBand + AiIfVerifier (text inline IN(...)).
Final cascade_count = |confident_pos| + |bq_yes_on_uncertain|. Pure Python aggregation.
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import (
    Cascade, MarginSignal, AlphaBand, AiIfVerifier,
    bq_client, per_row_cost, run_query,
    relative_error_score, build_profile, write_profile,
)

MOVIE_DIR    = os.path.abspath(os.path.join(_HERE, ".."))
EMB_PATH     = os.path.join(MOVIE_DIR, "data", "review_embeddings.npz")
REVIEWS_CSV  = os.path.join(MOVIE_DIR, "cache", "Reviews.csv")
PROFILE_PATH = os.path.join(MOVIE_DIR, "outputs", "Q3.json")

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

ALPHA = 0.2


def make_q3_verifier():
    """AI.IF over inline IN(uncertain rids) AND id='taken_3' — returns POS subset."""
    def verify_sql_template(rids):
        rid_list = ",".join(str(int(r)) for r in rids)
        return f"""
        SELECT reviewId AS id
        FROM movie.reviews AS r
        WHERE reviewId IN ({rid_list}) AND r.id = '{MOVIE_ID}'
          AND AI.IF(('{PROMPT}', r.reviewText),
                    connection_id => 'us.connection',
                    endpoint => 'gemini-2.5-flash')
        """
    return AiIfVerifier(verify_sql_template=verify_sql_template, id_column="id", coerce_id=int)




def per_row_cost_movie_q3(client):
    """Calibration: AI.GENERATE_BOOL on movie.reviews scoped id='taken_3' LIMIT 10."""
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
        scenario="movie", query_id=3, scale_factor=2000,
        prompt=PROMPT, params={"alpha": ALPHA},
        cascade_form="F-cascade: MarginSignal + AlphaBand + AiIfVerifier; client COUNT.",
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
    cal = per_row_cost_movie_q3(client)
    per_row = cal.per_row_cost_usd
    print(f"  per_row=${per_row:.6f}")
    profile["calibration"] = cal.to_dict()

    # ── Cascade: MarginSignal + AlphaBand + AiIfVerifier ──
    cascade = Cascade(
        embeddings=review_emb,
        ids=df["reviewId"].astype(int).tolist(),
        signal=MarginSignal(positive_prompts=POSITIVE, negative_prompts=NEGATIVE),
        band=AlphaBand(alpha=ALPHA),
        verifier=make_q3_verifier(),
    )
    print("\n=== Cascade (MarginSignal → AlphaBand → AiIfVerifier) ===")
    cres = cascade.run(client, per_row)

    n_confident_pos = len(cres.confident_pos_ids)
    bq_pos = cres.verifier_result.positive_ids
    n_uncertain = len(cres.uncertain_ids)
    cascade_count = n_confident_pos + len(bq_pos)
    cscore = relative_error_score(cascade_count, n_gt_pos)
    cwall = cres.total_wall_s
    ccalls = cres.verifier_result.n_calls
    ccost = cres.verifier_result.cost_usd
    print(f"  alpha={ALPHA}, uncertain={n_uncertain}, confident_pos={n_confident_pos}, "
          f"bq_yes_on_uncertain={len(bq_pos)}")
    print(f"  cascade_count={cascade_count} (GT={n_gt_pos})  score={cscore:.4f}")
    print(f"  wall={cwall:.2f}s  calls={ccalls}  cost=${ccost:.6f}")

    profile["dase_partition"] = cres.partition.to_dict() | {
        "n_confident_pos": n_confident_pos,
        "uncertain_reviewIds": list(cres.uncertain_ids),
    }


    profile["cascade"] = {
        "method": "F-cascade with COUNT aggregation: Cascade(MarginSignal, AlphaBand, AiIfVerifier).run()",
        "verifier": cres.verifier_result.to_dict(),
        "cascade_count": cascade_count,
        "cascade_count_breakdown": {
            "dase_confident_pos": n_confident_pos,
            "bq_uncertain_pos": len(bq_pos),
        },
        "score": {"relative_error": abs(cascade_count - n_gt_pos) / n_gt_pos if n_gt_pos else 0.0,
                  "score": cscore},
        "totals": {
            "wall_s": cwall,
            "wall_breakdown_s": {
                "dase": cres.timings_s.get("signal_compute", 0) + cres.timings_s.get("band_partition", 0),
                "bq_verify": cres.verifier_result.wall_s,
            },
            "slot_ms_bq_total": cres.verifier_result.slot_ms,
            "cost_usd": ccost,
            "n_llm_calls": ccalls,
        },
    }


    write_profile(profile, PROFILE_PATH)



if __name__ == "__main__":
    main()
