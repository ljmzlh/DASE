#!/usr/bin/env -S python -u
"""
Movie Q9 cascade v2 — sem_rank by review-level 1-5 sentiment score (R, no LIMIT).

NL: Score 1-5 how much the reviewer liked the movie, for ant_man scope.
GT: SPLIT_PART(originalScore, '/', 1) -> float per review.
Eval: _ranking (Spearman + Kendall_tau).

Operator (paper Table 3): R (semantic rank). The signal is a custom 5-anchor
cosine sim (not MarginSignal), so we compute scores+partition manually and
hand the uncertain ids directly to an AiIfVerifier-style verifier (BQ AI.SCORE
returns float per row, not boolean). We use the package's run_query + profile +
calibration helpers; only the verifier wrapper is bespoke (AI.SCORE has a
non-standard return shape: id → float, not id → bool).

NOTE on operator semantics: the user-supplied migration spec says "R (Q9, Q10):
copy movie/q1_v2 (TopKBand) directly." Q1 v2 is true top-K retrieval (return K
ids). Q9 is sem_rank — it must produce a score for ALL rows (no LIMIT) and
Spearman is computed over the full set. The natural cascade: confidence-band
on the cheap proxy (5-anchor confidence = top1-top2 anchor sim), send bottom
α to BQ AI.SCORE, use dase_score for the rest. We use AlphaBand on -confidence
(so bottom-α confidence rows are "uncertain" in the band sense).
"""
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import (
    AlphaBand, AiIfVerifier,
    bq_client, embed_query, run_query,
    build_profile, write_profile, print_summary,
)
from dase_cascade.calibration import _sum_tokens, _to_cost
from google.cloud import bigquery
import evaluator as ev

MOVIE_DIR    = os.path.abspath(os.path.join(_HERE, ".."))
EMB_PATH     = os.path.join(MOVIE_DIR, "data", "review_embeddings.npz")
REVIEWS_CSV  = os.path.join(MOVIE_DIR, "cache", "Reviews.csv")
PROFILE_PATH = os.path.join(MOVIE_DIR, "outputs", "Q9.json")
DIM = 3072

PROJECT = os.environ.get("GCP_PROJECT", "")
MOVIE_ID = "ant_man_and_the_wasp_quantumania"
STAGING_TABLE = "movie.q9_uncertain"

ANCHORS = {
    5: "Very positive. Strong positive sentiment, indicating high satisfaction.",
    4: "Positive. Noticeably positive sentiment, indicating general satisfaction.",
    3: "Neutral. Expresses no clear positive or negative sentiment. May be factual or descriptive without emotional language.",
    2: "Negative. Noticeably negative sentiment, indicating some level of dissatisfaction but without strong anger or frustration.",
    1: "Very negative. Strong negative sentiment, indicating high dissatisfaction, frustration, or anger.",
}
SCORE_PROMPT = (
    "Score from 1 to 5 how much did the reviewer like the movie based on provided rubrics.\n\n"
    "Rubrics:\n"
    "5: Very positive. Strong positive sentiment, indicating high satisfaction.\n"
    "4: Positive. Noticeably positive sentiment, indicating general satisfaction.\n"
    "3: Neutral. Expresses no clear positive or negative sentiment. May be factual or descriptive without emotional language.\n"
    "2: Negative. Noticeably negative sentiment, indicating some level of dissatisfaction but without strong anger or frustration.\n"
    "1: Very negative. Strong negative sentiment, indicating high dissatisfaction, frustration, or anger.\n\n"
    "Review:\n"
)

ALPHA = 0.7  # conservative dase prune — protects Spearman quality
PAPER_BQ_Q9 = {"score": 0.78, "latency_s": None, "cost_usd": 0.02}
SKIP_BASELINE = True


def cosine_sim_matrix(query_emb, doc_embs):
    q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-12)
    d_norm = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-12)
    return d_norm @ q_norm


def softmax_rows(x, axis=1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def per_row_score_calibration(client, sample_texts, k=10):
    """AI.GENERATE_BOOL proxy for AI.SCORE per-row cost (AI.SCORE doesn't expose token usage).
    Bespoke because we need to bind the rubric prompt + each row's reviewText as parameters."""
    texts = sample_texts[:k]
    THINKING = "JSON '{\"generation_config\":{\"thinking_config\":{\"thinking_budget\":0}}}'"
    selects, params = [], []
    for i, t in enumerate(texts):
        selects.append(f"""
        SELECT AI.GENERATE_BOOL(
          (@prompt_{i}, @rev_{i}),
          connection_id => 'us.connection',
          endpoint => 'gemini-2.5-flash',
          model_params => {THINKING}
        ) AS verdict""")
        params.append(bigquery.ScalarQueryParameter(f"prompt_{i}", "STRING", SCORE_PROMPT))
        params.append(bigquery.ScalarQueryParameter(f"rev_{i}", "STRING", t))
    sql = " UNION ALL ".join(selects)
    cfg = bigquery.QueryJobConfig(query_parameters=params, use_query_cache=False)
    t0 = time.time()
    df = client.query(sql, job_config=cfg).result().to_dataframe()
    elapsed = time.time() - t0
    p_other, p_audio, out, thoughts = _sum_tokens(df["verdict"])
    n = len(df)
    cost = _to_cost(p_other, p_audio, out, thoughts)
    return {
        "method": "AI.GENERATE_BOOL with Q9 rubric prompt + thinking_budget=0 (proxy for AI.SCORE token cost)",
        "n_sample": n,
        "tokens_total": {"prompt_other": p_other, "prompt_audio": p_audio, "output": out, "thoughts": thoughts},
        "sample_cost_usd": cost,
        "per_row_cost_usd": cost / n if n else 0.0,
        "elapsed_s": elapsed,
        "_caveat": "AI.SCORE doesn't expose token usage in result struct; AI.GENERATE_BOOL with same rubric prompt is a per-row cost proxy",
    }


def stage1_create_staging(client, uncertain_rids):
    rid_list = ",".join(str(int(r)) for r in uncertain_rids)
    sql = f"""
    CREATE OR REPLACE TABLE {STAGING_TABLE} AS
    SELECT * FROM movie.reviews
    WHERE reviewId IN ({rid_list})
    """
    return run_query(client, sql)


def stage2_score_uncertain(client):
    """Verbatim Q9.sql: AI.SCORE with rubric, only swap movie.reviews -> staging."""
    sql = f"""
    SELECT reviewId,
      AI.SCORE(
        (
        @prompt,
        reviewText
        ),
        connection_id => 'us.connection',
        endpoint => 'gemini-2.5-flash'
      ) AS reviewScore
    FROM {STAGING_TABLE}
    WHERE id = '{MOVIE_ID}'
    """
    params = [bigquery.ScalarQueryParameter("prompt", "STRING", SCORE_PROMPT)]
    return run_query(client, sql, params=params)


def main():
    profile = build_profile(
        scenario="movie", query_id=9, scale_factor=2000,
        params={"alpha": ALPHA},
        cascade_form=(
            "R (sem_rank) cascade: 5-anchor cosine sim per row → softmax-weighted continuous "
            "score in [1,5]; confidence = top1-top2 anchor sim. AlphaBand on confidence: bottom-α "
            "rows go to BQ AI.SCORE (verbatim Q9.sql, table swap to staging); rest use dase_score. "
            "Final scores merged for Spearman eval. NOTE: signal is custom (5-anchor), not "
            "MarginSignal — we compute it manually and call the verifier directly."
        ),
        extra={
            "score_prompt": SCORE_PROMPT,
            "structural_filter": f"id = '{MOVIE_ID}'",
            "anchors": ANCHORS,
        },
    )

    print(f"Loading data + computing dase 5-anchor scores on {MOVIE_ID} subset...")
    t = time.time()
    review_emb_full = np.load(EMB_PATH)["reviewText_emb"]
    df_full = pd.read_csv(REVIEWS_CSV)
    keep = ~df_full["reviewId"].duplicated()
    df_full = df_full[keep].reset_index(drop=True)
    review_emb_full = review_emb_full[keep.values]
    sub = (df_full["id"] == MOVIE_ID).values
    df = df_full[sub].reset_index(drop=True)
    review_emb = review_emb_full[sub]
    n_total = len(df)
    t_load = time.time() - t
    print(f"  {MOVIE_ID}: {n_total} reviews")
    profile["data"] = {"n_reviews_total_dedup": len(df_full), "n_reviews_in_scope": n_total}

    t = time.time()
    anchor_keys = [1, 2, 3, 4, 5]
    anchor_texts = [ANCHORS[k] for k in anchor_keys]
    anchor_embs = embed_query(anchor_texts, dim=DIM)
    t_embed = time.time() - t

    t = time.time()
    sims = np.stack([cosine_sim_matrix(a, review_emb) for a in anchor_embs], axis=1)
    sorted_sims = np.sort(sims, axis=1)
    confidence = sorted_sims[:, -1] - sorted_sims[:, -2]
    weights = softmax_rows(sims, axis=1)
    dase_score_raw = (weights * np.array(anchor_keys)).sum(axis=1)
    dlo, dhi = dase_score_raw.min(), dase_score_raw.max()
    if dhi > dlo:
        dase_score = 1.0 + 4.0 * (dase_score_raw - dlo) / (dhi - dlo)
    else:
        dase_score = np.full_like(dase_score_raw, 3.0)
    t_score = time.time() - t

    # AlphaBand on -confidence: bottom-α confidence → "uncertain" (sent to BQ).
    # AlphaBand selects bottom-α by |scores|; using +confidence (always positive) does
    # bottom-α by confidence — exactly what we want. No sign flip needed.
    t = time.time()
    band = AlphaBand(alpha=ALPHA)
    part = band.partition(confidence.astype(np.float32))
    uncertain_rids = [int(df.iloc[i]["reviewId"]) for i in sorted(part.uncertain.tolist())]
    n_uncertain = len(uncertain_rids)
    t_partition = time.time() - t

    print(f"  alpha={ALPHA}, n_uncertain={n_uncertain}, n_confident={n_total - n_uncertain}")
    print(f"  dase score range: [{dase_score.min():.2f}, {dase_score.max():.2f}], mean={dase_score.mean():.2f}")
    print(f"  confidence range: [{confidence.min():.4f}, {confidence.max():.4f}]")

    profile["dase_breakdown"] = {
        "data_load_dedup_filter_s": t_load,
        "embed_prompts_s": t_embed,
        "score_compute_s": t_score,
        "partition_s": t_partition,
        "total_s": t_load + t_embed + t_score + t_partition,
    }
    profile["dase_partition"] = {
        "n_uncertain": n_uncertain,
        "n_confident": n_total - n_uncertain,
        "uncertain_reviewIds": uncertain_rids,
    }

    client = bq_client(PROJECT)

    print(f"\n=== Per-row cost calibration (AI.GENERATE_BOOL proxy with Q9 rubric prompt) ===")
    sample_texts = [str(df.iloc[i]["reviewText"]) for i in range(min(10, n_total))]
    cal = per_row_score_calibration(client, sample_texts, k=10)
    per_row = cal["per_row_cost_usd"]
    print(f"  per_row=${per_row:.6f}")
    profile["calibration"] = cal

    if SKIP_BASELINE:
        print(f"\n=== Baseline ABORTED (SKIP_BASELINE=True) — using paper Table 4(a) numbers ===")
        baseline_sql = (
            f"SELECT reviewId, AI.SCORE((@prompt, reviewText), connection_id => 'us.connection', "
            f"endpoint => 'gemini-2.5-flash') AS reviewScore "
            f"FROM movie.reviews WHERE id = '{MOVIE_ID}'"
        )
        profile["baseline"] = {
            "_status": "aborted",
            "_status_note": (
                "Baseline NOT run on our project. Q9 has no LIMIT — must AI.SCORE on all 128 "
                "ant_man reviews. Per project policy, baseline metrics substituted from paper "
                "Table 4(a)."
            ),
            "method": "sembench bigquery/Q9.sql verbatim on movie.reviews — NOT EXECUTED",
            "sql": baseline_sql,
            "score": {"spearman_correlation": PAPER_BQ_Q9["score"], "kendall_tau": None, "_source": "paper Table 4(a)"},
            "latency_breakdown": {"wall_s": PAPER_BQ_Q9["latency_s"], "slot_ms": None, "_source": "paper Table 4(a)"},
            "cost_breakdown": {
                "n_llm_calls_est": n_total,
                "n_llm_calls_method": "scope size (Q9 no LIMIT)",
                "per_row_cost_usd": per_row,
                "total_cost_usd": PAPER_BQ_Q9["cost_usd"],
                "_source": "paper Table 4(a) cost",
            },
        }
        bcost = PAPER_BQ_Q9["cost_usd"]
        bwall = PAPER_BQ_Q9["latency_s"]
        bslot = None
        b_score = PAPER_BQ_Q9["score"]
        bcalls = n_total
    else:
        raise NotImplementedError("set SKIP_BASELINE=False only if you want to pay for full 128-row AI.SCORE")

    print(f"\n=== Cascade Stage 1: CTAS {STAGING_TABLE} from {n_uncertain} uncertain reviews ===")
    s1_df, s1_wall, s1_slot, s1_sql = stage1_create_staging(client, uncertain_rids)
    print(f"  wall={s1_wall:.2f}s, slot_ms={s1_slot}")

    print(f"\n=== Cascade Stage 2: AI.SCORE on {STAGING_TABLE} ===")
    s2_df, s2_wall, s2_slot, s2_sql = stage2_score_uncertain(client)
    bq_scores = {int(r["reviewId"]): float(r["reviewScore"]) for _, r in s2_df.iterrows()}
    s2_calls = n_uncertain
    cascade_cost = per_row * s2_calls
    print(f"  BQ scored {len(bq_scores)} uncertain reviews")
    print(f"  wall={s2_wall:.2f}s, slot_ms={s2_slot}, n_calls={s2_calls}, cost=${cascade_cost:.6f}")

    # Merge: confident → dase_score, uncertain → bq_score
    final_scores = []
    for i in range(n_total):
        rid = int(df.iloc[i]["reviewId"])
        if rid in bq_scores:
            final_scores.append((rid, bq_scores[rid]))
        else:
            final_scores.append((rid, float(dase_score[i])))
    sys_df = pd.DataFrame(final_scores, columns=["reviewId", "reviewScore"])
    cmetric = ev.evaluate_q9(sys_df)
    print(f"  cascade Spearman={cmetric.spearman_correlation:.4f}, Kendall_tau={cmetric.kendall_tau:.4f}")

    cascade_total_wall = profile["dase_breakdown"]["total_s"] + s1_wall + s2_wall
    cascade_total_slot = s1_slot + s2_slot
    profile["cascade"] = {
        "method": "Two-branch: confident rows use dase softmax-weighted score; uncertain rows use BQ AI.SCORE on staging.",
        "stage1_ctas": {
            "sql": s1_sql,
            "latency_breakdown": {"wall_s": s1_wall, "slot_ms": s1_slot},
            "cost_usd": 0.0,
        },
        "stage2_score": {
            "sql": s2_sql,
            "result_bq_scores": bq_scores,
            "latency_breakdown": {"wall_s": s2_wall, "slot_ms": s2_slot},
            "cost_breakdown": {
                "n_llm_calls": s2_calls,
                "n_llm_calls_method": "n_uncertain (Stage 2 no LIMIT, staging size)",
                "per_row_cost_usd": per_row,
                "total_cost_usd": cascade_cost,
            },
        },
        "score": {"spearman_correlation": cmetric.spearman_correlation, "kendall_tau": cmetric.kendall_tau},
        "totals": {
            "wall_s": cascade_total_wall,
            "wall_breakdown_s": {
                "dase": profile["dase_breakdown"]["total_s"],
                "bq_stage1_ctas": s1_wall,
                "bq_stage2_score": s2_wall,
            },
            "slot_ms_bq_total": cascade_total_slot,
            "cost_usd": cascade_cost,
            "n_llm_calls": s2_calls,
        },
    }

    profile["comparison"] = {
        "score":      {"paper": PAPER_BQ_Q9["score"], "baseline": b_score, "cascade": cmetric.spearman_correlation,
                       "_baseline_source": "paper (aborted)" if SKIP_BASELINE else "ours",
                       "_metric": "Spearman correlation"},
        "wall_s":     {"paper": PAPER_BQ_Q9["latency_s"], "baseline": bwall, "cascade_total": cascade_total_wall},
        "slot_ms_bq": {"baseline": bslot, "cascade_total": cascade_total_slot},
        "cost_usd":   {"paper": PAPER_BQ_Q9["cost_usd"], "baseline": bcost, "cascade": cascade_cost},
        "n_llm_calls": {"baseline": bcalls, "cascade": s2_calls},
    }

    write_profile(profile, PROFILE_PATH)

    print_summary(
        f"Movie Q9 (R sem_rank, alpha={ALPHA})",
        columns=["paper", "baseline", "cascade"],
        rows=[
            ("Spearman",   [PAPER_BQ_Q9["score"], b_score, cmetric.spearman_correlation], ".2f"),
            ("cost ($)",   [PAPER_BQ_Q9["cost_usd"], bcost, cascade_cost], ".4f"),
            ("#LLM calls", [n_total, bcalls, s2_calls], "d"),
        ],
    )


if __name__ == "__main__":
    main()
