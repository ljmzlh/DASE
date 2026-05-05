#!/usr/bin/env -S python -u
"""
Movie Q10 cascade v2 — sem_rank with movie-level aggregation (R, no LIMIT).

NL: Rank movies by audience preference based on review sentiment.
GT: SELECT M.id AS movieId, M.audienceScore AS movieScore FROM Movies AS M;
Eval: _ranking (Spearman) over 116 movies vs Movies.audienceScore.

Operator (paper Table 3): R (semantic rank). Same primitive as Q9 v2 but on the
full corpus (2000 reviews / 116 movies); per-review scores are aggregated
client-side (pandas groupby movieId mean).

Cascade: 5-anchor cosine sim per review → softmax-weighted continuous score in
[1,5]; AlphaBand on confidence (top1-top2 anchor sim); bottom-α rows go to BQ
AI.SCORE (verbatim Q9-style query on staging); confident rows use dase_score.
All review scores merged in Python; pandas GROUP BY movieId mean → per-movie
predicted score; Spearman vs Movies.audienceScore.

NOTE: signal is custom (5-anchor), not MarginSignal — we compute it manually
and hand uncertain ids directly to the staging+AI.SCORE flow. AI.SCORE returns
floats per row, which doesn't fit AiIfVerifier's bool API; we run the staging
CTAS + AI.SCORE query directly via run_query.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import (
    AlphaBand,
    bq_client, embed_query, run_query,
    build_profile, write_profile, print_summary,
)
from dase_cascade.calibration import _sum_tokens, _to_cost
from google.cloud import bigquery
import evaluator as ev

MOVIE_DIR    = os.path.abspath(os.path.join(_HERE, ".."))
EMB_PATH     = os.path.join(MOVIE_DIR, "data", "review_embeddings.npz")
REVIEWS_CSV  = os.path.join(MOVIE_DIR, "cache", "Reviews.csv")
PROFILE_PATH = os.path.join(MOVIE_DIR, "outputs", "Q10.json")
DIM = 3072

PROJECT = os.environ.get("GCP_PROJECT", "")
STAGING_TABLE = "movie.q10_uncertain"

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

ALPHA = 0.7
PAPER_BQ_Q10 = {"score": 0.44, "latency_s": 32.1, "cost_usd": 0.13}
SKIP_BASELINE = False  # try once; set True to skip


def cosine_sim_matrix(query_emb, doc_embs):
    q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-12)
    d_norm = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-12)
    return d_norm @ q_norm


def softmax_rows(x, axis=1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def per_row_cost_calibration(client, sample_texts, k=10):
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
        "method": "AI.GENERATE_BOOL with Q10 rubric prompt + thinking_budget=0 (proxy for AI.SCORE token cost)",
        "n_sample": n,
        "tokens_total": {"prompt_other": p_other, "prompt_audio": p_audio, "output": out, "thoughts": thoughts},
        "sample_cost_usd": cost,
        "per_row_cost_usd": cost / n if n else 0.0,
        "elapsed_s": elapsed,
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
    sql = f"""
    SELECT reviewId,
      AI.SCORE(
        (@prompt, reviewText),
        connection_id => 'us.connection',
        endpoint => 'gemini-2.5-flash'
      ) AS reviewScore,
      id
    FROM {STAGING_TABLE}
    """
    params = [bigquery.ScalarQueryParameter("prompt", "STRING", SCORE_PROMPT)]
    return run_query(client, sql, params=params)


def run_baseline(client):
    sql = """
    SELECT id AS movieId, AVG(reviewScore) AS movieScore
    FROM (
      SELECT id,
        AI.SCORE(
          (@prompt, reviewText),
          connection_id => 'us.connection',
          endpoint => 'gemini-2.5-flash'
        ) AS reviewScore
      FROM movie.reviews
    )
    GROUP BY id
    """
    params = [bigquery.ScalarQueryParameter("prompt", "STRING", SCORE_PROMPT)]
    return run_query(client, sql, params=params)


def main():
    profile = build_profile(
        scenario="movie", query_id=10, scale_factor=2000,
        params={"alpha": ALPHA},
        cascade_form=(
            "R (sem_rank) cascade with client-side aggregation: 5-anchor softmax-weighted "
            "score on full 2000 reviews, rescaled to [1,5]; confidence = top1-top2 anchor sim. "
            "AlphaBand on confidence → uncertain rows (BQ AI.SCORE on staging). All review "
            "scores merged in Python; pandas GROUP BY movieId mean for per-movie ranking."
        ),
        extra={"score_prompt": SCORE_PROMPT, "anchors": ANCHORS},
    )

    print("Loading data + computing dase 5-anchor scores on full reviews...")
    t = time.time()
    review_emb_full = np.load(EMB_PATH)["reviewText_emb"]
    df_full = pd.read_csv(REVIEWS_CSV)
    keep = ~df_full["reviewId"].duplicated()
    df = df_full[keep].reset_index(drop=True)
    review_emb = review_emb_full[keep.values]
    n_total = len(df)
    n_movies = df["id"].nunique()
    t_load = time.time() - t
    print(f"  {n_total} unique reviews across {n_movies} movies")
    profile["data"] = {
        "n_reviews_total_csv": len(df_full),
        "n_reviews_dedup": n_total,
        "n_movies": int(n_movies),
    }

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

    # AlphaBand on +confidence (always positive) → bottom-α confidence = "uncertain"
    t = time.time()
    band = AlphaBand(alpha=ALPHA)
    part = band.partition(confidence.astype(np.float32))
    uncertain_rids = [int(df.iloc[i]["reviewId"]) for i in sorted(part.uncertain.tolist())]
    n_uncertain = len(uncertain_rids)
    t_partition = time.time() - t

    print(f"  alpha={ALPHA}, n_uncertain={n_uncertain}, n_confident={n_total - n_uncertain}")
    print(f"  dase score range: [{dase_score.min():.2f}, {dase_score.max():.2f}], mean={dase_score.mean():.2f}")

    profile["dase_breakdown"] = {
        "data_load_dedup_s": t_load,
        "embed_prompts_s": t_embed,
        "score_compute_s": t_score,
        "partition_s": t_partition,
        "total_s": t_load + t_embed + t_score + t_partition,
    }
    profile["dase_partition"] = {"n_uncertain": n_uncertain, "n_confident": n_total - n_uncertain}

    client = bq_client(PROJECT)

    print(f"\n=== Per-row cost calibration (Q10 rubric prompt + thinking=0) ===")
    sample_texts = [str(df.iloc[i]["reviewText"]) for i in range(min(10, n_total))]
    cal = per_row_cost_calibration(client, sample_texts, k=10)
    per_row = cal["per_row_cost_usd"]
    print(f"  per_row=${per_row:.6f}")
    profile["calibration"] = cal

    if SKIP_BASELINE:
        print(f"\n=== Baseline ABORTED (SKIP_BASELINE=True) — using paper Table 4(a) numbers ===")
        baseline_sql = (
            "SELECT id AS movieId, AVG(reviewScore) AS movieScore FROM ("
            "SELECT id, AI.SCORE((@prompt, reviewText), connection_id => 'us.connection', "
            "endpoint => 'gemini-2.5-flash') AS reviewScore FROM movie.reviews) GROUP BY id"
        )
        profile["baseline"] = {
            "_status": "aborted",
            "_status_note": (
                "Baseline NOT run on our project. Q10 has 2000 AI.SCORE calls; if BQ slot "
                "allocation is poor today this would take 30+ minutes. Per project policy, "
                "baseline metrics are substituted from paper Table 4(a)."
            ),
            "method": "sembench bigquery/Q10.sql verbatim — NOT EXECUTED",
            "sql": baseline_sql,
            "score": {"spearman_correlation": PAPER_BQ_Q10["score"], "_source": "paper Table 4(a)"},
            "latency_breakdown": {"wall_s": PAPER_BQ_Q10["latency_s"], "slot_ms": None, "_source": "paper Table 4(a)"},
            "cost_breakdown": {
                "n_llm_calls": n_total,
                "per_row_cost_usd": per_row,
                "total_cost_usd": PAPER_BQ_Q10["cost_usd"],
                "_source": "paper Table 4(a) cost",
            },
        }
        b_score = PAPER_BQ_Q10["score"]
        bwall = PAPER_BQ_Q10["latency_s"]
        bslot = None
        bcost = PAPER_BQ_Q10["cost_usd"]
        bcalls = n_total
    else:
        print(f"\n=== Baseline (sembench bigquery/Q10.sql verbatim on movie.reviews) ===")
        bdf, bwall, bslot, bsql = run_baseline(client)
        bcalls = n_total
        bcost = per_row * bcalls
        bm = ev.evaluate_q10(bdf)
        b_score = bm.spearman_correlation
        print(f"  returned: {len(bdf)} movies")
        print(f"  wall={bwall:.2f}s, slot_ms={bslot}, n_calls={bcalls}, cost=${bcost:.6f}")
        print(f"  Spearman={b_score:.4f}, KendallTau={bm.kendall_tau:.4f}")
        profile["baseline"] = {
            "method": "sembench bigquery/Q10.sql verbatim on movie.reviews",
            "sql": bsql,
            "result_movies": [(str(r.iloc[0]), float(r.iloc[1])) for _, r in bdf.iterrows()],
            "score": {"spearman_correlation": b_score, "kendall_tau": bm.kendall_tau},
            "latency_breakdown": {"wall_s": bwall, "slot_ms": bslot},
            "cost_breakdown": {
                "n_llm_calls": bcalls,
                "n_llm_calls_method": "scope size (Q10 no LIMIT, all rows evaluated)",
                "per_row_cost_usd": per_row,
                "total_cost_usd": bcost,
            },
        }

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

    final_review_scores = []
    for i in range(n_total):
        rid = int(df.iloc[i]["reviewId"])
        if rid in bq_scores:
            final_review_scores.append((str(df.iloc[i]["id"]), rid, bq_scores[rid]))
        else:
            final_review_scores.append((str(df.iloc[i]["id"]), rid, float(dase_score[i])))
    review_df = pd.DataFrame(final_review_scores, columns=["id", "reviewId", "reviewScore"])

    movie_df = (
        review_df.groupby("id")["reviewScore"].mean().reset_index()
        .rename(columns={"id": "movieId", "reviewScore": "movieScore"})
    )
    cm = ev.evaluate_q10(movie_df)
    print(f"\n  Aggregated to {len(movie_df)} movies (client-side groupby).")
    print(f"  cascade Spearman={cm.spearman_correlation:.4f}, KendallTau={cm.kendall_tau:.4f}")

    cascade_total_wall = profile["dase_breakdown"]["total_s"] + s1_wall + s2_wall
    cascade_total_slot = s1_slot + s2_slot
    profile["cascade"] = {
        "method": "Two-branch sem_rank: dase confident reviews + BQ uncertain reviews; client-side groupby movieId mean",
        "stage1_ctas": {
            "sql": s1_sql,
            "latency_breakdown": {"wall_s": s1_wall, "slot_ms": s1_slot},
            "cost_usd": 0.0,
        },
        "stage2_score": {
            "sql": s2_sql,
            "n_bq_scores_returned": len(bq_scores),
            "latency_breakdown": {"wall_s": s2_wall, "slot_ms": s2_slot},
            "cost_breakdown": {
                "n_llm_calls": s2_calls,
                "n_llm_calls_method": "n_uncertain (Stage 2 no LIMIT, staging size)",
                "per_row_cost_usd": per_row,
                "total_cost_usd": cascade_cost,
            },
        },
        "score": {"spearman_correlation": cm.spearman_correlation, "kendall_tau": cm.kendall_tau},
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
        "score_spearman": {"paper": PAPER_BQ_Q10["score"], "baseline": b_score, "cascade": cm.spearman_correlation,
                           "_baseline_source": "paper (aborted)" if SKIP_BASELINE else "ours"},
        "wall_s":         {"paper": PAPER_BQ_Q10["latency_s"], "baseline": bwall, "cascade_total": cascade_total_wall},
        "slot_ms_bq":     {"baseline": bslot, "cascade_total": cascade_total_slot},
        "cost_usd":       {"paper": PAPER_BQ_Q10["cost_usd"], "baseline": bcost, "cascade": cascade_cost},
        "n_llm_calls":    {"baseline": bcalls, "cascade": s2_calls},
    }

    write_profile(profile, PROFILE_PATH)

    print_summary(
        f"Movie Q10 (R sem_rank, alpha={ALPHA})",
        columns=["paper", "baseline", "cascade"],
        rows=[
            ("Spearman",   [PAPER_BQ_Q10["score"], b_score, cm.spearman_correlation], ".2f"),
            ("wall (s)",   [PAPER_BQ_Q10["latency_s"], bwall, cascade_total_wall], ".2f"),
            ("cost ($)",   [PAPER_BQ_Q10["cost_usd"], bcost, cascade_cost], ".4f"),
            ("#LLM calls", [n_total, bcalls, s2_calls], "d"),
        ],
    )


if __name__ == "__main__":
    main()
