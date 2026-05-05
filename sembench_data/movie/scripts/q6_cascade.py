#!/usr/bin/env -S python -u
"""
Movie Q6 cascade v2 — opposite-sentiment review pairs for `ant_man` (J + R top-K pairs).

NL: 10 pairs of reviews with OPPOSITE sentiment for ant_man_and_the_wasp_quantumania.
GT: pairs (r1, r2) where r1.scoreSentiment != r2.scoreSentiment within ant_man scope.
Eval: _review_pairs_limit, limit=10.

Operator (paper Table 3): J + R. Same structure as Q5 v2 (only the pair prompt
differs — opposite vs same sentiment). PairCosineSignal scores pairs; top-K positive
(highest sim) + top-K negative (lowest sim, ≈ opposite sentiment) form the uncertain
pool. AiIfVerifier batched + LIMIT short-circuit.

"""
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import (
    PairCosineSignal, AiIfVerifier,
    bq_client, run_query,
    build_profile, write_profile, print_summary,
)
from dase_cascade.calibration import _sum_tokens, _to_cost
from google.cloud import bigquery
import evaluator as ev

MOVIE_DIR    = os.path.abspath(os.path.join(_HERE, ".."))
EMB_PATH     = os.path.join(MOVIE_DIR, "data", "review_embeddings.npz")
REVIEWS_CSV  = os.path.join(MOVIE_DIR, "cache", "Reviews.csv")
PROFILE_PATH = os.path.join(MOVIE_DIR, "outputs", "Q6.json")

PROJECT  = os.environ.get("GCP_PROJECT", "")
PAIR_PROMPT_PREFIX = "These two movie reviews express opposite sentiments - one is positive and the other is negative. Review 1: "
PAIR_PROMPT_SEP    = ", Review 2: "
MOVIE_ID = "ant_man_and_the_wasp_quantumania"

K_POS = 5
K_NEG = 5
TARGET_PAIRS = 10

PAPER_BQ_Q6 = {"score_f1": 0.69, "latency_s": 54.5, "cost_usd": 1.00}
SKIP_BASELINE = True


def per_pair_cost_calibration(client, sample_texts, n_pairs=5):
    rng = np.random.default_rng(0)
    chosen = []
    while len(chosen) < n_pairs:
        i, j = rng.integers(0, len(sample_texts)), rng.integers(0, len(sample_texts))
        if i != j:
            chosen.append((sample_texts[i], sample_texts[j]))

    THINKING = "JSON '{\"generation_config\":{\"thinking_config\":{\"thinking_budget\":0}}}'"
    selects, params = [], []
    for k, (t1, t2) in enumerate(chosen):
        selects.append(f"""
        SELECT AI.GENERATE_BOOL(
          ('{PAIR_PROMPT_PREFIX}', @t1_{k}, '{PAIR_PROMPT_SEP}', @t2_{k}),
          connection_id => 'us.connection',
          endpoint => 'gemini-2.5-flash',
          model_params => {THINKING}
        ) AS verdict
        """)
        params.append(bigquery.ScalarQueryParameter(f"t1_{k}", "STRING", t1))
        params.append(bigquery.ScalarQueryParameter(f"t2_{k}", "STRING", t2))
    sql = " UNION ALL ".join(selects)
    cfg = bigquery.QueryJobConfig(query_parameters=params, use_query_cache=False)

    t0 = time.time()
    df = client.query(sql, job_config=cfg).result().to_dataframe()
    elapsed = time.time() - t0
    p_other, p_audio, out, thoughts = _sum_tokens(df["verdict"])
    n = len(df)
    cost = _to_cost(p_other, p_audio, out, thoughts)
    return {
        "method": "AI.GENERATE_BOOL on opposite-sentiment pair prompt + thinking_budget=0",
        "n_sample_pairs": n,
        "tokens_total": {"prompt_other": p_other, "prompt_audio": p_audio, "output": out, "thoughts": thoughts},
        "sample_cost_usd": cost,
        "per_pair_cost_usd": cost / n if n else 0.0,
        "elapsed_s": elapsed,
    }


def select_top_pairs(pair_signal: PairCosineSignal, n: int, k_pos: int, k_neg: int):
    """Top-K_pos highest-sim self-pairs ∪ top-K_neg lowest-sim self-pairs (i!=j)."""
    L = np.arange(n, dtype=np.int64)
    R = np.arange(n, dtype=np.int64)
    Lm = pair_signal._left[L]
    Rm = pair_signal._right[R]
    S = Lm @ Rm.T
    np.fill_diagonal(S, -np.inf)
    flat = S.flatten()
    top_pos_idx = np.argpartition(-flat, k_pos)[:k_pos]
    flat_for_neg = np.where(np.isfinite(flat), flat, np.inf)
    top_neg_idx = np.argpartition(flat_for_neg, k_neg)[:k_neg]
    pos_pairs = [(int(idx // n), int(idx % n), float(flat[idx])) for idx in top_pos_idx]
    neg_pairs = [(int(idx // n), int(idx % n), float(flat[idx])) for idx in top_neg_idx]
    pos_pairs.sort(key=lambda t: -t[2])
    neg_pairs.sort(key=lambda t: t[2])
    return pos_pairs, neg_pairs


def make_q6_pair_verifier(df: pd.DataFrame):
    def verify_sql_template(pair_ids):
        rows = ",".join(
            f"STRUCT({int(a)} AS rid1, {int(b)} AS rid2, '{int(a)}-{int(b)}' AS pair_id)"
            for (a, b) in pair_ids
        )
        return f"""
        WITH pairs AS (
          SELECT rid1, rid2, pair_id FROM UNNEST([{rows}])
        )
        SELECT p.pair_id AS id
        FROM pairs p
        JOIN movie.reviews AS r1 ON r1.reviewId = p.rid1
        JOIN movie.reviews AS r2 ON r2.reviewId = p.rid2
        WHERE r1.id = '{MOVIE_ID}' AND r2.id = '{MOVIE_ID}' AND r1.reviewId <> r2.reviewId
          AND AI.IF(
            ('{PAIR_PROMPT_PREFIX}', r1.reviewText, '{PAIR_PROMPT_SEP}', r2.reviewText),
            connection_id => 'us.connection',
            endpoint => 'gemini-2.5-flash'
          )
        LIMIT {TARGET_PAIRS}
        """
    return AiIfVerifier(verify_sql_template=verify_sql_template, id_column="id", coerce_id=str)


def main():
    profile = build_profile(
        scenario="movie", query_id=6, scale_factor=2000,
        params={"K_pos": K_POS, "K_neg": K_NEG, "target_pairs": TARGET_PAIRS},
        cascade_form=(
            "J+R cascade: PairCosineSignal scores all ordered self-pairs in ant_man scope; "
            f"top-{K_POS} positive (highest sim) + top-{K_NEG} negative (lowest sim) form the "
            "uncertain pool. AiIfVerifier (CTAS pair tuples → join movie.reviews twice) with "
            f"LIMIT {TARGET_PAIRS}; opposite-sentiment prompt."
        ),
        extra={
            "pair_prompt_prefix": PAIR_PROMPT_PREFIX,
            "pair_prompt_sep": PAIR_PROMPT_SEP,
            "structural_filter": f"r1.id = '{MOVIE_ID}'",
        },
    )

    print(f"Loading data + computing dase pair sims on {MOVIE_ID} subset...")
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
    n_gt_neg = int((df["scoreSentiment"] == "NEGATIVE").sum())
    print(f"  {MOVIE_ID}: {n_total} reviews ({n_gt_pos} POS, {n_gt_neg} NEG)")
    profile["data"] = {
        "n_reviews_total_dedup": len(df_full),
        "n_reviews_in_scope": n_total,
        "n_gt_positive_in_scope": n_gt_pos,
        "n_gt_negative_in_scope": n_gt_neg,
    }

    t0 = time.time()
    pair_signal = PairCosineSignal(embeddings_left=review_emb)
    pos_pairs, neg_pairs = select_top_pairs(pair_signal, n_total, K_POS, K_NEG)
    t_dase = time.time() - t0
    pair_ids = [(a, b) for (i, j, _s) in pos_pairs + neg_pairs
                for (a, b) in [(int(df.iloc[i]["reviewId"]), int(df.iloc[j]["reviewId"]))]]
    print(f"  K_pos={K_POS} pos-pair sims: {[f'{s:+.3f}' for _, _, s in pos_pairs]}")
    print(f"  K_neg={K_NEG} neg-pair sims: {[f'{s:+.3f}' for _, _, s in neg_pairs]}")
    print(f"  uncertain pool: {len(pair_ids)} pairs")
    profile["dase_breakdown"] = {"pair_signal_compute_s": t_dase}
    profile["dase_partition"] = {
        "K_pos": K_POS, "K_neg": K_NEG,
        "n_uncertain_pairs": len(pair_ids),
        "pos_pair_rids": [(int(df.iloc[i]["reviewId"]), int(df.iloc[j]["reviewId"]), s) for i, j, s in pos_pairs],
        "neg_pair_rids": [(int(df.iloc[i]["reviewId"]), int(df.iloc[j]["reviewId"]), s) for i, j, s in neg_pairs],
    }

    client = bq_client(PROJECT)

    print(f"\n=== Per-pair cost calibration ===")
    sample_texts = [str(df.iloc[i]["reviewText"]) for i, _, _ in (pos_pairs + neg_pairs)]
    cal = per_pair_cost_calibration(client, sample_texts, n_pairs=5)
    per_pair = cal["per_pair_cost_usd"]
    print(f"  per_pair=${per_pair:.6f}")
    profile["calibration"] = cal

    bsql = (
        f"SELECT r1.id, r1.reviewId AS reviewId1, r2.reviewId AS reviewId2 "
        f"FROM movie.reviews AS r1 JOIN movie.reviews AS r2 "
        f"ON r1.id = r2.id AND r1.reviewId <> r2.reviewId "
        f"WHERE r1.id = '{MOVIE_ID}' AND AI.IF("
        f"('{PAIR_PROMPT_PREFIX}', r1.reviewText, '{PAIR_PROMPT_SEP}', r2.reviewText), "
        f"connection_id => 'us.connection', endpoint => 'gemini-2.5-flash') "
        f"LIMIT {TARGET_PAIRS}"
    )
    if SKIP_BASELINE:
        print(f"\n=== Baseline ABORTED (SKIP_BASELINE=True) — using paper Table 4(a) numbers ===")
        bcalls_est = round(PAPER_BQ_Q6["cost_usd"] / per_pair) if per_pair else 0
        bcost = PAPER_BQ_Q6["cost_usd"]
        bwall = PAPER_BQ_Q6["latency_s"]
        bslot = None
        b_f1 = PAPER_BQ_Q6["score_f1"]
        profile["baseline"] = {
            "_status": "aborted",
            "_status_note": (
                "Baseline NOT run on our project. BQ self-join over the ant_man scope is too "
                "slow on our project (Q5 attempt: 301.6s). Per project policy, baseline metrics "
                "substituted from paper Table 4(a)."
            ),
            "method": "sembench bigquery/Q6.sql verbatim on movie.reviews — NOT EXECUTED",
            "sql": bsql,
            "score": {"precision": None, "recall": None, "f1": PAPER_BQ_Q6["score_f1"], "_source": "paper Table 4(a)"},
            "latency_breakdown": {"wall_s": PAPER_BQ_Q6["latency_s"], "slot_ms": None, "_source": "paper Table 4(a)"},
            "cost_breakdown": {
                "n_llm_calls_est": bcalls_est,
                "n_llm_calls_method": "paper $1.00 / per_pair_cost (our calibration)",
                "per_pair_cost_usd": per_pair,
                "total_cost_usd": PAPER_BQ_Q6["cost_usd"],
                "_source": "paper Table 4(a) cost",
            },
        }
    else:
        raise NotImplementedError("set SKIP_BASELINE=False only if you can wait 5+ min")

    print("\n=== Cascade verifier (single AI.IF on K_pos+K_neg pairs, LIMIT short-circuit) ===")
    verifier = make_q6_pair_verifier(df)
    t0 = time.time()
    vres = verifier.verify(client, pair_ids, per_pair)
    t_verify = time.time() - t0

    accepted_pair_ids = [s for s in vres.positive_ids]
    accepted_pairs = []
    for s in accepted_pair_ids:
        a, b = s.split("-")
        accepted_pairs.append((MOVIE_ID, int(a), int(b)))
    sys_df = pd.DataFrame(accepted_pairs, columns=["id", "reviewId1", "reviewId2"])
    cmetric = ev.evaluate_q6(sys_df)

    s2_calls_est = round(vres.slot_ms / 2500) if vres.slot_ms else len(sys_df)
    ccalls = max(s2_calls_est, len(sys_df))
    ccalls = min(ccalls, len(pair_ids))
    cascade_cost = per_pair * ccalls
    cascade_total_wall = t_dase + t_verify
    print(f"  returned {len(sys_df)} pairs")
    print(f"  P={cmetric.precision:.4f} R={cmetric.recall:.4f} F1={cmetric.f1_score:.4f}")
    print(f"  wall={cascade_total_wall:.2f}s, calls={ccalls}, cost=${cascade_cost:.6f}")

    profile["cascade"] = {
        "method": (
            "J+R cascade: PairCosineSignal top-K pos/neg pairs → AiIfVerifier (single AI.IF "
            "on materialized pair tuples joined to movie.reviews) with LIMIT short-circuit; "
            "opposite-sentiment prompt"
        ),
        "verifier": vres.to_dict(),
        "result_pairs": accepted_pairs,
        "score": {"precision": cmetric.precision, "recall": cmetric.recall, "f1": cmetric.f1_score},
        "totals": {
            "wall_s": cascade_total_wall,
            "wall_breakdown_s": {"dase_pair_signal": t_dase, "bq_verify": vres.wall_s},
            "slot_ms_bq_total": vres.slot_ms,
            "cost_usd": cascade_cost,
            "n_llm_calls": ccalls,
        },
    }

    profile["comparison"] = {
        "score_f1":    {"paper": PAPER_BQ_Q6["score_f1"], "baseline": b_f1, "cascade": cmetric.f1_score,
                        "_baseline_source": "paper (aborted)" if SKIP_BASELINE else "ours"},
        "wall_s":      {"paper": PAPER_BQ_Q6["latency_s"], "baseline": bwall, "cascade_total": cascade_total_wall},
        "slot_ms_bq":  {"baseline": bslot, "cascade_total": vres.slot_ms},
        "cost_usd":    {"paper": PAPER_BQ_Q6["cost_usd"], "baseline": bcost, "cascade": cascade_cost},
        "n_llm_calls": {
            "paper_implied": round(PAPER_BQ_Q6["cost_usd"] / per_pair) if per_pair else None,
            "baseline_est": bcalls_est,
            "cascade": ccalls,
        },
    }

    write_profile(profile, PROFILE_PATH)

    print_summary(
        f"Movie Q6 (J+R, K_pos={K_POS}, K_neg={K_NEG})",
        columns=["paper", "baseline", "cascade"],
        rows=[
            ("score (F1)", [PAPER_BQ_Q6["score_f1"], b_f1, cmetric.f1_score], ".2f"),
            ("wall (s)",   [PAPER_BQ_Q6["latency_s"], bwall, cascade_total_wall], ".2f"),
            ("cost ($)",   [PAPER_BQ_Q6["cost_usd"], bcost, cascade_cost], ".4f"),
            ("#LLM calls", [None, bcalls_est, ccalls], "d"),
        ],
    )


if __name__ == "__main__":
    main()
