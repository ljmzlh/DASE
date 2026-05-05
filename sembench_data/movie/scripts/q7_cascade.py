#!/usr/bin/env -S python -u
"""
Movie Q7 cascade v2 — opposite-sentiment review pairs (J + R, NO LIMIT) for `ant_man`.

NL: ALL pairs of reviews with OPPOSITE sentiment for ant_man_and_the_wasp_quantumania.
GT: pairs (r1, r2) where r1.scoreSentiment != r2.scoreSentiment (no LIMIT).
Eval: _review_pairs (no limit) precision/recall/F1.

Operator (paper Table 3): J + R. PairCosineSignal scores all ordered self-pairs;
top-K_POS most-similar + top-K_NEG most-dissimilar form the uncertain pool;
AiIfVerifier runs the opposite-sentiment AI.IF on those pairs (NO LIMIT — Q7 returns
all positives in the uncertain pool).

NOTE on operator semantics: Q7's NL is "ALL opposite pairs" (no LIMIT, ~16k ordered
self-join pairs in the ant_man scope of ~128 reviews). PairCosineSignal+top-K only
materializes 2*K pairs as candidates, capping recall at 2*K / |GT_pairs|.
PairCosineSignal scores the Cartesian product and selects opposite-sentiment pairs by similarity threshold.
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
PROFILE_PATH = os.path.join(MOVIE_DIR, "outputs", "Q7.json")

PROJECT  = os.environ.get("GCP_PROJECT", "")
PAIR_PROMPT_PREFIX = "These two movie reviews express opposite sentiments - one is positive and the other is negative. Review 1: "
PAIR_PROMPT_SEP    = ", Review 2: "
MOVIE_ID = "ant_man_and_the_wasp_quantumania"

# Q7 has no LIMIT — use larger K to capture more candidate opposite pairs.
K_POS = 100
K_NEG = 1000   # extra mass on low-sim pairs (likely opposite-sentiment candidates)
PAPER_BQ_Q7 = {"score_f1": 0.70, "latency_s": 198.3, "cost_usd": 3.31}
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
    k_pos = min(k_pos, flat.size)
    k_neg = min(k_neg, flat.size)
    top_pos_idx = np.argpartition(-flat, k_pos - 1)[:k_pos]
    flat_for_neg = np.where(np.isfinite(flat), flat, np.inf)
    top_neg_idx = np.argpartition(flat_for_neg, k_neg - 1)[:k_neg]
    pos_pairs = [(int(idx // n), int(idx % n), float(flat[idx])) for idx in top_pos_idx]
    neg_pairs = [(int(idx // n), int(idx % n), float(flat[idx])) for idx in top_neg_idx]
    pos_pairs.sort(key=lambda t: -t[2])
    neg_pairs.sort(key=lambda t: t[2])
    return pos_pairs, neg_pairs


def make_q7_pair_verifier(df: pd.DataFrame):
    """AI.IF on pair tuples — NO LIMIT (Q7 returns all positives in candidate pool)."""
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
        """
    return AiIfVerifier(verify_sql_template=verify_sql_template, id_column="id", coerce_id=str)


def main():
    profile = build_profile(
        scenario="movie", query_id=7, scale_factor=2000,
        params={"K_pos": K_POS, "K_neg": K_NEG},
        cascade_form=(
            "J+R cascade (no LIMIT): PairCosineSignal scores all ordered self-pairs in ant_man scope; "
            f"top-{K_POS} positive (highest sim) + top-{K_NEG} negative (lowest sim, ≈ opposite) "
            "form the uncertain pool. AiIfVerifier runs opposite-sentiment AI.IF on those pairs "
            "with NO LIMIT (returns all positives). NOTE: original Q7 used Option D "
            "(per-row classify + Cartesian product) to get full recall; this v2 uses "
            "PairCosineSignal per migration spec — recall is capped by candidate pool size."
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
    print(f"  GT opposite pair count (ordered) = {2 * n_gt_pos * n_gt_neg}")
    profile["data"] = {
        "n_reviews_total_dedup": len(df_full),
        "n_reviews_in_scope": n_total,
        "n_gt_positive_in_scope": n_gt_pos,
        "n_gt_negative_in_scope": n_gt_neg,
        "n_gt_opposite_pairs_ordered": 2 * n_gt_pos * n_gt_neg,
    }

    t0 = time.time()
    pair_signal = PairCosineSignal(embeddings_left=review_emb)
    pos_pairs, neg_pairs = select_top_pairs(pair_signal, n_total, K_POS, K_NEG)
    t_dase = time.time() - t0
    pair_ids = [(int(df.iloc[i]["reviewId"]), int(df.iloc[j]["reviewId"]))
                for i, j, _s in pos_pairs + neg_pairs]
    # dedupe in case top-K pos and top-K neg overlap (unlikely)
    seen = set()
    pair_ids_dedup = []
    for p in pair_ids:
        if p not in seen:
            seen.add(p)
            pair_ids_dedup.append(p)
    pair_ids = pair_ids_dedup
    print(f"  K_pos={K_POS}, K_neg={K_NEG}, uncertain pool: {len(pair_ids)} unique pairs")
    profile["dase_breakdown"] = {"pair_signal_compute_s": t_dase}
    profile["dase_partition"] = {
        "K_pos": K_POS, "K_neg": K_NEG,
        "n_uncertain_pairs": len(pair_ids),
    }

    client = bq_client(PROJECT)

    print(f"\n=== Per-pair cost calibration ===")
    sample_texts = [str(df.iloc[i]["reviewText"]) for i in range(min(20, n_total))]
    cal = per_pair_cost_calibration(client, sample_texts, n_pairs=5)
    per_pair = cal["per_pair_cost_usd"]
    print(f"  per_pair=${per_pair:.6f}")
    profile["calibration"] = cal

    bsql = (
        "SELECT r1.id, r1.reviewId AS reviewId1, r2.reviewId AS reviewId2 "
        "FROM movie.reviews AS r1 JOIN movie.reviews AS r2 "
        "ON r1.id = r2.id AND r1.reviewId <> r2.reviewId "
        f"WHERE r1.id = '{MOVIE_ID}' AND AI.IF("
        f"('{PAIR_PROMPT_PREFIX}', r1.reviewText, '{PAIR_PROMPT_SEP}', r2.reviewText), "
        "connection_id => 'us.connection', endpoint => 'gemini-2.5-flash')"
    )
    if SKIP_BASELINE:
        print(f"\n=== Baseline ABORTED (SKIP_BASELINE=True) — using paper Table 4(a) numbers ===")
        bcalls_est = round(PAPER_BQ_Q7["cost_usd"] / per_pair) if per_pair else 0
        bcost = PAPER_BQ_Q7["cost_usd"]
        bwall = PAPER_BQ_Q7["latency_s"]
        bslot = None
        b_f1 = PAPER_BQ_Q7["score_f1"]
        profile["baseline"] = {
            "_status": "aborted",
            "_status_note": (
                "Baseline NOT run on our project. Q7 has no LIMIT — must AI.IF on all ~16k "
                "pairs (paper 198.3s, $3.31). Per project policy, baseline metrics substituted "
                "from paper Table 4(a)."
            ),
            "method": "sembench bigquery/Q7.sql verbatim on movie.reviews — NOT EXECUTED",
            "sql": bsql,
            "score": {"precision": None, "recall": None, "f1": PAPER_BQ_Q7["score_f1"], "_source": "paper Table 4(a)"},
            "latency_breakdown": {"wall_s": PAPER_BQ_Q7["latency_s"], "slot_ms": None, "_source": "paper Table 4(a)"},
            "cost_breakdown": {
                "n_llm_calls_est": bcalls_est,
                "n_llm_calls_method": "paper $3.31 / per_pair_cost (our calibration)",
                "per_pair_cost_usd": per_pair,
                "total_cost_usd": PAPER_BQ_Q7["cost_usd"],
                "_source": "paper Table 4(a) cost",
            },
        }
    else:
        raise NotImplementedError("set SKIP_BASELINE=False only if you can wait 30+ min and pay ~$3")

    print(f"\n=== Cascade verifier (AI.IF on {len(pair_ids)} pairs, NO LIMIT) ===")
    verifier = make_q7_pair_verifier(df)
    t0 = time.time()
    vres = verifier.verify(client, pair_ids, per_pair)
    t_verify = time.time() - t0

    accepted_pair_ids = [s for s in vres.positive_ids]
    accepted_pairs = [(MOVIE_ID, int(s.split("-")[0]), int(s.split("-")[1])) for s in accepted_pair_ids]
    sys_df = pd.DataFrame(accepted_pairs, columns=["id", "reviewId1", "reviewId2"])
    cmetric = ev.evaluate_q7(sys_df)

    ccalls = len(pair_ids)  # no LIMIT — all candidates evaluated
    cascade_cost = per_pair * ccalls
    cascade_total_wall = t_dase + t_verify
    print(f"  returned {len(sys_df)} pairs")
    print(f"  P={cmetric.precision:.4f} R={cmetric.recall:.4f} F1={cmetric.f1_score:.4f}")
    print(f"  wall={cascade_total_wall:.2f}s, calls={ccalls}, cost=${cascade_cost:.6f}")

    profile["cascade"] = {
        "method": (
            "J+R cascade (no LIMIT): PairCosineSignal top-K_pos/K_neg pairs → AiIfVerifier "
            "(opposite-sentiment AI.IF on materialized pair tuples, no LIMIT)"
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
        "score_f1":    {"paper": PAPER_BQ_Q7["score_f1"], "baseline": b_f1, "cascade": cmetric.f1_score,
                        "_baseline_source": "paper (aborted)" if SKIP_BASELINE else "ours"},
        "wall_s":      {"paper": PAPER_BQ_Q7["latency_s"], "baseline": bwall, "cascade_total": cascade_total_wall},
        "slot_ms_bq":  {"baseline": bslot, "cascade_total": vres.slot_ms},
        "cost_usd":    {"paper": PAPER_BQ_Q7["cost_usd"], "baseline": bcost, "cascade": cascade_cost},
        "n_llm_calls": {
            "paper_implied_via_per_pair": bcalls_est,
            "baseline_est": bcalls_est,
            "cascade": ccalls,
        },
    }

    write_profile(profile, PROFILE_PATH)

    print_summary(
        f"Movie Q7 (J+R, no LIMIT, K_pos={K_POS}, K_neg={K_NEG})",
        columns=["paper", "baseline", "cascade"],
        rows=[
            ("score (F1)", [PAPER_BQ_Q7["score_f1"], b_f1, cmetric.f1_score], ".2f"),
            ("wall (s)",   [PAPER_BQ_Q7["latency_s"], bwall, cascade_total_wall], ".2f"),
            ("cost ($)",   [PAPER_BQ_Q7["cost_usd"], bcost, cascade_cost], ".4f"),
            ("#LLM calls", [None, bcalls_est, ccalls], "d"),
        ],
    )


if __name__ == "__main__":
    main()
