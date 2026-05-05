#!/usr/bin/env -S python -u
"""
MMQA Q3a cascade — sem-filter (text-only): "Which movies are comedies?"

NL: 13 comedy titles in GT.
Operator: F (row-level binary classification on lizzy_caplan_text_data, 200 rows).

Refactored to use dase_cascade unified solver. Operator (paper Table 3): F.
Cascade(MarginSignal, AbsoluteBand, AiIfVerifier) drives prefilter → BQ AI.IF
on uncertain rows. Caller assembles final positive set as
confident_pos ∪ bq_yes (sem-filter union).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DASE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
SEMBENCH_MY = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, DASE_ROOT)
sys.path.insert(0, SEMBENCH_MY)

from google.cloud import bigquery  # noqa: E402

from dase_cascade import (  # noqa: E402
    Cascade, MarginSignal, AbsoluteBand, AiIfVerifier,
    bq_client, per_row_cost, run_query,
    f1_set, build_profile, write_profile, print_summary,
)

MMQA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(MMQA_DIR, "data")
NL_PATH = os.path.join(MMQA_DIR, "query", "natural_language", "q3a.json")
PROFILE_DIR = os.path.join(MMQA_DIR, "outputs")
PROFILE_PATH = os.path.join(PROFILE_DIR, "Q3a.json")

PROJECT = os.environ.get("GCP_PROJECT", "")
DATASET = "mmqa"
STAGING_TABLE = f"{DATASET}.q3a_uncertain"

POSITIVE_PROMPTS = [
    "a comedy movie",
    "a humorous comedy film",
    "a movie in the comedy genre",
]
NEGATIVE_PROMPTS = [
    "not a comedy movie",
    "this film is not a comedy and not primarily humorous",
    "not a funny or joke-driven comedy",
]

# Confidence thresholds calibrated on the signal's empirical std (held-out sample), not GT-tuned.
MARGIN_HI = 0.020
MARGIN_LO = -0.030

PAPER_BQ_Q3a = {"score": 0.7647, "latency_s": 13.3, "cost_usd": 0.0099}
PAPER_DASE_NN_Q3a = {"score": None, "latency_s": 1e-3, "cost_usd": 1e-9}
SKIP_BASELINE = False  # 200 rows single AI.IF agg, safe


def make_q3a_verifier():
    """CTAS staging from uncertain (title, text) tuples; AI.IF on staging."""
    def make_staging(ids):
        # ids here are (title, text) tuples — we encode them via array params.
        titles = [t for t, _ in ids]
        texts = [x for _, x in ids]
        # Inline as VALUES; BQ array params can't be embedded in CREATE OR
        # REPLACE TABLE ... AS SELECT FROM UNNEST(@titles) WITH OFFSET style
        # without query parameters, so we synthesize literal STRUCTs.
        def _esc(s):
            return s.replace("\\", "\\\\").replace("'", "\\'")
        structs = ",".join(
            f"STRUCT('{_esc(t)}' AS title, '{_esc(x)}' AS text)"
            for t, x in zip(titles, texts)
        )
        return f"""
        CREATE OR REPLACE TABLE {STAGING_TABLE} AS
        SELECT title, text FROM UNNEST([{structs}])
        """

    verify_sql = f"""
    SELECT title FROM {STAGING_TABLE}
    WHERE AI.IF(
      title || " is a comedy movie given their description: " || text,
      connection_id => 'us.connection', endpoint => 'gemini-2.5-flash')
    """
    return AiIfVerifier(
        verify_sql=verify_sql,
        make_staging_sql=make_staging,
        id_column="title", coerce_id=str,
    )


def run_baseline(client):
    """verbatim BQ q3a.sql (model_params stripped)."""
    sql = f"""
    SELECT title
    FROM {DATASET}.lizzy_caplan_text_data t
    WHERE AI.IF(
      t.title || " is a comedy movie given their description: " || t.text,
      connection_id => 'us.connection', endpoint => 'gemini-2.5-flash')
    """
    return run_query(client, sql)


def per_row_cost_q3a(client, sample_rows):
    """Q3a calibration uses bound (title, text) pair concatenated by a per-row
    prompt. The general per_row_cost(sample_texts=...) helper takes a single
    string per row, so we synthesize each row's full prompt body up-front."""
    sample_texts = [
        f"{t} is a comedy movie given their description: {x}"
        for t, x in sample_rows
    ]
    return per_row_cost(
        client,
        prompt="",  # prompt is fully embedded in the per-row text
        sample_texts=sample_texts,
        method_label="AI.GENERATE_BOOL inline title+text + thinking_budget=0",
        k=len(sample_texts),
    )


def main():
    profile = build_profile(
        scenario="mmqa", query_id="3a", scale_factor=200,
        params={"MARGIN_HI": MARGIN_HI, "MARGIN_LO": MARGIN_LO},
        cascade_form="F-cascade: MarginSignal + AbsoluteBand + AiIfVerifier (text); union confident_pos with BQ-verified uncertain.",
        extra={
            "operator": "sem-filter (text-only binary classification)",
            "cascade_strategy": "confidence-based skip (sem-filter)",
            "dase_prompts": {"positive": POSITIVE_PROMPTS, "negative": NEGATIVE_PROMPTS},
        },
    )

    print("Loading lizzy_caplan_text_data + GT (NL JSON)...")
    df = pd.read_parquet(os.path.join(DATA_DIR, "lizzy_caplan_text_data.parquet"))
    with open(NL_PATH) as f:
        gt_titles = json.load(f)["ground_truth"]
    n = len(df)
    print(f"  {n} rows; GT {len(gt_titles)} comedy titles")
    profile["data"] = {"n_rows": n, "n_gt": len(gt_titles), "gt_titles": gt_titles}

    text_emb = np.array(df["embedding"].tolist(), dtype=np.float32)
    # ids are (title, text) tuples so the verifier can rebuild the staging table
    row_ids = list(zip(df["title"].tolist(), df["text"].tolist()))

    client = bq_client(PROJECT)

    print("\n=== Cost calibration (5 sample rows) ===")
    sample_rows = list(zip(df.iloc[:5]["title"].tolist(), df.iloc[:5]["text"].tolist()))
    cal = per_row_cost_q3a(client, sample_rows)
    per_row = cal.per_row_cost_usd
    print(f"  per_row=${per_row:.6f}")
    profile["calibration"] = cal.to_dict()

    # ── Cascade ──
    cascade = Cascade(
        embeddings=text_emb,
        ids=row_ids,
        signal=MarginSignal(positive_prompts=POSITIVE_PROMPTS, negative_prompts=NEGATIVE_PROMPTS),
        band=AbsoluteBand(tau_low=MARGIN_LO, tau_high=MARGIN_HI),
        verifier=make_q3a_verifier(),
    )
    print("\n=== Cascade (MarginSignal → AbsoluteBand → AiIfVerifier) ===")
    cres = cascade.run(client, per_row)

    confident_pos_titles = [t for t, _ in cres.confident_pos_ids]
    bq_pass_titles = sorted(cres.verifier_result.positive_ids)
    cascade_titles = sorted(set(confident_pos_titles) | set(bq_pass_titles))
    cscore_f1, cp, cr = f1_set(cascade_titles, gt_titles)
    # Reorder to (f1, p, r) → original print used (f1, p, r)
    cp_p, cr_p, cscore_f1 = cscore_f1, cp, cr
    # f1_set returns (p, r, f1); rebind unambiguously
    cp_v, cr_v, cf1_v = f1_set(cascade_titles, gt_titles)
    print(f"  margin partition: confident_pos={cres.partition.to_dict()['n_confident_pos']}, "
          f"uncertain={cres.partition.to_dict()['n_uncertain']}, "
          f"confident_neg={cres.partition.to_dict()['n_confident_neg']}")
    print(f"  dase confident_pos titles (preview 5): {confident_pos_titles[:5]}")
    print(f"  bq verified: {len(bq_pass_titles)} titles")
    print(f"  cascade output: {len(cascade_titles)} titles, F1={cf1_v:.4f} P={cp_v:.4f} R={cr_v:.4f}")

    profile["dase_partition"] = cres.partition.to_dict() | {
        "confident_pos_titles": confident_pos_titles,
    }

    n_calls_cascade = cres.verifier_result.n_calls
    cascade_cost = cres.verifier_result.cost_usd
    cascade_wall = cres.verifier_result.ctas_wall_s + cres.verifier_result.wall_s
    cascade_slot = cres.verifier_result.ctas_slot_ms + cres.verifier_result.slot_ms

    # ── ours BQ baseline (verbatim) ──
    if SKIP_BASELINE:
        bcost = PAPER_BQ_Q3a["cost_usd"]; bwall = PAPER_BQ_Q3a["latency_s"]; bslot = None
        bscore_f1 = PAPER_BQ_Q3a["score"]; bcalls = n; b_titles = None
        profile["baseline"] = {
            "_status": "skipped", "score": {"f1": bscore_f1, "_source": "paper"},
            "latency_breakdown": {"wall_s": bwall, "_source": "paper"},
            "cost_breakdown": {"n_llm_calls": bcalls, "total_cost_usd": bcost, "_source": "paper"},
        }
    else:
        print(f"\n=== Baseline (verbatim BQ q3a.sql, 200 AI.IF) ===")
        bdf, bwall, bslot, bsql = run_baseline(client)
        b_titles = bdf["title"].tolist()
        bcalls = n
        bcost = per_row * n
        bp_v, br_v, bscore_f1 = f1_set(b_titles, gt_titles)
        print(f"  baseline returned {len(b_titles)} titles, F1={bscore_f1:.4f} P={bp_v:.4f} R={br_v:.4f}")
        print(f"  wall={bwall:.2f}s slot_ms={bslot} cost=${bcost:.6f}")
        profile["baseline"] = {
            "method": "verbatim BQ q3a.sql", "sql": bsql, "result_titles": b_titles,
            "score": {"f1": bscore_f1, "precision": bp_v, "recall": br_v},
            "latency_breakdown": {"wall_s": bwall, "slot_ms": bslot},
            "cost_breakdown": {"n_llm_calls": bcalls, "per_row_cost_usd": per_row, "total_cost_usd": bcost},
        }

    profile["cascade"] = {
        "method": "F-cascade: Cascade(MarginSignal, AbsoluteBand, AiIfVerifier).run() — union confident_pos with BQ-verified uncertain",
        "verifier": cres.verifier_result.to_dict(),
        "result_titles": cascade_titles,
        "score": {"f1": cf1_v, "precision": cp_v, "recall": cr_v},
        "totals": {"wall_s": cascade_wall, "slot_ms_bq_total": cascade_slot,
                   "cost_usd": cascade_cost, "n_llm_calls": n_calls_cascade},
    }

    profile["comparison"] = {
        "score": {"paper_BQ": PAPER_BQ_Q3a["score"], "paper_DASE_NN": PAPER_DASE_NN_Q3a["score"],
                   "ours_BQ": bscore_f1, "ours_cascade": cf1_v},
        "wall_s": {"paper_BQ": PAPER_BQ_Q3a["latency_s"], "paper_DASE_NN": PAPER_DASE_NN_Q3a["latency_s"],
                    "ours_BQ": bwall, "ours_cascade": cascade_wall},
        "cost_usd": {"paper_BQ": PAPER_BQ_Q3a["cost_usd"], "paper_DASE_NN": PAPER_DASE_NN_Q3a["cost_usd"],
                      "ours_BQ": bcost, "ours_cascade": cascade_cost},
        "n_llm_calls": {"paper_BQ": n, "paper_DASE_NN": 0,
                         "ours_BQ": bcalls, "ours_cascade": n_calls_cascade},
    }

    write_profile(profile, PROFILE_PATH)

    print_summary(
        f"MMQA Q3a (MARGIN_HI={MARGIN_HI}, MARGIN_LO={MARGIN_LO})",
        columns=["paper BQ", "DASE+NN", "ours BQ", "ours cascade"],
        rows=[
            ("score (F1)", [PAPER_BQ_Q3a["score"], PAPER_DASE_NN_Q3a["score"], bscore_f1, cf1_v], ".2f"),
            ("cost ($)",   [PAPER_BQ_Q3a["cost_usd"], PAPER_DASE_NN_Q3a["cost_usd"], bcost, cascade_cost], ".4f"),
            ("wall (s)",   [PAPER_BQ_Q3a["latency_s"], PAPER_DASE_NN_Q3a["latency_s"], bwall, cascade_wall], ".2f"),
            ("#LLM calls", [n, 0, bcalls, n_calls_cascade], "d"),
        ],
    )


if __name__ == "__main__":
    main()
