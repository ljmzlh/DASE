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
    f1_set, build_profile, write_profile,
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


    profile["cascade"] = {
        "method": "F-cascade: Cascade(MarginSignal, AbsoluteBand, AiIfVerifier).run() — union confident_pos with BQ-verified uncertain",
        "verifier": cres.verifier_result.to_dict(),
        "result_titles": cascade_titles,
        "score": {"f1": cf1_v, "precision": cp_v, "recall": cr_v},
        "totals": {"wall_s": cascade_wall, "slot_ms_bq_total": cascade_slot,
                   "cost_usd": cascade_cost, "n_llm_calls": n_calls_cascade},
    }


    write_profile(profile, PROFILE_PATH)



if __name__ == "__main__":
    main()
