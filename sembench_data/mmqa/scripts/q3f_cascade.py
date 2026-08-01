#!/usr/bin/env -S python -u
"""MMQA Q3f cascade — sem-filter on lizzy_caplan_text_data: rom-com filter.

Operator: F (text-only binary classification, subset of Q3a comedy).
Refactored to use dase_cascade primitives:
Cascade(MarginSignal, AbsoluteBand, AiIfVerifier).
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
NL_PATH = os.path.join(MMQA_DIR, "query", "natural_language", "q3f.json")
PROFILE_DIR = os.path.join(MMQA_DIR, "outputs")
PROFILE_PATH = os.path.join(PROFILE_DIR, "Q3f.json")
PROJECT = os.environ.get("GCP_PROJECT", "")
DATASET = "mmqa"
STAGING_TABLE = f"{DATASET}.q3f_uncertain"

POSITIVE_PROMPTS = ["a romantic comedy movie", "a lighthearted rom-com film", "a movie in the romantic comedy genre"]
NEGATIVE_PROMPTS = ["not a romantic comedy movie", "this film is not a rom-com", "not a love story told as a light comedy"]
MARGIN_HI = 0.013
MARGIN_LO = -0.030


def make_q3f_verifier():
    def make_staging(ids):
        def _esc(s):
            return s.replace("\\", "\\\\").replace("'", "\\'")
        structs = ",".join(
            f"STRUCT('{_esc(t)}' AS title, '{_esc(x)}' AS text)" for t, x in ids
        )
        return f"""
        CREATE OR REPLACE TABLE {STAGING_TABLE} AS
        SELECT title, text FROM UNNEST([{structs}])
        """
    verify_sql = f"""SELECT title FROM {STAGING_TABLE}
    WHERE AI.IF(title || " is a romantic comedy given their description: " || text,
      connection_id => 'us.connection', endpoint => 'gemini-2.5-flash')"""
    return AiIfVerifier(
        verify_sql=verify_sql, make_staging_sql=make_staging,
        id_column="title", coerce_id=str,
    )




def per_row_cost_q3f(client, sample_rows):
    sample_texts = [
        f"{t} is a romantic comedy given their description: {x}" for t, x in sample_rows
    ]
    return per_row_cost(
        client, prompt="", sample_texts=sample_texts,
        method_label="AI.GENERATE_BOOL inline title+text + thinking_budget=0",
        k=len(sample_texts),
    )


def main():
    profile = build_profile(
        scenario="mmqa", query_id="3f", scale_factor=200,
        params={"MARGIN_HI": MARGIN_HI, "MARGIN_LO": MARGIN_LO},
        cascade_form="F-cascade: MarginSignal + AbsoluteBand + AiIfVerifier (text); union confident_pos with BQ-verified uncertain.",
        extra={
            "operator": "sem-filter",
            "cascade_strategy": "confidence-based skip",
            "dase_prompts": {"positive": POSITIVE_PROMPTS, "negative": NEGATIVE_PROMPTS},
        },
    )
    df = pd.read_parquet(os.path.join(DATA_DIR, "lizzy_caplan_text_data.parquet"))
    gt = json.load(open(NL_PATH))["ground_truth"]
    n = len(df)
    print(f"  {n} rows; GT {len(gt)} rom-com titles")
    profile["data"] = {"n_rows": n, "n_gt": len(gt), "gt_titles": gt}

    text_emb = np.array(df["embedding"].tolist(), dtype=np.float32)
    row_ids = list(zip(df["title"].tolist(), df["text"].tolist()))

    client = bq_client(PROJECT)
    cal = per_row_cost_q3f(
        client, list(zip(df.iloc[:5]["title"].tolist(), df.iloc[:5]["text"].tolist())),
    )
    per_row = cal.per_row_cost_usd
    print(f"  per_row=${per_row:.6f}")
    profile["calibration"] = cal.to_dict()

    cascade = Cascade(
        embeddings=text_emb, ids=row_ids,
        signal=MarginSignal(positive_prompts=POSITIVE_PROMPTS, negative_prompts=NEGATIVE_PROMPTS),
        band=AbsoluteBand(tau_low=MARGIN_LO, tau_high=MARGIN_HI),
        verifier=make_q3f_verifier(),
    )
    cres = cascade.run(client, per_row)

    cp_titles = [t for t, _ in cres.confident_pos_ids]
    bq_pass = sorted(cres.verifier_result.positive_ids)
    print(f"  partition: cp={cres.partition.to_dict()['n_confident_pos']}, "
          f"uncertain={cres.partition.to_dict()['n_uncertain']}, "
          f"cn={cres.partition.to_dict()['n_confident_neg']}; cp_titles={cp_titles}")
    profile["dase_partition"] = cres.partition.to_dict() | {
        "confident_pos_titles": cp_titles,
    }


    cascade_titles = sorted(set(cp_titles) | set(bq_pass))
    cp_v, cr_v, cscore = f1_set(cascade_titles, gt)
    print(f"  cascade {len(cascade_titles)} titles, F1={cscore:.4f} P={cp_v:.4f} R={cr_v:.4f}")
    n_cas = cres.verifier_result.n_calls
    cas_cost = cres.verifier_result.cost_usd
    cas_wall = cres.verifier_result.ctas_wall_s + cres.verifier_result.wall_s
    cas_slot = cres.verifier_result.ctas_slot_ms + cres.verifier_result.slot_ms

    profile["cascade"] = {
        "method": "Cascade(MarginSignal, AbsoluteBand, AiIfVerifier).run() — union confident_pos with BQ-verified uncertain",
        "verifier": cres.verifier_result.to_dict(),
        "result_titles": cascade_titles,
        "score": {"f1": cscore, "precision": cp_v, "recall": cr_v},
        "totals": {"wall_s": cas_wall, "slot_ms_bq_total": cas_slot,
                   "cost_usd": cas_cost, "n_llm_calls": n_cas},
    }
    write_profile(profile, PROFILE_PATH)


if __name__ == "__main__":
    main()
