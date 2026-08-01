#!/usr/bin/env -S python -u
"""
Cars Q10 cascade — sem_classify (24-class, AI.CLASSIFY).

NL: For all complaints, classify which car component is problematic.
GT: 19657 (car_id, problem_category) rows, 24 categories.
Eval: macro F1 across 24 classes.

Refactored. Operator (paper Table 3): C (multi-class classify).
Pattern: anchor-set argmax + top1−top2 confidence is a standard ConfidenceMargin
signal that doesn't quite fit MarginSignal/RoleMarginSignal — inlined as a
small helper here. AlphaBand on the confidence + AiGenerateVerifier on the
uncertain half (returns id→class via .values), then merge dase argmax for
confident half + BQ class for uncertain half.
"""
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import (
    AlphaBand, AiGenerateVerifier, ConfidenceMarginSignal,
    bq_client, embed_query, per_row_cost, run_query,
    build_profile, write_profile,
)

CARS_DIR = os.path.abspath(os.path.join(_HERE, ".."))
TEXT_PARQUET        = os.path.join(CARS_DIR, "data", "text_complaints.parquet")
GT_CSV              = os.path.join(CARS_DIR, "ground_truth", "Q10.csv")
PROFILE_PATH        = os.path.join(CARS_DIR, "outputs", "Q10.json")

PROJECT       = os.environ.get("GCP_PROJECT", "")
DATASET       = "cars_dataset"
STAGING_TABLE = f"{DATASET}.q10_uncertain_complaints"

CATEGORIES = [
    "ELECTRICAL SYSTEM", "POWER TRAIN", "ENGINE", "STEERING", "SERVICE BRAKES",
    "STRUCTURE", "AIR BAGS", "ENGINE AND ENGINE COOLING", "VEHICLE SPEED CONTROL",
    "VISIBILITY/WIPER", "FUEL/PROPULSION SYSTEM", "FORWARD COLLISION AVOIDANCE",
    "EXTERIOR LIGHTING", "SUSPENSION", "FUEL SYSTEM", "VISIBILITY", "WHEELS",
    "SEAT BELTS", "BACK OVER PREVENTION", "TIRES", "SEATS", "LATCHES/LOCKS/LINKAGES",
    "LANE DEPARTURE", "EQUIPMENT",
]
ANCHOR_PROMPTS = [f"complaint about {c.lower()}" for c in CATEGORIES]
PROMPT_TEMPLATE = "Classify car complaint to one of given problem categories. Answer only one of given problem categories, nothing more. Complaint: %s"

ALPHA = 0.5


def trunc2(x): return f"{math.floor(x * 100) / 100:.2f}"


def make_q10_verifier():
    """Stage1 CTAS uncertain complaints; Stage2 verbatim Q10 AI.CLASSIFY → id→class."""
    def make_staging(complaint_ids):
        cid_list = ",".join(str(int(c)) for c in complaint_ids)
        return f"""
        CREATE OR REPLACE TABLE {STAGING_TABLE} AS
        SELECT * FROM {DATASET}.complaints WHERE complaint_id IN ({cid_list})
        """
    cats_sql = ", ".join(f'"{c}"' for c in CATEGORIES)
    verify_sql = f"""
    SELECT s.complaint_id AS id, AI.CLASSIFY(
        FORMAT('{PROMPT_TEMPLATE}', s.summary),
        categories => [{cats_sql}],
        connection_id => 'us.connection',
        endpoint => 'gemini-2.5-flash'
    ) AS category
    FROM {STAGING_TABLE} AS s
    """
    return AiGenerateVerifier(
        verify_sql=verify_sql, make_staging_sql=make_staging,
        id_column="id", value_column="category",
        coerce_id=int, coerce_value=lambda v: str(v).lower().replace("\n", ""),
    )


def macro_f1(gt_df, sys_df, id_col="car_id", label_col="problem_category"):
    gt = gt_df.sort_values(id_col).reset_index(drop=True)
    sys = sys_df.sort_values(id_col).reset_index(drop=True)
    sys[label_col] = sys[label_col].apply(lambda x: str(x).lower().replace("\n", ""))
    p, r, f1, _ = precision_recall_fscore_support(
        gt[label_col], sys[label_col], average="macro", zero_division=0)
    return p, r, f1




def main():
    profile = build_profile(
        scenario="cars", query_id=10, scale_factor=19672,
        prompt=PROMPT_TEMPLATE, params={"alpha": ALPHA, "K_classes": len(CATEGORIES)},
        cascade_form=("C: 24-anchor argmax + top1−top2 confidence → AlphaBand → "
                      "AiGenerateVerifier(AI.CLASSIFY) on uncertain half; merge "
                      "dase class on confident + BQ class on uncertain."),
        extra={"operator": "C", "categories": CATEGORIES,
               "dase_prompts": {"anchors": ANCHOR_PROMPTS}},
    )

    print("Loading text_complaints + GT...")
    df = pd.read_parquet(TEXT_PARQUET)
    n_total = len(df)
    gt = pd.read_csv(GT_CSV)
    gt_aligned = gt.set_index("car_id").reindex(df["car_id"]).reset_index()
    print(f"  {n_total} complaints; GT rows: {len(gt)}")
    profile["data"] = {"n_complaints": n_total, "n_gt_rows": len(gt)}

    # ── Signal: anchor argmax + top1−top2 confidence ──
    t = time.time()
    text_emb = np.stack(df["embedding"].tolist()).astype(np.float32)
    signal = ConfidenceMarginSignal(anchors=ANCHOR_PROMPTS)
    confidence = signal.compute(text_emb)
    dase_classes = [CATEGORIES[k].lower() for k in signal.last_argmax]
    t_signal = time.time() - t

    # ── Band: AlphaBand on confidence (bottom alpha → uncertain) ──
    t = time.time()
    part = AlphaBand(alpha=ALPHA).partition(confidence)
    uncertain_set = set(int(i) for i in part.uncertain.tolist())
    uncertain_complaint_ids = [int(df.iloc[i]["complaint_id"]) for i in sorted(uncertain_set)]
    n_uncertain = len(uncertain_complaint_ids)
    t_partition = time.time() - t

    print(f"  alpha={ALPHA}: n_uncertain={n_uncertain}, n_confident={n_total - n_uncertain}")
    print(f"  confidence min={confidence.min():.4f}, median={np.median(confidence):.4f}, max={confidence.max():.4f}")
    profile["dase_breakdown_s"] = {"signal_s": t_signal, "partition_s": t_partition,
                                    "total": t_signal + t_partition}
    profile["dase_partition"] = {
        "n_uncertain": n_uncertain, "n_confident": n_total - n_uncertain,
        "confidence_min": float(confidence.min()),
        "confidence_median": float(np.median(confidence)),
        "confidence_max": float(confidence.max()),
    }

    client = bq_client(PROJECT)

    # ── Cost calibration ──
    print("\n=== Per-row cost calibration ===")
    sample_summaries = [str(df.iloc[i]["summary"]) for i in range(min(10, n_total))]
    cal = per_row_cost(
        client, prompt=PROMPT_TEMPLATE, sample_texts=sample_summaries,
        method_label="AI.GENERATE_BOOL with Q10 FORMAT prompt approx + thinking_budget=0", k=10,
    )
    per_row = cal.per_row_cost_usd
    print(f"  per_row=${per_row:.6f}")
    profile["calibration"] = cal.to_dict()


    # ── Cascade: verifier on uncertain ──
    print(f"\n=== Cascade: AiGenerateVerifier(AI.CLASSIFY) on {n_uncertain} uncertain complaints ===")
    verifier = make_q10_verifier()
    vres = verifier.verify(client, uncertain_complaint_ids, per_row)
    bq_class_by_complaint = vres.values  # complaint_id → class label
    print(f"  CTAS wall={vres.ctas_wall_s:.2f}s slot={vres.ctas_slot_ms}; "
          f"AI.CLASSIFY wall={vres.wall_s:.2f}s slot={vres.slot_ms}; "
          f"calls={vres.n_calls} cost=${vres.cost_usd:.6f}")

    # ── Merge: dase argmax for confident, BQ class for uncertain ──
    cascade_classes = []
    for i in range(n_total):
        if i in uncertain_set:
            cid = int(df.iloc[i]["complaint_id"])
            cascade_classes.append(bq_class_by_complaint.get(cid, dase_classes[i]))
        else:
            cascade_classes.append(dase_classes[i])
    cascade_sys = pd.DataFrame({"car_id": df["car_id"].values, "problem_category": cascade_classes})
    cp, cr, c_f1 = macro_f1(gt_aligned, cascade_sys)
    print(f"  cascade macro P={cp:.4f} R={cr:.4f} F1={c_f1:.4f}")

    cascade_total_wall = profile["dase_breakdown_s"]["total"] + vres.ctas_wall_s + vres.wall_s
    cascade_total_slot = vres.ctas_slot_ms + vres.slot_ms

    profile["cascade"] = {
        "method": "C: 24-anchor argmax + top1−top2 confidence + AlphaBand + AiGenerateVerifier on uncertain half; merge dase + BQ classes",
        "verifier": vres.to_dict(),
        "score": {"precision": cp, "recall": cr, "f1_score": c_f1},
        "totals": {
            "wall_s": cascade_total_wall,
            "wall_breakdown_s": {"dase": profile["dase_breakdown_s"]["total"],
                                  "bq_stage1_ctas": vres.ctas_wall_s, "bq_stage2_aiif": vres.wall_s},
            "slot_ms_bq_total": cascade_total_slot,
            "cost_usd": vres.cost_usd, "n_llm_calls": vres.n_calls,
        },
    }
    write_profile(profile, PROFILE_PATH)



if __name__ == "__main__":
    main()
