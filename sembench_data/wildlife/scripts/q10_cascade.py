#!/usr/bin/env -S python -u
"""
Wildlife Q10 cascade — sem_filter (zebra) + GROUP BY (City, StationID) + argmax.

NL: City and station with most zebra pictures (ties broken arbitrarily).
GT: argmax_(City,StationID) count(Species LIKE '%ZEBRA%').
Eval: F1 = 1.0 if pred (city, station) is one of GT-tied tops, else fraction match.

Refactored to use dase_cascade unified solver. Operator (paper Table 3): F+L.
Like Q3 but the GROUP BY key is the 2-tuple (City, StationID). Cascade returns
the subset of uncertain ImagePath URIs that are zebras; the GROUP BY + argmax
runs client-side via pandas after merging dase confident-pos rows.
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
    build_profile, write_profile, print_summary,
)

# ─── Paths / scenario constants ──────────────────────────────────────────
WILDLIFE_DIR = os.path.abspath(os.path.join(_HERE, ".."))
IMAGE_CSV    = os.path.join(WILDLIFE_DIR, "cache", "image_data.csv")
EMB_PATH     = os.path.join(WILDLIFE_DIR, "data", "image_embeddings.npz")
PROFILE_PATH = os.path.join(WILDLIFE_DIR, "outputs", "Q10.json")

PROJECT      = os.environ.get("GCP_PROJECT", "")
BUCKET       = f"{PROJECT}-animals_dataset"
DATASET      = "animals_dataset"
STAGING      = f"{DATASET}.q10_uncertain_mm"

PROMPT = "Does this image contain a zebra?"

ZEBRA_POSITIVE = [
    "a photograph of a zebra",
    "a wildlife camera trap image showing a zebra",
    "an animal with black and white stripes, a zebra",
]
ZEBRA_NEGATIVE = [
    "a photograph that does not contain a zebra",
    "a wildlife camera trap image of an animal that is not a zebra",
    "an animal scene with no zebra in it",
]

ALPHA = 0.5  # argmax over GROUP BY: misclassifying one row can flip winner; protect quality
PAPER_BQ_Q10 = {"score": 1.00, "latency_s": 87.8, "cost_usd": 0.11}
PAPER_DASE_NN_Q10 = {"score": 1.00, "latency_s": 8e-4, "cost_usd": 1e-9}
SKIP_BASELINE = False


def make_zebra_verifier():
    """Verifier returns the subset of uncertain ImagePath URIs that are zebras.
    Client-side GROUP BY (City, StationID) joins the result with the input dataframe."""
    def make_staging(uris):
        items = ",".join(f"'{u}'" for u in uris)
        return f"""
        CREATE OR REPLACE TABLE {STAGING} AS
        SELECT m.ImagePath AS uri, m.Species, m.City, m.StationID, ot.ref AS image
        FROM {DATASET}.image_data_images m
        JOIN {DATASET}.image_data_external ot ON ot.uri = m.ImagePath
        WHERE m.ImagePath IN UNNEST([{items}])
        """

    verify_sql = f"""
    SELECT uri AS id FROM {STAGING}
    WHERE AI.IF(('{PROMPT}', image),
                connection_id => 'us.connection',
                endpoint => 'gemini-2.5-flash')
    """
    return AiIfVerifier(
        verify_sql=verify_sql, make_staging_sql=make_staging,
        id_column="id", coerce_id=str,
    )


def run_baseline(client):
    sql = f"""
    SELECT City AS city, StationID AS stationID
    FROM {DATASET}.image_data_mm
    WHERE AI.IF(('{PROMPT}', image), connection_id => 'us.connection', endpoint => 'gemini-2.5-flash')
    GROUP BY City, StationID
    ORDER BY COUNT(*) DESC
    LIMIT 1
    """
    return run_query(client, sql)


def argmax_with_ties(zebra_df):
    if zebra_df.empty:
        return set()
    counts = zebra_df.groupby(["City", "StationID"]).size()
    mx = counts.max()
    return set(counts[counts == mx].index.tolist())


def f1_argmax(pred_set, gt_set):
    if not gt_set:
        return 1.0 if not pred_set else 0.0
    if not pred_set:
        return 0.0
    return len(pred_set & gt_set) / len(pred_set)


def main():
    profile = build_profile(
        scenario="wildlife", query_id=10, scale_factor=200,
        prompt=PROMPT, params={"alpha": ALPHA},
        cascade_form=("F+L cascade: Cascade(MarginSignal, AlphaBand, AiIfVerifier) + "
                      "client-side GROUP BY (City, StationID) argmax."),
        extra={"dase_prompts": {"positive": ZEBRA_POSITIVE, "negative": ZEBRA_NEGATIVE}},
    )

    print("Loading image data and embeddings...")
    df = pd.read_csv(IMAGE_CSV)
    emb = np.load(EMB_PATH)["caption_emb"]
    df["GcsUri"] = df["ImagePath"].apply(lambda p: f"gs://{BUCKET}/animal_images/{os.path.basename(p)}")
    n = len(df)

    gt_z = df[df["Species"].str.contains("ZEBRA")]
    gt_counts = gt_z.groupby(["City", "StationID"]).size()
    gt_max = gt_counts.max() if not gt_counts.empty else 0
    gt_tied = set(gt_counts[gt_counts == gt_max].index.tolist()) if gt_max else set()
    print(f"  {n} images, gt zebra rows: {len(gt_z)}")
    print(f"  GT (City,StationID) tied at max count={gt_max}: {sorted(gt_tied)}")
    profile["data"] = {"n_images": n, "n_gt_zebra_rows": int(len(gt_z)),
                       "gt_max_count": int(gt_max), "gt_tied_top": sorted([list(t) for t in gt_tied])}

    client = bq_client(PROJECT)

    print("\n=== Per-row cost calibration ===")
    cal = per_row_cost(client, PROMPT,
        sample_uris=df["GcsUri"].head(5).tolist(),
        ext_table=f"{DATASET}.image_data_external")
    per_row = cal.per_row_cost_usd
    print(f"  per_row=${per_row:.6f}")
    profile["calibration"] = cal.to_dict()

    cascade = Cascade(
        embeddings=emb,
        ids=df["GcsUri"].tolist(),  # gs:// — matches m.ImagePath in BQ image_data_images
        signal=MarginSignal(positive_prompts=ZEBRA_POSITIVE, negative_prompts=ZEBRA_NEGATIVE),
        band=AlphaBand(alpha=ALPHA),
        verifier=make_zebra_verifier(),
    )
    print("\n=== Cascade (Signal → Band → Verifier) ===")
    cres = cascade.run(client, per_row)

    n_uncertain = len(cres.uncertain_ids)
    path_to_keys = dict(zip(df["ImagePath"], zip(df["City"], df["StationID"])))
    dase_pos_rows = [path_to_keys[uri] for uri in cres.confident_pos_ids]
    bq_pos_rows = [path_to_keys[uri] for uri in cres.verifier_result.positive_ids]
    cascade_zebra_df = pd.DataFrame(dase_pos_rows + bq_pos_rows, columns=["City", "StationID"])
    print(f"  alpha={ALPHA}, uncertain={n_uncertain}")
    print(f"  dase confident_pos rows: {len(dase_pos_rows)}, bq verified rows: {len(bq_pos_rows)}")
    profile["dase_partition"] = {"n_uncertain": n_uncertain,
                                 "n_dase_confident_pos": len(dase_pos_rows)}

    if SKIP_BASELINE:
        b_score = PAPER_BQ_Q10["score"]; bwall = PAPER_BQ_Q10["latency_s"]; bslot = None
        bcost = PAPER_BQ_Q10["cost_usd"]; bcalls_total = round(bcost / per_row); b_pred = None
        profile["baseline"] = {"_status": "aborted", "score": {"score": b_score, "_source": "paper"},
                                "latency_breakdown": {"wall_s": bwall, "_source": "paper"},
                                "cost_breakdown": {"n_llm_calls": bcalls_total, "total_cost_usd": bcost, "_source": "paper"}}
    else:
        print(f"\n=== Baseline (sembench Q10.sql verbatim) ===")
        bdf, bwall, bslot, bsql = run_baseline(client)
        b_pred = set([(r["city"], r["stationID"]) for _, r in bdf.iterrows()])
        bcalls_total = n
        bcost = per_row * n
        b_score = f1_argmax(b_pred, gt_tied)
        print(f"  baseline pred: {sorted(b_pred)} (GT-tied: {sorted(gt_tied)})  F1={b_score:.4f}")
        profile["baseline"] = {
            "method": "sembench bigquery/Q10.sql verbatim",
            "sql": bsql, "result": sorted([list(t) for t in b_pred]),
            "score": {"f1": b_score},
            "latency_breakdown": {"wall_s": bwall, "slot_ms": bslot},
            "cost_breakdown": {"n_llm_calls": bcalls_total, "per_row_cost_usd": per_row, "total_cost_usd": bcost},
        }

    cascade_pred = argmax_with_ties(cascade_zebra_df)
    cscore = f1_argmax(cascade_pred, gt_tied)
    print(f"  cascade pred: {sorted(cascade_pred)} (GT-tied: {sorted(gt_tied)})  F1={cscore:.4f}")

    cascade_total_wall = cres.verifier_result.ctas_wall_s + cres.verifier_result.wall_s
    cascade_total_slot = cres.verifier_result.ctas_slot_ms + cres.verifier_result.slot_ms
    profile["cascade"] = {
        "method": "F+L cascade (Cascade primitive) + client-side GROUP BY (City, StationID) argmax",
        "verifier": cres.verifier_result.to_dict(),
        "n_dase_pos_rows": len(dase_pos_rows),
        "n_bq_verified_rows": len(bq_pos_rows),
        "cascade_pred_top": sorted([list(t) for t in cascade_pred]),
        "score": {"f1": cscore},
        "totals": {"wall_s": cascade_total_wall, "slot_ms_bq_total": cascade_total_slot,
                   "cost_usd": cres.verifier_result.cost_usd, "n_llm_calls": cres.verifier_result.n_calls},
    }

    profile["comparison"] = {
        "score": {"paper_BQ": PAPER_BQ_Q10["score"], "paper_DASE_NN": PAPER_DASE_NN_Q10["score"], "ours_BQ": b_score, "ours_cascade": cscore},
        "wall_s": {"paper_BQ": PAPER_BQ_Q10["latency_s"], "paper_DASE_NN": PAPER_DASE_NN_Q10["latency_s"], "ours_BQ": bwall, "ours_cascade": cascade_total_wall},
        "cost_usd": {"paper_BQ": PAPER_BQ_Q10["cost_usd"], "paper_DASE_NN": PAPER_DASE_NN_Q10["cost_usd"], "ours_BQ": bcost, "ours_cascade": cres.verifier_result.cost_usd},
        "n_llm_calls": {"paper_BQ": round(PAPER_BQ_Q10["cost_usd"] / per_row), "paper_DASE_NN": 0, "ours_BQ": bcalls_total, "ours_cascade": cres.verifier_result.n_calls},
    }

    write_profile(profile, PROFILE_PATH)

    print_summary(
        f"Wildlife Q10 (alpha={ALPHA})",
        columns=["paper BQ", "DASE+NN", "ours BQ", "ours cascade"],
        rows=[
            ("score (F1)", [PAPER_BQ_Q10["score"], PAPER_DASE_NN_Q10["score"], b_score, cscore], ".2f"),
            ("wall (s)",   [PAPER_BQ_Q10["latency_s"], PAPER_DASE_NN_Q10["latency_s"], bwall, cascade_total_wall], ".2f"),
            ("cost ($)",   [PAPER_BQ_Q10["cost_usd"], PAPER_DASE_NN_Q10["cost_usd"], bcost, cres.verifier_result.cost_usd], ".4f"),
            ("#LLM calls", [round(PAPER_BQ_Q10["cost_usd"] / per_row), 0, bcalls_total, cres.verifier_result.n_calls], "d"),
        ],
    )


if __name__ == "__main__":
    main()
