#!/usr/bin/env -S python -u
"""
Wildlife Q3 cascade — sem_filter (zebra) + GROUP BY City + argmax (F+L).

NL: City where we captured most pictures of zebras.
GT: argmax_city COUNT(Species LIKE '%ZEBRA%'). Tie-set possible.
Eval: cascade output city correct iff in GT-tied set.

Refactored to use dase_cascade unified solver. Operator (paper Table 3): F+L.
The Cascade primitive (Signal+Band+Verifier) drives the prefilter+BQ stage on
ImagePath → returns the subset of URIs BQ confirmed as zebra. The GROUP BY
City + argmax happens client-side after merging dase confident-pos rows.

Cost / call accounting matches the original: BQ stage processes n_uncertain
URIs at one AI.IF call each → s2_calls = n_uncertain, same as original Q3.
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
PROFILE_PATH = os.path.join(WILDLIFE_DIR, "outputs", "Q3.json")

PROJECT      = os.environ.get("GCP_PROJECT", "")
BUCKET       = f"{PROJECT}-animals_dataset"
DATASET      = "animals_dataset"
STAGING      = f"{DATASET}.q3_uncertain_mm"

PROMPT = "Does this image contain a zebra?"

POSITIVE = [
    "a photograph of a zebra",
    "a wildlife camera trap image showing a zebra",
    "an animal with black and white stripes, a zebra",
]
NEGATIVE = [
    "a photograph that does not contain a zebra",
    "a wildlife camera trap image of an animal that is not a zebra",
    "an animal scene with no zebra in it",
]

ALPHA = 0.2
PAPER_BQ_Q3 = {"score": 0.00, "latency_s": 25.7, "cost_usd": 0.11}
PAPER_DASE_NN_Q3 = {"score": 0.00, "latency_s": 8e-4, "cost_usd": 1e-9}
SKIP_BASELINE = False


def make_zebra_verifier():
    """Verifier returns the subset of uncertain ImagePath URIs that are zebras.
    Client-side GROUP BY City joins the result with the input dataframe."""
    def make_staging(uris):
        items = ",".join(f"'{u}'" for u in uris)
        return f"""
        CREATE OR REPLACE TABLE {STAGING} AS
        SELECT m.ImagePath AS uri, m.City, m.StationID, ot.ref AS image
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
    """Verbatim sembench Q3.sql on full image_data_mm — argmax city."""
    sql = f"""
    SELECT City AS city
    FROM {DATASET}.image_data_mm
    WHERE AI.IF(('{PROMPT}', image),
                connection_id => 'us.connection',
                endpoint => 'gemini-2.5-flash')
    GROUP BY City
    ORDER BY COUNT(*) DESC
    LIMIT 1
    """
    return run_query(client, sql)


def main():
    profile = build_profile(
        scenario="wildlife", query_id=3, scale_factor=200,
        prompt=PROMPT, params={"alpha": ALPHA},
        cascade_form="F+L cascade: Cascade(MarginSignal, AlphaBand, AiIfVerifier) + client-side GROUP BY City argmax.",
        extra={"dase_prompts": {"positive": POSITIVE, "negative": NEGATIVE}},
    )

    print("Loading image data + caption embeddings...")
    df = pd.read_csv(IMAGE_CSV)
    image_emb = np.load(EMB_PATH)["caption_emb"]
    df["GcsUri"] = df["ImagePath"].apply(
        lambda p: f"gs://{BUCKET}/animal_images/{os.path.basename(p)}")
    n_total = len(df)

    zebra_df = df[df["Species"].str.contains("ZEBRA")]
    city_counts = zebra_df.groupby("City").size().to_dict()
    max_zebra = max(city_counts.values()) if city_counts else 0
    gt_cities = sorted([c for c, n in city_counts.items() if n == max_zebra])
    print(f"  {n_total} images, zebra-by-city: {city_counts}, GT cities (tied): {gt_cities}")
    profile["data"] = {"n_images": n_total, "zebra_count_by_city": city_counts, "gt_cities": gt_cities}

    client = bq_client(PROJECT)

    # ── Per-row cost calibration ──
    print("\n=== Per-row cost calibration ===")
    cal = per_row_cost(
        client, PROMPT,
        sample_uris=df["GcsUri"].head(5).tolist(),
        ext_table=f"{DATASET}.image_data_external",
    )
    per_row = cal.per_row_cost_usd
    print(f"  per_row=${per_row:.6f}")
    profile["calibration"] = cal.to_dict()

    # ── Cascade: dase prefilter on ImagePath, BQ verifier on uncertain ──
    cascade = Cascade(
        embeddings=image_emb,
        ids=df["GcsUri"].tolist(),  # gs:// — matches m.ImagePath in BQ image_data_images
        signal=MarginSignal(positive_prompts=POSITIVE, negative_prompts=NEGATIVE),
        band=AlphaBand(alpha=ALPHA),
        verifier=make_zebra_verifier(),
    )
    print("\n=== Cascade (Signal → Band → Verifier) ===")
    cres = cascade.run(client, per_row)

    # Map ImagePath → City via the input df
    path_to_city = dict(zip(df["ImagePath"], df["City"]))

    # dase confident_pos by city
    dase_confident_pos_by_city = {}
    for uri in cres.confident_pos_ids:
        c = path_to_city[uri]
        dase_confident_pos_by_city[c] = dase_confident_pos_by_city.get(c, 0) + 1

    # bq verified by city (subset of uncertain ImagePaths)
    bq_pos_by_city = {}
    for uri in cres.verifier_result.positive_ids:
        c = path_to_city[uri]
        bq_pos_by_city[c] = bq_pos_by_city.get(c, 0) + 1

    n_uncertain = len(cres.uncertain_ids)
    print(f"  alpha={ALPHA}, n_uncertain={n_uncertain}")
    print(f"  dase confident_pos by city: {dase_confident_pos_by_city}")
    print(f"  bq_pos_by_city on uncertain: {bq_pos_by_city}")
    profile["dase_partition"] = {"n_uncertain": n_uncertain,
                                 "dase_confident_pos_by_city": dase_confident_pos_by_city}

    # ── Baseline (verbatim sembench Q3.sql) ──
    if SKIP_BASELINE:
        bcost = PAPER_BQ_Q3["cost_usd"]; bwall = PAPER_BQ_Q3["latency_s"]; bslot = None
        bcity = None; b_score = PAPER_BQ_Q3["score"]; bcalls = round(bcost / per_row)
        profile["baseline"] = {"_status": "aborted",
                                "score": {"score": b_score, "_source": "paper"},
                                "latency_breakdown": {"wall_s": bwall, "_source": "paper"},
                                "cost_breakdown": {"n_llm_calls": bcalls, "per_row_cost_usd": per_row, "total_cost_usd": bcost, "_source": "paper"}}
    else:
        print(f"\n=== Baseline (sembench Q3.sql verbatim) ===")
        bdf, bwall, bslot, bsql = run_baseline(client)
        bcity = bdf.iloc[0]["city"] if len(bdf) > 0 else None
        bcalls = n_total
        bcost = per_row * bcalls
        b_score = 1.0 if bcity in gt_cities else 0.0
        print(f"  baseline city: {bcity}  (GT cities: {gt_cities})  score={b_score:.4f}")
        profile["baseline"] = {
            "method": "sembench bigquery/Q3.sql verbatim on image_data_mm",
            "sql": bsql, "result_city": str(bcity),
            "score": {"score": b_score, "in_gt": bcity in gt_cities},
            "latency_breakdown": {"wall_s": bwall, "slot_ms": bslot},
            "cost_breakdown": {"n_llm_calls": bcalls, "per_row_cost_usd": per_row, "total_cost_usd": bcost},
        }

    # ── Combine: total counts across dase confident + BQ uncertain ──
    total_by_city = dict(dase_confident_pos_by_city)
    for city, n in bq_pos_by_city.items():
        total_by_city[city] = total_by_city.get(city, 0) + n
    if total_by_city:
        max_count = max(total_by_city.values())
        cascade_city = sorted([c for c, v in total_by_city.items() if v == max_count])[0]
    else:
        cascade_city = None
    cscore = 1.0 if cascade_city in gt_cities else 0.0
    print(f"  total_by_city: {total_by_city}")
    print(f"  cascade output city: {cascade_city}  score={cscore:.4f}")

    # Cascade wall: omit dase (it's <1s); match original's stage1+stage2 measure.
    cascade_total_wall = (cres.verifier_result.ctas_wall_s
                          + cres.verifier_result.wall_s)
    cascade_total_slot = cres.verifier_result.ctas_slot_ms + cres.verifier_result.slot_ms
    profile["cascade"] = {
        "method": "F+L cascade (Cascade primitive) + client-side GROUP BY argmax",
        "verifier": cres.verifier_result.to_dict(),
        "bq_pos_by_city": bq_pos_by_city,
        "total_count_by_city": total_by_city,
        "cascade_city": cascade_city,
        "score": {"score": cscore, "in_gt": cascade_city in gt_cities},
        "totals": {"wall_s": cascade_total_wall, "slot_ms_bq_total": cascade_total_slot,
                   "cost_usd": cres.verifier_result.cost_usd, "n_llm_calls": cres.verifier_result.n_calls},
    }

    profile["comparison"] = {
        "score": {"paper_BQ": PAPER_BQ_Q3["score"], "paper_DASE_NN": PAPER_DASE_NN_Q3["score"], "ours_BQ": b_score, "ours_cascade": cscore},
        "wall_s": {"paper_BQ": PAPER_BQ_Q3["latency_s"], "paper_DASE_NN": PAPER_DASE_NN_Q3["latency_s"], "ours_BQ": bwall, "ours_cascade": cascade_total_wall},
        "cost_usd": {"paper_BQ": PAPER_BQ_Q3["cost_usd"], "paper_DASE_NN": PAPER_DASE_NN_Q3["cost_usd"], "ours_BQ": bcost, "ours_cascade": cres.verifier_result.cost_usd},
        "n_llm_calls": {"paper_BQ": round(PAPER_BQ_Q3["cost_usd"] / per_row), "paper_DASE_NN": 0, "ours_BQ": bcalls, "ours_cascade": cres.verifier_result.n_calls},
    }

    write_profile(profile, PROFILE_PATH)

    print_summary(
        f"Wildlife Q3 (alpha={ALPHA})",
        columns=["paper BQ", "DASE+NN", "ours BQ", "ours cascade"],
        rows=[
            ("score",     [PAPER_BQ_Q3["score"], PAPER_DASE_NN_Q3["score"], b_score, cscore], ".2f"),
            ("city",      [None, None, str(bcity) if bcity is not None else None, str(cascade_city) if cascade_city is not None else None]),
            ("wall (s)",  [PAPER_BQ_Q3["latency_s"], PAPER_DASE_NN_Q3["latency_s"], bwall, cascade_total_wall], ".2f"),
            ("cost ($)",  [PAPER_BQ_Q3["cost_usd"], PAPER_DASE_NN_Q3["cost_usd"], bcost, cres.verifier_result.cost_usd], ".4f"),
            ("#LLM calls",[round(PAPER_BQ_Q3["cost_usd"] / per_row), 0, bcalls, cres.verifier_result.n_calls], "d"),
        ],
    )


if __name__ == "__main__":
    main()
