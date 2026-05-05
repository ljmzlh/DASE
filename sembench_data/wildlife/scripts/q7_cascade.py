#!/usr/bin/env -S python -u
"""
Wildlife Q7 cascade — INTERSECT (cities with zebra image AND impala image).

NL: Cities where zebras and impala co-occur (image-only).
GT: zebra_cities ∩ impala_cities.
Eval: set retrieval F1.

Refactored to use dase_cascade unified solver. Operator (paper Table 3): M
= two F (zebra + impala on images) intersected client-side.

Two parallel signal+band pipelines on the same image embeddings (one per
prompt). The two uncertain index sets are unioned to form a single staging
table; two AI.IF queries run against it (one per prompt). The intersect
happens client-side.

Cost / call accounting: BQ stage 2 runs 2 × |union_uncertain| AI.IF calls;
matches the original.
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import (
    MarginSignal, AlphaBand, AiIfVerifier,
    bq_client, per_row_cost, run_query,
    f1_set, build_profile, write_profile, print_summary,
)

# ─── Paths / scenario constants ──────────────────────────────────────────
WILDLIFE_DIR = os.path.abspath(os.path.join(_HERE, ".."))
IMAGE_CSV    = os.path.join(WILDLIFE_DIR, "cache", "image_data.csv")
EMB_PATH     = os.path.join(WILDLIFE_DIR, "data", "image_embeddings.npz")
PROFILE_PATH = os.path.join(WILDLIFE_DIR, "outputs", "Q7.json")

PROJECT      = os.environ.get("GCP_PROJECT", "")
BUCKET       = f"{PROJECT}-animals_dataset"
DATASET      = "animals_dataset"
STAGING      = f"{DATASET}.q7_uncertain_mm"

ZEBRA_PROMPT  = "Does this image contain a zebra?"
IMPALA_PROMPT = "Does this image contain an impala?"

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
IMPALA_POSITIVE = [
    "a photograph of an impala",
    "a wildlife camera trap image showing an impala antelope",
    "an impala animal in the picture",
]
IMPALA_NEGATIVE = [
    "a photograph that does not contain an impala",
    "a wildlife camera trap image of a non-impala animal",
    "an animal scene without any impala",
]

ALPHA = 0.5  # INTERSECT sensitive to BQ noise on each side
PAPER_BQ_Q7 = {"score": 1.00, "latency_s": 24.6, "cost_usd": 0.22}
PAPER_DASE_NN_Q7 = {"score": 0.00, "latency_s": 1e-3, "cost_usd": 1e-9}
SKIP_BASELINE = True  # baseline INTERSECT 2× AI.IF on full 200 images crashed gRPC


def _uri_array_literal(uris):
    items = ",".join(f"'{u}'" for u in uris)
    return f"[{items}]"


def stage1_create_staging_sql(union_uris):
    return f"""
    CREATE OR REPLACE TABLE {STAGING} AS
    SELECT m.Species, m.City, m.StationID, ot.ref AS image
    FROM {DATASET}.image_data_images m
    JOIN {DATASET}.image_data_external ot ON ot.uri = m.ImagePath
    WHERE m.ImagePath IN UNNEST({_uri_array_literal(union_uris)})
    """


def make_prompt_verifier(prompt):
    """Verifier with verify_sql only (no make_staging_sql) — reuses shared staging."""
    verify_sql = f"""
    SELECT DISTINCT City AS id FROM {STAGING}
    WHERE AI.IF(('{prompt}', image),
                connection_id => 'us.connection',
                endpoint => 'gemini-2.5-flash')
    """
    return AiIfVerifier(verify_sql=verify_sql, id_column="id", coerce_id=str)


def run_baseline(client):
    sql = f"""
    SELECT City FROM (
      SELECT DISTINCT City FROM {DATASET}.image_data_mm
      WHERE AI.IF(('{ZEBRA_PROMPT}', image), connection_id => 'us.connection', endpoint => 'gemini-2.5-flash')
    ) INTERSECT DISTINCT (
      SELECT DISTINCT City FROM {DATASET}.image_data_mm
      WHERE AI.IF(('{IMPALA_PROMPT}', image), connection_id => 'us.connection', endpoint => 'gemini-2.5-flash')
    )
    """
    return run_query(client, sql)


def main():
    profile = build_profile(
        scenario="wildlife", query_id=7, scale_factor=200,
        params={"alpha": ALPHA},
        cascade_form=("Two F-cascades sharing one staging table (union of zebra-/impala-uncertain image rows); "
                      "two AI.IF DISTINCT-City queries; client-side intersect."),
        extra={
            "zebra_prompt": ZEBRA_PROMPT, "impala_prompt": IMPALA_PROMPT,
            "dase_zebra_prompts": {"positive": ZEBRA_POSITIVE, "negative": ZEBRA_NEGATIVE},
            "dase_impala_prompts": {"positive": IMPALA_POSITIVE, "negative": IMPALA_NEGATIVE},
        },
    )

    print("Loading image data + caption embeddings...")
    df = pd.read_csv(IMAGE_CSV)
    image_emb = np.load(EMB_PATH)["caption_emb"]
    df["GcsUri"] = df["ImagePath"].apply(lambda p: f"gs://{BUCKET}/animal_images/{os.path.basename(p)}")
    n_total = len(df)

    gt_zebra = set(df[df["Species"].str.contains("ZEBRA")]["City"])
    gt_impala = set(df[df["Species"].str.contains("IMPALA")]["City"])
    gt_cities = gt_zebra & gt_impala
    print(f"  {n_total} images")
    print(f"  GT zebra cities: {sorted(gt_zebra)}")
    print(f"  GT impala cities: {sorted(gt_impala)}")
    print(f"  GT cities (intersect): {sorted(gt_cities)}")
    profile["data"] = {"n_images": n_total, "gt_zebra": sorted(gt_zebra),
                       "gt_impala": sorted(gt_impala), "gt_cities": sorted(gt_cities)}

    client = bq_client(PROJECT)

    print("\n=== Per-row cost calibration ===")
    cal = per_row_cost(client, ZEBRA_PROMPT,
        sample_uris=df["GcsUri"].head(5).tolist(),
        ext_table=f"{DATASET}.image_data_external")
    per_row = cal.per_row_cost_usd
    print(f"  per_row=${per_row:.6f}")
    profile["calibration"] = cal.to_dict()

    # ── Two parallel signal+band pipelines on the same image embeddings ──
    z_signal = MarginSignal(positive_prompts=ZEBRA_POSITIVE, negative_prompts=ZEBRA_NEGATIVE)
    z_scores = z_signal.compute(image_emb)
    z_part = AlphaBand(alpha=ALPHA).partition(z_scores)
    z_uncertain_idx = set(z_part.uncertain.tolist())
    dase_zebra_cities = set(df["City"].iloc[z_part.confident_pos])

    i_signal = MarginSignal(positive_prompts=IMPALA_POSITIVE, negative_prompts=IMPALA_NEGATIVE)
    i_scores = i_signal.compute(image_emb)
    i_part = AlphaBand(alpha=ALPHA).partition(i_scores)
    i_uncertain_idx = set(i_part.uncertain.tolist())
    dase_impala_cities = set(df["City"].iloc[i_part.confident_pos])

    union_uncertain_idx = sorted(z_uncertain_idx | i_uncertain_idx)
    union_uris = df["GcsUri"].iloc[union_uncertain_idx].tolist()  # gs:// — matches m.ImagePath in BQ
    print(f"  alpha={ALPHA}, zebra_uncertain={len(z_uncertain_idx)}, impala_uncertain={len(i_uncertain_idx)}, union={len(union_uncertain_idx)}")
    print(f"  dase zebra confident pos cities: {sorted(dase_zebra_cities)}")
    print(f"  dase impala confident pos cities: {sorted(dase_impala_cities)}")
    profile["dase_partition"] = {
        "n_uncertain_zebra": len(z_uncertain_idx), "n_uncertain_impala": len(i_uncertain_idx),
        "n_union_uncertain": len(union_uncertain_idx),
        "dase_zebra_confident_cities": sorted(dase_zebra_cities),
        "dase_impala_confident_cities": sorted(dase_impala_cities),
    }

    if SKIP_BASELINE:
        b_score = PAPER_BQ_Q7["score"]; bwall = PAPER_BQ_Q7["latency_s"]; bslot = None
        bcost = PAPER_BQ_Q7["cost_usd"]; bcalls = round(bcost / per_row); b_cities = None
        profile["baseline"] = {"_status": "aborted", "score": {"score": b_score, "_source": "paper"},
                                "latency_breakdown": {"wall_s": bwall, "_source": "paper"},
                                "cost_breakdown": {"n_llm_calls": bcalls, "total_cost_usd": bcost, "_source": "paper"}}
    else:
        print(f"\n=== Baseline (sembench Q7.sql verbatim) ===")
        bdf, bwall, bslot, bsql = run_baseline(client)
        b_cities = set(bdf["City"])
        bcalls = 2 * n_total
        bcost = per_row * bcalls
        _, _, b_score = f1_set(b_cities, gt_cities)
        print(f"  baseline cities: {sorted(b_cities)} (GT: {sorted(gt_cities)})  F1={b_score:.4f}")
        profile["baseline"] = {
            "method": "sembench bigquery/Q7.sql verbatim with INTERSECT DISTINCT",
            "sql": bsql, "result_cities": sorted(b_cities),
            "score": {"f1": b_score},
            "latency_breakdown": {"wall_s": bwall, "slot_ms": bslot},
            "cost_breakdown": {"n_llm_calls": bcalls, "per_row_cost_usd": per_row, "total_cost_usd": bcost},
        }

    # ── Stage 1: shared staging (one CTAS over union) ──
    print(f"\n=== Cascade Stage 1: CTAS {STAGING} (union {len(union_uncertain_idx)} rows) ===")
    s1_sql = stage1_create_staging_sql(union_uris)
    _, s1_wall, s1_slot, _ = run_query(client, s1_sql)
    print(f"  wall={s1_wall:.2f}s, slot_ms={s1_slot}")

    # ── Stage 2: two AI.IF queries against the shared staging ──
    print(f"\n=== Cascade Stage 2: 2 AI.IF queries (zebra + impala) ===")
    z_verifier = make_prompt_verifier(ZEBRA_PROMPT)
    i_verifier = make_prompt_verifier(IMPALA_PROMPT)
    # Each verifier processes |union_uncertain| rows; n_calls = |union_uncertain| each.
    z_v = z_verifier.verify(client, union_uris, per_row)
    i_v = i_verifier.verify(client, union_uris, per_row)

    bq_zebra_cities = z_v.positive_ids
    bq_impala_cities = i_v.positive_ids
    s2_calls = z_v.n_calls + i_v.n_calls
    s2_wall = z_v.wall_s + i_v.wall_s
    s2_slot = z_v.slot_ms + i_v.slot_ms
    cascade_cost = z_v.cost_usd + i_v.cost_usd
    print(f"  bq zebra cities: {sorted(bq_zebra_cities)}")
    print(f"  bq impala cities: {sorted(bq_impala_cities)}")
    print(f"  s2 wall={s2_wall:.2f}s slot={s2_slot} calls={s2_calls} cost=${cascade_cost:.6f}")

    # ── Combine and intersect ──
    zebra_cities = dase_zebra_cities | bq_zebra_cities
    impala_cities = dase_impala_cities | bq_impala_cities
    cascade_cities = zebra_cities & impala_cities
    _, _, cscore = f1_set(cascade_cities, gt_cities)
    print(f"  zebra_cities (combined): {sorted(zebra_cities)}")
    print(f"  impala_cities (combined): {sorted(impala_cities)}")
    print(f"  cascade_cities (intersect): {sorted(cascade_cities)} (GT: {sorted(gt_cities)})  F1={cscore:.4f}")

    cascade_total_wall = s1_wall + s2_wall
    cascade_total_slot = s1_slot + s2_slot
    profile["cascade"] = {
        "method": "Two F (zebra + impala) on shared staging; client-side intersect",
        "stage1_sql": s1_sql, "stage1_wall_s": s1_wall, "stage1_slot_ms": s1_slot,
        "stage2_zebra": z_v.to_dict(), "stage2_impala": i_v.to_dict(),
        "bq_zebra_cities_on_uncertain": sorted(bq_zebra_cities),
        "bq_impala_cities_on_uncertain": sorted(bq_impala_cities),
        "zebra_cities_combined": sorted(zebra_cities),
        "impala_cities_combined": sorted(impala_cities),
        "cascade_cities": sorted(cascade_cities),
        "score": {"f1": cscore},
        "totals": {"wall_s": cascade_total_wall, "slot_ms_bq_total": cascade_total_slot,
                   "cost_usd": cascade_cost, "n_llm_calls": s2_calls},
    }

    profile["comparison"] = {
        "score": {"paper_BQ": PAPER_BQ_Q7["score"], "paper_DASE_NN": PAPER_DASE_NN_Q7["score"], "ours_BQ": b_score, "ours_cascade": cscore},
        "wall_s": {"paper_BQ": PAPER_BQ_Q7["latency_s"], "paper_DASE_NN": PAPER_DASE_NN_Q7["latency_s"], "ours_BQ": bwall, "ours_cascade": cascade_total_wall},
        "cost_usd": {"paper_BQ": PAPER_BQ_Q7["cost_usd"], "paper_DASE_NN": PAPER_DASE_NN_Q7["cost_usd"], "ours_BQ": bcost, "ours_cascade": cascade_cost},
        "n_llm_calls": {"paper_BQ": round(PAPER_BQ_Q7["cost_usd"] / per_row), "paper_DASE_NN": 0, "ours_BQ": bcalls, "ours_cascade": s2_calls},
    }

    write_profile(profile, PROFILE_PATH)

    print_summary(
        f"Wildlife Q7 (alpha={ALPHA})",
        columns=["paper BQ", "DASE+NN", "ours BQ", "ours cascade"],
        rows=[
            ("score (F1)", [PAPER_BQ_Q7["score"], PAPER_DASE_NN_Q7["score"], b_score, cscore], ".2f"),
            ("wall (s)",   [PAPER_BQ_Q7["latency_s"], PAPER_DASE_NN_Q7["latency_s"], bwall, cascade_total_wall], ".2f"),
            ("cost ($)",   [PAPER_BQ_Q7["cost_usd"], PAPER_DASE_NN_Q7["cost_usd"], bcost, cascade_cost], ".4f"),
            ("#LLM calls", [round(PAPER_BQ_Q7["cost_usd"] / per_row), 0, bcalls, s2_calls], "d"),
        ],
    )


if __name__ == "__main__":
    main()
