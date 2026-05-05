#!/usr/bin/env -S python -u
"""
Wildlife Q1 cascade — count zebras (single-modal F + COUNT aggregation).

NL: How many pictures of zebras do we have in our database?
GT: Species LIKE '%ZEBRA%' → 11 images.
Eval: relative_error_score on COUNT.

Refactored to use dase_cascade unified solver. Operator (paper Table 3): F.
The Cascade primitive (Signal+Band+Verifier) drives the prefilter+BQ stage;
the COUNT aggregation is client-side (just len of the positive set).
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
    relative_error_score, build_profile, write_profile, print_summary,
)

# ─── Paths / scenario constants ──────────────────────────────────────────
WILDLIFE_DIR = os.path.abspath(os.path.join(_HERE, ".."))
IMAGE_CSV    = os.path.join(WILDLIFE_DIR, "cache", "image_data.csv")
EMB_PATH     = os.path.join(WILDLIFE_DIR, "data", "image_embeddings.npz")
PROFILE_PATH = os.path.join(WILDLIFE_DIR, "outputs", "Q1.json")

PROJECT      = os.environ.get("GCP_PROJECT", "")
BUCKET       = f"{PROJECT}-animals_dataset"
DATASET      = "animals_dataset"
STAGING      = f"{DATASET}.q1_uncertain_mm"

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
PAPER_BQ_Q1 = {"score": 0.79, "latency_s": 32.0, "cost_usd": 0.11}
SKIP_BASELINE = False


def make_zebra_verifier():
    """Build the BQ verifier for Q1.

    Stage 1 (CTAS): join uncertain ImagePaths against image_data_external to
                    get the multimodal `image` ref column.
    Stage 2 (AI.IF): SELECT GcsUri FROM staging WHERE AI.IF — returns the
                    subset of uncertain URIs that BQ confirmed contain a zebra.
                    (We return URIs rather than COUNT so the verifier slots
                    into the standard Cascade abstraction; the caller takes
                    `len(positive_set)` for the cascade COUNT.)
    """
    def make_staging(uris):
        items = ",".join(f"'{u}'" for u in uris)
        return f"""
        CREATE OR REPLACE TABLE {STAGING} AS
        SELECT m.ImagePath AS uri, ot.ref AS image
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
    """Verbatim sembench Q1.sql on the full image_data_mm — returns ZEBRA count."""
    sql = f"""
    SELECT COUNT(*) AS count
    FROM {DATASET}.image_data_mm
    WHERE AI.IF(('{PROMPT}', image),
                connection_id => 'us.connection',
                endpoint => 'gemini-2.5-flash')
    """
    return run_query(client, sql)


def main():
    profile = build_profile(
        scenario="wildlife", query_id=1, scale_factor=200,
        prompt=PROMPT, params={"alpha": ALPHA},
        cascade_form="F-cascade: MarginSignal + AlphaBand + AiIfVerifier; client COUNT.",
        extra={"dase_prompts": {"positive": POSITIVE, "negative": NEGATIVE}},
    )

    print("Loading image data + caption embeddings...")
    df = pd.read_csv(IMAGE_CSV)
    image_emb = np.load(EMB_PATH)["caption_emb"]
    assert len(df) == image_emb.shape[0]
    df["GcsUri"] = df["ImagePath"].apply(
        lambda p: f"gs://{BUCKET}/animal_images/{os.path.basename(p)}")
    n_total = len(df)
    n_gt = int(df["Species"].str.contains("ZEBRA").sum())
    print(f"  {n_total} images, GT zebra count = {n_gt}")
    profile["data"] = {"n_images": n_total, "n_gt_zebra": n_gt}

    client = bq_client(PROJECT)

    # ── Per-row cost calibration ──
    print("\n=== Per-row cost calibration ===")
    cal = per_row_cost(
        client, PROMPT,
        sample_uris=df["GcsUri"].head(5).tolist(),
        ext_table=f"{DATASET}.image_data_external",
    )
    per_row = cal.per_row_cost_usd
    print(f"  per_row=${per_row:.6f}, sample_cost=${cal.sample_cost_usd:.6f}, elapsed={cal.elapsed_s:.1f}s")
    profile["calibration"] = cal.to_dict()

    # ── Cascade: Signal+Band on embeddings, Verifier on uncertain URIs ──
    cascade = Cascade(
        embeddings=image_emb,
        ids=df["GcsUri"].tolist(),  # gs:// — matches m.ImagePath in BQ image_data_images
        signal=MarginSignal(positive_prompts=POSITIVE, negative_prompts=NEGATIVE),
        band=AlphaBand(alpha=ALPHA),
        verifier=make_zebra_verifier(),
    )
    print("\n=== Cascade (Signal → Band → Verifier) ===")
    cres = cascade.run(client, per_row)

    # Split confident_pos/uncertain — these are URIs (== ImagePath); we already
    # know n_confident_pos, and BQ returned the subset of uncertain that's positive.
    n_confident_pos = len(cres.confident_pos_ids)
    bq_pos = cres.verifier_result.positive_ids   # subset of uncertain URIs
    n_uncertain = len(cres.uncertain_ids)
    cascade_count = n_confident_pos + len(bq_pos)
    cscore = relative_error_score(cascade_count, n_gt)
    cascade_total_wall = cres.total_wall_s
    cascade_total_slot = cres.verifier_result.ctas_slot_ms + cres.verifier_result.slot_ms
    print(f"  alpha={ALPHA}, uncertain={n_uncertain}, confident_pos={n_confident_pos}, "
          f"bq_yes_on_uncertain={len(bq_pos)}")
    print(f"  cascade_count={cascade_count} (GT={n_gt})  score={cscore:.4f}")
    print(f"  wall={cascade_total_wall:.2f}s  slot={cascade_total_slot}  "
          f"calls={cres.verifier_result.n_calls}  cost=${cres.verifier_result.cost_usd:.6f}")

    profile["dase_partition"] = cres.partition.to_dict() | {
        "n_confident_pos": n_confident_pos,
    }

    # ── Baseline (verbatim sembench Q1.sql) ──
    if SKIP_BASELINE:
        b_score, bwall, bslot = PAPER_BQ_Q1["score"], PAPER_BQ_Q1["latency_s"], None
        bcost, bcount = PAPER_BQ_Q1["cost_usd"], None
        bcalls = round(bcost / per_row)
        profile["baseline"] = {"_status": "aborted",
                               "score": {"score": b_score, "_source": "paper"},
                               "latency_breakdown": {"wall_s": bwall, "_source": "paper"},
                               "cost_breakdown": {"n_llm_calls": bcalls, "total_cost_usd": bcost, "_source": "paper"}}
    else:
        print(f"\n=== Baseline (sembench Q1.sql verbatim) ===")
        bdf, bwall, bslot, bsql = run_baseline(client)
        bcount = int(bdf.iloc[0]["count"])
        bcalls = n_total
        bcost = per_row * bcalls
        b_score = relative_error_score(bcount, n_gt)
        print(f"  count={bcount} (GT={n_gt})")
        print(f"  wall={bwall:.2f}s  slot={bslot}  calls={bcalls}  cost=${bcost:.6f}  score={b_score:.4f}")
        profile["baseline"] = {
            "method": "sembench bigquery/Q1.sql verbatim on image_data_mm", "sql": bsql,
            "result_count": bcount, "score": {"score": b_score},
            "latency_breakdown": {"wall_s": bwall, "slot_ms": bslot},
            "cost_breakdown": {"n_llm_calls": bcalls, "per_row_cost_usd": per_row, "total_cost_usd": bcost},
        }

    profile["cascade"] = {
        "method": "F-cascade with COUNT aggregation: Cascade(MarginSignal, AlphaBand, AiIfVerifier).run()",
        "verifier": cres.verifier_result.to_dict(),
        "cascade_count": cascade_count,
        "cascade_count_breakdown": {"dase_confident_pos": n_confident_pos, "bq_uncertain_pos": len(bq_pos)},
        "score": {"score": cscore},
        "totals": {"wall_s": cascade_total_wall, "slot_ms_bq_total": cascade_total_slot,
                   "cost_usd": cres.verifier_result.cost_usd,
                   "n_llm_calls": cres.verifier_result.n_calls},
    }

    profile["comparison"] = {
        "score":       {"paper": PAPER_BQ_Q1["score"],   "baseline": b_score, "cascade": cscore},
        "wall_s":      {"paper": PAPER_BQ_Q1["latency_s"], "baseline": bwall,  "cascade_total": cascade_total_wall},
        "cost_usd":    {"paper": PAPER_BQ_Q1["cost_usd"], "baseline": bcost,  "cascade": cres.verifier_result.cost_usd},
        "n_llm_calls": {"baseline": bcalls, "cascade": cres.verifier_result.n_calls},
    }

    write_profile(profile, PROFILE_PATH)

    print_summary(
        f"Wildlife Q1 (alpha={ALPHA})",
        columns=["paper Tbl4c", "baseline", "cascade"],
        rows=[
            ("score",      [PAPER_BQ_Q1["score"], b_score, cscore], ".2f"),
            ("count",      [None, bcount, cascade_count]),
            ("wall (s)",   [PAPER_BQ_Q1["latency_s"], bwall, cascade_total_wall], ".2f"),
            ("cost ($)",   [PAPER_BQ_Q1["cost_usd"], bcost, cres.verifier_result.cost_usd], ".4f"),
            ("#LLM calls", [None, bcalls, cres.verifier_result.n_calls], "d"),
        ],
    )


if __name__ == "__main__":
    main()
