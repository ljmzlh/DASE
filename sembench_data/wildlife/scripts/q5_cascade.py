#!/usr/bin/env -S python -u
"""
Wildlife Q5 cascade — UNION across image + audio modalities.

NL: List cities with elephant images OR elephant audio.
GT: SET of cities (Species LIKE '%ELEPHANT%' UNION Animal='Elephant') per city.
Eval: set retrieval F1.

Refactored to use sembench_data/dase_cascade unified solver.
Operator (paper Table 3): M = composition of two F (image + audio) with
client-side set union over City — not a primitive on its own.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

# Bootstrap sys.path so `import dase_cascade` (which lives at
# sembench_data/dase_cascade/) resolves; the package then patches sys.path
# further to expose `tools.llm_tool` at the dase_clean root.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import (
    Cascade, MarginSignal, AlphaBand, AiIfVerifier,
    bq_client, per_row_cost, run_query,
    f1_set, build_profile, write_profile, print_summary,
)

# ─── Paths / scenario constants ──────────────────────────────────────────
WILDLIFE_DIR = os.path.abspath(os.path.join(_HERE, ".."))
IMAGE_CSV    = os.path.join(WILDLIFE_DIR, "cache", "image_data.csv")
AUDIO_CSV    = os.path.join(WILDLIFE_DIR, "cache", "audio_data.csv")
IMG_EMB_PATH = os.path.join(WILDLIFE_DIR, "data", "image_embeddings.npz")
AUD_EMB_PATH = os.path.join(WILDLIFE_DIR, "data", "audio_embeddings.npz")
PROFILE_PATH = os.path.join(WILDLIFE_DIR, "outputs", "Q5.json")

PROJECT = os.environ.get("GCP_PROJECT", "")
BUCKET  = f"{PROJECT}-animals_dataset"
DATASET = "animals_dataset"
IMG_STAGING = f"{DATASET}.q5_image_uncertain_mm"
AUD_STAGING = f"{DATASET}.q5_audio_uncertain_mm"

IMG_PROMPT = "Does this image contain an elephant? "
AUD_PROMPT = "Does this audio contain an elephant sound? "

IMG_POSITIVE = [
    "a photograph of an elephant",
    "a wildlife camera trap image showing an elephant",
    "an elephant captured in the photo",
]
IMG_NEGATIVE = [
    "a photograph that does not contain an elephant",
    "a wildlife camera trap image of a non-elephant animal",
    "an animal photo without any elephant",
]
AUD_POSITIVE = [
    "a sound recording of an elephant",
    "audio of an elephant trumpeting or vocalizing",
    "elephant call sound clip",
]
AUD_NEGATIVE = [
    "a sound recording of an animal that is not an elephant",
    "audio of a non-elephant animal vocalization",
    "animal sound clip without any elephant",
]

ALPHA = 0.2
PAPER_BQ_Q5 = {"score": 0.75, "latency_s": 19.2, "cost_usd": 0.12}
PAPER_DASE_NN_Q5 = {"score": 1.00, "latency_s": 1e-3, "cost_usd": 1e-9}
SKIP_BASELINE = False


# ─── Side-specific BQ verifier setup ─────────────────────────────────────
def make_image_verifier(uncertain_uris):
    """CTAS staging from uncertain image URIs, then AI.IF returns Cities for image-elephant rows."""
    def make_staging(_ids_unused):
        # We use uncertain_uris (closure) as the param; signature must match (ids,).
        return f"""
        CREATE OR REPLACE TABLE {IMG_STAGING} AS
        SELECT m.Species, m.City, m.StationID, ot.ref AS image
        FROM {DATASET}.image_data_images m
        JOIN {DATASET}.image_data_external ot ON ot.uri = m.ImagePath
        WHERE m.ImagePath IN UNNEST({_uri_array_literal(uncertain_uris)})
        """

    verify_sql = f"""
    SELECT DISTINCT City AS id FROM {IMG_STAGING}
    WHERE AI.IF(('{IMG_PROMPT}', image),
                connection_id => 'us.connection',
                endpoint => 'gemini-2.5-flash')
    """
    return AiIfVerifier(
        verify_sql=verify_sql,
        make_staging_sql=make_staging,
        id_column="id",
        coerce_id=str,
    )


def make_audio_verifier(uncertain_uris):
    def make_staging(_ids_unused):
        return f"""
        CREATE OR REPLACE TABLE {AUD_STAGING} AS
        SELECT m.Animal, m.City, m.StationID, ot.ref AS audio
        FROM {DATASET}.audio_data_files m
        JOIN {DATASET}.audio_data_external ot ON ot.uri = m.AudioPath
        WHERE m.AudioPath IN UNNEST({_uri_array_literal(uncertain_uris)})
        """

    verify_sql = f"""
    SELECT DISTINCT City AS id FROM {AUD_STAGING}
    WHERE AI.IF(('{AUD_PROMPT}', audio),
                connection_id => 'us.connection',
                endpoint => 'gemini-2.5-flash')
    """
    return AiIfVerifier(
        verify_sql=verify_sql,
        make_staging_sql=make_staging,
        id_column="id",
        coerce_id=str,
    )


def _uri_array_literal(uris):
    """Inline a python list of uris as BQ ARRAY<STRING>. Cheap; uncertain count is ~10s."""
    items = ",".join(f"'{u}'" for u in uris)
    return f"[{items}]"


def run_baseline(client):
    sql = f"""
    SELECT DISTINCT city FROM (
      SELECT City AS city FROM {DATASET}.image_data_mm
      WHERE AI.IF(('{IMG_PROMPT}', image), connection_id => 'us.connection', endpoint => 'gemini-2.5-flash')
      UNION ALL
      SELECT City AS city FROM {DATASET}.audio_data_mm
      WHERE AI.IF(('{AUD_PROMPT}', audio), connection_id => 'us.connection', endpoint => 'gemini-2.5-flash')
    )
    """
    return run_query(client, sql)


def main():
    profile = build_profile(
        scenario="wildlife", query_id=5, scale_factor=200,
        params={"alpha": ALPHA},
        cascade_form="Two F-cascades (image + audio, AlphaBand, AI.IF) + client-side City union",
        extra={
            "image_prompt": IMG_PROMPT, "audio_prompt": AUD_PROMPT,
            "dase_image_prompts": {"positive": IMG_POSITIVE, "negative": IMG_NEGATIVE},
            "dase_audio_prompts": {"positive": AUD_POSITIVE, "negative": AUD_NEGATIVE},
        },
    )

    print("Loading image + audio data and embeddings...")
    img_df = pd.read_csv(IMAGE_CSV)
    aud_df = pd.read_csv(AUDIO_CSV)
    img_emb = np.load(IMG_EMB_PATH)["caption_emb"]
    aud_emb = np.load(AUD_EMB_PATH)["caption_emb"]
    img_df["GcsUri"] = img_df["ImagePath"].apply(lambda p: f"gs://{BUCKET}/animal_images/{os.path.basename(p)}")
    aud_df["GcsUri"] = aud_df["AudioPath"].apply(lambda p: f"gs://{BUCKET}/animal_audio/{os.path.basename(p)}")

    n_img, n_aud = len(img_df), len(aud_df)
    gt_img_cities = set(img_df[img_df["Species"].str.contains("ELEPHANT")]["City"])
    gt_aud_cities = set(aud_df[aud_df["Animal"] == "Elephant"]["City"])
    gt_cities = gt_img_cities | gt_aud_cities
    print(f"  {n_img} images, {n_aud} audios; GT cities (union): {sorted(gt_cities)}")
    profile["data"] = {"n_images": n_img, "n_audios": n_aud, "gt_cities": sorted(gt_cities)}

    client = bq_client(PROJECT)

    # ── Per-side cost calibration ──
    print("\n=== Per-row cost calibration (image + audio) ===")
    img_cal = per_row_cost(
        client, IMG_PROMPT,
        sample_uris=img_df["GcsUri"].head(5).tolist(),
        ext_table=f"{DATASET}.image_data_external",
    )
    aud_cal = per_row_cost(
        client, AUD_PROMPT,
        sample_uris=aud_df["GcsUri"].head(5).tolist(),
        ext_table=f"{DATASET}.audio_data_external",
    )
    print(f"  image per_row=${img_cal.per_row_cost_usd:.6f}, audio per_row=${aud_cal.per_row_cost_usd:.6f}")
    profile["calibration"] = {"image": img_cal.to_dict(), "audio": aud_cal.to_dict()}

    # ── Image cascade ──
    # The Cascade composes Signal+Band, but the Verifier needs the URI list of
    # uncertain rows (not just ids) to CTAS the staging table. We do the dase
    # part first (signal + band → uncertain indices), then build the verifier
    # from those URIs and run BQ stage explicitly via cascade.run().
    img_signal = MarginSignal(positive_prompts=IMG_POSITIVE, negative_prompts=IMG_NEGATIVE)
    img_band = AlphaBand(alpha=ALPHA)
    # We invoke signal + band manually here so we can hand uncertain URIs to the
    # verifier; the BQ stage is then run via a one-shot direct call, not Cascade.
    # (Future ergonomics: a `RowsCascade` that exposes the uncertain rows to a
    # verifier-builder, but for now this two-step pattern is explicit and clear.)
    img_scores = img_signal.compute(img_emb)
    img_part = img_band.partition(img_scores)
    img_uncertain_uris = img_df["GcsUri"].iloc[img_part.uncertain].tolist()
    img_confident_cities = set(img_df["City"].iloc[img_part.confident_pos])
    print(f"\n  image  alpha={ALPHA}: confident_pos cities={sorted(img_confident_cities)}, uncertain={len(img_uncertain_uris)}")

    # ── Audio cascade ──
    aud_signal = MarginSignal(positive_prompts=AUD_POSITIVE, negative_prompts=AUD_NEGATIVE)
    aud_scores = aud_signal.compute(aud_emb)
    aud_part = AlphaBand(alpha=ALPHA).partition(aud_scores)
    aud_uncertain_uris = aud_df["GcsUri"].iloc[aud_part.uncertain].tolist()
    aud_confident_cities = set(aud_df["City"].iloc[aud_part.confident_pos])
    print(f"  audio  alpha={ALPHA}: confident_pos cities={sorted(aud_confident_cities)}, uncertain={len(aud_uncertain_uris)}")

    profile["dase_partition"] = {
        "image": {"n_uncertain": len(img_uncertain_uris),
                  "confident_pos_cities": sorted(img_confident_cities)},
        "audio": {"n_uncertain": len(aud_uncertain_uris),
                  "confident_pos_cities": sorted(aud_confident_cities)},
    }

    # ── Baseline (verbatim sembench Q5.sql) ──
    if SKIP_BASELINE:
        b_score, bwall, bslot = PAPER_BQ_Q5["score"], PAPER_BQ_Q5["latency_s"], None
        bcost = PAPER_BQ_Q5["cost_usd"]; b_cities = None
        bcalls_total = round(bcost / ((img_cal.per_row_cost_usd + aud_cal.per_row_cost_usd) / 2))
        profile["baseline"] = {"_status": "aborted", "score": {"f1": b_score, "_source": "paper"},
                               "latency_breakdown": {"wall_s": bwall, "_source": "paper"},
                               "cost_breakdown": {"total_cost_usd": bcost, "_source": "paper"}}
    else:
        print(f"\n=== Baseline (sembench Q5.sql verbatim) ===")
        bdf, bwall, bslot, bsql = run_baseline(client)
        b_cities = set(bdf["city"])
        bcalls_total = n_img + n_aud
        bcost = img_cal.per_row_cost_usd * n_img + aud_cal.per_row_cost_usd * n_aud
        _, _, b_score = f1_set(b_cities, gt_cities)
        print(f"  cities={sorted(b_cities)} (GT={sorted(gt_cities)})")
        print(f"  wall={bwall:.2f}s, slot={bslot}, calls={bcalls_total}, cost=${bcost:.6f}, F1={b_score:.4f}")
        profile["baseline"] = {
            "method": "sembench bigquery/Q5.sql verbatim", "sql": bsql,
            "result_cities": sorted(b_cities), "score": {"f1": b_score},
            "latency_breakdown": {"wall_s": bwall, "slot_ms": bslot},
            "cost_breakdown": {"n_llm_calls": bcalls_total,
                               "img_per_row": img_cal.per_row_cost_usd,
                               "aud_per_row": aud_cal.per_row_cost_usd,
                               "total_cost_usd": bcost},
        }

    # ── BQ stage for both modalities (via verifiers) ──
    print(f"\n=== Cascade BQ stages: 2 CTAS + 2 AI.IF ===")
    img_verifier = make_image_verifier(img_uncertain_uris)
    aud_verifier = make_audio_verifier(aud_uncertain_uris)

    img_v = img_verifier.verify(client, img_uncertain_uris, img_cal.per_row_cost_usd)
    print(f"  image  CTAS wall={img_v.ctas_wall_s:.2f}s slot={img_v.ctas_slot_ms}; "
          f"AI.IF wall={img_v.wall_s:.2f}s slot={img_v.slot_ms}; YES cities={sorted(img_v.positive_ids)}")
    aud_v = aud_verifier.verify(client, aud_uncertain_uris, aud_cal.per_row_cost_usd)
    print(f"  audio  CTAS wall={aud_v.ctas_wall_s:.2f}s slot={aud_v.ctas_slot_ms}; "
          f"AI.IF wall={aud_v.wall_s:.2f}s slot={aud_v.slot_ms}; YES cities={sorted(aud_v.positive_ids)}")

    # ── M-merge (client-side City union) ──
    cascade_cities = set(img_v.positive_ids) | set(aud_v.positive_ids) | img_confident_cities | aud_confident_cities
    _, _, cscore = f1_set(cascade_cities, gt_cities)
    cascade_cost = img_v.cost_usd + aud_v.cost_usd
    cascade_total_wall = img_v.ctas_wall_s + aud_v.ctas_wall_s + img_v.wall_s + aud_v.wall_s
    cascade_total_slot = img_v.ctas_slot_ms + aud_v.ctas_slot_ms + img_v.slot_ms + aud_v.slot_ms
    s2_calls = img_v.n_calls + aud_v.n_calls

    print(f"\n  cascade cities={sorted(cascade_cities)} (GT={sorted(gt_cities)})  F1={cscore:.4f}")
    print(f"  cascade wall={cascade_total_wall:.2f}s slot={cascade_total_slot} calls={s2_calls} cost=${cascade_cost:.6f}")

    profile["cascade"] = {
        "method": "Two F-cascades (img+aud, AlphaBand, AI.IF on uncertain) + client-side City union",
        "image": img_v.to_dict(), "audio": aud_v.to_dict(),
        "cascade_cities": sorted(cascade_cities),
        "score": {"f1": cscore},
        "totals": {"wall_s": cascade_total_wall, "slot_ms_bq_total": cascade_total_slot,
                   "cost_usd": cascade_cost, "n_llm_calls": s2_calls},
    }

    profile["comparison"] = {
        "score":       {"paper_BQ": PAPER_BQ_Q5["score"],   "paper_DASE_NN": PAPER_DASE_NN_Q5["score"],   "ours_BQ": b_score, "ours_cascade": cscore},
        "wall_s":      {"paper_BQ": PAPER_BQ_Q5["latency_s"], "paper_DASE_NN": PAPER_DASE_NN_Q5["latency_s"], "ours_BQ": bwall,   "ours_cascade": cascade_total_wall},
        "cost_usd":    {"paper_BQ": PAPER_BQ_Q5["cost_usd"], "paper_DASE_NN": PAPER_DASE_NN_Q5["cost_usd"], "ours_BQ": bcost,    "ours_cascade": cascade_cost},
        "n_llm_calls": {"paper_BQ": round(PAPER_BQ_Q5["cost_usd"] / ((img_cal.per_row_cost_usd + aud_cal.per_row_cost_usd) / 2)),
                        "paper_DASE_NN": 0, "ours_BQ": bcalls_total, "ours_cascade": s2_calls},
    }

    write_profile(profile, PROFILE_PATH)

    print_summary(
        f"Wildlife Q5 (alpha={ALPHA})",
        columns=["paper BQ", "DASE+NN", "ours BQ", "ours cascade"],
        rows=[
            ("score (F1)", [PAPER_BQ_Q5["score"], PAPER_DASE_NN_Q5["score"], b_score, cscore], ".2f"),
            ("wall (s)",   [PAPER_BQ_Q5["latency_s"], PAPER_DASE_NN_Q5["latency_s"], bwall, cascade_total_wall], ".2f"),
            ("cost ($)",   [PAPER_BQ_Q5["cost_usd"], PAPER_DASE_NN_Q5["cost_usd"], bcost, cascade_cost], ".4f"),
            ("#LLM calls", [round(PAPER_BQ_Q5["cost_usd"] / ((img_cal.per_row_cost_usd + aud_cal.per_row_cost_usd) / 2)),
                            0, bcalls_total, s2_calls], "d"),
        ],
    )


if __name__ == "__main__":
    main()
