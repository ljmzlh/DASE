#!/usr/bin/env -S python -u
"""
Wildlife Q6 cascade — set DIFFERENCE (cities with monkey IMAGE − monkey AUDIO).

NL: Cities with monkey images but NO monkey audio recordings.
GT: image_monkey_cities − audio_monkey_cities.
Eval: set retrieval F1.

Refactored to use dase_cascade unified solver. Operator (paper Table 3): M
= composition of two F (image + audio) with client-side set difference over City.

Same shape as q5_v2.py: two parallel signal+band+verifier pipelines on the two
modalities, then a client-side set algebra step on the per-modality positive
City sets.
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
AUDIO_CSV    = os.path.join(WILDLIFE_DIR, "cache", "audio_data.csv")
IMG_EMB_PATH = os.path.join(WILDLIFE_DIR, "data", "image_embeddings.npz")
AUD_EMB_PATH = os.path.join(WILDLIFE_DIR, "data", "audio_embeddings.npz")
PROFILE_PATH = os.path.join(WILDLIFE_DIR, "outputs", "Q6.json")

PROJECT = os.environ.get("GCP_PROJECT", "")
BUCKET  = f"{PROJECT}-animals_dataset"
DATASET = "animals_dataset"
IMG_STAGING = f"{DATASET}.q6_image_uncertain_mm"
AUD_STAGING = f"{DATASET}.q6_audio_uncertain_mm"

IMG_PROMPT = "Does this image contain a monkey? "
AUD_PROMPT = "Does this audio contain a monkey sound? "

IMG_POSITIVE = [
    "a photograph of a monkey",
    "a wildlife camera trap image showing a monkey",
    "a monkey captured in the photo",
]
IMG_NEGATIVE = [
    "a photograph that does not contain a monkey",
    "a wildlife camera trap image of a non-monkey animal",
    "an animal photo without any monkey",
]
AUD_POSITIVE = [
    "a sound recording of a monkey",
    "audio of monkey vocalizations or calls",
    "monkey howling or chittering sound clip",
]
AUD_NEGATIVE = [
    "a sound recording of an animal that is not a monkey",
    "audio of a non-monkey animal vocalization",
    "animal sound clip without any monkey",
]

ALPHA = 0.5  # set-diff sensitive to BQ over-pred → expand band so BQ sees more rows
PAPER_BQ_Q6 = {"score": 0.20, "latency_s": 24.3, "cost_usd": 0.12}
PAPER_DASE_NN_Q6 = {"score": 0.00, "latency_s": 1e-3, "cost_usd": 1e-9}
SKIP_BASELINE = False


def _uri_array_literal(uris):
    items = ",".join(f"'{u}'" for u in uris)
    return f"[{items}]"


def make_image_verifier(uncertain_uris):
    def make_staging(_ids_unused):
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
        verify_sql=verify_sql, make_staging_sql=make_staging,
        id_column="id", coerce_id=str,
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
        verify_sql=verify_sql, make_staging_sql=make_staging,
        id_column="id", coerce_id=str,
    )


def run_baseline(client):
    sql = f"""
    SELECT DISTINCT I.City AS city
    FROM {DATASET}.image_data_mm I
    WHERE AI.IF(('{IMG_PROMPT}', I.image), connection_id => 'us.connection', endpoint => 'gemini-2.5-flash')
    AND NOT EXISTS (
      SELECT * FROM {DATASET}.audio_data_mm A
      WHERE A.City = I.City
      AND AI.IF(('{AUD_PROMPT}', A.audio), connection_id => 'us.connection', endpoint => 'gemini-2.5-flash')
    )
    """
    return run_query(client, sql)


def main():
    profile = build_profile(
        scenario="wildlife", query_id=6, scale_factor=200,
        params={"alpha": ALPHA},
        cascade_form="Two F-cascades (image + audio, AlphaBand, AI.IF) + client-side City set-difference (img - aud)",
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
    gt_img_monkey = set(img_df[img_df["Species"].str.contains("MONKEY")]["City"])
    gt_aud_monkey = set(aud_df[aud_df["Animal"] == "Monkey"]["City"])
    gt_cities = gt_img_monkey - gt_aud_monkey
    print(f"  {n_img} images, {n_aud} audios")
    print(f"  GT image monkey cities: {sorted(gt_img_monkey)}")
    print(f"  GT audio monkey cities: {sorted(gt_aud_monkey)}")
    print(f"  GT cities (img - aud): {sorted(gt_cities)}")
    profile["data"] = {"n_images": n_img, "n_audios": n_aud,
                       "gt_image_monkey_cities": sorted(gt_img_monkey),
                       "gt_audio_monkey_cities": sorted(gt_aud_monkey),
                       "gt_cities": sorted(gt_cities)}

    client = bq_client(PROJECT)

    # ── Per-side cost calibration ──
    print("\n=== Per-row cost calibration (image + audio) ===")
    img_cal = per_row_cost(client, IMG_PROMPT,
        sample_uris=img_df["GcsUri"].head(5).tolist(),
        ext_table=f"{DATASET}.image_data_external")
    aud_cal = per_row_cost(client, AUD_PROMPT,
        sample_uris=aud_df["GcsUri"].head(5).tolist(),
        ext_table=f"{DATASET}.audio_data_external")
    print(f"  image per_row=${img_cal.per_row_cost_usd:.6f}, audio per_row=${aud_cal.per_row_cost_usd:.6f}")
    profile["calibration"] = {"image": img_cal.to_dict(), "audio": aud_cal.to_dict()}

    # ── Image side: signal + band locally; verifier needs URI list (closure) ──
    img_signal = MarginSignal(positive_prompts=IMG_POSITIVE, negative_prompts=IMG_NEGATIVE)
    img_scores = img_signal.compute(img_emb)
    img_part = AlphaBand(alpha=ALPHA).partition(img_scores)
    img_uncertain_uris = img_df["GcsUri"].iloc[img_part.uncertain].tolist()
    dase_img_monkey_cities = set(img_df["City"].iloc[img_part.confident_pos])
    print(f"\n  image alpha={ALPHA}: confident_pos cities={sorted(dase_img_monkey_cities)}, uncertain={len(img_uncertain_uris)}")

    # ── Audio side ──
    aud_signal = MarginSignal(positive_prompts=AUD_POSITIVE, negative_prompts=AUD_NEGATIVE)
    aud_scores = aud_signal.compute(aud_emb)
    aud_part = AlphaBand(alpha=ALPHA).partition(aud_scores)
    aud_uncertain_uris = aud_df["GcsUri"].iloc[aud_part.uncertain].tolist()
    dase_aud_monkey_cities = set(aud_df["City"].iloc[aud_part.confident_pos])
    print(f"  audio alpha={ALPHA}: confident_pos cities={sorted(dase_aud_monkey_cities)}, uncertain={len(aud_uncertain_uris)}")

    profile["dase_partition"] = {
        "image_uncertain": len(img_uncertain_uris), "audio_uncertain": len(aud_uncertain_uris),
        "dase_image_monkey_cities": sorted(dase_img_monkey_cities),
        "dase_audio_monkey_cities": sorted(dase_aud_monkey_cities),
    }

    # ── Baseline ──
    if SKIP_BASELINE:
        b_score = PAPER_BQ_Q6["score"]; bwall = PAPER_BQ_Q6["latency_s"]; bslot = None
        bcost = PAPER_BQ_Q6["cost_usd"]
        bcalls_total = round(bcost / ((img_cal.per_row_cost_usd + aud_cal.per_row_cost_usd) / 2))
        b_cities = None
        profile["baseline"] = {"_status": "aborted", "score": {"score": b_score, "_source": "paper"},
                                "latency_breakdown": {"wall_s": bwall, "_source": "paper"},
                                "cost_breakdown": {"n_llm_calls": bcalls_total, "total_cost_usd": bcost, "_source": "paper"}}
    else:
        print(f"\n=== Baseline (sembench Q6.sql verbatim, NOT EXISTS) ===")
        bdf, bwall, bslot, bsql = run_baseline(client)
        b_cities = set(bdf["city"])
        # Q6 baseline correlated subquery: rough lower bound n_img + n_aud
        bcalls_total = n_img + n_aud
        bcost = img_cal.per_row_cost_usd * n_img + aud_cal.per_row_cost_usd * n_aud
        _, _, b_score = f1_set(b_cities, gt_cities)
        print(f"  baseline cities: {sorted(b_cities)} (GT: {sorted(gt_cities)})  F1={b_score:.4f}")
        profile["baseline"] = {
            "method": "sembench bigquery/Q6.sql verbatim with NOT EXISTS",
            "sql": bsql, "result_cities": sorted(b_cities),
            "score": {"f1": b_score},
            "latency_breakdown": {"wall_s": bwall, "slot_ms": bslot},
            "cost_breakdown": {"n_llm_calls_rough_lower_bound": bcalls_total, "total_cost_usd_rough": bcost},
        }

    # ── BQ stage for both modalities (via verifiers) ──
    print(f"\n=== Cascade BQ stages: 2 CTAS + 2 AI.IF ===")
    img_verifier = make_image_verifier(img_uncertain_uris)
    aud_verifier = make_audio_verifier(aud_uncertain_uris)
    img_v = img_verifier.verify(client, img_uncertain_uris, img_cal.per_row_cost_usd)
    aud_v = aud_verifier.verify(client, aud_uncertain_uris, aud_cal.per_row_cost_usd)
    print(f"  image  CTAS={img_v.ctas_wall_s:.2f}s  AI.IF={img_v.wall_s:.2f}s  YES cities={sorted(img_v.positive_ids)}")
    print(f"  audio  CTAS={aud_v.ctas_wall_s:.2f}s  AI.IF={aud_v.wall_s:.2f}s  YES cities={sorted(aud_v.positive_ids)}")

    # ── M-merge: client-side set difference ──
    image_monkey_cities = set(img_v.positive_ids) | dase_img_monkey_cities
    audio_monkey_cities = set(aud_v.positive_ids) | dase_aud_monkey_cities
    cascade_cities = image_monkey_cities - audio_monkey_cities
    _, _, cscore = f1_set(cascade_cities, gt_cities)
    cascade_cost = img_v.cost_usd + aud_v.cost_usd
    cascade_total_wall = img_v.ctas_wall_s + aud_v.ctas_wall_s + img_v.wall_s + aud_v.wall_s
    cascade_total_slot = img_v.ctas_slot_ms + aud_v.ctas_slot_ms + img_v.slot_ms + aud_v.slot_ms
    s2_calls = img_v.n_calls + aud_v.n_calls
    print(f"\n  cascade_cities (img - aud): {sorted(cascade_cities)} (GT: {sorted(gt_cities)})  F1={cscore:.4f}")
    print(f"  wall={cascade_total_wall:.2f}s slot={cascade_total_slot} calls={s2_calls} cost=${cascade_cost:.6f}")

    profile["cascade"] = {
        "method": "Two F-cascades (img+aud, AlphaBand, AI.IF on uncertain) + client-side City set-difference",
        "image": img_v.to_dict(), "audio": aud_v.to_dict(),
        "image_monkey_cities_combined": sorted(image_monkey_cities),
        "audio_monkey_cities_combined": sorted(audio_monkey_cities),
        "cascade_cities": sorted(cascade_cities),
        "score": {"f1": cscore},
        "totals": {"wall_s": cascade_total_wall, "slot_ms_bq_total": cascade_total_slot,
                   "cost_usd": cascade_cost, "n_llm_calls": s2_calls},
    }

    profile["comparison"] = {
        "score":       {"paper_BQ": PAPER_BQ_Q6["score"], "paper_DASE_NN": PAPER_DASE_NN_Q6["score"], "ours_BQ": b_score, "ours_cascade": cscore},
        "wall_s":      {"paper_BQ": PAPER_BQ_Q6["latency_s"], "paper_DASE_NN": PAPER_DASE_NN_Q6["latency_s"], "ours_BQ": bwall, "ours_cascade": cascade_total_wall},
        "cost_usd":    {"paper_BQ": PAPER_BQ_Q6["cost_usd"], "paper_DASE_NN": PAPER_DASE_NN_Q6["cost_usd"], "ours_BQ": bcost, "ours_cascade": cascade_cost},
        "n_llm_calls": {"paper_BQ": round(PAPER_BQ_Q6["cost_usd"] / ((img_cal.per_row_cost_usd + aud_cal.per_row_cost_usd) / 2)),
                        "paper_DASE_NN": 0, "ours_BQ": bcalls_total, "ours_cascade": s2_calls},
    }

    write_profile(profile, PROFILE_PATH)

    print_summary(
        f"Wildlife Q6 (alpha={ALPHA})",
        columns=["paper BQ", "DASE+NN", "ours BQ", "ours cascade"],
        rows=[
            ("score (F1)", [PAPER_BQ_Q6["score"], PAPER_DASE_NN_Q6["score"], b_score, cscore], ".2f"),
            ("wall (s)",   [PAPER_BQ_Q6["latency_s"], PAPER_DASE_NN_Q6["latency_s"], bwall, cascade_total_wall], ".2f"),
            ("cost ($)",   [PAPER_BQ_Q6["cost_usd"], PAPER_DASE_NN_Q6["cost_usd"], bcost, cascade_cost], ".4f"),
            ("#LLM calls", [round(PAPER_BQ_Q6["cost_usd"] / ((img_cal.per_row_cost_usd + aud_cal.per_row_cost_usd) / 2)),
                            0, bcalls_total, s2_calls], "d"),
        ],
    )


if __name__ == "__main__":
    main()
