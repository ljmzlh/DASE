#!/usr/bin/env -S python -u
"""
Cars Q5 cascade (v2) — composed F F (audio + image) → AND on car_id → COUNT,
                       via dase_cascade.

NL: How many automatic cars are damaged according to both audio and images?
GT: 5 @ sf_19672.
Eval: aggregation_single (relative_error → score=1/(1+rel_err)).

Refactored to use dase_cascade. Operator (paper Table 3): F F (composed).
Two independent Cascades on the (Automatic ⨝ audio ⨝ image) joint scope:
  cascade_audio = Cascade(MarginSignal(audio prompts), AlphaBand(α=1.0), AiIfVerifier_audio)
  cascade_image = Cascade(MarginSignal(image prompts), AlphaBand(α=0.5), AiIfVerifier_image)
Client-side AND merges the per-modality yes-car sets. Multi-stage composition
stays in this script (paper §5.1).
"""
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from google.cloud import bigquery

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import (
    Cascade, MarginSignal, AlphaBand, AiIfVerifier,
    bq_client, per_row_cost, run_query,
    relative_error_score, build_profile, write_profile, print_summary,
)

CARS_DIR = os.path.abspath(os.path.join(_HERE, ".."))
CARS_PARQUET = os.path.join(CARS_DIR, "data", "cars.parquet")
AUDIO_PARQUET = os.path.join(CARS_DIR, "data", "audio_cars.parquet")
IMAGE_PARQUET = os.path.join(CARS_DIR, "data", "image_cars.parquet")
GT_CSV = os.path.join(CARS_DIR, "ground_truth", "Q5.csv")
PROFILE_PATH = os.path.join(CARS_DIR, "outputs", "Q5.json")

PROJECT = os.environ.get("GCP_PROJECT", "")
BUCKET = f"{PROJECT}-cars_dataset"
DATASET = "cars_dataset"
STAGING_AUDIO = f"{DATASET}.q5_uncertain_audio_mm"
STAGING_IMAGE = f"{DATASET}.q5_uncertain_car_mm"

PROMPT_AUDIO = "You are given an audio recording of car diagnostics. Return true if the recording captures an audio of a damaged car."
PROMPT_IMAGE = "You are given an image of a vehicle or its parts. Return true if car is damaged."

POS_AUDIO = [
    "audio recording of a damaged or malfunctioning car",
    "engine, brake, or mechanical fault sounds",
    "audio of a car with abnormal mechanical problems",
]
NEG_AUDIO = [
    "audio of a healthy car running normally",
    "engine sounds with no fault",
    "normal car operation audio",
]
POS_IMAGE = [
    "an image of a damaged or wrecked car",
    "a car with dents, scratches, or broken parts",
    "a vehicle showing signs of accident damage",
]
NEG_IMAGE = [
    "an image of an undamaged intact car",
    "a clean car in good condition",
    "a vehicle with no visible damage",
]

ALPHA_AUDIO = 1.0
ALPHA_IMAGE = 0.5
PAPER_BQ_Q5 = {"score": 1.00, "latency_s": 58.9, "cost_usd": 1.47}
PAPER_DASE_NN_Q5 = {"score": 0.50, "latency_s": 1.1, "cost_usd": 1e-5}
SKIP_BASELINE = False


def trunc2(x):
    return f"{math.floor(x * 100) / 100:.2f}"


Q5_BASELINE_SQL = f"""
SELECT transmission, COUNT(*) AS count
FROM (
  SELECT DISTINCT p.car_id, p.transmission
  FROM {DATASET}.cars AS p, {DATASET}.audio_mm AS a, {DATASET}.car_mm AS x
  WHERE p.transmission = "Automatic"
    AND p.car_id = x.car_id AND p.car_id = a.car_id
    AND AI.IF(
      ('{PROMPT_AUDIO}', a.image),
      connection_id => 'us.connection',
      endpoint => 'gemini-2.5-flash')
    AND AI.IF(
      ('{PROMPT_IMAGE}', x.image),
      connection_id => 'us.connection',
      endpoint => 'gemini-2.5-flash')
)
GROUP BY transmission
"""

Q5_STAGE2_AUDIO_SQL = f"""
SELECT DISTINCT a.car_id AS id
FROM {STAGING_AUDIO} AS a
WHERE AI.IF(
  ('{PROMPT_AUDIO}', a.image),
  connection_id => 'us.connection',
  endpoint => 'gemini-2.5-flash'
)
"""

Q5_STAGE2_IMAGE_SQL = f"""
SELECT DISTINCT x.car_id AS id
FROM {STAGING_IMAGE} AS x
WHERE AI.IF(
  ('{PROMPT_IMAGE}', x.image),
  connection_id => 'us.connection',
  endpoint => 'gemini-2.5-flash'
)
"""


def make_audio_verifier():
    def make_staging(uris):
        items = ",".join(f"'{u}'" for u in uris)
        return f"""
        CREATE OR REPLACE TABLE {STAGING_AUDIO} AS
        SELECT m.audio_id, m.car_id, ot.ref AS image
        FROM {DATASET}.car_audio m
        JOIN {DATASET}.cars_audios ot ON ot.uri = m.audio_path
        WHERE m.audio_path IN UNNEST([{items}])
        """
    return AiIfVerifier(verify_sql=Q5_STAGE2_AUDIO_SQL,
                        make_staging_sql=make_staging,
                        id_column="id", coerce_id=int)


def make_image_verifier():
    def make_staging(uris):
        items = ",".join(f"'{u}'" for u in uris)
        return f"""
        CREATE OR REPLACE TABLE {STAGING_IMAGE} AS
        SELECT m.image_id, m.car_id, ot.ref AS image
        FROM {DATASET}.car_images m
        JOIN {DATASET}.cars_images ot ON ot.uri = m.image_path
        WHERE m.image_path IN UNNEST([{items}])
        """
    return AiIfVerifier(verify_sql=Q5_STAGE2_IMAGE_SQL,
                        make_staging_sql=make_staging,
                        id_column="id", coerce_id=int)


def main():
    profile = build_profile(
        scenario="cars", query_id=5, scale_factor=19672,
        params={"alpha_audio": ALPHA_AUDIO, "alpha_image": ALPHA_IMAGE,
                "_note": "alpha_audio=1.0 forces all audio to BQ (dase audio caption signal not reliable enough to drop)"},
        cascade_form=(
            "Composed F F via dase_cascade: two Cascade(MarginSignal, AlphaBand, AiIfVerifier(CTAS staging)) "
            "instances (audio + image) → client AND-merge on car_id → COUNT scalar"
        ),
        extra={
            "operator": "F F",
            "prompt": {"audio": PROMPT_AUDIO, "image": PROMPT_IMAGE},
            "structural_filter": "p.transmission='Automatic' AND p.car_id=x.car_id AND p.car_id=a.car_id",
            "dase_prompts": {
                "audio": {"positive": POS_AUDIO, "negative": NEG_AUDIO},
                "image": {"positive": POS_IMAGE, "negative": NEG_IMAGE},
            },
        },
    )

    print("Loading cars + audio + image, applying structural prefilter (Automatic ⨝ audio ⨝ image)...")
    t = time.time()
    cars = pd.read_parquet(CARS_PARQUET)
    audio = pd.read_parquet(AUDIO_PARQUET)
    img = pd.read_parquet(IMAGE_PARQUET)
    auto_ids = set(cars[cars["transmission"] == "Automatic"]["car_id"])
    audio_auto = audio[audio["car_id"].isin(auto_ids)].reset_index(drop=True)
    img_auto = img[img["car_id"].isin(auto_ids)].reset_index(drop=True)
    common_cars = set(audio_auto["car_id"]) & set(img_auto["car_id"])
    audio_scope = audio_auto[audio_auto["car_id"].isin(common_cars)].reset_index(drop=True)
    image_scope = img_auto[img_auto["car_id"].isin(common_cars)].reset_index(drop=True)
    n_audio = len(audio_scope)
    n_image = len(image_scope)
    n_cars = len(common_cars)
    gt_count = int(pd.read_csv(GT_CSV)["count"].iloc[0])
    t_load = time.time() - t

    print(f"  cars (Automatic): {len(auto_ids)}, audio rows in scope: {n_audio}, image rows in scope: {n_image}")
    print(f"  unique cars in joint scope: {n_cars}; GT count = {gt_count}")
    profile["data"] = {
        "n_automatic_cars": len(auto_ids),
        "n_audio_in_scope": n_audio,
        "n_image_in_scope": n_image,
        "n_unique_cars_in_scope": n_cars,
        "gt_count": gt_count,
    }

    audio_emb = np.stack(audio_scope["embedding"].tolist()).astype(np.float32)
    image_emb = np.stack(image_scope["embedding"].tolist()).astype(np.float32)
    audio_uris = [f"gs://{BUCKET}/car_audios/{os.path.basename(p)}" for p in audio_scope["audio_path"]]
    image_uris = [f"gs://{BUCKET}/car_images/{os.path.basename(p)}" for p in image_scope["image_path"]]
    car_by_audio_uri = dict(zip(audio_uris, audio_scope["car_id"].astype(int)))
    car_by_image_uri = dict(zip(image_uris, image_scope["car_id"].astype(int)))

    client = bq_client(PROJECT)

    print("\n=== Per-row cost calibration (audio k=5) ===")
    audio_sample = audio_uris[:5] if audio_uris else []
    cal_audio = per_row_cost(
        client, PROMPT_AUDIO,
        sample_uris=audio_sample,
        ext_table=f"{DATASET}.cars_audios",
        method_label="AI.GENERATE_BOOL multimodal (audio) + thinking_budget=0",
        k=5,
    )
    per_row_audio = cal_audio.per_row_cost_usd
    print(f"  audio per_row=${per_row_audio:.6f}")

    print("\n=== Per-row cost calibration (image k=5) ===")
    image_sample = image_uris[:5] if image_uris else []
    cal_image = per_row_cost(
        client, PROMPT_IMAGE,
        sample_uris=image_sample,
        ext_table=f"{DATASET}.cars_images",
        method_label="AI.GENERATE_BOOL multimodal (image) + thinking_budget=0",
        k=5,
    )
    per_row_image = cal_image.per_row_cost_usd
    print(f"  image per_row=${per_row_image:.6f}")
    profile["calibration"] = {"audio": cal_audio.to_dict(), "image": cal_image.to_dict()}

    # ── Two F-cascades ──
    print("\n=== Cascade audio (MarginSignal → AlphaBand → AiIfVerifier) ===")
    cascade_audio = Cascade(
        embeddings=audio_emb, ids=audio_uris,
        signal=MarginSignal(positive_prompts=POS_AUDIO, negative_prompts=NEG_AUDIO),
        band=AlphaBand(alpha=ALPHA_AUDIO),
        verifier=make_audio_verifier(),
    )
    cres_a = cascade_audio.run(client, per_row_audio)
    audio_dase_pos_cars = {car_by_audio_uri[u] for u in cres_a.confident_pos_ids}
    audio_bq_yes_cars = set(cres_a.verifier_result.positive_ids)
    audio_yes_cars = audio_dase_pos_cars | audio_bq_yes_cars

    print("\n=== Cascade image (MarginSignal → AlphaBand → AiIfVerifier) ===")
    cascade_image = Cascade(
        embeddings=image_emb, ids=image_uris,
        signal=MarginSignal(positive_prompts=POS_IMAGE, negative_prompts=NEG_IMAGE),
        band=AlphaBand(alpha=ALPHA_IMAGE),
        verifier=make_image_verifier(),
    )
    cres_i = cascade_image.run(client, per_row_image)
    image_dase_pos_cars = {car_by_image_uri[u] for u in cres_i.confident_pos_ids}
    image_bq_yes_cars = set(cres_i.verifier_result.positive_ids)
    image_yes_cars = image_dase_pos_cars | image_bq_yes_cars

    # AND merge → COUNT
    cascade_cars = audio_yes_cars & image_yes_cars
    cascade_count = len(cascade_cars)
    c_score = relative_error_score(cascade_count, gt_count)
    c_rel = abs(cascade_count - gt_count) / abs(gt_count) if gt_count else 0.0

    s2a_calls = len(cres_a.uncertain_ids)
    s2b_calls = len(cres_i.uncertain_ids)
    s2_calls = s2a_calls + s2b_calls
    cascade_cost = cres_a.verifier_result.cost_usd + cres_i.verifier_result.cost_usd
    s1a_wall = cres_a.verifier_result.ctas_wall_s; s1a_slot = cres_a.verifier_result.ctas_slot_ms
    s1b_wall = cres_i.verifier_result.ctas_wall_s; s1b_slot = cres_i.verifier_result.ctas_slot_ms
    s2a_wall = cres_a.verifier_result.wall_s; s2a_slot = cres_a.verifier_result.slot_ms
    s2b_wall = cres_i.verifier_result.wall_s; s2b_slot = cres_i.verifier_result.slot_ms
    s1_wall = s1a_wall + s1b_wall; s1_slot = s1a_slot + s1b_slot
    s2_wall = s2a_wall + s2b_wall; s2_slot = s2a_slot + s2b_slot
    t_dase = (cres_a.timings_s.get("signal_compute", 0.0) + cres_a.timings_s.get("band_partition", 0.0)
              + cres_i.timings_s.get("signal_compute", 0.0) + cres_i.timings_s.get("band_partition", 0.0))
    t_dase_total = t_load + t_dase

    print(f"\n  [audio] confident_pos cars: {len(audio_dase_pos_cars)}, uncertain={len(cres_a.uncertain_ids)}, bq_yes={len(audio_bq_yes_cars)}")
    print(f"  [image] confident_pos cars: {len(image_dase_pos_cars)}, uncertain={len(cres_i.uncertain_ids)}, bq_yes={len(image_bq_yes_cars)}")
    print(f"  audio_yes_cars: {len(audio_yes_cars)}, image_yes_cars: {len(image_yes_cars)}")
    print(f"  cascade AND-cars: {cascade_count}, GT={gt_count}, rel_err={c_rel:.4f}, score={c_score:.4f}")

    cascade_total_wall = t_dase_total + s1_wall + s2_wall
    cascade_total_slot = s1_slot + s2_slot

    profile["dase_breakdown_s"] = {"total": t_dase_total, "data_load": t_load, "dase_compute": t_dase}
    profile["dase_partition"] = {
        "audio": {"n_confident_pos_cars": len(audio_dase_pos_cars),
                  "n_confident_neg": len(cres_a.confident_neg_ids),
                  "n_uncertain": len(cres_a.uncertain_ids)},
        "image": {"n_confident_pos_cars": len(image_dase_pos_cars),
                  "n_confident_neg": len(cres_i.confident_neg_ids),
                  "n_uncertain": len(cres_i.uncertain_ids)},
    }

    # ── Baseline (cached or run) ──
    cached_profile = None
    if os.path.exists(PROFILE_PATH):
        try:
            cached_profile = json.load(open(PROFILE_PATH))
            if cached_profile.get("baseline", {}).get("result_count") is None:
                cached_profile = None
        except Exception:
            cached_profile = None

    if SKIP_BASELINE:
        print(f"\n=== Baseline ABORTED — paper Table 4(e) numbers ===")
        b_count = None; b_rel = None
        b_score = PAPER_BQ_Q5["score"]; bwall = PAPER_BQ_Q5["latency_s"]; bslot = None
        bcost = PAPER_BQ_Q5["cost_usd"]
        bcalls = round(bcost / ((per_row_audio + per_row_image) / 2)) if (per_row_audio + per_row_image) else (n_audio + n_image)
        profile["baseline"] = {
            "_status": "aborted",
            "score": {"score": b_score, "_source": "paper Table 4(e)"},
            "latency_breakdown": {"wall_s": bwall, "slot_ms": None, "_source": "paper"},
            "cost_breakdown": {"n_llm_calls": bcalls, "total_cost_usd": bcost, "_source": "paper"},
            "method": "sembench bigquery/Q5.sql verbatim — NOT EXECUTED",
            "sql": Q5_BASELINE_SQL.strip(),
        }
    elif cached_profile is not None:
        cb = cached_profile["baseline"]
        b_count = int(cb["result_count"])
        b_score = relative_error_score(b_count, gt_count)
        b_rel = abs(b_count - gt_count) / abs(gt_count) if gt_count else 0.0
        bwall = float(cb["latency_breakdown"]["wall_s"])
        bslot = cb["latency_breakdown"]["slot_ms"]
        bcalls = int(cb["cost_breakdown"]["n_llm_calls"])
        bcost = per_row_audio * n_audio + per_row_image * n_image
        print(f"\n=== Baseline (cached from prev Q5.json) ===")
        print(f"  count={b_count}, GT={gt_count}, rel_err={b_rel:.4f}, score={b_score:.4f}")
        print(f"  wall={bwall:.2f}s (cached), slot_ms={bslot}, n_calls={bcalls}, cost=${bcost:.6f}")
        profile["baseline"] = {
            **cb,
            "_status": "cached_from_prev_profile",
            "result_count": b_count,
            "score": {"relative_error": b_rel, "score": b_score},
        }
    else:
        print("\n=== Baseline (sembench Q5.sql verbatim, scalar COUNT) ===")
        bdf, bwall, bslot, _ = run_query(client, Q5_BASELINE_SQL)
        if len(bdf) and "count" in bdf.columns:
            b_count = int(bdf.iloc[0]["count"])
        else:
            b_count = 0
        b_score = relative_error_score(b_count, gt_count)
        b_rel = abs(b_count - gt_count) / abs(gt_count) if gt_count else 0.0
        bcalls = n_audio + n_image
        bcost = per_row_audio * n_audio + per_row_image * n_image
        print(f"  baseline count={b_count}, GT={gt_count}, rel_err={b_rel:.4f}, score={b_score:.4f}")
        print(f"  wall={bwall:.2f}s, slot_ms={bslot}, n_calls={bcalls}, cost=${bcost:.6f}")
        profile["baseline"] = {
            "method": "sembench bigquery/Q5.sql verbatim",
            "sql": Q5_BASELINE_SQL.strip(),
            "result_count": b_count,
            "score": {"relative_error": b_rel, "score": b_score},
            "latency_breakdown": {"wall_s": bwall, "slot_ms": bslot},
            "cost_breakdown": {
                "n_llm_calls": bcalls,
                "n_llm_calls_method": "n_audio + n_image (both AI.IF per row)",
                "per_row_cost_usd_audio": per_row_audio,
                "per_row_cost_usd_image": per_row_image,
                "total_cost_usd": bcost,
            },
        }

    profile["cascade"] = {
        "method": "Composed F F via dase_cascade: 2× Cascade(MarginSignal, AlphaBand, AiIfVerifier) (audio+image) → AND-merge on car_id",
        "stage1_ctas": {
            "audio": {"sql": cres_a.verifier_result.ctas_sql, "latency_breakdown": {"wall_s": s1a_wall, "slot_ms": s1a_slot}, "cost_usd": 0.0},
            "image": {"sql": cres_i.verifier_result.ctas_sql, "latency_breakdown": {"wall_s": s1b_wall, "slot_ms": s1b_slot}, "cost_usd": 0.0},
        },
        "stage2_run": {
            "audio": {"sql": Q5_STAGE2_AUDIO_SQL.strip(), "result_yes_cars": sorted(list(audio_bq_yes_cars)),
                      "latency_breakdown": {"wall_s": s2a_wall, "slot_ms": s2a_slot},
                      "cost_breakdown": {"n_llm_calls": s2a_calls, "per_row_cost_usd": per_row_audio,
                                         "total_cost_usd": per_row_audio * s2a_calls}},
            "image": {"sql": Q5_STAGE2_IMAGE_SQL.strip(), "result_yes_cars": sorted(list(image_bq_yes_cars)),
                      "latency_breakdown": {"wall_s": s2b_wall, "slot_ms": s2b_slot},
                      "cost_breakdown": {"n_llm_calls": s2b_calls, "per_row_cost_usd": per_row_image,
                                         "total_cost_usd": per_row_image * s2b_calls}},
        },
        "result_count": cascade_count,
        "result_cars": sorted(list(cascade_cars)),
        "score": {"relative_error": c_rel, "score": c_score},
        "totals": {
            "wall_s": cascade_total_wall,
            "wall_breakdown_s": {
                "dase": t_dase_total,
                "bq_stage1_ctas": s1_wall,
                "bq_stage2_aiif": s2_wall,
            },
            "slot_ms_bq_total": cascade_total_slot,
            "cost_usd": cascade_cost,
            "n_llm_calls": s2_calls,
        },
    }

    paper_n_calls = round(PAPER_BQ_Q5["cost_usd"] / ((per_row_audio + per_row_image) / 2)) if (per_row_audio + per_row_image) else None
    profile["comparison"] = {
        "score": {"paper_BQ": PAPER_BQ_Q5["score"], "paper_DASE_NN": PAPER_DASE_NN_Q5["score"],
                  "ours_BQ": b_score, "ours_cascade": c_score,
                  "_baseline_source": "paper (aborted)" if SKIP_BASELINE else "ours"},
        "wall_s": {"paper_BQ": PAPER_BQ_Q5["latency_s"], "paper_DASE_NN": PAPER_DASE_NN_Q5["latency_s"],
                   "ours_BQ": bwall, "ours_cascade": cascade_total_wall},
        "slot_ms_bq": {"ours_BQ": bslot, "ours_cascade": cascade_total_slot},
        "cost_usd": {"paper_BQ": PAPER_BQ_Q5["cost_usd"], "paper_DASE_NN": PAPER_DASE_NN_Q5["cost_usd"],
                     "ours_BQ": bcost, "ours_cascade": cascade_cost},
        "n_llm_calls": {"paper_BQ": paper_n_calls, "paper_DASE_NN": 0,
                        "ours_BQ": bcalls, "ours_cascade": s2_calls},
    }

    write_profile(profile, PROFILE_PATH)

    print_summary(
        f"Cars Q5 (α_a={ALPHA_AUDIO},α_i={ALPHA_IMAGE})",
        columns=["paper BQ", "DASE+NN", "ours BQ", "ours cascade"],
        rows=[
            ("score",      [PAPER_BQ_Q5["score"], PAPER_DASE_NN_Q5["score"], b_score, c_score], ".2f"),
            ("wall (s)",   [PAPER_BQ_Q5["latency_s"], PAPER_DASE_NN_Q5["latency_s"], bwall, cascade_total_wall], ".2f"),
            ("cost ($)",   [PAPER_BQ_Q5["cost_usd"], PAPER_DASE_NN_Q5["cost_usd"], bcost, cascade_cost], ".4f"),
            ("#LLM calls", [None, 0, bcalls, s2_calls], "d"),
        ],
    )


if __name__ == "__main__":
    main()
