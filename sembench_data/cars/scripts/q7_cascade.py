#!/usr/bin/env -S python -u
"""
Cars Q7 cascade (v2) — composed F F F (audio-brakes + image-dented + text-electrical)
                       → UNION merge, via dase_cascade.

NL: Find cars that are either dented (image), have worn out brakes (audio),
    or have electrical system problems (text).
GT: 4034 cars.
Eval: F1 on car_id set.

Refactored to use dase_cascade. Operator (paper Table 3): F (composed F F F).
Three Cascades + UNION merge: cascade_cars = audio_yes ∪ image_yes ∪ text_yes.
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
    f1_set, build_profile, write_profile, print_summary,
)

CARS_DIR = os.path.abspath(os.path.join(_HERE, ".."))
CARS_PARQUET = os.path.join(CARS_DIR, "data", "cars.parquet")
AUDIO_PARQUET = os.path.join(CARS_DIR, "data", "audio_cars.parquet")
IMAGE_PARQUET = os.path.join(CARS_DIR, "data", "image_cars.parquet")
TEXT_PARQUET = os.path.join(CARS_DIR, "data", "text_complaints.parquet")
GT_CSV = os.path.join(CARS_DIR, "ground_truth", "Q7.csv")
PROFILE_PATH = os.path.join(CARS_DIR, "outputs", "Q7.json")
BASELINE_CACHE_PATH = os.path.join(CARS_DIR, "outputs", "Q7_baseline_cache.json")

PROJECT = os.environ.get("GCP_PROJECT", "")
BUCKET = f"{PROJECT}-cars_dataset"
DATASET = "cars_dataset"
STAGING_AUDIO = f"{DATASET}.q7_uncertain_audio_mm"
STAGING_IMAGE = f"{DATASET}.q7_uncertain_car_mm"
STAGING_TEXT = f"{DATASET}.q7_uncertain_complaints"

PROMPT_AUDIO = "You are given an audio recording of car diagnostics. Return true if the car from the recording has worn out brakes."
PROMPT_IMAGE = "You are given an image of a vehicle or its parts. Return true if car is dented."
PROMPT_TEXT_ELEC = "In the complaint, the car has some problems with electrical system / connected to electrical system. Complaint: %s."

POS_AUDIO = [
    "audio of worn out brakes",
    "squealing or grinding brake noise from a car",
    "audio of degraded or failing brake pads",
]
NEG_AUDIO = [
    "audio of healthy brakes with no noise",
    "engine sounds unrelated to brakes",
    "normal car operation audio",
]
POS_IMAGE = [
    "an image of a dented car",
    "a vehicle with body dents or panel damage",
    "a car with visible dents on its body",
]
NEG_IMAGE = [
    "an image of an undamaged car without dents",
    "a clean car with smooth body panels",
    "a vehicle in pristine condition",
]
POS_TEXT_ELEC = [
    "complaint about electrical system problem",
    "issue with electrical components or wiring",
    "complaint about car electrical malfunction",
]
NEG_TEXT_ELEC = [
    "complaint about mechanical issue not electrical",
    "engine, brake, or non-electrical problem",
    "issues unrelated to car electrical system",
]

ALPHA_AUDIO = 1.0
ALPHA_IMAGE = 0.5
ALPHA_TEXT = 0.5

PAPER_BQ_Q7 = {"score_f1": 0.45, "latency_s": 86.0, "cost_usd": 3.17}
PAPER_DASE_NN_Q7 = {"score_f1": 0.47, "latency_s": 1.3, "cost_usd": 1e-5}
SKIP_BASELINE = False


def trunc2(x):
    return f"{math.floor(x * 100) / 100:.2f}"


Q7_BASELINE_SQL = f"""
WITH sick_audio AS (
  SELECT DISTINCT p.car_id, year FROM {DATASET}.cars AS p
  JOIN {DATASET}.audio_mm AS a ON p.car_id = a.car_id
  WHERE AI.IF(('{PROMPT_AUDIO}', a.image),
    connection_id => 'us.connection', endpoint => 'gemini-2.5-flash')
),
sick_text AS (
  SELECT p.car_id, year FROM {DATASET}.cars AS p
  JOIN {DATASET}.complaints AS s ON p.car_id = s.car_id
  WHERE AI.IF(FORMAT('{PROMPT_TEXT_ELEC}', s.summary),
    connection_id => 'us.connection', endpoint => 'gemini-2.5-flash')
),
sick_image AS (
  SELECT p.car_id, year FROM {DATASET}.car_mm as x
  JOIN {DATASET}.cars AS p ON p.car_id = x.car_id
  WHERE AI.IF(('{PROMPT_IMAGE}', x.image),
    connection_id => 'us.connection', endpoint => 'gemini-2.5-flash')
)
SELECT sick_audio.car_id FROM sick_audio
UNION DISTINCT
SELECT sick_text.car_id FROM sick_text
UNION DISTINCT
SELECT sick_image.car_id FROM sick_image
"""

Q7_STAGE2_AUDIO_SQL = f"""
SELECT DISTINCT a.car_id AS id FROM {STAGING_AUDIO} a
WHERE AI.IF(('{PROMPT_AUDIO}', a.image), connection_id => 'us.connection', endpoint => 'gemini-2.5-flash')
"""
Q7_STAGE2_IMAGE_SQL = f"""
SELECT DISTINCT x.car_id AS id FROM {STAGING_IMAGE} x
WHERE AI.IF(('{PROMPT_IMAGE}', x.image), connection_id => 'us.connection', endpoint => 'gemini-2.5-flash')
"""
Q7_STAGE2_TEXT_SQL = f"""
SELECT DISTINCT s.car_id AS id FROM {STAGING_TEXT} s
WHERE AI.IF(FORMAT('{PROMPT_TEXT_ELEC}', s.summary), connection_id => 'us.connection', endpoint => 'gemini-2.5-flash')
"""


def make_audio_verifier():
    def make_staging(uris):
        items = ",".join(f"'{u}'" for u in uris)
        return f"""
        CREATE OR REPLACE TABLE {STAGING_AUDIO} AS
        SELECT m.audio_id, m.car_id, ot.ref AS image
        FROM {DATASET}.car_audio m JOIN {DATASET}.cars_audios ot ON ot.uri = m.audio_path
        WHERE m.audio_path IN UNNEST([{items}])
        """
    return AiIfVerifier(verify_sql=Q7_STAGE2_AUDIO_SQL, make_staging_sql=make_staging,
                        id_column="id", coerce_id=int)


def make_image_verifier():
    def make_staging(uris):
        items = ",".join(f"'{u}'" for u in uris)
        return f"""
        CREATE OR REPLACE TABLE {STAGING_IMAGE} AS
        SELECT m.image_id, m.car_id, ot.ref AS image
        FROM {DATASET}.car_images m JOIN {DATASET}.cars_images ot ON ot.uri = m.image_path
        WHERE m.image_path IN UNNEST([{items}])
        """
    return AiIfVerifier(verify_sql=Q7_STAGE2_IMAGE_SQL, make_staging_sql=make_staging,
                        id_column="id", coerce_id=int)


def make_text_verifier():
    def make_staging(cids):
        items = ",".join(str(int(c)) for c in cids)
        return f"""
        CREATE OR REPLACE TABLE {STAGING_TEXT} AS
        SELECT * FROM {DATASET}.complaints
        WHERE complaint_id IN UNNEST([{items}])
        """
    return AiIfVerifier(verify_sql=Q7_STAGE2_TEXT_SQL, make_staging_sql=make_staging,
                        id_column="id", coerce_id=int)


def per_row_cost_text_elec(client, sample_summaries, k=10):
    """Q7 text-electrical calibration uses verbatim FORMAT('...', @s_i)."""
    THINKING = "JSON '{\"generation_config\":{\"thinking_config\":{\"thinking_budget\":0}}}'"
    selects, params = [], []
    for i, s in enumerate(sample_summaries[:k]):
        selects.append(f"""
        SELECT AI.GENERATE_BOOL(
          FORMAT('{PROMPT_TEXT_ELEC}', @s_{i}),
          connection_id => 'us.connection',
          endpoint => 'gemini-2.5-flash',
          model_params => {THINKING}
        ) AS verdict""")
        params.append(bigquery.ScalarQueryParameter(f"s_{i}", "STRING", s))
    sql = " UNION ALL ".join(selects)
    cfg = bigquery.QueryJobConfig(query_parameters=params, use_query_cache=False)
    from dase_cascade.calibration import _sum_tokens, _to_cost, CalibrationResult
    t0 = time.time()
    df = client.query(sql, job_config=cfg).result().to_dataframe()
    elapsed = time.time() - t0
    p_other, p_audio, out, thoughts = _sum_tokens(df["verdict"])
    n = len(df)
    cost = _to_cost(p_other, p_audio, out, thoughts)
    return CalibrationResult(
        method="AI.GENERATE_BOOL (text-electrical) + thinking_budget=0",
        n_sample=n,
        tokens_total={"prompt_other": p_other, "prompt_audio": p_audio, "output": out, "thoughts": thoughts},
        sample_cost_usd=cost,
        per_row_cost_usd=cost / n if n else 0.0,
        elapsed_s=elapsed,
    )


def main():
    profile = build_profile(
        scenario="cars", query_id=7, scale_factor=19672,
        params={"alpha_audio": ALPHA_AUDIO, "alpha_image": ALPHA_IMAGE, "alpha_text": ALPHA_TEXT,
                "_note": "alpha_audio=1.0 (per-modality memory: audio caption noisy); alpha_image=alpha_text=0.5 (UNION error propagation per §2)"},
        cascade_form=(
            "Composed F F F via dase_cascade: 3× Cascade(MarginSignal, AlphaBand, AiIfVerifier) "
            "(audio-brakes + image-dented + text-electrical) → UNION DISTINCT car_id"
        ),
        extra={
            "operator": "F",
            "prompt": {"audio": PROMPT_AUDIO, "image": PROMPT_IMAGE, "text_electrical": PROMPT_TEXT_ELEC},
            "structural_filter": "",
            "dase_prompts": {
                "audio_brakes": {"positive": POS_AUDIO, "negative": NEG_AUDIO},
                "image_dented": {"positive": POS_IMAGE, "negative": NEG_IMAGE},
                "text_electrical": {"positive": POS_TEXT_ELEC, "negative": NEG_TEXT_ELEC},
            },
        },
    )

    print("Loading 3 modalities + GT...")
    t = time.time()
    audio = pd.read_parquet(AUDIO_PARQUET)
    img = pd.read_parquet(IMAGE_PARQUET)
    text = pd.read_parquet(TEXT_PARQUET)
    audio_scope = audio.reset_index(drop=True)
    image_scope = img.reset_index(drop=True)
    text_scope = text.reset_index(drop=True)
    gt_cars = set(int(x) for x in pd.read_csv(GT_CSV)["car_id"])
    t_load = time.time() - t

    print(f"  audio={len(audio_scope)}, image={len(image_scope)}, text={len(text_scope)}; GT cars={len(gt_cars)}")
    profile["data"] = {
        "n_audio": len(audio_scope), "n_image": len(image_scope), "n_text": len(text_scope),
        "n_gt_cars": len(gt_cars),
    }

    audio_emb = np.stack(audio_scope["embedding"].tolist()).astype(np.float32)
    image_emb = np.stack(image_scope["embedding"].tolist()).astype(np.float32)
    text_emb = np.stack(text_scope["embedding"].tolist()).astype(np.float32)
    audio_uris = [f"gs://{BUCKET}/car_audios/{os.path.basename(p)}" for p in audio_scope["audio_path"]]
    image_uris = [f"gs://{BUCKET}/car_images/{os.path.basename(p)}" for p in image_scope["image_path"]]
    text_cids = text_scope["complaint_id"].astype(int).tolist()
    car_by_audio_uri = dict(zip(audio_uris, audio_scope["car_id"].astype(int)))
    car_by_image_uri = dict(zip(image_uris, image_scope["car_id"].astype(int)))
    car_by_text_cid = dict(zip(text_cids, text_scope["car_id"].astype(int)))

    client = bq_client(PROJECT)

    print("\n=== Calibration ===")
    cal_audio = per_row_cost(client, PROMPT_AUDIO,
                             sample_uris=audio_uris[:5] if audio_uris else [],
                             ext_table=f"{DATASET}.cars_audios",
                             method_label="AI.GENERATE_BOOL (audio-brakes) + thinking_budget=0", k=5)
    per_row_audio = cal_audio.per_row_cost_usd
    print(f"  audio per_row=${per_row_audio:.6f}")
    cal_image = per_row_cost(client, PROMPT_IMAGE,
                             sample_uris=image_uris[:5] if image_uris else [],
                             ext_table=f"{DATASET}.cars_images",
                             method_label="AI.GENERATE_BOOL (image-dented) + thinking_budget=0", k=5)
    per_row_image = cal_image.per_row_cost_usd
    print(f"  image per_row=${per_row_image:.6f}")
    text_sample = [str(text_scope.iloc[i]["summary"]) for i in range(min(10, len(text_scope)))]
    cal_text = per_row_cost_text_elec(client, text_sample, k=10)
    per_row_text = cal_text.per_row_cost_usd
    print(f"  text per_row=${per_row_text:.6f}")
    profile["calibration"] = {"audio": cal_audio.to_dict(), "image": cal_image.to_dict(), "text": cal_text.to_dict()}

    # ── 3 Cascades ──
    print("\n=== Cascade audio ===")
    cres_a = Cascade(embeddings=audio_emb, ids=audio_uris,
                     signal=MarginSignal(positive_prompts=POS_AUDIO, negative_prompts=NEG_AUDIO),
                     band=AlphaBand(alpha=ALPHA_AUDIO),
                     verifier=make_audio_verifier()).run(client, per_row_audio)
    audio_dase_pos = {car_by_audio_uri[u] for u in cres_a.confident_pos_ids}
    audio_bq_yes = set(cres_a.verifier_result.positive_ids)
    audio_yes = audio_dase_pos | audio_bq_yes

    print("\n=== Cascade image ===")
    cres_i = Cascade(embeddings=image_emb, ids=image_uris,
                     signal=MarginSignal(positive_prompts=POS_IMAGE, negative_prompts=NEG_IMAGE),
                     band=AlphaBand(alpha=ALPHA_IMAGE),
                     verifier=make_image_verifier()).run(client, per_row_image)
    image_dase_pos = {car_by_image_uri[u] for u in cres_i.confident_pos_ids}
    image_bq_yes = set(cres_i.verifier_result.positive_ids)
    image_yes = image_dase_pos | image_bq_yes

    print("\n=== Cascade text-electrical ===")
    cres_t = Cascade(embeddings=text_emb, ids=text_cids,
                     signal=MarginSignal(positive_prompts=POS_TEXT_ELEC, negative_prompts=NEG_TEXT_ELEC),
                     band=AlphaBand(alpha=ALPHA_TEXT),
                     verifier=make_text_verifier()).run(client, per_row_text)
    text_dase_pos = {car_by_text_cid[c] for c in cres_t.confident_pos_ids}
    text_bq_yes = set(cres_t.verifier_result.positive_ids)
    text_yes = text_dase_pos | text_bq_yes

    print(f"\n  [audio] α={ALPHA_AUDIO}: pos={len(audio_dase_pos)} uncertain={len(cres_a.uncertain_ids)} bq_yes={len(audio_bq_yes)}")
    print(f"  [image] α={ALPHA_IMAGE}: pos={len(image_dase_pos)} uncertain={len(cres_i.uncertain_ids)} bq_yes={len(image_bq_yes)}")
    print(f"  [text]  α={ALPHA_TEXT}: pos={len(text_dase_pos)} uncertain={len(cres_t.uncertain_ids)} bq_yes={len(text_bq_yes)}")

    s2a_calls = len(cres_a.uncertain_ids); s2b_calls = len(cres_i.uncertain_ids); s2c_calls = len(cres_t.uncertain_ids)
    s2_calls = s2a_calls + s2b_calls + s2c_calls
    cascade_cost = per_row_audio * s2a_calls + per_row_image * s2b_calls + per_row_text * s2c_calls
    s1a_wall, s1a_slot = cres_a.verifier_result.ctas_wall_s, cres_a.verifier_result.ctas_slot_ms
    s1b_wall, s1b_slot = cres_i.verifier_result.ctas_wall_s, cres_i.verifier_result.ctas_slot_ms
    s1c_wall, s1c_slot = cres_t.verifier_result.ctas_wall_s, cres_t.verifier_result.ctas_slot_ms
    s2a_wall, s2a_slot = cres_a.verifier_result.wall_s, cres_a.verifier_result.slot_ms
    s2b_wall, s2b_slot = cres_i.verifier_result.wall_s, cres_i.verifier_result.slot_ms
    s2c_wall, s2c_slot = cres_t.verifier_result.wall_s, cres_t.verifier_result.slot_ms
    s1_wall = s1a_wall + s1b_wall + s1c_wall; s1_slot = s1a_slot + s1b_slot + s1c_slot
    s2_wall = s2a_wall + s2b_wall + s2c_wall; s2_slot = s2a_slot + s2b_slot + s2c_slot
    t_dase = (cres_a.timings_s.get("signal_compute", 0.0) + cres_a.timings_s.get("band_partition", 0.0)
              + cres_i.timings_s.get("signal_compute", 0.0) + cres_i.timings_s.get("band_partition", 0.0)
              + cres_t.timings_s.get("signal_compute", 0.0) + cres_t.timings_s.get("band_partition", 0.0))
    t_dase_total = t_load + t_dase

    profile["dase_breakdown_s"] = {"data_load": t_load, "dase_compute": t_dase, "total": t_dase_total}
    profile["dase_partition"] = {
        "audio": {"n_confident_pos_cars": len(audio_dase_pos), "n_confident_neg": len(cres_a.confident_neg_ids), "n_uncertain": len(cres_a.uncertain_ids)},
        "image": {"n_confident_pos_cars": len(image_dase_pos), "n_confident_neg": len(cres_i.confident_neg_ids), "n_uncertain": len(cres_i.uncertain_ids)},
        "text":  {"n_confident_pos_cars": len(text_dase_pos),  "n_confident_neg": len(cres_t.confident_neg_ids), "n_uncertain": len(cres_t.uncertain_ids)},
    }

    # ── Baseline (cached or run) ──
    cached_baseline = None
    if os.path.exists(BASELINE_CACHE_PATH):
        try:
            cached_baseline = json.load(open(BASELINE_CACHE_PATH))
        except Exception:
            cached_baseline = None

    if SKIP_BASELINE:
        print(f"\n=== Baseline ABORTED — paper Table 4(e) ===")
        b_p = b_r = None
        b_f1 = PAPER_BQ_Q7["score_f1"]; bwall = PAPER_BQ_Q7["latency_s"]; bslot = None
        bcost = PAPER_BQ_Q7["cost_usd"]; bres_cars = set()
        bcalls = round(bcost / per_row_image) if per_row_image else (len(audio_scope) + len(image_scope) + len(text_scope))
        profile["baseline"] = {
            "_status": "aborted",
            "score": {"f1_score": b_f1, "_source": "paper Table 4(e)"},
            "latency_breakdown": {"wall_s": bwall, "slot_ms": None, "_source": "paper"},
            "cost_breakdown": {"n_llm_calls": bcalls, "total_cost_usd": bcost, "_source": "paper"},
            "method": "Q7.sql verbatim — NOT EXECUTED", "sql": Q7_BASELINE_SQL.strip(),
        }
    elif cached_baseline is not None:
        bres_cars = set(int(x) for x in cached_baseline["result_ids"])
        bwall = float(cached_baseline["wall_s"])
        bslot = int(cached_baseline.get("slot_ms") or 0)
        b_p, b_r, b_f1 = f1_set(bres_cars, gt_cars)
        bcalls = len(audio_scope) + len(image_scope) + len(text_scope)
        bcost = (per_row_audio * len(audio_scope) + per_row_image * len(image_scope) + per_row_text * len(text_scope))
        print(f"\n=== Baseline (cached) ===")
        print(f"  returned {len(bres_cars)} cars; P={b_p:.4f} R={b_r:.4f} F1={b_f1:.4f}")
        print(f"  wall={bwall:.2f}s (cached), slot_ms={bslot}, n_calls={bcalls}, cost=${bcost:.6f}")
        profile["baseline"] = {
            "method": "Q7.sql verbatim (cached from prior run)",
            "sql": Q7_BASELINE_SQL.strip(),
            "result_ids": sorted(list(bres_cars)),
            "score": {"precision": b_p, "recall": b_r, "f1_score": b_f1},
            "latency_breakdown": {"wall_s": bwall, "slot_ms": bslot, "_status": "cached"},
            "cost_breakdown": {"n_llm_calls": bcalls,
                               "n_llm_calls_method": "n_audio + n_image + n_text (3 sick_* CTEs)",
                               "per_row_cost_usd_audio": per_row_audio,
                               "per_row_cost_usd_image": per_row_image,
                               "per_row_cost_usd_text": per_row_text,
                               "total_cost_usd": bcost},
        }
    else:
        print("\n=== Baseline (Q7.sql verbatim) ===")
        bdf, bwall, bslot, _ = run_query(client, Q7_BASELINE_SQL)
        bres_cars = set(int(x) for x in bdf["car_id"])
        b_p, b_r, b_f1 = f1_set(bres_cars, gt_cars)
        bcalls = len(audio_scope) + len(image_scope) + len(text_scope)
        bcost = (per_row_audio * len(audio_scope) + per_row_image * len(image_scope) + per_row_text * len(text_scope))
        print(f"  returned {len(bres_cars)} cars; P={b_p:.4f} R={b_r:.4f} F1={b_f1:.4f}")
        print(f"  wall={bwall:.2f}s, slot_ms={bslot}, n_calls={bcalls}, cost=${bcost:.6f}")
        os.makedirs(os.path.dirname(BASELINE_CACHE_PATH), exist_ok=True)
        with open(BASELINE_CACHE_PATH, "w") as f:
            json.dump({"result_ids": sorted(list(bres_cars)), "wall_s": bwall, "slot_ms": bslot}, f)
        print(f"  baseline cache saved to {BASELINE_CACHE_PATH}")
        profile["baseline"] = {
            "method": "Q7.sql verbatim",
            "sql": Q7_BASELINE_SQL.strip(),
            "result_ids": sorted(list(bres_cars)),
            "score": {"precision": b_p, "recall": b_r, "f1_score": b_f1},
            "latency_breakdown": {"wall_s": bwall, "slot_ms": bslot},
            "cost_breakdown": {"n_llm_calls": bcalls,
                               "n_llm_calls_method": "n_audio + n_image + n_text (3 sick_* CTEs)",
                               "per_row_cost_usd_audio": per_row_audio,
                               "per_row_cost_usd_image": per_row_image,
                               "per_row_cost_usd_text": per_row_text,
                               "total_cost_usd": bcost},
        }

    # ── UNION merge ──
    cascade_cars = audio_yes | image_yes | text_yes
    cp, cr, c_f1 = f1_set(cascade_cars, gt_cars)
    print(f"\n  audio_yes_cars={len(audio_yes)}, image_yes_cars={len(image_yes)}, text_yes_cars={len(text_yes)}")
    print(f"  cascade UNION cars: {len(cascade_cars)}, GT={len(gt_cars)}; P={cp:.4f} R={cr:.4f} F1={c_f1:.4f}")

    cascade_total_wall = t_dase_total + s1_wall + s2_wall
    cascade_total_slot = s1_slot + s2_slot

    profile["cascade"] = {
        "method": "3 independent dase_cascade.Cascades (audio+image+text-electrical) → UNION DISTINCT car_id",
        "stage1_ctas": {
            "audio": {"sql": cres_a.verifier_result.ctas_sql, "latency_breakdown": {"wall_s": s1a_wall, "slot_ms": s1a_slot}, "cost_usd": 0.0},
            "image": {"sql": cres_i.verifier_result.ctas_sql, "latency_breakdown": {"wall_s": s1b_wall, "slot_ms": s1b_slot}, "cost_usd": 0.0},
            "text":  {"sql": cres_t.verifier_result.ctas_sql, "latency_breakdown": {"wall_s": s1c_wall, "slot_ms": s1c_slot}, "cost_usd": 0.0},
        },
        "stage2_run": {
            "audio": {"sql": Q7_STAGE2_AUDIO_SQL.strip(), "result_yes_cars": sorted(list(audio_bq_yes)),
                      "latency_breakdown": {"wall_s": s2a_wall, "slot_ms": s2a_slot},
                      "cost_breakdown": {"n_llm_calls": s2a_calls, "per_row_cost_usd": per_row_audio,
                                         "total_cost_usd": per_row_audio * s2a_calls}},
            "image": {"sql": Q7_STAGE2_IMAGE_SQL.strip(), "result_yes_cars": sorted(list(image_bq_yes)),
                      "latency_breakdown": {"wall_s": s2b_wall, "slot_ms": s2b_slot},
                      "cost_breakdown": {"n_llm_calls": s2b_calls, "per_row_cost_usd": per_row_image,
                                         "total_cost_usd": per_row_image * s2b_calls}},
            "text":  {"sql": Q7_STAGE2_TEXT_SQL.strip(), "result_yes_cars": sorted(list(text_bq_yes)),
                      "latency_breakdown": {"wall_s": s2c_wall, "slot_ms": s2c_slot},
                      "cost_breakdown": {"n_llm_calls": s2c_calls, "per_row_cost_usd": per_row_text,
                                         "total_cost_usd": per_row_text * s2c_calls}},
        },
        "result_ids": sorted(list(cascade_cars)),
        "score": {"precision": cp, "recall": cr, "f1_score": c_f1},
        "totals": {
            "wall_s": cascade_total_wall,
            "wall_breakdown_s": {"dase": t_dase_total, "bq_stage1_ctas": s1_wall, "bq_stage2_aiif": s2_wall},
            "slot_ms_bq_total": cascade_total_slot,
            "cost_usd": cascade_cost,
            "n_llm_calls": s2_calls,
        },
    }

    paper_n_calls = round(PAPER_BQ_Q7["cost_usd"] / per_row_image) if per_row_image else None
    profile["comparison"] = {
        "score": {"paper_BQ": PAPER_BQ_Q7["score_f1"], "paper_DASE_NN": PAPER_DASE_NN_Q7["score_f1"],
                  "ours_BQ": b_f1, "ours_cascade": c_f1,
                  "_baseline_source": "paper (aborted)" if SKIP_BASELINE else "ours"},
        "wall_s": {"paper_BQ": PAPER_BQ_Q7["latency_s"], "paper_DASE_NN": PAPER_DASE_NN_Q7["latency_s"],
                   "ours_BQ": bwall, "ours_cascade": cascade_total_wall},
        "slot_ms_bq": {"ours_BQ": bslot, "ours_cascade": cascade_total_slot},
        "cost_usd": {"paper_BQ": PAPER_BQ_Q7["cost_usd"], "paper_DASE_NN": PAPER_DASE_NN_Q7["cost_usd"],
                     "ours_BQ": bcost, "ours_cascade": cascade_cost},
        "n_llm_calls": {"paper_BQ": paper_n_calls, "paper_DASE_NN": 0, "ours_BQ": bcalls, "ours_cascade": s2_calls},
    }

    write_profile(profile, PROFILE_PATH)

    print_summary(
        "Cars Q7 (composed F F F + UNION)",
        columns=["paper BQ", "DASE+NN", "ours BQ", "ours cascade"],
        rows=[
            ("score (F1)", [PAPER_BQ_Q7["score_f1"], PAPER_DASE_NN_Q7["score_f1"], b_f1, c_f1], ".2f"),
            ("wall (s)",   [PAPER_BQ_Q7["latency_s"], PAPER_DASE_NN_Q7["latency_s"], bwall, cascade_total_wall], ".2f"),
            ("cost ($)",   [PAPER_BQ_Q7["cost_usd"], PAPER_DASE_NN_Q7["cost_usd"], bcost, cascade_cost], ".4f"),
            ("#LLM calls", [None, 0, bcalls, s2_calls], "d"),
        ],
    )


if __name__ == "__main__":
    main()
