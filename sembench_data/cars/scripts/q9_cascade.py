#!/usr/bin/env -S python -u
"""
Cars Q9 cascade (v2) — drop-only F (single AI.IF on image+audio pair BLOBs)
                       via dase_cascade primitives.

NL: Find cars that are torn according to images and have bad ignition
    according to audio.
GT: empty (F1=1.0 for sys=∅ AND GT=∅).
Eval: F1 on car_id set.

Refactored to use dase_cascade primitives directly (Signal + Band + Verifier),
because the standard Cascade.run() only forwards uncertain to the verifier;
Q9 is "drop-only": image confident_neg pairs are dropped, but image
confident_pos AND uncertain pairs both go to BQ. We orchestrate manually:

  scores = MarginSignal(image prompts).compute(image_emb)
  part   = AlphaBand(0.5).partition(scores)
  kept   = pairs whose image idx ∈ part.confident_pos ∪ part.uncertain
  result = AiIfVerifier(verify_sql_template=lambda ids: ... IN(...) AI.IF(image,audio))
           .verify(client, kept_car_ids, per_row_pair)
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
    MarginSignal, AlphaBand, AiIfVerifier,
    bq_client, run_query,
    f1_set, build_profile, write_profile, print_summary,
)
from dase_cascade.calibration import _sum_tokens, _to_cost, CalibrationResult

CARS_DIR = os.path.abspath(os.path.join(_HERE, ".."))
CARS_PARQUET = os.path.join(CARS_DIR, "data", "cars.parquet")
AUDIO_PARQUET = os.path.join(CARS_DIR, "data", "audio_cars.parquet")
IMAGE_PARQUET = os.path.join(CARS_DIR, "data", "image_cars.parquet")
GT_CSV = os.path.join(CARS_DIR, "ground_truth", "Q9.csv")
PROFILE_PATH = os.path.join(CARS_DIR, "outputs", "Q9.json")
BASELINE_CACHE_PATH = os.path.join(CARS_DIR, "outputs", "Q9_baseline_cache.json")

PROJECT = os.environ.get("GCP_PROJECT", "")
BUCKET = f"{PROJECT}-cars_dataset"
DATASET = "cars_dataset"

PROMPT = "You are given an image of a vehicle and an audio recording of car diagnostics. Return true if car is torn according to image and has bad ignition according to audio."

POS_IMAGE = [
    "a torn car",
    "vehicle with torn or ripped material",
    "image of a car with tear damage",
]
NEG_IMAGE = [
    "an undamaged car",
    "a vehicle without tears or rips",
    "intact car body without damage",
]
POS_AUDIO = [
    "audio of a car with bad ignition",
    "ignition starting problem in a car",
    "engine fails to ignite",
]
NEG_AUDIO = [
    "audio of normal car ignition",
    "engine starts normally",
    "healthy engine startup",
]

ALPHA_IMAGE = 0.5
PAPER_BQ_Q9 = {"score_f1": 0.00, "latency_s": 13.6, "cost_usd": 0.03}
PAPER_DASE_NN_Q9 = {"score_f1": 1.00, "latency_s": 0.7, "cost_usd": 5e-6}
SKIP_BASELINE = False


def trunc2(x):
    return f"{math.floor(x * 100) / 100:.2f}"


Q9_BASELINE_SQL = f"""
SELECT DISTINCT p.car_id
FROM {DATASET}.car_mm as x, {DATASET}.cars AS p, {DATASET}.audio_mm as a
WHERE p.car_id = x.car_id AND p.car_id = a.car_id
  AND AI.IF(
    ('{PROMPT}', x.image, a.image),
    connection_id => 'us.connection',
    endpoint => 'gemini-2.5-flash')
"""


def make_q9_verifier():
    """car_id IN (kept) AI.IF(image, audio) on the (cars ⨝ image ⨝ audio) join."""
    def verify_sql_template(car_ids):
        id_list = ",".join(str(int(c)) for c in car_ids)
        return f"""
        SELECT DISTINCT p.car_id AS id
        FROM {DATASET}.car_mm AS x, {DATASET}.cars AS p, {DATASET}.audio_mm AS a
        WHERE p.car_id = x.car_id AND p.car_id = a.car_id
          AND p.car_id IN ({id_list})
          AND AI.IF(
            ('{PROMPT}', x.image, a.image),
            connection_id => 'us.connection',
            endpoint => 'gemini-2.5-flash')
        """
    return AiIfVerifier(verify_sql_template=verify_sql_template,
                        id_column="id", coerce_id=int)


def per_row_cost_pair(client, sample_pairs, k=5):
    """Custom pair calibration with (image_ref, audio_ref) inputs."""
    THINKING = "JSON '{\"generation_config\":{\"thinking_config\":{\"thinking_budget\":0}}}'"
    selects, params = [], []
    for i, (img_uri, aud_uri) in enumerate(sample_pairs[:k]):
        params.append(bigquery.ScalarQueryParameter(f"img_{i}", "STRING", img_uri))
        params.append(bigquery.ScalarQueryParameter(f"aud_{i}", "STRING", aud_uri))
        selects.append(f"""
        SELECT AI.GENERATE_BOOL(
          ('{PROMPT}', img.ref, aud.ref),
          connection_id => 'us.connection',
          endpoint => 'gemini-2.5-flash',
          model_params => {THINKING}
        ) AS verdict
        FROM {DATASET}.cars_images img, {DATASET}.cars_audios aud
        WHERE img.uri = @img_{i} AND aud.uri = @aud_{i}""")
    sql = " UNION ALL ".join(selects)
    cfg = bigquery.QueryJobConfig(query_parameters=params, use_query_cache=False)
    t0 = time.time()
    df = client.query(sql, job_config=cfg).result().to_dataframe()
    elapsed = time.time() - t0
    p_other, p_audio, out, thoughts = _sum_tokens(df["verdict"])
    n = len(df)
    cost = _to_cost(p_other, p_audio, out, thoughts)
    return CalibrationResult(
        method="AI.GENERATE_BOOL multimodal pair (image, audio, Q9 prompt) + thinking_budget=0",
        n_sample=n,
        tokens_total={"prompt_other": p_other, "prompt_audio": p_audio, "output": out, "thoughts": thoughts},
        sample_cost_usd=cost,
        per_row_cost_usd=cost / n if n else 0.0,
        elapsed_s=elapsed,
    )


def main():
    profile = build_profile(
        scenario="cars", query_id=9, scale_factor=19672,
        prompt=PROMPT,
        params={"alpha_image": ALPHA_IMAGE, "alpha_audio": 1.0,
                "_note": "drop-only: image confident_neg dropped; audio fully BQ; do not trust dase POS for AND"},
        cascade_form=(
            "Drop-only F via dase_cascade primitives: MarginSignal(image)+AlphaBand → "
            "drop image confident_neg pairs → AiIfVerifier on remaining (cars⨝image⨝audio AI.IF(image, audio))"
        ),
        extra={
            "operator": "F",
            "structural_filter": "p.car_id = x.car_id AND p.car_id = a.car_id",
            "dase_prompts": {
                "image_torn": {"positive": POS_IMAGE, "negative": NEG_IMAGE},
                "audio_bad_ignition": {"positive": POS_AUDIO, "negative": NEG_AUDIO},
            },
        },
    )

    print("Loading cars + audio + image + GT (cars ⨝ image ⨝ audio)...")
    t = time.time()
    cars = pd.read_parquet(CARS_PARQUET)
    audio = pd.read_parquet(AUDIO_PARQUET)
    img = pd.read_parquet(IMAGE_PARQUET)
    common = set(audio["car_id"]) & set(img["car_id"])
    audio_scope = audio[audio["car_id"].isin(common)].drop_duplicates("car_id").reset_index(drop=True)
    image_scope = img[img["car_id"].isin(common)].drop_duplicates("car_id").reset_index(drop=True)
    image_scope = image_scope.set_index("car_id").loc[sorted(common)].reset_index()
    audio_scope = audio_scope.set_index("car_id").loc[sorted(common)].reset_index()
    n_pairs = len(common)
    gt_df = pd.read_csv(GT_CSV)
    gt_cars = set(int(x) for x in gt_df["car_id"]) if "car_id" in gt_df.columns and len(gt_df) > 0 else set()
    t_load = time.time() - t

    print(f"  pairs: {n_pairs}; GT cars: {len(gt_cars)} (empty expected)")
    profile["data"] = {"n_pairs": n_pairs, "n_gt_cars": len(gt_cars)}

    img_emb = np.stack(image_scope["embedding"].tolist()).astype(np.float32)

    # ── Manual Signal + Band (drop-only: keep POS ∪ uncertain) ──
    t = time.time()
    signal = MarginSignal(positive_prompts=POS_IMAGE, negative_prompts=NEG_IMAGE)
    image_margins = signal.compute(img_emb)
    band = AlphaBand(alpha=ALPHA_IMAGE)
    part = band.partition(image_margins)
    t_dase = time.time() - t

    image_pos_idx = set(part.confident_pos.tolist())
    image_neg_idx = set(part.confident_neg.tolist())
    uncertain_idx = set(part.uncertain.tolist())

    kept_idx = sorted(image_pos_idx | uncertain_idx)
    kept_car_ids = [int(image_scope.iloc[i]["car_id"]) for i in kept_idx]

    print(f"  alpha_image={ALPHA_IMAGE}: image_pos={len(image_pos_idx)}, image_neg={len(image_neg_idx)}, uncertain={len(uncertain_idx)}")
    print(f"  drop-only: kept {len(kept_car_ids)}/{n_pairs} pairs for BQ (dropped {len(image_neg_idx)} image_confident_neg)")

    profile["dase_breakdown_s"] = {"data_load": t_load, "dase_compute": t_dase, "total": t_load + t_dase}
    profile["dase_partition"] = {
        "image": {
            "n_confident_pos": len(image_pos_idx),
            "n_confident_neg": len(image_neg_idx),
            "n_uncertain": len(uncertain_idx),
        },
        "kept_for_bq": len(kept_car_ids),
    }

    client = bq_client(PROJECT)

    # ── Calibration: pair multimodal ──
    print("\n=== Per-row cost calibration (k=5 multimodal pair) ===")
    sample_pairs = []
    for i in range(min(5, n_pairs)):
        img_uri = f"gs://{BUCKET}/car_images/{os.path.basename(image_scope.iloc[i]['image_path'])}"
        aud_uri = f"gs://{BUCKET}/car_audios/{os.path.basename(audio_scope.iloc[i]['audio_path'])}"
        sample_pairs.append((img_uri, aud_uri))
    cal = per_row_cost_pair(client, sample_pairs, k=5)
    per_row = cal.per_row_cost_usd
    print(f"  per_row=${per_row:.6f}, elapsed={cal.elapsed_s:.1f}s")
    profile["calibration"] = cal.to_dict()

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
        b_f1 = PAPER_BQ_Q9["score_f1"]; bwall = PAPER_BQ_Q9["latency_s"]; bslot = None
        bcost = PAPER_BQ_Q9["cost_usd"]; bcalls = round(bcost / per_row) if per_row else n_pairs
        bres_cars = set()
        profile["baseline"] = {
            "_status": "aborted",
            "score": {"f1_score": b_f1, "_source": "paper Table 4(e)"},
            "latency_breakdown": {"wall_s": bwall, "slot_ms": None, "_source": "paper"},
            "cost_breakdown": {"n_llm_calls": bcalls, "total_cost_usd": bcost, "_source": "paper"},
            "method": "Q9.sql verbatim — NOT EXECUTED", "sql": Q9_BASELINE_SQL.strip(),
        }
    elif cached_baseline is not None:
        bres_cars = set(int(x) for x in cached_baseline["result_ids"])
        bwall = float(cached_baseline["wall_s"])
        bslot = int(cached_baseline.get("slot_ms") or 0)
        b_p, b_r, b_f1 = f1_set(bres_cars, gt_cars)
        bcalls = n_pairs
        bcost = per_row * bcalls
        print(f"\n=== Baseline (cached) ===")
        print(f"  returned {len(bres_cars)} cars; P={b_p:.4f} R={b_r:.4f} F1={b_f1:.4f}")
        print(f"  wall={bwall:.2f}s (cached), slot_ms={bslot}, n_calls={bcalls}, cost=${bcost:.6f}")
        profile["baseline"] = {
            "method": "Q9.sql verbatim (cached)", "sql": Q9_BASELINE_SQL.strip(),
            "result_ids": sorted(list(bres_cars)),
            "score": {"precision": b_p, "recall": b_r, "f1_score": b_f1},
            "latency_breakdown": {"wall_s": bwall, "slot_ms": bslot, "_status": "cached"},
            "cost_breakdown": {"n_llm_calls": bcalls, "n_llm_calls_method": "scope size (pair AI.IF)",
                               "per_row_cost_usd": per_row, "total_cost_usd": bcost},
        }
    else:
        print("\n=== Baseline (Q9.sql verbatim, multimodal pair AI.IF on full scope) ===")
        bdf, bwall, bslot, _ = run_query(client, Q9_BASELINE_SQL)
        bres_cars = set(int(x) for x in bdf["car_id"])
        b_p, b_r, b_f1 = f1_set(bres_cars, gt_cars)
        bcalls = n_pairs
        bcost = per_row * bcalls
        print(f"  returned {len(bres_cars)} cars; P={b_p:.4f} R={b_r:.4f} F1={b_f1:.4f}")
        print(f"  wall={bwall:.2f}s, slot_ms={bslot}, n_calls={bcalls}, cost=${bcost:.6f}")
        os.makedirs(os.path.dirname(BASELINE_CACHE_PATH), exist_ok=True)
        with open(BASELINE_CACHE_PATH, "w") as f:
            json.dump({"result_ids": sorted(list(bres_cars)), "wall_s": bwall, "slot_ms": bslot}, f)
        print(f"  baseline cache saved to {BASELINE_CACHE_PATH}")
        profile["baseline"] = {
            "method": "Q9.sql verbatim", "sql": Q9_BASELINE_SQL.strip(),
            "result_ids": sorted(list(bres_cars)),
            "score": {"precision": b_p, "recall": b_r, "f1_score": b_f1},
            "latency_breakdown": {"wall_s": bwall, "slot_ms": bslot},
            "cost_breakdown": {"n_llm_calls": bcalls, "n_llm_calls_method": "scope size (pair AI.IF)",
                               "per_row_cost_usd": per_row, "total_cost_usd": bcost},
        }

    # ── Cascade: AiIfVerifier on kept pairs ──
    print(f"\n=== Cascade: BQ AI.IF(image, audio) on {len(kept_car_ids)} kept pairs ===")
    verifier = make_q9_verifier()
    if kept_car_ids:
        vres = verifier.verify(client, kept_car_ids, per_row)
    else:
        from dase_cascade import VerifierResult
        vres = VerifierResult(positive_ids=set())
    cascade_cars = set(vres.positive_ids)
    s2_calls = len(kept_car_ids)
    cascade_cost = vres.cost_usd
    cwall = vres.wall_s
    cslot = vres.slot_ms
    cp, cr, c_f1 = f1_set(cascade_cars, gt_cars)
    print(f"  cascade returned {len(cascade_cars)} cars; P={cp:.4f} R={cr:.4f} F1={c_f1:.4f}")
    print(f"  wall={cwall:.2f}s, slot_ms={cslot}, n_calls={s2_calls}, cost=${cascade_cost:.6f}")

    cascade_total_wall = profile["dase_breakdown_s"]["total"] + cwall

    profile["cascade"] = {
        "method": "Drop-only F via dase_cascade primitives: MarginSignal+AlphaBand drop image_neg → AiIfVerifier(IN(kept)) on cars⨝image⨝audio AI.IF(image, audio)",
        "stage1_ctas": {"latency_breakdown": {"wall_s": 0.0, "slot_ms": 0}, "cost_usd": 0.0,
                        "_note": "no staging; IN-list passed directly"},
        "stage2_run": {
            "sql": vres.sql, "result_yes_cars": sorted(list(cascade_cars)),
            "latency_breakdown": {"wall_s": cwall, "slot_ms": cslot},
            "cost_breakdown": {"n_llm_calls": s2_calls, "per_row_cost_usd": per_row, "total_cost_usd": cascade_cost},
        },
        "result_ids": sorted(list(cascade_cars)),
        "score": {"precision": cp, "recall": cr, "f1_score": c_f1},
        "totals": {
            "wall_s": cascade_total_wall,
            "wall_breakdown_s": {"dase": profile["dase_breakdown_s"]["total"],
                                 "bq_stage1_ctas": 0.0, "bq_stage2_aiif": cwall},
            "slot_ms_bq_total": cslot, "cost_usd": cascade_cost, "n_llm_calls": s2_calls,
        },
    }

    paper_n_calls = round(PAPER_BQ_Q9["cost_usd"] / per_row) if per_row else None
    profile["comparison"] = {
        "score": {"paper_BQ": PAPER_BQ_Q9["score_f1"], "paper_DASE_NN": PAPER_DASE_NN_Q9["score_f1"],
                  "ours_BQ": b_f1, "ours_cascade": c_f1,
                  "_baseline_source": "paper (aborted)" if SKIP_BASELINE else "ours"},
        "wall_s": {"paper_BQ": PAPER_BQ_Q9["latency_s"], "paper_DASE_NN": PAPER_DASE_NN_Q9["latency_s"],
                   "ours_BQ": bwall, "ours_cascade": cascade_total_wall},
        "slot_ms_bq": {"ours_BQ": bslot, "ours_cascade": cslot},
        "cost_usd": {"paper_BQ": PAPER_BQ_Q9["cost_usd"], "paper_DASE_NN": PAPER_DASE_NN_Q9["cost_usd"],
                     "ours_BQ": bcost, "ours_cascade": cascade_cost},
        "n_llm_calls": {"paper_BQ": paper_n_calls, "paper_DASE_NN": 0,
                        "ours_BQ": bcalls, "ours_cascade": s2_calls},
    }

    write_profile(profile, PROFILE_PATH)

    print_summary(
        "Cars Q9 (drop-only F)",
        columns=["paper BQ", "DASE+NN", "ours BQ", "ours cascade"],
        rows=[
            ("score (F1)", [PAPER_BQ_Q9["score_f1"], PAPER_DASE_NN_Q9["score_f1"], b_f1, c_f1], ".2f"),
            ("wall (s)",   [PAPER_BQ_Q9["latency_s"], PAPER_DASE_NN_Q9["latency_s"], bwall, cascade_total_wall], ".2f"),
            ("cost ($)",   [PAPER_BQ_Q9["cost_usd"], PAPER_DASE_NN_Q9["cost_usd"], bcost, cascade_cost], ".4f"),
            ("#LLM calls", [None, 0, bcalls, s2_calls], "d"),
        ],
    )


if __name__ == "__main__":
    main()
