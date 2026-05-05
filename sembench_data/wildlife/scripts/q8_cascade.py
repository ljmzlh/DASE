#!/usr/bin/env -S python -u
"""
Wildlife Q8 cascade — (elephant UNION) INTERSECT (monkey UNION).

NL: Cities with elephant (image OR audio) AND monkey (image OR audio).
GT: (img_eleph ∪ aud_eleph) ∩ (img_monk ∪ aud_monk).
Eval: set retrieval F1.

Refactored to use dase_cascade unified solver. Operator (paper Table 3): M
= 4 F (img×eleph, aud×eleph, img×monk, aud×monk) combined client-side via
union-then-intersect.

Like q7_v2: shared image staging table (union of img_eleph_uncertain ∪
img_monk_uncertain) and shared audio staging table; 4 AI.IF verifiers run on
those staging tables (2 image, 2 audio). Set algebra is client-side.

Cost / call accounting matches the original:
  s2_calls = 2*|img_unc_union| + 2*|aud_unc_union|
  cost     = img_per_row*2*|img_unc_union| + aud_per_row*2*|aud_unc_union|
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
PROFILE_PATH = os.path.join(WILDLIFE_DIR, "outputs", "Q8.json")

PROJECT = os.environ.get("GCP_PROJECT", "")
BUCKET  = f"{PROJECT}-animals_dataset"
DATASET = "animals_dataset"
IMG_STAGING = f"{DATASET}.q8_image_uncertain_mm"
AUD_STAGING = f"{DATASET}.q8_audio_uncertain_mm"

IMG_ELEPH_PROMPT = "Does this image contain an elephant? "
AUD_ELEPH_PROMPT = "Does this audio contain an elephant sound? "
IMG_MONK_PROMPT = "Does this image contain a monkey? "
AUD_MONK_PROMPT = "Does this audio contain a monkey sound? "

IMG_ELEPH_POS = ["a photograph of an elephant", "a wildlife camera trap image showing an elephant", "an elephant captured in the photo"]
IMG_ELEPH_NEG = ["a photograph that does not contain an elephant", "a wildlife camera trap image of a non-elephant animal", "an animal photo without any elephant"]
AUD_ELEPH_POS = ["a sound recording of an elephant", "audio of an elephant trumpeting or vocalizing", "elephant call sound clip"]
AUD_ELEPH_NEG = ["a sound recording of an animal that is not an elephant", "audio of a non-elephant animal vocalization", "animal sound clip without any elephant"]
IMG_MONK_POS = ["a photograph of a monkey", "a wildlife camera trap image showing a monkey", "a monkey captured in the photo"]
IMG_MONK_NEG = ["a photograph that does not contain a monkey", "a wildlife camera trap image of a non-monkey animal", "an animal photo without any monkey"]
AUD_MONK_POS = ["a sound recording of a monkey", "audio of monkey vocalizations or calls", "monkey howling or chittering sound clip"]
AUD_MONK_NEG = ["a sound recording of an animal that is not a monkey", "audio of a non-monkey animal vocalization", "animal sound clip without any monkey"]

ALPHA = 0.5
PAPER_BQ_Q8 = {"score": 0.75, "latency_s": 35.5, "cost_usd": 0.23}
PAPER_DASE_NN_Q8 = {"score": 0.00, "latency_s": 2e-3, "cost_usd": 1e-9}
SKIP_BASELINE = True


def _uri_array_literal(uris):
    items = ",".join(f"'{u}'" for u in uris)
    return f"[{items}]"


def stage1_image_sql(uris):
    return f"""
    CREATE OR REPLACE TABLE {IMG_STAGING} AS
    SELECT m.Species, m.City, m.StationID, ot.ref AS image
    FROM {DATASET}.image_data_images m
    JOIN {DATASET}.image_data_external ot ON ot.uri = m.ImagePath
    WHERE m.ImagePath IN UNNEST({_uri_array_literal(uris)})
    """


def stage1_audio_sql(uris):
    return f"""
    CREATE OR REPLACE TABLE {AUD_STAGING} AS
    SELECT m.Animal, m.City, m.StationID, ot.ref AS audio
    FROM {DATASET}.audio_data_files m
    JOIN {DATASET}.audio_data_external ot ON ot.uri = m.AudioPath
    WHERE m.AudioPath IN UNNEST({_uri_array_literal(uris)})
    """


def make_table_verifier(table, ref_col, prompt):
    """Verifier with verify_sql only (no make_staging_sql) — reuses pre-built staging."""
    verify_sql = f"""
    SELECT DISTINCT City AS id FROM {table}
    WHERE AI.IF(('{prompt}', {ref_col}),
                connection_id => 'us.connection',
                endpoint => 'gemini-2.5-flash')
    """
    return AiIfVerifier(verify_sql=verify_sql, id_column="id", coerce_id=str)


def main():
    profile = build_profile(
        scenario="wildlife", query_id=8, scale_factor=200,
        params={"alpha": ALPHA},
        cascade_form="4-prompt × dual-modality + client-side intersect (Cascade primitive in shared-staging mode)",
    )

    print("Loading data + embeddings...")
    img_df = pd.read_csv(IMAGE_CSV)
    img_emb = np.load(IMG_EMB_PATH)["caption_emb"]
    aud_df = pd.read_csv(AUDIO_CSV)
    aud_emb = np.load(AUD_EMB_PATH)["caption_emb"]
    img_df["GcsUri"] = img_df["ImagePath"].apply(lambda p: f"gs://{BUCKET}/animal_images/{os.path.basename(p)}")
    aud_df["GcsUri"] = aud_df["AudioPath"].apply(lambda p: f"gs://{BUCKET}/animal_audio/{os.path.basename(p)}")
    n_img, n_aud = len(img_df), len(aud_df)

    gt_img_eleph = set(img_df[img_df["Species"].str.contains("ELEPHANT")]["City"])
    gt_aud_eleph = set(aud_df[aud_df["Animal"] == "Elephant"]["City"])
    gt_img_monk = set(img_df[img_df["Species"].str.contains("MONKEY")]["City"])
    gt_aud_monk = set(aud_df[aud_df["Animal"] == "Monkey"]["City"])
    gt_eleph = gt_img_eleph | gt_aud_eleph
    gt_monk = gt_img_monk | gt_aud_monk
    gt_cities = gt_eleph & gt_monk
    print(f"  GT elephant cities (img∪aud): {sorted(gt_eleph)}")
    print(f"  GT monkey   cities (img∪aud): {sorted(gt_monk)}")
    print(f"  GT cities (intersect): {sorted(gt_cities)}")
    profile["data"] = {"n_img": n_img, "n_aud": n_aud,
                       "gt_elephant_cities": sorted(gt_eleph),
                       "gt_monkey_cities": sorted(gt_monk),
                       "gt_cities": sorted(gt_cities)}

    # ── 4 signal+band passes (shared embeddings per modality) ──
    def signal_band(emb, pos, neg, df):
        s = MarginSignal(positive_prompts=pos, negative_prompts=neg).compute(emb)
        part = AlphaBand(alpha=ALPHA).partition(s)
        unc_idx = set(part.uncertain.tolist())
        confident_cities = set(df["City"].iloc[part.confident_pos])
        return unc_idx, confident_cities

    img_eleph_unc, dase_img_eleph_cities = signal_band(img_emb, IMG_ELEPH_POS, IMG_ELEPH_NEG, img_df)
    aud_eleph_unc, dase_aud_eleph_cities = signal_band(aud_emb, AUD_ELEPH_POS, AUD_ELEPH_NEG, aud_df)
    img_monk_unc,  dase_img_monk_cities  = signal_band(img_emb, IMG_MONK_POS,  IMG_MONK_NEG,  img_df)
    aud_monk_unc,  dase_aud_monk_cities  = signal_band(aud_emb, AUD_MONK_POS,  AUD_MONK_NEG,  aud_df)

    img_unc_union = sorted(img_eleph_unc | img_monk_unc)
    aud_unc_union = sorted(aud_eleph_unc | aud_monk_unc)
    img_uri_unc = img_df["GcsUri"].iloc[img_unc_union].tolist()  # gs:// — matches m.ImagePath in BQ
    aud_uri_unc = aud_df["GcsUri"].iloc[aud_unc_union].tolist()  # gs:// — matches m.AudioPath in BQ

    print(f"  alpha={ALPHA}")
    print(f"  image uncertain: eleph={len(img_eleph_unc)}, monk={len(img_monk_unc)}, union={len(img_unc_union)}")
    print(f"  audio uncertain: eleph={len(aud_eleph_unc)}, monk={len(aud_monk_unc)}, union={len(aud_unc_union)}")
    print(f"  dase img_eleph cities: {sorted(dase_img_eleph_cities)}")
    print(f"  dase aud_eleph cities: {sorted(dase_aud_eleph_cities)}")
    print(f"  dase img_monk cities:  {sorted(dase_img_monk_cities)}")
    print(f"  dase aud_monk cities:  {sorted(dase_aud_monk_cities)}")
    profile["dase_partition"] = {
        "img_uncertain_union": len(img_unc_union), "aud_uncertain_union": len(aud_unc_union),
        "dase_img_eleph_cities": sorted(dase_img_eleph_cities),
        "dase_aud_eleph_cities": sorted(dase_aud_eleph_cities),
        "dase_img_monk_cities": sorted(dase_img_monk_cities),
        "dase_aud_monk_cities": sorted(dase_aud_monk_cities),
    }

    client = bq_client(PROJECT)
    print("\n=== Per-row cost calibration ===")
    img_cal = per_row_cost(client, IMG_ELEPH_PROMPT,
        sample_uris=img_df["GcsUri"].head(5).tolist(),
        ext_table=f"{DATASET}.image_data_external")
    aud_cal = per_row_cost(client, AUD_ELEPH_PROMPT,
        sample_uris=aud_df["GcsUri"].head(5).tolist(),
        ext_table=f"{DATASET}.audio_data_external")
    img_per_row = img_cal.per_row_cost_usd; aud_per_row = aud_cal.per_row_cost_usd
    print(f"  image per_row=${img_per_row:.6f}, audio per_row=${aud_per_row:.6f}")
    profile["calibration"] = {"image_per_row": img_per_row, "audio_per_row": aud_per_row}

    if SKIP_BASELINE:
        print("\n=== Baseline skipped (4-way AI.IF run is long; using paper numbers) ===")
        bcost = PAPER_BQ_Q8["cost_usd"]; bwall = PAPER_BQ_Q8["latency_s"]; bslot = None
        b_score = PAPER_BQ_Q8["score"]; b_cities = None
        bcalls = round(bcost / ((img_per_row + aud_per_row)/2))
        profile["baseline"] = {"_status": "aborted",
                                "score": {"score": b_score, "_source": "paper"},
                                "latency_breakdown": {"wall_s": bwall, "_source": "paper"},
                                "cost_breakdown": {"n_llm_calls": bcalls, "total_cost_usd": bcost, "_source": "paper"}}

    # ── Stage 1: 2 staging tables (image + audio) ──
    print(f"\n=== Cascade Stage 1: 2 staging tables ===")
    img_s1_sql = stage1_image_sql(img_uri_unc)
    aud_s1_sql = stage1_audio_sql(aud_uri_unc)
    _, img_s1_wall, img_s1_slot, _ = run_query(client, img_s1_sql)
    _, aud_s1_wall, aud_s1_slot, _ = run_query(client, aud_s1_sql)
    s1_wall = img_s1_wall + aud_s1_wall
    s1_slot = img_s1_slot + aud_s1_slot
    print(f"  combined wall={s1_wall:.2f}s, slot_ms={s1_slot}")

    # ── Stage 2: 4 AI.IF queries against the 2 staging tables ──
    print(f"\n=== Cascade Stage 2: 4 AI.IF queries ===")
    v_img_eleph = make_table_verifier(IMG_STAGING, "image", IMG_ELEPH_PROMPT)
    v_aud_eleph = make_table_verifier(AUD_STAGING, "audio", AUD_ELEPH_PROMPT)
    v_img_monk  = make_table_verifier(IMG_STAGING, "image", IMG_MONK_PROMPT)
    v_aud_monk  = make_table_verifier(AUD_STAGING, "audio", AUD_MONK_PROMPT)

    # Each image verifier processes |img_unc_union| rows; each audio verifier |aud_unc_union|.
    r_img_e = v_img_eleph.verify(client, img_uri_unc, img_per_row)
    r_aud_e = v_aud_eleph.verify(client, aud_uri_unc, aud_per_row)
    r_img_m = v_img_monk.verify (client, img_uri_unc, img_per_row)
    r_aud_m = v_aud_monk.verify (client, aud_uri_unc, aud_per_row)

    bq_img_eleph = r_img_e.positive_ids
    bq_aud_eleph = r_aud_e.positive_ids
    bq_img_monk  = r_img_m.positive_ids
    bq_aud_monk  = r_aud_m.positive_ids

    s2_wall = r_img_e.wall_s + r_aud_e.wall_s + r_img_m.wall_s + r_aud_m.wall_s
    s2_slot = r_img_e.slot_ms + r_aud_e.slot_ms + r_img_m.slot_ms + r_aud_m.slot_ms
    s2_calls = r_img_e.n_calls + r_aud_e.n_calls + r_img_m.n_calls + r_aud_m.n_calls
    cascade_cost = r_img_e.cost_usd + r_aud_e.cost_usd + r_img_m.cost_usd + r_aud_m.cost_usd

    print(f"  bq img_eleph: {sorted(bq_img_eleph)}")
    print(f"  bq aud_eleph: {sorted(bq_aud_eleph)}")
    print(f"  bq img_monk:  {sorted(bq_img_monk)}")
    print(f"  bq aud_monk:  {sorted(bq_aud_monk)}")
    print(f"  combined wall={s2_wall:.2f}s, slot_ms={s2_slot}, calls={s2_calls}, cost=${cascade_cost:.6f}")

    elephant_cities = dase_img_eleph_cities | dase_aud_eleph_cities | bq_img_eleph | bq_aud_eleph
    monkey_cities   = dase_img_monk_cities  | dase_aud_monk_cities  | bq_img_monk  | bq_aud_monk
    cascade_cities  = elephant_cities & monkey_cities
    _, _, cscore = f1_set(cascade_cities, gt_cities)
    print(f"  elephant_cities (combined): {sorted(elephant_cities)}")
    print(f"  monkey_cities   (combined): {sorted(monkey_cities)}")
    print(f"  cascade_cities  (intersect): {sorted(cascade_cities)} (GT: {sorted(gt_cities)})  F1={cscore:.4f}")

    cascade_total_wall = s1_wall + s2_wall
    cascade_total_slot = s1_slot + s2_slot
    profile["cascade"] = {
        "method": "4 AI.IF verifiers on 2 shared staging tables; client-side intersect",
        "stage1_wall_s": s1_wall, "stage1_slot_ms": s1_slot,
        "stage2_results": {
            "img_eleph": r_img_e.to_dict(), "aud_eleph": r_aud_e.to_dict(),
            "img_monk":  r_img_m.to_dict(), "aud_monk":  r_aud_m.to_dict(),
        },
        "bq_results": {"img_eleph": sorted(bq_img_eleph), "aud_eleph": sorted(bq_aud_eleph),
                       "img_monk": sorted(bq_img_monk), "aud_monk": sorted(bq_aud_monk)},
        "stage2_wall_s": s2_wall, "stage2_slot_ms": s2_slot,
        "elephant_cities_combined": sorted(elephant_cities),
        "monkey_cities_combined": sorted(monkey_cities),
        "cascade_cities": sorted(cascade_cities),
        "score": {"f1": cscore},
        "totals": {"wall_s": cascade_total_wall, "slot_ms_bq_total": cascade_total_slot,
                   "cost_usd": cascade_cost, "n_llm_calls": s2_calls},
    }

    profile["comparison"] = {
        "score": {"paper_BQ": PAPER_BQ_Q8["score"], "paper_DASE_NN": PAPER_DASE_NN_Q8["score"], "ours_BQ": b_score, "ours_cascade": cscore},
        "wall_s": {"paper_BQ": PAPER_BQ_Q8["latency_s"], "paper_DASE_NN": PAPER_DASE_NN_Q8["latency_s"], "ours_BQ": bwall, "ours_cascade": cascade_total_wall},
        "cost_usd": {"paper_BQ": PAPER_BQ_Q8["cost_usd"], "paper_DASE_NN": PAPER_DASE_NN_Q8["cost_usd"], "ours_BQ": bcost, "ours_cascade": cascade_cost},
        "n_llm_calls": {"paper_BQ": bcalls if b_cities is None else 0, "paper_DASE_NN": 0, "ours_BQ": bcalls, "ours_cascade": s2_calls},
    }

    write_profile(profile, PROFILE_PATH)

    print_summary(
        f"Wildlife Q8 (alpha={ALPHA})",
        columns=["paper BQ", "DASE+NN", "ours BQ", "ours cascade"],
        rows=[
            ("score (F1)", [PAPER_BQ_Q8["score"], PAPER_DASE_NN_Q8["score"], b_score, cscore], ".2f"),
            ("wall (s)",   [PAPER_BQ_Q8["latency_s"], PAPER_DASE_NN_Q8["latency_s"], bwall, cascade_total_wall], ".2f"),
            ("cost ($)",   [PAPER_BQ_Q8["cost_usd"], PAPER_DASE_NN_Q8["cost_usd"], bcost, cascade_cost], ".4f"),
            ("#LLM calls", [bcalls, 0, bcalls, s2_calls], "d"),
        ],
    )


if __name__ == "__main__":
    main()
