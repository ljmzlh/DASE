#!/usr/bin/env -S python -u
"""
Ecomm Q10 cascade — 3-way join (shoe + lower + upper, same color+brand).

NL: matching outfits with shared brand+color, each ≤ 1000 INR, in 4 base colors.
GT: SF=500 → 8 GT triples.
BQ template baseline: 113³ × 5 AI.IF = 7M calls — INFEASIBLE.

Refactored. Operator (paper Table 3): J (multi-way semantic join). Composes:

  Stage 0:  3 × Cascade (RoleMarginSignal + AbsoluteBand + AiIfVerifier(unary))
            ↳ pool[role] = confident_yes ∪ bq_yes for each role.
  Stage 1:  per-role BQ AI.IF on uncertain images (cached in Q10_stage1_cache.json)
  Stage 2:  PairCosineSignal — drop pairs with sim ≤ PAIR_TAU_LOW, keep rest as uncertain.
  Stage 3:  AiIfVerifier on uncertain pairs (2 batched queries: (s,l) and (l,u)).
  Stage 4:  client-side triple assembly: (s,l) × (l,u) joined on l.

The dase_cascade primitives drive every stage; orchestration (multi-stage,
caching, triple assembly) stays in this script — paper §5.1 explicitly leaves
multi-stage composition to the operator scheduler.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import (
    Cascade, RoleMarginSignal, AbsoluteBand, PairCosineSignal,
    AiIfVerifier, bq_client, per_row_cost, run_query,
    f1_set, build_profile, write_profile, print_summary,
)
from dase_cascade.calibration import _sum_tokens, _to_cost          # for pair calibration
from google.cloud import bigquery

ECOMM_DIR = os.path.abspath(os.path.join(_HERE, ".."))
PRODUCTS_IMAGE_PARQUET = os.path.join(ECOMM_DIR, "data", "products_image.parquet")
STYLES_PARQUET = os.path.join(ECOMM_DIR, "cache", "sf_500", "styles_details.parquet")
PROFILE_PATH = os.path.join(ECOMM_DIR, "outputs", "Q10.json")
STAGE1_CACHE_PATH = os.path.join(ECOMM_DIR, "outputs", "Q10_stage1_cache.json")

PROJECT = os.environ.get("GCP_PROJECT", "")
DATASET = "fashion_product_images"
GCS_BUCKET = f"{PROJECT}-mmb-fashion-product-images-bucket"

PRICE_LIMIT = 1000
BASE_COLOURS = ["Black", "Blue", "Red", "White"]
TAU_HIGH = 0.05
TAU_LOW = -0.02
PAIR_TAU_LOW = 0.75
STAGE3_BATCH = 300
PAPER_DASE_NN_Q10 = {"score_f1": None, "latency_s": 0.7, "cost_usd": 1e-5}

# Verbatim sembench BQ template prompts
SHOE_PROMPT = (
    "The image depicts a (pair of) shoe(s), sandal(s), flip-flop(s). "
    "If there are multiple products in the picture, always refer to the most promiment one."
)
LOWER_PROMPT = (
    "The image depicts a piece of apparel that can be worn on the lower part of the body, "
    "like pants, shorts, skirts, ... "
    "If there are multiple products in the picture, always refer to the most promiment one."
)
UPPER_PROMPT = (
    "The image depicts a piece of apparel that can be worn on the upper part of the body, "
    "like t-shirts, shirts, pullovers, hoodies, but still require some sort of clothing on "
    "the lower body, which means, e.g., not a dress. "
    "If there are multiple products in the picture, always refer to the most promiment one."
)
JOIN_PROMPT_PREFIX = (
    "The images depict products with the same primary base color, e.g., both are black, "
    "both are white, and both products are from the same brand. "
    "The description of the first product is "
)
ROLE_PROMPTS = {"shoe": SHOE_PROMPT, "lower": LOWER_PROMPT, "upper": UPPER_PROMPT}


def _colour_in_clause():
    return "(" + ", ".join(f"'{c}'" for c in BASE_COLOURS) + ")"


def _product_selection_cte():
    return f"""
WITH images AS (
  SELECT
    images.*,
    productDisplayName AS title,
    productDescriptors.description.value AS descr
  FROM {DATASET}.STYLES_DETAILS styles_details
  JOIN {DATASET}.IMAGE_MAPPING mapping
    ON styles_details.styleImages.default.imageURL = mapping.link
  JOIN EXTERNAL_OBJECT_TRANSFORM(TABLE `{DATASET}.IMAGES`, ['SIGNED_URL']) as images
    ON ARRAY_LAST(SPLIT(images.uri, '/')) = mapping.filename
  WHERE true
    AND styles_details.baseColour IN {_colour_in_clause()}
    AND price <= {PRICE_LIMIT}
)
"""


def stage1_unary_sql(role_prompt: str, uncertain_ids):
    id_list = ",".join(str(int(i)) for i in uncertain_ids)
    return f"""
{_product_selection_cte()}
SELECT
  ARRAY_FIRST(SPLIT(ARRAY_LAST(SPLIT(uri, '/')), '.')) AS id
FROM images
WHERE CAST(ARRAY_FIRST(SPLIT(ARRAY_LAST(SPLIT(uri, '/')), '.')) AS INT64) IN ({id_list})
  AND AI.IF(('{role_prompt}', ref),
            connection_id => 'us.connection',
            endpoint => 'gemini-2.5-flash')
"""


def stage3_pair_sql(staging_table: str):
    return f"""
{_product_selection_cte()},
pairs_typed AS (SELECT left_id, right_id FROM {staging_table})
SELECT
  CONCAT(CAST(pairs_typed.left_id AS STRING), '-', CAST(pairs_typed.right_id AS STRING)) AS pair_id
FROM pairs_typed
JOIN images i1
  ON CAST(ARRAY_FIRST(SPLIT(ARRAY_LAST(SPLIT(i1.uri, '/')), '.')) AS INT64) = pairs_typed.left_id
JOIN images i2
  ON CAST(ARRAY_FIRST(SPLIT(ARRAY_LAST(SPLIT(i2.uri, '/')), '.')) AS INT64) = pairs_typed.right_id
WHERE AI.IF(
    ('{JOIN_PROMPT_PREFIX}',
     i1.title, ' ', i1.descr, ' and the image of the first product is ', i1.ref,
     ' The description of the second product is ',
     i2.title, ' ', i2.descr, ' and the image of the second product is ', i2.ref),
    connection_id => 'us.connection',
    endpoint => 'gemini-2.5-flash'
  )
"""


def stage3_create_pair_table(client, table_name, pairs):
    if not pairs:
        return run_query(client, f"CREATE OR REPLACE TABLE {table_name} (left_id INT64, right_id INT64)")
    structs = ",".join(f"STRUCT({int(l)} AS left_id, {int(r)} AS right_id)" for l, r in pairs)
    return run_query(client, f"CREATE OR REPLACE TABLE {table_name} AS SELECT left_id, right_id FROM UNNEST([{structs}])")


# Pair calibration is bespoke — the prompt binds 4 STRING params + 2 image refs.
# Cleanest to just inline here (not generalizable into per_row_cost).
def per_row_pair_calibration(client, sample_pairs, k=10):
    THINKING = "JSON '{\"generation_config\":{\"thinking_config\":{\"thinking_budget\":0}}}'"
    selects, params = [], []
    for i, (u1, u2, t1, t2, d1, d2) in enumerate(sample_pairs[:k]):
        selects.append(f"""
        SELECT AI.GENERATE_BOOL(
          ('{JOIN_PROMPT_PREFIX}',
           @t1_{i}, ' ', @d1_{i}, ' and the image of the first product is ', img1.ref,
           ' The description of the second product is ',
           @t2_{i}, ' ', @d2_{i}, ' and the image of the second product is ', img2.ref),
          connection_id => 'us.connection',
          endpoint => 'gemini-2.5-flash',
          model_params => {THINKING}
        ) AS verdict
        FROM EXTERNAL_OBJECT_TRANSFORM(TABLE {DATASET}.IMAGES, ['SIGNED_URL']) AS img1
        JOIN EXTERNAL_OBJECT_TRANSFORM(TABLE {DATASET}.IMAGES, ['SIGNED_URL']) AS img2
          ON img1.uri = @u1_{i} AND img2.uri = @u2_{i}""")
        params += [
            bigquery.ScalarQueryParameter(f"u1_{i}", "STRING", u1),
            bigquery.ScalarQueryParameter(f"u2_{i}", "STRING", u2),
            bigquery.ScalarQueryParameter(f"t1_{i}", "STRING", t1),
            bigquery.ScalarQueryParameter(f"t2_{i}", "STRING", t2),
            bigquery.ScalarQueryParameter(f"d1_{i}", "STRING", d1),
            bigquery.ScalarQueryParameter(f"d2_{i}", "STRING", d2),
        ]
    sql = " UNION ALL ".join(selects)
    cfg = bigquery.QueryJobConfig(query_parameters=params, use_query_cache=False)
    t0 = time.time()
    df = client.query(sql, job_config=cfg).result().to_dataframe()
    elapsed = time.time() - t0
    p_other, p_audio, out, thoughts = _sum_tokens(df["verdict"])
    n = len(df)
    cost = _to_cost(p_other, p_audio, out, thoughts)
    return {"per_row_cost_usd": cost / n if n else 0.0, "n_sample": n, "elapsed_s": elapsed,
            "tokens": {"prompt_other": p_other, "prompt_audio": p_audio, "output": out, "thoughts": thoughts}}


def main():
    profile = build_profile(
        scenario="ecomm", query_id=10, scale_factor=500,
        params={"price_limit": PRICE_LIMIT, "base_colours": BASE_COLOURS,
                "tau_high_role": TAU_HIGH, "tau_low_role": TAU_LOW, "pair_tau_low": PAIR_TAU_LOW},
        cascade_form=("J cascade: 3× role Cascade(RoleMarginSignal, AbsoluteBand, AiIfVerifier) → "
                      "PairCosineSignal threshold drop → AiIfVerifier on uncertain pairs (batched) → "
                      "client-side triple assembly."),
    )

    print("Loading + filtering scope ...")
    sdf = pd.read_parquet(STYLES_PARQUET)
    sdf = sdf[sdf["baseColour"].isin(BASE_COLOURS) & (sdf["price"] <= PRICE_LIMIT)].copy()
    sdf["id"] = sdf["id"].astype(np.int64)
    keep_ids = sdf["id"].tolist()
    keep_set = set(keep_ids)

    pdf = pd.read_parquet(PRODUCTS_IMAGE_PARQUET)
    pdf["Id"] = pdf["Id"].astype(np.int64)
    pdf = pdf[pdf["Id"].isin(keep_set)].copy()
    pdf = pdf.set_index("Id").loc[keep_ids].reset_index()
    n = len(pdf)
    img_emb = np.stack(pdf["embedding"].tolist()).astype(np.float32)
    ids_arr = pdf["Id"].astype(int).tolist()
    print(f"  in-scope: {n} products (baseColour∈{BASE_COLOURS}, price≤{PRICE_LIMIT})")
    profile["data"] = {"n_products_in_scope": n,
                       "scope_filter": f"baseColour IN {BASE_COLOURS} AND price <= {PRICE_LIMIT}"}

    # ── GT triples for eval ──
    sdf_full = pd.read_parquet(STYLES_PARQUET)
    sdf_full["id"] = sdf_full["id"].astype(np.int64)
    sdf_full["mc"] = sdf_full["masterCategory"].apply(lambda x: x.get("typeName") if isinstance(x, dict) else None)
    sdf_full["sc"] = sdf_full["subCategory"].apply(lambda x: x.get("typeName") if isinstance(x, dict) else None)
    shoes_gt = sdf_full[(sdf_full["mc"] == "Footwear") & (sdf_full["price"] <= PRICE_LIMIT)]
    lower_gt = sdf_full[(sdf_full["mc"] == "Apparel") & (sdf_full["sc"] == "Bottomwear") & (sdf_full["price"] <= PRICE_LIMIT)]
    upper_gt = sdf_full[(sdf_full["mc"] == "Apparel") & (sdf_full["sc"] == "Topwear") & (sdf_full["price"] <= PRICE_LIMIT)]
    gt_triples = set()
    for _, s in shoes_gt.iterrows():
        if s["baseColour"] not in BASE_COLOURS:
            continue
        cand_l = lower_gt[(lower_gt["baseColour"] == s["baseColour"]) & (lower_gt["brandName"] == s["brandName"])]
        for _, l in cand_l.iterrows():
            cand_u = upper_gt[(upper_gt["baseColour"] == l["baseColour"]) & (upper_gt["brandName"] == l["brandName"])]
            for _, u in cand_u.iterrows():
                gt_triples.add(f"{int(s['id'])}-{int(l['id'])}-{int(u['id'])}")
    print(f"  GT triples: {len(gt_triples)}")
    profile["data"]["n_gt_triples"] = len(gt_triples)

    client = bq_client(PROJECT)

    # ── Cost calibration ──
    print("\n=== Per-row cost calibration ===")
    sample_uris = [f"gs://{GCS_BUCKET}/{int(i)}.jpg" for i in ids_arr[:10]]
    cal_unary = per_row_cost(client, SHOE_PROMPT, sample_uris=sample_uris,
                             ext_table=f"EXTERNAL_OBJECT_TRANSFORM(TABLE {DATASET}.IMAGES, ['SIGNED_URL']) AS",
                             k=10)
    per_row_unary = cal_unary.per_row_cost_usd
    print(f"  unary per_row=${per_row_unary:.6f}")

    # Pair calibration (bespoke; needs 4 string params + 2 image refs)
    sample_pairs = []
    for kk in range(min(10, n - 1)):
        i, j = kk, (kk + 1) % n
        u1 = f"gs://{GCS_BUCKET}/{int(ids_arr[i])}.jpg"
        u2 = f"gs://{GCS_BUCKET}/{int(ids_arr[j])}.jpg"
        s_i = sdf[sdf["id"] == int(ids_arr[i])].iloc[0]
        s_j = sdf[sdf["id"] == int(ids_arr[j])].iloc[0]
        t1 = s_i["productDisplayName"] or ""
        t2 = s_j["productDisplayName"] or ""
        try:
            d1 = (s_i["productDescriptors"] or {}).get("description", {}).get("value", "") or ""
        except Exception:
            d1 = ""
        try:
            d2 = (s_j["productDescriptors"] or {}).get("description", {}).get("value", "") or ""
        except Exception:
            d2 = ""
        sample_pairs.append((u1, u2, t1, t2, d1[:500], d2[:500]))
    cal_pair = per_row_pair_calibration(client, sample_pairs, k=10)
    per_row_pair = cal_pair["per_row_cost_usd"]
    print(f"  pair  per_row=${per_row_pair:.6f}")
    profile["calibration"] = {"unary": cal_unary.to_dict(), "pair": cal_pair}

    # ── Stage 0 + Stage 1: per-role Cascade with caching ──
    print("\n=== Stage 0+1: per-role Cascade(RoleMarginSignal, AbsoluteBand, AiIfVerifier) ===")
    have_cache = os.path.isfile(STAGE1_CACHE_PATH)
    cache = json.load(open(STAGE1_CACHE_PATH)) if have_cache else None

    pool = {}                         # role → sorted list of pids in role's pool (yes ∪ bq_yes)
    s0_partition_summary = {}
    bq_yes_per_role: dict = {}        # role → set(bq yes pids); needed for cache write
    s1_walls, s1_slots, s1_calls_total = {}, {}, 0
    t_dase0_total = 0.0
    for role in ROLE_PROMPTS:
        signal = RoleMarginSignal(role_prompts=ROLE_PROMPTS, target_role=role)
        # Compute scores+partition manually so we can populate s0_partition_summary
        # AND skip BQ entirely on cache hit. Cascade.run() would unconditionally call BQ.
        t0 = time.time()
        scores = signal.compute(img_emb)
        part = AbsoluteBand(tau_low=TAU_LOW, tau_high=TAU_HIGH).partition(scores)
        t_dase0_total += time.time() - t0
        confident_yes_ids = [int(ids_arr[i]) for i in part.confident_pos]
        uncertain_ids     = [int(ids_arr[i]) for i in part.uncertain]
        s0_partition_summary[role] = {
            "n_confident_yes": len(confident_yes_ids),
            "n_uncertain":     len(uncertain_ids),
            "n_confident_no":  int(part.confident_neg.size),
        }
        print(f"  {role:>5}: confident_yes={len(confident_yes_ids):>3}, "
              f"uncertain={len(uncertain_ids):>3}, confident_no={int(part.confident_neg.size):>3}")

        if have_cache:
            bq_yes = set(int(x) for x in cache["bq_yes"][role])
            s1_walls[role] = cache["walls"][role]
            s1_slots[role] = cache["slots"][role]
            print(f"    {role}: cached BQ yes={len(bq_yes)}")
        elif not uncertain_ids:
            bq_yes = set()
            s1_walls[role], s1_slots[role] = 0.0, 0
        else:
            verifier = AiIfVerifier(
                verify_sql_template=lambda ids, _r=ROLE_PROMPTS[role]: stage1_unary_sql(_r, ids),
                id_column="id", coerce_id=int,
            )
            vres = verifier.verify(client, uncertain_ids, per_row_unary)
            bq_yes = set(vres.positive_ids)
            s1_walls[role] = vres.wall_s
            s1_slots[role] = vres.slot_ms
            s1_calls_total += vres.n_calls
            print(f"    {role}: BQ yes={len(bq_yes)}, wall={vres.wall_s:.1f}s slot={vres.slot_ms}")

        bq_yes_per_role[role] = bq_yes
        pool[role] = sorted(set(confident_yes_ids) | bq_yes)

    if have_cache:
        s1_calls_total = cache["s1_calls"]
    else:
        with open(STAGE1_CACHE_PATH, "w") as f:
            json.dump({
                "bq_yes": {r: sorted(int(x) for x in bq_yes_per_role[r]) for r in ROLE_PROMPTS},
                "walls":  s1_walls,
                "slots":  s1_slots,
                "s1_calls": s1_calls_total,
            }, f, indent=2)
        print(f"  cached stage1 → {STAGE1_CACHE_PATH}")

    print(f"\n  Pools: shoe={len(pool['shoe'])}, lower={len(pool['lower'])}, upper={len(pool['upper'])}")
    profile["dase_breakdown"] = {"data_load_dase_s": t_dase0_total}
    profile["dase_partition"] = s0_partition_summary

    # ── Stage 2: PairCosineSignal — drop pairs with sim ≤ PAIR_TAU_LOW ──
    print(f"\n=== Stage 2: PairCosineSignal threshold (PAIR_TAU_LOW={PAIR_TAU_LOW}) ===")
    pid_to_idx = {int(ids_arr[k]): k for k in range(n)}
    pair_signal = PairCosineSignal(embeddings_left=img_emb)

    def keep_pairs(left_pool, right_pool):
        L = np.array([pid_to_idx[p] for p in left_pool], dtype=np.int64)
        R = np.array([pid_to_idx[p] for p in right_pool], dtype=np.int64)
        triples = pair_signal.all_pairs_above(L, R, PAIR_TAU_LOW)
        # Convert (idx, idx, sim) → (pid, pid)
        return [(int(ids_arr[i]), int(ids_arr[j])) for i, j, _ in triples]

    sl_pairs_uncertain = keep_pairs(pool["shoe"], pool["lower"])
    lu_pairs_uncertain = keep_pairs(pool["lower"], pool["upper"])
    sl_drop = len(pool["shoe"]) * len(pool["lower"]) - len(sl_pairs_uncertain)
    lu_drop = len(pool["lower"]) * len(pool["upper"]) - len(lu_pairs_uncertain)
    print(f"  (s,l): drop={sl_drop}, uncertain={len(sl_pairs_uncertain)}")
    print(f"  (l,u): drop={lu_drop}, uncertain={len(lu_pairs_uncertain)}")

    # ── Stage 3: AiIfVerifier on uncertain pairs (batched) ──
    print("\n=== Stage 3: AiIfVerifier on uncertain pairs (2 batched join queries) ===")
    def run_pair_stage(pair_list, table_prefix, label):
        if not pair_list:
            return set(), 0.0, 0, 0.0, 0
        yes = set()
        total_wall = 0.0; total_slot = 0
        ctas_wall = 0.0; ctas_slot = 0
        for bi in range(0, len(pair_list), STAGE3_BATCH):
            batch = pair_list[bi:bi + STAGE3_BATCH]
            tab = f"{DATASET}.q10_uncertain_{table_prefix}_pairs"
            _, cw, cs, _ = stage3_create_pair_table(client, tab, batch)
            ctas_wall += cw; ctas_slot += cs
            df_b, b_wall, b_slot, _ = run_query(client, stage3_pair_sql(tab))
            for x in df_b["pair_id"]:
                a, b = x.split('-')
                yes.add((int(a), int(b)))
            total_wall += b_wall; total_slot += b_slot
            print(f"    {label} batch [{bi}:{bi+len(batch)}] → YES so far={len(yes)}, batch_wall={b_wall:.1f}s")
        return yes, total_wall, total_slot, ctas_wall, ctas_slot

    sl_yes, sl_wall, sl_slot, ctas_sl_wall, ctas_sl_slot = run_pair_stage(sl_pairs_uncertain, "sl", "(s,l)")
    lu_yes, lu_wall, lu_slot, ctas_lu_wall, ctas_lu_slot = run_pair_stage(lu_pairs_uncertain, "lu", "(l,u)")

    # ── Stage 4: triple assembly ──
    triples = set()
    for s, l in sl_yes:
        for l2, u in lu_yes:
            if l2 == l:
                triples.add(f"{s}-{l}-{u}")
    cp, cr, c_f1 = f1_set(triples, gt_triples)
    print(f"\n  cascade triples: {len(triples)}; P={cp:.4f} R={cr:.4f} F1={c_f1:.4f}")

    s3_total_calls = len(sl_pairs_uncertain) + len(lu_pairs_uncertain)
    cascade_cost = per_row_unary * s1_calls_total + per_row_pair * s3_total_calls
    cascade_total_wall = (
        t_dase0_total + sum(s1_walls.values())
        + ctas_sl_wall + ctas_lu_wall + sl_wall + lu_wall
    )
    cascade_total_slot = (
        sum(s1_slots.values()) + ctas_sl_slot + ctas_lu_slot + sl_slot + lu_slot
    )

    profile["baseline"] = {
        "_status": "aborted",
        "_status_note": "BQ template = 113³×5 AI.IF = 7M calls; infeasible. paper Q10 reports X.",
    }
    profile["cascade"] = {
        "method": "Stage0 RoleMarginSignal+AbsoluteBand → Stage1 unary AI.IF → Stage2 PairCosineSignal → Stage3 pair AI.IF → Stage4 triple assemble",
        "stage1_unary_walls_s": s1_walls, "stage1_unary_slots_ms": s1_slots,
        "stage3_pair_walls_s": {"sl": sl_wall, "lu": lu_wall},
        "stage3_pair_slots_ms": {"sl": sl_slot, "lu": lu_slot},
        "n_returned_triples": len(triples),
        "score": {"precision": cp, "recall": cr, "f1_score": c_f1},
        "totals": {
            "wall_s": cascade_total_wall,
            "wall_breakdown_s": {
                "dase_stage0": t_dase0_total,
                "bq_stage1_unary": sum(s1_walls.values()),
                "bq_stage_ctas": ctas_sl_wall + ctas_lu_wall,
                "bq_stage3_pair": sl_wall + lu_wall,
            },
            "slot_ms_bq_total": cascade_total_slot,
            "cost_usd": cascade_cost,
            "n_llm_calls": s1_calls_total + s3_total_calls,
            "n_llm_calls_breakdown": {"stage1_unary": s1_calls_total, "stage3_pair": s3_total_calls},
        },
    }
    profile["comparison"] = {
        "score":       {"paper_BQ": None, "paper_DASE_NN": None, "ours_cascade": c_f1},
        "wall_s":      {"paper_BQ": None, "paper_DASE_NN": PAPER_DASE_NN_Q10["latency_s"], "ours_cascade": cascade_total_wall},
        "cost_usd":    {"paper_BQ": None, "paper_DASE_NN": PAPER_DASE_NN_Q10["cost_usd"], "ours_cascade": cascade_cost},
        "n_llm_calls": {"paper_BQ": None, "paper_DASE_NN": 0, "ours_cascade": s1_calls_total + s3_total_calls},
    }

    write_profile(profile, PROFILE_PATH)

    print_summary(
        "Ecomm Q10 (J: 3-way join cascade)",
        columns=["paper BQ", "DASE+NN", "ours cascade"],
        rows=[
            ("F1",         [None, None, c_f1], ".2f"),
            ("wall (s)",   [None, PAPER_DASE_NN_Q10["latency_s"], cascade_total_wall], ".2f"),
            ("cost ($)",   [None, PAPER_DASE_NN_Q10["cost_usd"], cascade_cost], ".4f"),
            ("#LLM calls", [None, 0, s1_calls_total + s3_total_calls], "d"),
        ],
    )


if __name__ == "__main__":
    main()
