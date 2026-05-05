#!/usr/bin/env -S python -u
"""
Ecomm Q11 cascade — 4-way join (shoes / lower / upper / accessory) all-black + same brand.

NL: matching outfits (4 items) all-black, same brand, accessory price ≤ 500.
GT: 18 quadruples (SF=500).
BQ baseline: 500³ × 89 × 7 AI.IF ≈ 78B calls — INFEASIBLE.

Refactored. Operator (paper Table 3): J (4-way semantic join). Identical
algorithmic skeleton to ecomm/Q10 v2 — one extra role (accessory, with
price ≤ 500 prefilter), one extra pair set (u,a), and quadruple assembly
instead of triple. Composes:

  Stage 0: 4× RoleMarginSignal + AbsoluteBand (accessory side OR'd with price filter)
  Stage 1: per-role AiIfVerifier on uncertain (cached in Q11_stage1_cache.json)
  Stage 2: PairCosineSignal — drop pairs with sim ≤ PAIR_TAU_LOW for 3 pair sets
  Stage 3: AiIfVerifier(same-brand AI.IF) on uncertain pairs (3 batched query series)
  Stage 4: client-side quadruple assembly via SL ⨝ LU ⨝ UA on shared l, u
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
    RoleMarginSignal, AbsoluteBand, PairCosineSignal,
    AiIfVerifier, bq_client, per_row_cost, run_query,
    f1_set, build_profile, write_profile, print_summary,
)
from dase_cascade.calibration import _sum_tokens, _to_cost
from google.cloud import bigquery

ECOMM_DIR = os.path.abspath(os.path.join(_HERE, ".."))
PRODUCTS_IMAGE_PARQUET = os.path.join(ECOMM_DIR, "data", "products_image.parquet")
STYLES_PARQUET = os.path.join(ECOMM_DIR, "cache", "sf_500", "styles_details.parquet")
PROFILE_PATH = os.path.join(ECOMM_DIR, "outputs", "Q11.json")
STAGE1_CACHE_PATH = os.path.join(ECOMM_DIR, "outputs", "Q11_stage1_cache.json")

PROJECT = os.environ.get("GCP_PROJECT", "")
DATASET = "fashion_product_images"
GCS_BUCKET = f"{PROJECT}-mmb-fashion-product-images-bucket"

ACC_PRICE_LIMIT = 500
TAU_HIGH = 0.05
TAU_LOW  = -0.02
PAIR_TAU_LOW = 0.75
STAGE_BATCH = 300
PAPER_BQ_Q11      = {"score_f1": None, "latency_s": None, "cost_usd": None}
PAPER_DASE_NN_Q11 = {"score_f1": None, "latency_s": None, "cost_usd": None}

SHOE_PROMPT = """
    You will receive an image and a description of a product.
    Determine whether the product can be worn on the feet, like shoes, sandals, flip-flops, ...
    The predominant color of the depicted product should be black.
    If there are multiple products in the picture, always refer to the most promiment one.
    The description of the product is as follows: """
LOWER_PROMPT = """
    You will receive an image and a description of a product.
    Determine whether the product can be worn on the lower part of the body, like pants, shorts, skirts, ...
    The predominant color of the depicted product should be black.
    Do not consider swimwear.
    If there are multiple products in the picture, always refer to the most promiment one.
    The description of the product is as follows: """
UPPER_PROMPT = """
    You will receive an image and a description of a product.
    Determine whether the product can be worn on the upper part of the body, like t-shirts, shirts, pullovers, hoodies, but still require some sort of clothing on the lower body, which means, e.g., not a dress.
    The predominant color of the depicted product should be black.
    Do not consider swimwear.
    If there are multiple products in the picture, always refer to the most promiment one.
    The description of the product is as follows: """
ACCESSORY_PROMPT = """
    You will receive an image and a description of a product.
    Determine whether the product a watch or some jewellery or a bag.
    A bag might be a handbag or a (gym) backpack or some other type of bag.
    If there are multiple products in the picture, always refer to the most promiment one.
    The description of the product is as follows: """
SAME_BRAND_PROMPT_PREFIX = (
    "You will receive and image and the description of two products. "
    "Determine whether they are from the same brand. "
    "The description of the first product is as follows: "
)
ROLE_PROMPTS = {"shoe": SHOE_PROMPT, "lower": LOWER_PROMPT, "upper": UPPER_PROMPT, "accessory": ACCESSORY_PROMPT}
ROLE_PRICE_LIMIT = {"shoe": None, "lower": None, "upper": None, "accessory": ACC_PRICE_LIMIT}


def _product_selection_cte():
    return f"""
WITH images AS (
  SELECT
    images.*,
    productDisplayName AS title,
    productDescriptors.description.value AS descr,
    price
  FROM {DATASET}.STYLES_DETAILS styles_details
  JOIN {DATASET}.IMAGE_MAPPING mapping
    ON styles_details.styleImages.default.imageURL = mapping.link
  JOIN EXTERNAL_OBJECT_TRANSFORM(TABLE `{DATASET}.IMAGES`, ['SIGNED_URL']) as images
    ON ARRAY_LAST(SPLIT(images.uri, '/')) = mapping.filename
)
"""


def stage1_unary_sql(role_prompt: str, uncertain_ids, price_limit):
    id_list = ",".join(str(int(i)) for i in uncertain_ids)
    price_clause = f"  AND price <= {price_limit}\n" if price_limit is not None else ""
    return f"""
{_product_selection_cte()}
SELECT ARRAY_FIRST(SPLIT(ARRAY_LAST(SPLIT(uri, '/')), '.')) AS id
FROM images
WHERE CAST(ARRAY_FIRST(SPLIT(ARRAY_LAST(SPLIT(uri, '/')), '.')) AS INT64) IN ({id_list})
{price_clause}  AND AI.IF(
    ('''{role_prompt}''', title, ' ', descr, ' ', ref),
    connection_id => 'us.connection',
    endpoint => 'gemini-2.5-flash'
  )
"""


def stage3_pair_sql(staging_table: str):
    return f"""
{_product_selection_cte()},
pairs_typed AS (SELECT left_id, right_id FROM {staging_table})
SELECT CONCAT(CAST(pairs_typed.left_id AS STRING), '-', CAST(pairs_typed.right_id AS STRING)) AS pair_id
FROM pairs_typed
JOIN images i1 ON CAST(ARRAY_FIRST(SPLIT(ARRAY_LAST(SPLIT(i1.uri, '/')), '.')) AS INT64) = pairs_typed.left_id
JOIN images i2 ON CAST(ARRAY_FIRST(SPLIT(ARRAY_LAST(SPLIT(i2.uri, '/')), '.')) AS INT64) = pairs_typed.right_id
WHERE AI.IF(
    ('{SAME_BRAND_PROMPT_PREFIX}',
     i1.title, ' ', i1.descr, ' And the image of the first product is ', i1.ref,
     'The description of the second product is as follows: ',
     i2.title, ' ', i2.descr, ' And the image of the second product is ', i2.ref),
    connection_id => 'us.connection',
    endpoint => 'gemini-2.5-flash'
  )
"""


def stage_create_pair_table(client, table_name, pairs):
    if not pairs:
        return run_query(client, f"CREATE OR REPLACE TABLE {table_name} (left_id INT64, right_id INT64)")
    structs = ",".join(f"STRUCT({int(l)} AS left_id, {int(r)} AS right_id)" for l, r in pairs)
    return run_query(client, f"CREATE OR REPLACE TABLE {table_name} AS SELECT left_id, right_id FROM UNNEST([{structs}])")


# Pair calibration is bespoke (4 string params + 2 image refs); shape-match Q10's helper.
def per_row_pair_calibration(client, sample_pairs, k=10):
    THINKING = "JSON '{\"generation_config\":{\"thinking_config\":{\"thinking_budget\":0}}}'"
    selects, params = [], []
    for i, (u1, u2, t1, t2, d1, d2) in enumerate(sample_pairs[:k]):
        selects.append(f"""
        SELECT AI.GENERATE_BOOL(
          ('{SAME_BRAND_PROMPT_PREFIX}',
           @t1_{i}, ' ', @d1_{i}, ' And the image of the first product is ', img1.ref,
           'The description of the second product is as follows: ',
           @t2_{i}, ' ', @d2_{i}, ' And the image of the second product is ', img2.ref),
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
        scenario="ecomm", query_id=11, scale_factor=500,
        params={"acc_price_limit": ACC_PRICE_LIMIT,
                "tau_high_role": TAU_HIGH, "tau_low_role": TAU_LOW,
                "pair_tau_low": PAIR_TAU_LOW, "stage_batch": STAGE_BATCH},
        cascade_form=("J 4-way: 4× Cascade(RoleMarginSignal, AbsoluteBand, AiIfVerifier) → "
                      "PairCosineSignal threshold drop on 3 pair sets → AiIfVerifier(same-brand) "
                      "on uncertain pairs (batched) → quadruple assembly."),
    )

    print("Loading + computing dase 4-role contrastive margins...")
    sdf = pd.read_parquet(STYLES_PARQUET)
    sdf["id"] = sdf["id"].astype(np.int64)
    keep_ids = sdf["id"].tolist()
    n = len(keep_ids)

    pdf = pd.read_parquet(PRODUCTS_IMAGE_PARQUET)
    pdf["Id"] = pdf["Id"].astype(np.int64)
    pdf = pdf[pdf["Id"].isin(set(keep_ids))].copy()
    pdf = pdf.set_index("Id").loc[keep_ids].reset_index()
    img_emb = np.stack(pdf["embedding"].tolist()).astype(np.float32)
    ids_arr = pdf["Id"].astype(int).tolist()
    sdf_indexed = sdf.set_index("id")
    price_ok = sdf_indexed.loc[keep_ids, "price"].to_numpy() <= ACC_PRICE_LIMIT
    print(f"  total products: {n}")

    # ── GT quadruples for eval ──
    def gm(x): return x.get("typeName") if isinstance(x, dict) else None
    sdf2 = sdf.copy()
    sdf2["mc"] = sdf2["masterCategory"].apply(gm)
    sdf2["sc"] = sdf2["subCategory"].apply(gm)
    sdf2["at"] = sdf2["articleType"].apply(gm)
    shoes_gt = sdf2[(sdf2["mc"] == "Footwear") & (sdf2["baseColour"] == "Black")]
    lower_gt = sdf2[(sdf2["mc"] == "Apparel") & (sdf2["sc"] == "Bottomwear") & (sdf2["at"] != "Swimwear") & (sdf2["baseColour"] == "Black")]
    upper_gt = sdf2[(sdf2["mc"] == "Apparel") & (sdf2["sc"] == "Topwear") & (sdf2["at"] != "Swimwear") & (sdf2["baseColour"] == "Black")]
    acc_gt = sdf2[(sdf2["mc"] == "Accessories") & (sdf2["sc"].isin(["Watches", "Jewellery", "Bags"])) & (sdf2["price"] <= ACC_PRICE_LIMIT)]
    gt_quads = set()
    for _, s in shoes_gt.iterrows():
        for _, l in lower_gt[lower_gt["brandName"] == s["brandName"]].iterrows():
            for _, u in upper_gt[upper_gt["brandName"] == l["brandName"]].iterrows():
                for _, a in acc_gt[acc_gt["brandName"] == u["brandName"]].iterrows():
                    gt_quads.add(f"{int(s['id'])}-{int(l['id'])}-{int(u['id'])}-{int(a['id'])}")
    print(f"  GT quadruples: {len(gt_quads)}")
    profile["data"] = {"n_products_in_scope": n, "n_gt_quadruples": len(gt_quads)}

    client = bq_client(PROJECT)

    # ── Calibration (unary + pair) ──
    print("\n=== Per-row cost calibration ===")
    sample_uris = [f"gs://{GCS_BUCKET}/{int(i)}.jpg" for i in ids_arr[:10]]
    cal_unary = per_row_cost(client, SHOE_PROMPT, sample_uris=sample_uris,
                             ext_table=f"EXTERNAL_OBJECT_TRANSFORM(TABLE {DATASET}.IMAGES, ['SIGNED_URL']) AS",
                             k=10)
    per_row_unary = cal_unary.per_row_cost_usd
    print(f"  unary per_row=${per_row_unary:.6f}")

    sample_pairs = []
    for kk in range(min(10, n - 1)):
        i, j = kk, (kk + 1) % n
        pi, pj = int(ids_arr[i]), int(ids_arr[j])
        u1 = f"gs://{GCS_BUCKET}/{pi}.jpg"; u2 = f"gs://{GCS_BUCKET}/{pj}.jpg"
        ri, rj = sdf_indexed.loc[pi], sdf_indexed.loc[pj]
        t1 = str(ri["productDisplayName"] or ""); t2 = str(rj["productDisplayName"] or "")
        try:
            d1 = (ri["productDescriptors"] or {}).get("description", {}).get("value", "") or ""
        except Exception:
            d1 = ""
        try:
            d2 = (rj["productDescriptors"] or {}).get("description", {}).get("value", "") or ""
        except Exception:
            d2 = ""
        sample_pairs.append((u1, u2, t1, t2, d1[:500], d2[:500]))
    cal_pair = per_row_pair_calibration(client, sample_pairs, k=10)
    per_row_pair = cal_pair["per_row_cost_usd"]
    print(f"  pair  per_row=${per_row_pair:.6f}")
    profile["calibration"] = {"unary": cal_unary.to_dict(), "pair": cal_pair}

    # ── Stage 0 + Stage 1: per-role Cascade (manual to support cache + accessory price filter) ──
    print("\n=== Stage 0+1: per-role Cascade(RoleMarginSignal, AbsoluteBand, AiIfVerifier) ===")
    have_cache = os.path.isfile(STAGE1_CACHE_PATH)
    cache = json.load(open(STAGE1_CACHE_PATH)) if have_cache else None

    pool = {}
    s0_partition_summary = {}
    bq_yes_per_role: dict = {}
    s1_walls, s1_slots, s1_calls_total = {r: 0.0 for r in ROLE_PROMPTS}, {r: 0 for r in ROLE_PROMPTS}, 0
    t_dase0_total = 0.0
    for role in ROLE_PROMPTS:
        signal = RoleMarginSignal(role_prompts=ROLE_PROMPTS, target_role=role)
        t0 = time.time()
        scores = signal.compute(img_emb)
        # Accessory side: force confident_no on rows that fail price filter.
        if role == "accessory":
            scores = scores.copy()
            scores[~price_ok] = TAU_LOW - 1.0   # below tau_low → confident_no
        part = AbsoluteBand(tau_low=TAU_LOW, tau_high=TAU_HIGH).partition(scores)
        t_dase0_total += time.time() - t0
        confident_yes_ids = [int(ids_arr[i]) for i in part.confident_pos]
        uncertain_ids     = [int(ids_arr[i]) for i in part.uncertain]
        s0_partition_summary[role] = {"n_confident_yes": len(confident_yes_ids),
                                      "n_uncertain": len(uncertain_ids),
                                      "n_confident_no": int(part.confident_neg.size)}
        print(f"  {role:>9}: confident_yes={len(confident_yes_ids):>3}, "
              f"uncertain={len(uncertain_ids):>3}, confident_no={int(part.confident_neg.size):>3}")

        if have_cache:
            bq_yes = set(int(x) for x in cache["bq_yes"][role])
            s1_walls[role] = cache["walls"][role]; s1_slots[role] = cache["slots"][role]
            print(f"    {role}: cached BQ yes={len(bq_yes)}")
        elif not uncertain_ids:
            bq_yes = set()
        else:
            # Process uncertain in batches (Q11 has 4-role + many uncertain → must batch)
            verifier = AiIfVerifier(
                verify_sql_template=lambda ids, _r=ROLE_PROMPTS[role], _pl=ROLE_PRICE_LIMIT[role]:
                    stage1_unary_sql(_r, ids, _pl),
                id_column="id", coerce_id=int,
            )
            bq_yes = set()
            for bi in range(0, len(uncertain_ids), STAGE_BATCH):
                batch = uncertain_ids[bi:bi + STAGE_BATCH]
                vres = verifier.verify(client, batch, per_row_unary)
                bq_yes.update(vres.positive_ids)
                s1_walls[role] += vres.wall_s; s1_slots[role] += vres.slot_ms
                s1_calls_total += vres.n_calls
                print(f"    {role} batch [{bi}:{bi+len(batch)}] → BQ_yes so far={len(bq_yes)}, batch_wall={vres.wall_s:.1f}s")

        bq_yes_per_role[role] = bq_yes
        pool[role] = sorted(set(confident_yes_ids) | bq_yes)

    if have_cache:
        s1_calls_total = cache["s1_calls"]
    else:
        with open(STAGE1_CACHE_PATH, "w") as f:
            json.dump({"bq_yes": {r: sorted(int(x) for x in bq_yes_per_role[r]) for r in ROLE_PROMPTS},
                       "walls": s1_walls, "slots": s1_slots, "s1_calls": s1_calls_total}, f, indent=2)
        print(f"  cached → {STAGE1_CACHE_PATH}")

    print(f"\n  Pools: shoe={len(pool['shoe'])}, lower={len(pool['lower'])}, "
          f"upper={len(pool['upper'])}, accessory={len(pool['accessory'])}")
    profile["dase_breakdown"] = {"data_load_dase_s": t_dase0_total}
    profile["dase_partition"] = s0_partition_summary

    # ── Stage 2: PairCosineSignal threshold drop on 3 pair sets ──
    print(f"\n=== Stage 2: PairCosineSignal threshold drop (PAIR_TAU_LOW={PAIR_TAU_LOW}) ===")
    pid_to_idx = {int(ids_arr[k]): k for k in range(n)}
    pair_signal = PairCosineSignal(embeddings_left=img_emb)

    def keep_pairs(left_pool, right_pool, label):
        L = np.array([pid_to_idx[p] for p in left_pool], dtype=np.int64)
        R = np.array([pid_to_idx[p] for p in right_pool], dtype=np.int64)
        triples = pair_signal.all_pairs_above(L, R, PAIR_TAU_LOW)
        kept = [(int(ids_arr[i]), int(ids_arr[j])) for i, j, _ in triples]
        drop = len(left_pool) * len(right_pool) - len(kept)
        print(f"    {label}: drop={drop}, uncertain={len(kept)}")
        return kept

    sl_unc = keep_pairs(pool["shoe"], pool["lower"], "(s,l)")
    lu_unc = keep_pairs(pool["lower"], pool["upper"], "(l,u)")
    ua_unc = keep_pairs(pool["upper"], pool["accessory"], "(u,a)")

    # ── Stage 3: AiIfVerifier on uncertain pairs ──
    print("\n=== Stage 3: AiIfVerifier(same-brand) on uncertain pairs (3 × batched queries) ===")
    def run_pair_stage(pair_list, table_prefix, label):
        if not pair_list:
            return set(), 0.0, 0, 0.0, 0
        yes = set(); total_wall = 0.0; total_slot = 0; ctas_wall = 0.0; ctas_slot = 0
        for bi in range(0, len(pair_list), STAGE_BATCH):
            batch = pair_list[bi:bi + STAGE_BATCH]
            tab = f"{DATASET}.q11_uncertain_{table_prefix}_pairs"
            _, cw, cs, _ = stage_create_pair_table(client, tab, batch)
            ctas_wall += cw; ctas_slot += cs
            df_b, b_wall, b_slot, _ = run_query(client, stage3_pair_sql(tab))
            for x in df_b["pair_id"]:
                a, b = x.split('-')
                yes.add((int(a), int(b)))
            total_wall += b_wall; total_slot += b_slot
            print(f"    {label} batch [{bi}:{bi+len(batch)}] → YES so far={len(yes)}, batch_wall={b_wall:.1f}s")
        return yes, total_wall, total_slot, ctas_wall, ctas_slot

    sl_yes, sl_wall, sl_slot, sl_ctas_wall, sl_ctas_slot = run_pair_stage(sl_unc, "sl", "(s,l)")
    lu_yes, lu_wall, lu_slot, lu_ctas_wall, lu_ctas_slot = run_pair_stage(lu_unc, "lu", "(l,u)")
    ua_yes, ua_wall, ua_slot, ua_ctas_wall, ua_ctas_slot = run_pair_stage(ua_unc, "ua", "(u,a)")

    # ── Stage 4: quadruple assembly via SL ⨝ LU ⨝ UA on shared l, u ──
    sl_by_l: dict = {}
    for s, l in sl_yes:
        sl_by_l.setdefault(l, []).append(s)
    lu_by_l: dict = {}
    for l, u in lu_yes:
        lu_by_l.setdefault(l, []).append(u)
    ua_by_u: dict = {}
    for u, a in ua_yes:
        ua_by_u.setdefault(u, []).append(a)

    quads = set()
    for l in sl_by_l:
        if l not in lu_by_l: continue
        for s in sl_by_l[l]:
            for u in lu_by_l[l]:
                if u not in ua_by_u: continue
                for a in ua_by_u[u]:
                    quads.add(f"{s}-{l}-{u}-{a}")
    cp, cr, c_f1 = f1_set(quads, gt_quads)
    print(f"\n  cascade quadruples: {len(quads)}; P={cp:.4f} R={cr:.4f} F1={c_f1:.4f}")

    s3_total_calls = len(sl_unc) + len(lu_unc) + len(ua_unc)
    cascade_cost = per_row_unary * s1_calls_total + per_row_pair * s3_total_calls
    cascade_total_wall = (
        t_dase0_total + sum(s1_walls.values())
        + sl_ctas_wall + lu_ctas_wall + ua_ctas_wall + sl_wall + lu_wall + ua_wall
    )
    cascade_total_slot = (
        sum(s1_slots.values()) + sl_ctas_slot + lu_ctas_slot + ua_ctas_slot + sl_slot + lu_slot + ua_slot
    )

    profile["baseline"] = {
        "_status": "aborted",
        "_status_note": "BQ template = 500³ × 89 × 7 AI.IF ≈ 78B calls, infeasible. paper Q11 reports X.",
    }
    profile["cascade"] = {
        "method": "Stage0 RoleMarginSignal × 4 + AbsoluteBand → Stage1 unary AI.IF (cached batched) → Stage2 PairCosineSignal × 3 → Stage3 same-brand AI.IF batched → Stage4 quadruple assembly",
        "stage1_unary_walls_s": s1_walls, "stage1_unary_slots_ms": s1_slots,
        "stage3_pair_walls_s": {"sl": sl_wall, "lu": lu_wall, "ua": ua_wall},
        "stage3_pair_slots_ms": {"sl": sl_slot, "lu": lu_slot, "ua": ua_slot},
        "stage_ctas_walls_s": {"sl": sl_ctas_wall, "lu": lu_ctas_wall, "ua": ua_ctas_wall},
        "n_returned_quadruples": len(quads),
        "score": {"precision": cp, "recall": cr, "f1_score": c_f1},
        "totals": {
            "wall_s": cascade_total_wall,
            "wall_breakdown_s": {
                "dase_stage0": t_dase0_total,
                "bq_stage1_unary": sum(s1_walls.values()),
                "bq_stage_ctas": sl_ctas_wall + lu_ctas_wall + ua_ctas_wall,
                "bq_stage3_pair": sl_wall + lu_wall + ua_wall,
            },
            "slot_ms_bq_total": cascade_total_slot,
            "cost_usd": cascade_cost,
            "n_llm_calls": s1_calls_total + s3_total_calls,
            "n_llm_calls_breakdown": {"stage1_unary": s1_calls_total, "stage3_pair": s3_total_calls},
        },
    }
    profile["comparison"] = {
        "score":       {"paper_BQ": None, "paper_DASE_NN": None, "ours_BQ": None, "ours_cascade": c_f1},
        "wall_s":      {"paper_BQ": None, "paper_DASE_NN": None, "ours_BQ": None, "ours_cascade": cascade_total_wall},
        "cost_usd":    {"paper_BQ": None, "paper_DASE_NN": None, "ours_BQ": None, "ours_cascade": cascade_cost},
        "n_llm_calls": {"paper_BQ": None, "paper_DASE_NN": 0, "ours_BQ": None, "ours_cascade": s1_calls_total + s3_total_calls},
    }
    write_profile(profile, PROFILE_PATH)

    print_summary(
        "Ecomm Q11 (J: 4-way join cascade)",
        columns=["paper BQ", "DASE+NN", "ours cascade"],
        rows=[
            ("F1",         [None, None, c_f1], ".2f"),
            ("wall (s)",   [None, None, cascade_total_wall], ".2f"),
            ("cost ($)",   [None, None, cascade_cost], ".4f"),
            ("#LLM calls", [None, 0, s1_calls_total + s3_total_calls], "d"),
        ],
    )


if __name__ == "__main__":
    main()
