#!/usr/bin/env -S python -u
"""
Ecomm Q12 cascade — sem_filter + sem_map (Adidas/Puma → JSON {id, brand, category}).

NL: For Adidas/Puma products in mc∈{Accessories, Apparel, Footwear}, output
    JSON {"id":..,"brand":..,"category":..}.
GT: 58 GT JSONs.
Eval: F1 over JSON id strings.

Refactored to use dase_cascade.ClusterCascade. Operator (paper Table 3): M
(cluster-based prefilter for SEM_MAP, distinct primitive — see ClusterCascade).

Pipeline (mirrors Q3):
  1. AgglomerativeClustering(metric=cosine, linkage=complete, distance_threshold=TAU)
     groups 477 in-scope products into K tight clusters.
  2. ClusterCascade picks one centroid-nearest representative per cluster.
  3. Verifier wraps the verbatim Q12 SQL (AI.IF in WHERE + AI.GENERATE in SELECT)
     on staging table of K reps. Output is rep_id + parsed JSON for filter-pass reps.
  4. Cluster → (brand, category) propagation: each member gets own id but rep's bc.
  5. F1 vs GT JSONs.

Note Q12 has 2 LLM calls per row (filter + generate). Cost is filter*K + generate*n_pass.
We feed AiGenerateVerifier the combined per-row rate (filter+generate); the
n_calls field captures K reps (filter calls). True per-stage breakdown is in
profile["calibration"] for downstream use.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import (
    ClusterCascade, AiGenerateVerifier,
    bq_client, run_query,
    f1_set, build_profile, write_profile, print_summary,
)
from dase_cascade.calibration import _sum_tokens, _to_cost
from google.cloud import bigquery

ECOMM_DIR = os.path.abspath(os.path.join(_HERE, ".."))
PRODUCTS_IMAGE_PARQUET = os.path.join(ECOMM_DIR, "data", "products_image.parquet")
STYLES_PARQUET = os.path.join(ECOMM_DIR, "cache", "sf_500", "styles_details.parquet")
PROFILE_PATH = os.path.join(ECOMM_DIR, "outputs", "Q12.json")
BASELINE_CACHE = os.path.join(ECOMM_DIR, "outputs", "Q12_baseline_cache.json")

PROJECT = os.environ.get("GCP_PROJECT", "")
DATASET = "fashion_product_images"
GCS_BUCKET = f"{PROJECT}-mmb-fashion-product-images-bucket"
STAGING_TABLE = f"{DATASET}.q12_reps"

ALLOWED_MC = ["Accessories", "Apparel", "Footwear"]
TAU_DIST = 0.05
PAPER_BQ_Q12 = {"score_f1": 0.97, "latency_s": 31.1, "cost_usd": 0.10}
PAPER_DASE_NN_Q12 = {"score_f1": 0.85, "latency_s": 0.7, "cost_usd": 6e-6}

GENERATE_PROMPT = """
    You are given a product description and an image of the product as well as the product id.
    The product contains a fashion item (clothing, shoes, accessories, etc).
    There might be multiple fashion items in the image, especially when a model is presenting them.
    If this is the case, focus only on the primary fashion item and use the description to determine which item in the image is of interest.

    For each product, generate the following JSON:
    ```
    {
        "id": <product id> (integer),
        "brand": <extract the brand name from the description and/or image. use lower-case letters for the brand name>",
        "category": <classify the images into ''accessories'', ''apparel'', ''footwear''>
    }
    ```

    Output the json in a single line.
    Keep the order of the keys in the JSON as given in the description.
    Do not use spaces between { or keys and values in the JSON, i.e., do no use spaces anywhere in the JSON structure.
    Use normal quotes in the JSON; do not use single quotes.

    The id, description and the image are as follows:
    """
FILTER_PROMPT = "Does the following description describe a product from either Adidas or Puma?"


def _q12_sql_for(table: str, with_rep_id: bool) -> str:
    """Verbatim sembench Q12 SQL with optional rep_id projection (cascade variant)."""
    extra = ", styles_details.id AS rep_id" if with_rep_id else ""
    return f"""
SELECT
  AI.GENERATE(
    ('''{GENERATE_PROMPT}''',
     CAST(styles_details.id AS STRING), ' ',
     styles_details.productDisplayName, ' ',
     styles_details.productDescriptors.description.value, ' ',
     images.ref),
    connection_id => 'us.connection',
    endpoint => 'gemini-2.5-flash'
  ).result AS id{extra}
FROM {table} AS styles_details
JOIN {DATASET}.IMAGE_MAPPING AS image_mapping
  ON image_mapping.link = styles_details.styleImages.default.imageURL
JOIN EXTERNAL_OBJECT_TRANSFORM(TABLE `{DATASET}.IMAGES`, ['SIGNED_URL']) as images
  ON ARRAY_LAST(SPLIT(images.uri, '/')) = image_mapping.filename
WHERE true
  AND styles_details.masterCategory.typeName in ('Accessories', 'Apparel', 'Footwear')
  AND AI.IF(
    ('{FILTER_PROMPT}',
     styles_details.productDisplayName, ' ',
     styles_details.productDescriptors.description.value),
    connection_id => 'us.connection',
    endpoint => 'gemini-2.5-flash'
  )
"""


# Bespoke calibration: filter (text-only) + generate (image+text).
def per_row_cost_q12(client, sample_uris, sample_titles, sample_descrs, k=10):
    THINKING = "JSON '{\"generation_config\":{\"thinking_config\":{\"thinking_budget\":0}}}'"
    selects_filter = []
    selects_gen = []
    params = []
    for i in range(k):
        selects_filter.append(f"""
        SELECT AI.GENERATE_BOOL(
          ('{FILTER_PROMPT}', @t_{i}, ' ', @d_{i}),
          connection_id => 'us.connection',
          endpoint => 'gemini-2.5-flash',
          model_params => {THINKING}
        ) AS verdict""")
        selects_gen.append(f"""
        SELECT AI.GENERATE_BOOL(
          ('Is this a fashion product? Just check the image and description: ',
           @t_{i}, ' ', @d_{i}, ' ', img.ref),
          connection_id => 'us.connection',
          endpoint => 'gemini-2.5-flash',
          model_params => {THINKING}
        ) AS verdict
        FROM EXTERNAL_OBJECT_TRANSFORM(TABLE {DATASET}.IMAGES, ['SIGNED_URL']) AS img
        WHERE img.uri = @uri_{i}""")
        params += [
            bigquery.ScalarQueryParameter(f"uri_{i}", "STRING", sample_uris[i]),
            bigquery.ScalarQueryParameter(f"t_{i}", "STRING", sample_titles[i]),
            bigquery.ScalarQueryParameter(f"d_{i}", "STRING", sample_descrs[i]),
        ]

    def _eval(sql_pieces, label):
        sql = " UNION ALL ".join(sql_pieces)
        cfg = bigquery.QueryJobConfig(query_parameters=params, use_query_cache=False)
        import time as _t
        t0 = _t.time()
        df = client.query(sql, job_config=cfg).result().to_dataframe()
        elapsed = _t.time() - t0
        p_other, p_audio, out, thoughts = _sum_tokens(df["verdict"])
        n = len(df)
        cost = _to_cost(p_other, p_audio, out, thoughts)
        return {"label": label, "n": n, "elapsed_s": elapsed, "cost_usd": cost,
                "tokens": {"prompt_other": p_other, "prompt_audio": p_audio,
                           "output": out, "thoughts": thoughts}}

    f_cal = _eval(selects_filter, "filter (text-only AI.IF proxy)")
    g_cal = _eval(selects_gen, "generate (image+text AI.GENERATE proxy)")
    per_row_filter = f_cal["cost_usd"] / f_cal["n"]
    per_row_gen = g_cal["cost_usd"] / g_cal["n"]
    return {
        "method": "AI.GENERATE_BOOL proxy: filter rate + generate rate (sum)",
        "filter_per_row_usd": per_row_filter,
        "generate_per_row_usd": per_row_gen,
        "per_row_cost_usd": per_row_filter + per_row_gen,
        "filter_calibration": f_cal,
        "generate_calibration": g_cal,
    }


def parse_brand_category(json_str):
    """Parse BQ AI.GENERATE output JSON to (brand, category). Return None on failure."""
    try:
        s = json_str.strip()
        if s.startswith("```"):
            s = s.strip("`").lstrip("json").strip()
        d = json.loads(s)
        b = d.get("brand")
        c = d.get("category")
        if b is None or c is None:
            return None
        return (str(b).lower(), str(c).lower())
    except Exception:
        return None


def make_json_id(pid, brand, category):
    return json.dumps({"id": int(pid), "brand": brand, "category": category},
                      separators=(",", ":"), ensure_ascii=False)


def main():
    profile = build_profile(
        scenario="ecomm", query_id=12, scale_factor=500,
        params={"tau_distance_cosine": TAU_DIST, "rep_strategy": "centroid_nearest"},
        cascade_form=(
            f"M-cascade (cluster + propagate, sem_filter+sem_map): "
            f"AgglomerativeClustering(image-cap emb, cosine, complete, "
            f"distance_threshold={TAU_DIST}) on 477 in-scope; "
            "AiGenerateVerifier wrapping Q12 SQL on K reps; per-cluster rep filter+JSON "
            "→ propagate (brand, category) to all members with their own id."
        ),
    )

    print("Loading + clustering ...")
    sdf = pd.read_parquet(STYLES_PARQUET)
    def gm(x): return x.get("typeName") if isinstance(x, dict) else None
    sdf["mc"] = sdf["masterCategory"].apply(gm)
    sdf = sdf[sdf["mc"].isin(ALLOWED_MC)].copy()
    sdf["id"] = sdf["id"].astype(np.int64)
    keep_ids = sdf["id"].tolist()
    keep_set = set(keep_ids)

    pdf = pd.read_parquet(PRODUCTS_IMAGE_PARQUET)
    pdf["Id"] = pdf["Id"].astype(np.int64)
    pdf = pdf[pdf["Id"].isin(keep_set)].copy()
    pdf = pdf.set_index("Id").loc[keep_ids].reset_index()
    n = len(pdf)
    img_emb = np.stack(pdf["embedding"].tolist()).astype(np.float32)
    ids = pdf["Id"].astype(int).tolist()
    print(f"  in-scope: {n} products (mc IN {ALLOWED_MC})")

    # GT
    gt_rows = sdf[sdf["brandName"].str.lower().isin(["adidas", "puma"])]
    gt_jsons = set()
    for _, r in gt_rows.iterrows():
        gt_jsons.add(make_json_id(int(r["id"]), str(r["brandName"]).lower(), str(r["mc"]).lower()))
    print(f"  GT JSONs: {len(gt_jsons)}")
    profile["data"] = {"n_products_in_scope": n,
                       "scope_filter": f"masterCategory IN {ALLOWED_MC}",
                       "n_gt_jsons": len(gt_jsons)}

    client = bq_client(PROJECT)

    # Calibration
    print("\n=== Per-row cost calibration ===")
    sample_uris, sample_titles, sample_descrs = [], [], []
    sdf_indexed = sdf.set_index("id")
    for i in range(min(10, n)):
        pid = int(ids[i])
        sample_uris.append(f"gs://{GCS_BUCKET}/{pid}.jpg")
        row = sdf_indexed.loc[pid]
        sample_titles.append(str(row["productDisplayName"] or ""))
        try:
            d = (row["productDescriptors"] or {}).get("description", {}).get("value", "") or ""
        except Exception:
            d = ""
        sample_descrs.append(d[:500])
    cal = per_row_cost_q12(client, sample_uris, sample_titles, sample_descrs, k=10)
    per_row = cal["per_row_cost_usd"]
    print(f"  per_row=${per_row:.6f} (filter ${cal['filter_per_row_usd']:.6f} + "
          f"generate ${cal['generate_per_row_usd']:.6f})")
    profile["calibration"] = cal

    # Custom verifier returning rep_id + parsed (brand, category) tuple as the value.
    # We let AiGenerateVerifier hand us a value_column, then post-process to a tuple.
    # ClusterCascade will propagate this tuple via cluster_to_value mapping.
    def make_staging(rep_ids_):
        id_list = ",".join(str(int(i)) for i in rep_ids_)
        return f"""
        CREATE OR REPLACE TABLE {STAGING_TABLE} AS
        SELECT * FROM {DATASET}.STYLES_DETAILS WHERE id IN ({id_list})
        """
    verifier = AiGenerateVerifier(
        verify_sql=_q12_sql_for(STAGING_TABLE, with_rep_id=True),
        make_staging_sql=make_staging,
        id_column="rep_id", value_column="id",   # 'id' col is the AI.GENERATE JSON
        coerce_id=int,
        coerce_value=lambda x: parse_brand_category(str(x)),  # may return None
    )

    # ── Cluster cascade ──
    print(f"\n=== ClusterCascade(Agglomerative, distance_threshold={TAU_DIST}) ===")
    clusterer = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="complete",
        distance_threshold=TAU_DIST,
    )
    cluster_cascade = ClusterCascade(
        embeddings=img_emb, ids=ids,
        clusterer=clusterer,
        verifier=verifier,
        rep_strategy="centroid_nearest",
    )
    cres = cluster_cascade.run(client, per_row)
    print(f"  K clusters = {cres.n_clusters}; size stats = {cres.cluster_size_stats()}")
    print(f"  verifier: wall={cres.verifier_result.wall_s:.2f}s slot={cres.verifier_result.slot_ms} "
          f"calls={cres.verifier_result.n_calls}")

    # Build cascade JSON set: every member of a cluster whose rep parsed → JSON.
    K = cres.n_clusters
    rep_id_to_json = {rid: v for rid, v in cres.verifier_result.values.items() if v}
    n_rep_pass = len(rep_id_to_json)
    print(f"  parsed {n_rep_pass} JSONs from {K} reps")

    cascade_jsons = set()
    for pid, bc in cres.predicted.items():
        if bc is None or bc == "UNKNOWN" or not isinstance(bc, tuple):
            continue
        brand, cat = bc
        cascade_jsons.add(make_json_id(int(pid), brand, cat))

    cp, cr, c_f1 = f1_set(cascade_jsons, gt_jsons)
    print(f"  cascade {len(cascade_jsons)} JSONs; P={cp:.4f} R={cr:.4f} F1={c_f1:.4f}")

    # Cost: filter*K + generate*n_rep_pass
    s2_filter_calls = K
    s2_generate_calls = n_rep_pass
    s2_calls = s2_filter_calls + s2_generate_calls
    cascade_cost = (cal["filter_per_row_usd"] * s2_filter_calls
                    + cal["generate_per_row_usd"] * s2_generate_calls)
    cascade_total_wall = cres.timings_s["cluster_fit"] + cres.timings_s["verify_total"]
    cascade_total_slot = cres.verifier_result.ctas_slot_ms + cres.verifier_result.slot_ms

    profile["dase_partition"] = cres.to_dict()

    # Baseline (cached)
    if os.path.exists(BASELINE_CACHE):
        print(f"\n=== Baseline (cached from {BASELINE_CACHE}) ===")
        with open(BASELINE_CACHE) as f:
            cache = json.load(f)
        bres_jsons = set(cache["bres_jsons"])
        bwall = cache["wall_s"]; bslot = cache.get("slot_ms")
    else:
        print("\n=== Baseline (sembench q12.sql verbatim on STYLES_DETAILS) ===")
        bdf, bwall, bslot, _ = run_query(client, _q12_sql_for(f"{DATASET}.STYLES_DETAILS", with_rep_id=False))
        bres_jsons = set()
        for _, row in bdf.iterrows():
            s = str(row["id"]).strip()
            try:
                d = json.loads(s)
                bres_jsons.add(make_json_id(int(d["id"]), str(d["brand"]).lower(), str(d["category"]).lower()))
            except Exception:
                pass
        with open(BASELINE_CACHE, "w") as f:
            json.dump({"bres_jsons": sorted(bres_jsons), "wall_s": bwall, "slot_ms": bslot}, f, indent=2)
        print(f"  cached to {BASELINE_CACHE}")
    bp, br, b_f1 = f1_set(bres_jsons, gt_jsons)
    n_pass_baseline = len(bres_jsons)
    bcalls = n + n_pass_baseline
    bcost = cal["filter_per_row_usd"] * n + cal["generate_per_row_usd"] * n_pass_baseline
    print(f"  returned {len(bres_jsons)} JSONs; P={bp:.4f} R={br:.4f} F1={b_f1:.4f}")
    print(f"  wall={bwall:.2f}s slot={bslot} n_calls={bcalls} cost=${bcost:.6f}")
    profile["baseline"] = {
        "method": "sembench bigquery/q12.sql verbatim on STYLES_DETAILS",
        "sql": _q12_sql_for(f"{DATASET}.STYLES_DETAILS", with_rep_id=False).strip(),
        "score": {"precision": bp, "recall": br, "f1_score": b_f1},
        "latency_breakdown": {"wall_s": bwall, "slot_ms": bslot},
        "cost_breakdown": {"n_llm_calls": bcalls,
                           "n_llm_calls_method": "n_in_scope filter + generate per filter-pass row",
                           "per_row_cost_usd": per_row, "total_cost_usd": bcost},
    }

    profile["cascade"] = {
        "method": ("M-cascade ClusterCascade(Agglomerative) → AiGenerateVerifier wrapping Q12 SQL "
                   "on K reps → propagate (brand, category) to all cluster members."),
        "verifier": cres.verifier_result.to_dict(),
        "n_reps_total": K, "n_reps_filter_pass": n_rep_pass,
        "score": {"precision": cp, "recall": cr, "f1_score": c_f1},
        "totals": {
            "wall_s": cascade_total_wall, "slot_ms_bq_total": cascade_total_slot,
            "cost_usd": cascade_cost, "n_llm_calls": s2_calls,
            "n_llm_calls_breakdown": {"filter": s2_filter_calls, "generate": s2_generate_calls},
        },
    }

    profile["comparison"] = {
        "score": {"paper_BQ": PAPER_BQ_Q12["score_f1"], "paper_DASE_NN": PAPER_DASE_NN_Q12["score_f1"],
                  "ours_BQ": b_f1, "ours_cascade": c_f1},
        "wall_s": {"paper_BQ": PAPER_BQ_Q12["latency_s"], "paper_DASE_NN": PAPER_DASE_NN_Q12["latency_s"],
                   "ours_BQ": bwall, "ours_cascade": cascade_total_wall},
        "cost_usd": {"paper_BQ": PAPER_BQ_Q12["cost_usd"], "paper_DASE_NN": PAPER_DASE_NN_Q12["cost_usd"],
                     "ours_BQ": bcost, "ours_cascade": cascade_cost},
        "n_llm_calls": {"paper_BQ": bcalls, "paper_DASE_NN": 0,
                        "ours_BQ": bcalls, "ours_cascade": s2_calls},
    }

    write_profile(profile, PROFILE_PATH)

    print_summary(
        f"Ecomm Q12 (ClusterCascade)",
        columns=["paper BQ", "DASE+NN", "ours BQ", "ours cascade"],
        rows=[
            ("F1",         [PAPER_BQ_Q12["score_f1"], PAPER_DASE_NN_Q12["score_f1"], b_f1, c_f1], ".2f"),
            ("wall (s)",   [PAPER_BQ_Q12["latency_s"], PAPER_DASE_NN_Q12["latency_s"], bwall, cascade_total_wall], ".2f"),
            ("cost ($)",   [PAPER_BQ_Q12["cost_usd"], PAPER_DASE_NN_Q12["cost_usd"], bcost, cascade_cost], ".4f"),
            ("#LLM calls", [bcalls, 0, bcalls, s2_calls], "d"),
        ],
    )


if __name__ == "__main__":
    main()
