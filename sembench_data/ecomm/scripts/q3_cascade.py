#!/usr/bin/env -S python -u
"""
Ecomm Q3 cascade — clustering + representative-sampling for SEM_MAP.

NL: Extract brand name from each product description.
GT: 500 (id, brandName) pairs; ~171 unique brands.
Eval: Adjusted Rand Index (ARI) between predicted and GT brand clustering.

Refactored to use dase_cascade.ClusterCascade + AiGenerateVerifier.
Operator (paper Table 3): M (cluster-based prefilter for SEM_MAP, distinct
algorithmic primitive from F/J — see ClusterCascade docstring).

Pipeline:
  1. AgglomerativeClustering(metric=cosine, linkage=complete, distance_threshold=TAU)
     groups 500 products into K tight clusters (sim > 1-TAU).
  2. ClusterCascade picks one centroid-nearest representative per cluster.
  3. AiGenerateVerifier runs sembench's verbatim AI.GENERATE on the K reps.
  4. Label propagation: every product in cluster c → brand of rep(c).
  5. ARI vs GT.
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
    bq_client, per_row_cost, run_query,
    ari_score, build_profile, write_profile,
)

ECOMM_DIR = os.path.abspath(os.path.join(_HERE, ".."))
PRODUCTS_PARQUET = os.path.join(ECOMM_DIR, "data", "products_text.parquet")
STYLES_PARQUET   = os.path.join(ECOMM_DIR, "cache", "sf_500", "styles_details.parquet")
PROFILE_PATH     = os.path.join(ECOMM_DIR, "outputs", "Q3.json")

PROJECT = os.environ.get("GCP_PROJECT", "")
DATASET = "fashion_product_images"
STAGING = f"{DATASET}.q3_reps"

TAU_DIST = 0.10                                        # cosine distance; sim > 0.90

# Verbatim sembench Q3 BQ template — cascade reuses with table swap.
def _q3_sql_for(table: str) -> str:
    return f"""
    SELECT
      id,
      AI.GENERATE(
        ('Extract the brand name from the following product description. Only return the brand name, nothing else: ',
         styles_details.productDisplayName, ' ',
         styles_details.productDescriptors.description.value),
        connection_id => 'us.connection',
        endpoint => 'gemini-2.5-flash'
      ).result AS category
    FROM {table} AS styles_details
    """


def make_q3_verifier():
    """ClusterCascade hands K rep ids to the verifier; CTAS staging table from
    those ids, then run verbatim Q3 AI.GENERATE on staging."""
    def make_staging(ids):
        id_list = ",".join(str(int(i)) for i in ids)
        return f"""
        CREATE OR REPLACE TABLE {STAGING} AS
        SELECT * FROM {DATASET}.STYLES_DETAILS WHERE id IN ({id_list})
        """
    return AiGenerateVerifier(
        verify_sql=_q3_sql_for(STAGING),
        make_staging_sql=make_staging,
        id_column="id", value_column="category",
        coerce_id=int,
    )


def main():
    profile = build_profile(
        scenario="ecomm", query_id=3, scale_factor=500,
        params={"tau_distance_cosine": TAU_DIST, "rep_strategy": "centroid_nearest"},
        cascade_form=(
            f"ClusterCascade(AgglomerativeClustering(cosine, complete, threshold={TAU_DIST})) "
            "+ AiGenerateVerifier on K representatives + label propagation per cluster."
        ),
    )

    print("Loading products + clustering...")
    pdf = pd.read_parquet(PRODUCTS_PARQUET)
    sdf = pd.read_parquet(STYLES_PARQUET)
    n_total = len(pdf)
    embeddings = np.stack(pdf["embedding"].tolist()).astype(np.float32)
    ids = pdf["Id"].astype(int).tolist()
    gt_map = {int(row["id"]): str(row.get("brandName") or "") for _, row in sdf.iterrows()}
    n_gt_brands = len(set(v for v in gt_map.values() if v))
    print(f"  {n_total} products; {n_gt_brands} unique GT brands")
    profile["data"] = {"n_products": n_total, "n_gt_unique_brands": n_gt_brands}

    client = bq_client(PROJECT)

    print("\n=== Per-row cost calibration ===")
    sample_texts = [str(pdf.iloc[i]["text"]) for i in range(min(10, n_total))]
    cal = per_row_cost(
        client,
        prompt="Is this product description meaningful? ",
        sample_texts=sample_texts,
        method_label="AI.GENERATE_BOOL proxy with similar product-text prompt",
        k=10,
    )
    per_row = cal.per_row_cost_usd
    print(f"  per_row=${per_row:.6f}")
    profile["calibration"] = cal.to_dict()

    # ── Cluster cascade ──
    print(f"\n=== ClusterCascade(Agglomerative, distance_threshold={TAU_DIST}) ===")
    clusterer = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="complete",
        distance_threshold=TAU_DIST,
    )
    cluster_cascade = ClusterCascade(
        embeddings=embeddings, ids=ids,
        clusterer=clusterer,
        verifier=make_q3_verifier(),
        rep_strategy="centroid_nearest",
    )
    cres = cluster_cascade.run(client, per_row)
    print(f"  K clusters = {cres.n_clusters}; size stats = {cres.cluster_size_stats()}")
    print(f"  verifier: wall={cres.verifier_result.wall_s:.2f}s slot={cres.verifier_result.slot_ms} "
          f"calls={cres.verifier_result.n_calls} cost=${cres.verifier_result.cost_usd:.6f}")

    # ── Eval ARI ──
    common_ids = sorted(set(cres.predicted.keys()) & set(gt_map.keys()))
    pred_labels = [cres.predicted[i] for i in common_ids]
    gt_labels   = [gt_map[i] for i in common_ids]
    c_ari = ari_score(pred_labels, gt_labels)
    print(f"  cascade ARI={c_ari:.4f}")

    cascade_total_wall = (
        cres.timings_s["cluster_fit"]
        + cres.timings_s["verify_total"]
    )
    cascade_total_slot = cres.verifier_result.ctas_slot_ms + cres.verifier_result.slot_ms
    profile["dase_partition"] = cres.to_dict()
    profile["cascade"] = {
        "method": "ClusterCascade(Agglomerative) → AiGenerateVerifier on reps + label propagation",
        "verifier": cres.verifier_result.to_dict(),
        "score": {"ari": c_ari},
        "totals": {"wall_s": cascade_total_wall, "slot_ms_bq_total": cascade_total_slot,
                   "cost_usd": cres.verifier_result.cost_usd, "n_llm_calls": cres.verifier_result.n_calls},
    }



    write_profile(profile, PROFILE_PATH)



if __name__ == "__main__":
    main()
