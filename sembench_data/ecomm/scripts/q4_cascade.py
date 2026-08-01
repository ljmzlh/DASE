#!/usr/bin/env -S python -u
"""
Ecomm Q4 cascade — sem_map color extraction (image), 6-anchor argmax + BQ on uncertain.

NL: Extract primary color of each product (6 colors).
GT: 294 (id, baseColour) pairs.
Eval: ARI between predicted and GT color labels.

Refactored to use dase_cascade. Operator (paper Table 3): M (sem_map).

The dase prefilter is multi-anchor *argmax classification* with confidence =
top1 − top2; this isn't directly modeled by MarginSignal (binary). We compute
the n-class score inline as a small helper, then partition with AbsoluteBand
(confidence > TAU_HIGH → confident, else uncertain → BQ).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import (
    AbsoluteBand, AiGenerateVerifier,
    bq_client, embed_query, per_row_cost, run_query,
    ari_score, build_profile, write_profile, cosine_sim_batch,
)

ECOMM_DIR = os.path.abspath(os.path.join(_HERE, ".."))
IMAGES_PARQUET = os.path.join(ECOMM_DIR, "data", "products_image.parquet")
STYLES_PARQUET = os.path.join(ECOMM_DIR, "cache", "sf_500", "styles_details.parquet")
PROFILE_PATH = os.path.join(ECOMM_DIR, "outputs", "Q4.json")

PROJECT = os.environ.get("GCP_PROJECT", "")
DATASET = "fashion_product_images"
GCS_BUCKET = f"{PROJECT}-mmb-fashion-product-images-bucket"
STAGING_TABLE = f"{DATASET}.q4_uncertain"

COLORS = ["Black", "Blue", "Red", "White", "Orange", "Green"]
COLOR_ANCHORS = {
    "Black":  "a primarily black-colored fashion product",
    "Blue":   "a primarily blue-colored fashion product",
    "Red":    "a primarily red-colored fashion product",
    "White":  "a primarily white-colored fashion product",
    "Orange": "a primarily orange-colored fashion product",
    "Green":  "a primarily green-colored fashion product",
}

TAU_HIGH = 0.05


def _q4_sql_for(table: str) -> str:
    return f"""
WITH product_selection AS (
  SELECT images.*
  FROM {table} styles_details
  JOIN {DATASET}.IMAGE_MAPPING mapping
    ON styles_details.styleImages.default.imageURL = mapping.link
  JOIN EXTERNAL_OBJECT_TRANSFORM(TABLE `{DATASET}.IMAGES`, ['SIGNED_URL']) as images
    ON ARRAY_LAST(SPLIT(images.uri, '/')) = mapping.filename
  WHERE TRUE
    AND baseColour IN ('Black', 'Blue', 'Red', 'White', 'Orange', 'Green')
)
SELECT
  ARRAY_FIRST(SPLIT(ARRAY_LAST(SPLIT(images.uri, '/')), '.')) as id,
  AI.GENERATE(
    ('Extract the primary color of the product in the image. Only return the base color, nothing else: ',
     images.ref),
    connection_id => 'us.connection',
    endpoint => 'gemini-2.5-flash'
  ).result AS category
FROM product_selection as images
"""


def make_q4_verifier():
    def make_staging(ids):
        id_list = ",".join(str(int(i)) for i in ids)
        return f"""
        CREATE OR REPLACE TABLE {STAGING_TABLE} AS
        SELECT * FROM {DATASET}.STYLES_DETAILS WHERE id IN ({id_list})
        """
    # For Q4 the AI.GENERATE returns BQ id as STRING; coerce to int after squeeze.
    return AiGenerateVerifier(
        verify_sql=_q4_sql_for(STAGING_TABLE),
        make_staging_sql=make_staging,
        id_column="id", value_column="category",
        coerce_id=lambda x: int(x),
    )


def argmax_classify(embeddings: np.ndarray, anchor_texts):
    """Compute (argmax_label_idx, confidence=top1−top2) per row over n_anchors."""
    anchor_embs = embed_query(anchor_texts)
    sims = np.stack([cosine_sim_batch(a, embeddings) for a in anchor_embs], axis=1)  # (N, K)
    argmax_idx = sims.argmax(axis=1)
    sorted_sims = np.sort(sims, axis=1)
    confidence = sorted_sims[:, -1] - sorted_sims[:, -2]
    return argmax_idx, confidence


def main():
    profile = build_profile(
        scenario="ecomm", query_id=4, scale_factor=500,
        params={"tau_high": TAU_HIGH, "colors": COLORS},
        cascade_form=(
            "M-cascade: 6-anchor argmax classification (cosine sim) on image-cap emb; "
            "confidence = top1−top2; AbsoluteBand on confidence (>TAU_HIGH → confident, else BQ); "
            "AiGenerateVerifier on uncertain ids; merge dase argmax (confident) ∪ BQ output."
        ),
        extra={"color_anchors": COLOR_ANCHORS},
    )

    print("Loading products + computing 6-anchor argmax ...")
    pdf = pd.read_parquet(IMAGES_PARQUET)
    sdf = pd.read_parquet(STYLES_PARQUET)
    base_ok = sdf["baseColour"].isin(COLORS)
    valid_ids = set(sdf.loc[base_ok, "id"].astype(int).tolist())
    in_scope = pdf["Id"].isin(valid_ids)
    pdf_scope = pdf[in_scope].reset_index(drop=True)
    n_total = len(pdf_scope)
    embeddings = np.stack(pdf_scope["embedding"].tolist()).astype(np.float32)
    gt_map = {int(r["id"]): str(r["baseColour"]) for _, r in sdf[base_ok].iterrows()}
    print(f"  scope (baseColour in 6): {n_total} products")
    profile["data"] = {"n_products_in_scope": n_total,
                       "scope_filter": "baseColour IN (Black, Blue, Red, White, Orange, Green)"}

    import time as _t
    t0 = _t.time()
    anchor_texts = [COLOR_ANCHORS[c] for c in COLORS]
    argmax_idx, confidence = argmax_classify(embeddings, anchor_texts)
    dase_color = [COLORS[i] for i in argmax_idx]
    band = AbsoluteBand(tau_low=-1.0, tau_high=TAU_HIGH)
    part = band.partition(confidence)
    confident_mask = np.zeros(n_total, dtype=bool); confident_mask[part.confident_pos] = True
    confident_idx = np.where(confident_mask)[0].tolist()
    uncertain_idx = np.where(~confident_mask)[0].tolist()
    uncertain_ids = [int(pdf_scope.iloc[i]["Id"]) for i in uncertain_idx]
    t_dase = _t.time() - t0

    print(f"  dase confident: {len(confident_idx)}, uncertain (→BQ): {len(uncertain_idx)}")
    print(f"  dase confidence range: [{confidence.min():.4f}, {confidence.max():.4f}]")
    print(f"  dase argmax color distribution (confident only):")
    for c in COLORS:
        n_c = sum(1 for i in confident_idx if dase_color[i] == c)
        print(f"    {c}: {n_c}")

    profile["dase_breakdown"] = {"dase_compute_s": t_dase, "total_s": t_dase}
    profile["dase_partition"] = {
        "n_confident": len(confident_idx),
        "n_uncertain": len(uncertain_idx),
        "tau_high": TAU_HIGH,
        "uncertain_ids": uncertain_ids,
    }

    client = bq_client(PROJECT)

    print("\n=== Per-row cost calibration ===")
    sample_uris = [f"gs://{GCS_BUCKET}/{int(pdf_scope.iloc[i]['Id'])}.jpg" for i in range(min(10, n_total))]
    cal = per_row_cost(
        client,
        prompt="Is this image primarily showing a fashion product?",
        sample_uris=sample_uris,
        ext_table=f"EXTERNAL_OBJECT_TRANSFORM(TABLE {DATASET}.IMAGES, ['SIGNED_URL']) AS",
        method_label="AI.GENERATE_BOOL on image-ref proxy + thinking_budget=0",
        k=10,
    )
    per_row = cal.per_row_cost_usd
    print(f"  per_row=${per_row:.6f}")
    profile["calibration"] = cal.to_dict() | {
        "_caveat": "Q4 uses AI.GENERATE (free-form color string); per-row cost dominated by image input + ~3-token output. Proxy is close.",
    }


    # ── Cascade verifier on uncertain ──
    print(f"\n=== Cascade: AiGenerateVerifier on {len(uncertain_ids)} uncertain ids ===")
    verifier = make_q4_verifier()
    if uncertain_ids:
        vres = verifier.verify(client, uncertain_ids, per_row)
        bq_color_map = {int(k): v for k, v in vres.values.items()}
    else:
        from dase_cascade import VerifierResult
        vres = VerifierResult(positive_ids=set())
        bq_color_map = {}
    print(f"  BQ returned {len(bq_color_map)}; wall={vres.wall_s:.2f}s "
          f"slot={vres.slot_ms} cost=${vres.cost_usd:.6f}")

    # Merge
    cascade_pred = {}
    uncertain_set = set(uncertain_idx)
    for i in range(n_total):
        pid = int(pdf_scope.iloc[i]["Id"])
        if i in uncertain_set:
            cascade_pred[pid] = bq_color_map.get(pid, "UNKNOWN")
        else:
            cascade_pred[pid] = dase_color[i]
    ids_sorted = sorted(cascade_pred.keys() & gt_map.keys())
    c_ari = ari_score([cascade_pred[i] for i in ids_sorted], [gt_map[i] for i in ids_sorted])
    print(f"\n  cascade ARI={c_ari:.4f}")

    cascade_total_wall = t_dase + vres.ctas_wall_s + vres.wall_s
    cascade_total_slot = vres.ctas_slot_ms + vres.slot_ms
    profile["cascade"] = {
        "method": "M-cascade: 6-anchor argmax classification + threshold + AiGenerateVerifier on uncertain",
        "verifier": vres.to_dict(),
        "score": {"ari": float(c_ari)},
        "totals": {
            "wall_s": cascade_total_wall, "slot_ms_bq_total": cascade_total_slot,
            "cost_usd": vres.cost_usd, "n_llm_calls": vres.n_calls,
        },
    }

    write_profile(profile, PROFILE_PATH)



if __name__ == "__main__":
    main()
