#!/usr/bin/env -S python -u
"""
Ecomm Q6 cascade — sem_classify on image (5 categories), 5-anchor argmax + BQ on uncertain.

Q6 = Q5 with image input. Same 5 categories, same scope (228 Apparel ∖ excl), same GT.
Eval: ARI.

Refactored to use dase_cascade. Operator (paper Table 3): M (sem_classify).
Note: Despite the original docstring banner, Q6 is M (multi-class classification),
not C (count). Its Eval is ARI.

Inlined argmax helper (n-class top1−top2 confidence) + AbsoluteBand on confidence
+ AiGenerateVerifier wrapping AI.CLASSIFY (image variant).
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
PROFILE_PATH = os.path.join(ECOMM_DIR, "outputs", "Q6.json")

PROJECT = os.environ.get("GCP_PROJECT", "")
DATASET = "fashion_product_images"
GCS_BUCKET = f"{PROJECT}-mmb-fashion-product-images-bucket"
STAGING_TABLE = f"{DATASET}.q6_uncertain"

CATEGORIES = ["Dress", "Bottomwear", "Socks", "Topwear", "Innerwear"]
CATEGORY_ANCHORS = {
    "Dress": "Dress: A dress is a one-piece outer garment that is worn on the torso, hangs down over the legs, and often consist of a bodice attached to a skirt.",
    "Bottomwear": "Bottomwear: Bottomwear refers to clothing worn on the lower part of the body, such as trousers, jeans, skirts, shorts, and leggings.",
    "Socks": "Socks: Socks are a type of clothing worn on the feet, typically made of soft fabric, designed to provide comfort and warmth.",
    "Topwear": "Topwear: Topwear refers to clothing worn on the upper part of the body, such as shirts, blouses, t-shirts, and jackets.",
    "Innerwear": "Innerwear: Innerwear refers to clothing worn beneath outer garments, typically close to the skin, such as underwear, bras, and undershirts.",
}

TAU_HIGH = 0.05


def _q6_sql_for(table: str) -> str:
    return f"""
WITH product_selection AS (
  SELECT images.*
  FROM {table} styles_details
  JOIN {DATASET}.IMAGE_MAPPING mapping
    ON styles_details.styleImages.default.imageURL = mapping.link
  JOIN EXTERNAL_OBJECT_TRANSFORM(TABLE `{DATASET}.IMAGES`, ['SIGNED_URL']) as images
    ON ARRAY_LAST(SPLIT(images.uri, '/')) = mapping.filename
  WHERE TRUE
    AND masterCategory.typeName = 'Apparel'
    AND subCategory.typeName NOT IN ('Saree', 'Apparel Set', 'Loungewear and Nightwear')
)
SELECT
  ARRAY_FIRST(SPLIT(ARRAY_LAST(SPLIT(images.uri, '/')), '.')) AS id,
  AI.CLASSIFY(
    ('You are given an image of a product. Your task is to classify the product. ', images.ref),
    categories => [
      ('Dress', 'A dress is a one-piece outer garment that is worn on the torso, hangs down over the legs, and often consist of a bodice attached to a skirt.'),
      ('Bottomwear', 'Bottomwear refers to clothing worn on the lower part of the body, such as trousers, jeans, skirts, shorts, and leggings.'),
      ('Socks', 'Socks are a type of clothing worn on the feet, typically made of soft fabric, designed to provide comfort and warmth.'),
      ('Topwear', 'Topwear refers to clothing worn on the upper part of the body, such as shirts, blouses, t-shirts, and jackets'),
      ('Innerwear', 'Innerwear refers to clothing worn beneath outer garments, typically close to the skin, such as underwear, bras, and undershirts.')
    ],
    connection_id => 'us.connection',
    endpoint => 'gemini-2.5-flash'
  ) AS category
FROM product_selection images
"""


def make_q6_verifier():
    def make_staging(ids):
        id_list = ",".join(str(int(i)) for i in ids)
        return f"""
        CREATE OR REPLACE TABLE {STAGING_TABLE} AS
        SELECT * FROM {DATASET}.STYLES_DETAILS WHERE id IN ({id_list})
        """
    # Q6 returns id as STRING from ARRAY_FIRST(SPLIT(...))
    return AiGenerateVerifier(
        verify_sql=_q6_sql_for(STAGING_TABLE),
        make_staging_sql=make_staging,
        id_column="id", value_column="category",
        coerce_id=lambda x: int(x),
    )


def argmax_classify(embeddings, anchor_texts):
    anchor_embs = embed_query(anchor_texts)
    sims = np.stack([cosine_sim_batch(a, embeddings) for a in anchor_embs], axis=1)
    argmax_idx = sims.argmax(axis=1)
    sorted_sims = np.sort(sims, axis=1)
    confidence = sorted_sims[:, -1] - sorted_sims[:, -2]
    return argmax_idx, confidence


def main():
    profile = build_profile(
        scenario="ecomm", query_id=6, scale_factor=500,
        params={"tau_high": TAU_HIGH, "categories": CATEGORIES},
        cascade_form=(
            "M-cascade: 5-anchor argmax classification on image-cap emb; AbsoluteBand on top1−top2 "
            "confidence (>TAU_HIGH → confident, else BQ); AiGenerateVerifier wrapping AI.CLASSIFY."
        ),
        extra={"category_anchors": CATEGORY_ANCHORS},
    )

    print("Loading + computing dase 5-class on image embedding ...")
    pdf_full = pd.read_parquet(IMAGES_PARQUET)
    sdf = pd.read_parquet(STYLES_PARQUET)
    def get_typename(x): return x.get("typeName") if isinstance(x, dict) else None
    sdf["m"] = sdf["masterCategory"].apply(get_typename)
    sdf["s"] = sdf["subCategory"].apply(get_typename)
    excluded = {"Saree", "Apparel Set", "Loungewear and Nightwear"}
    in_scope = (sdf["m"] == "Apparel") & (~sdf["s"].isin(excluded))
    valid_ids = set(sdf.loc[in_scope, "id"].astype(int).tolist())
    pdf_scope = pdf_full[pdf_full["Id"].isin(valid_ids)].reset_index(drop=True)
    n_total = len(pdf_scope)
    embeddings = np.stack(pdf_scope["embedding"].tolist()).astype(np.float32)
    gt_map = {int(r["id"]): str(r["s"]) for _, r in sdf[in_scope].iterrows()}
    print(f"  scope: {n_total} products")
    profile["data"] = {"n_products_in_scope": n_total,
                       "scope_filter": "Apparel - excluded subCategories"}

    import time as _t
    t0 = _t.time()
    anchor_texts = [CATEGORY_ANCHORS[c] for c in CATEGORIES]
    argmax_idx, confidence = argmax_classify(embeddings, anchor_texts)
    dase_cat = [CATEGORIES[i] for i in argmax_idx]
    band = AbsoluteBand(tau_low=-1.0, tau_high=TAU_HIGH)
    part = band.partition(confidence)
    confident_mask = np.zeros(n_total, dtype=bool); confident_mask[part.confident_pos] = True
    confident_idx = np.where(confident_mask)[0].tolist()
    uncertain_idx = np.where(~confident_mask)[0].tolist()
    uncertain_ids = [int(pdf_scope.iloc[i]["Id"]) for i in uncertain_idx]
    t_dase = _t.time() - t0

    print(f"  dase confident: {len(confident_idx)}, uncertain (→BQ): {len(uncertain_idx)}")
    print(f"  dase confidence range: [{confidence.min():.4f}, {confidence.max():.4f}]")
    print(f"  dase argmax distribution (confident only):")
    for c in CATEGORIES:
        n_c = sum(1 for i in confident_idx if dase_cat[i] == c)
        print(f"    {c}: {n_c}")

    profile["dase_breakdown"] = {"dase_compute_s": t_dase, "total_s": t_dase}
    profile["dase_partition"] = {"n_confident":len(confident_idx),"n_uncertain":len(uncertain_idx),
                                  "tau_high":TAU_HIGH,"uncertain_ids":uncertain_ids}

    client = bq_client(PROJECT)

    print("\n=== Per-row cost calibration ===")
    sample_uris = [f"gs://{GCS_BUCKET}/{int(pdf_scope.iloc[i]['Id'])}.jpg"
                   for i in range(min(10, n_total))]
    cal = per_row_cost(
        client,
        prompt="Is this an image of a fashion product?",
        sample_uris=sample_uris,
        ext_table=f"EXTERNAL_OBJECT_TRANSFORM(TABLE {DATASET}.IMAGES, ['SIGNED_URL']) AS",
        method_label="AI.GENERATE_BOOL on image-ref proxy + thinking=0",
        k=10,
    )
    per_row = cal.per_row_cost_usd
    print(f"  per_row=${per_row:.6f}")
    profile["calibration"] = cal.to_dict() | {
        "_caveat": "Q6 uses AI.CLASSIFY on image; per-row cost dominated by image input.",
    }


    print(f"\n=== Cascade: AiGenerateVerifier on {len(uncertain_ids)} uncertain ids ===")
    verifier = make_q6_verifier()
    if uncertain_ids:
        vres = verifier.verify(client, uncertain_ids, per_row)
        bq_cat_map = {int(k): v for k, v in vres.values.items()}
    else:
        from dase_cascade import VerifierResult
        vres = VerifierResult(positive_ids=set())
        bq_cat_map = {}
    print(f"  BQ returned {len(bq_cat_map)}; wall={vres.wall_s:.2f}s "
          f"slot={vres.slot_ms} cost=${vres.cost_usd:.6f}")

    cascade_pred = {}
    uncertain_set = set(uncertain_idx)
    for i in range(n_total):
        pid = int(pdf_scope.iloc[i]["Id"])
        if i in uncertain_set:
            cascade_pred[pid] = bq_cat_map.get(pid, "UNKNOWN")
        else:
            cascade_pred[pid] = dase_cat[i]
    ids_sorted = sorted(cascade_pred.keys() & gt_map.keys())
    c_ari = ari_score([cascade_pred[i] for i in ids_sorted], [gt_map[i] for i in ids_sorted])
    print(f"\n  cascade ARI={c_ari:.4f}")

    cascade_total_wall = t_dase + vres.ctas_wall_s + vres.wall_s
    cascade_total_slot = vres.ctas_slot_ms + vres.slot_ms
    profile["cascade"] = {
        "method":"M-cascade: 5-anchor argmax on image emb + threshold + AiGenerateVerifier on uncertain",
        "verifier": vres.to_dict(),
        "score":{"ari":float(c_ari)},
        "totals":{"wall_s":cascade_total_wall,"slot_ms_bq_total":cascade_total_slot,
                  "cost_usd":vres.cost_usd,"n_llm_calls":vres.n_calls},
    }

    write_profile(profile, PROFILE_PATH)



if __name__ == "__main__":
    main()
