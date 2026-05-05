#!/usr/bin/env -S python -u
"""
Ecomm Q5 cascade — sem_classify on text (5 categories), 5-anchor argmax + BQ on uncertain.

NL: Classify each Apparel product into Dress / Bottomwear / Socks / Topwear / Innerwear.
GT: 228 products (Apparel ∖ excluded subCategories).
Eval: ARI.

Refactored to use dase_cascade. Operator (paper Table 3): M (sem_classify).

Multi-anchor argmax classification (top1−top2 confidence) is inlined as a
helper — there's no n-class Signal class in the package yet. AbsoluteBand on
confidence partitions; AiGenerateVerifier wraps AI.CLASSIFY (returns id +
chosen category as a string column).
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
    ari_score, build_profile, write_profile, print_summary, cosine_sim_batch,
)

ECOMM_DIR = os.path.abspath(os.path.join(_HERE, ".."))
PRODUCTS_PARQUET = os.path.join(ECOMM_DIR, "data", "products_text.parquet")
STYLES_PARQUET   = os.path.join(ECOMM_DIR, "cache", "sf_500", "styles_details.parquet")
PROFILE_PATH     = os.path.join(ECOMM_DIR, "outputs", "Q5.json")
BASELINE_CACHE   = os.path.join(ECOMM_DIR, "outputs", "Q5_baseline_cache.json")

PROJECT = os.environ.get("GCP_PROJECT", "")
DATASET = "fashion_product_images"
STAGING_TABLE = f"{DATASET}.q5_uncertain"

CATEGORIES = ["Dress", "Bottomwear", "Socks", "Topwear", "Innerwear"]
CATEGORY_ANCHORS = {
    "Dress": "Dress: A dress is a one-piece outer garment that is worn on the torso, hangs down over the legs, and often consist of a bodice attached to a skirt.",
    "Bottomwear": "Bottomwear: Bottomwear refers to clothing worn on the lower part of the body, such as trousers, jeans, skirts, shorts, and leggings.",
    "Socks": "Socks: Socks are a type of clothing worn on the feet, typically made of soft fabric, designed to provide comfort and warmth.",
    "Topwear": "Topwear: Topwear refers to clothing worn on the upper part of the body, such as shirts, blouses, t-shirts, and jackets.",
    "Innerwear": "Innerwear: Innerwear refers to clothing worn beneath outer garments, typically close to the skin, such as underwear, bras, and undershirts.",
}

TAU_HIGH = 0.05
PAPER_BQ_Q5 = {"score_ari": 0.98, "latency_s": 25.7, "cost_usd": 0.17}
PAPER_DASE_NN_Q5 = {"score_ari": 0.96, "latency_s": 0.7, "cost_usd": 4e-6}
SKIP_BASELINE = False


def _q5_sql_for(table: str) -> str:
    return f"""
WITH product_selection AS (
  SELECT *
  FROM {table} styles_details
  WHERE TRUE
    AND masterCategory.typeName = 'Apparel'
    AND subCategory.typeName NOT IN ('Saree', 'Apparel Set', 'Loungewear and Nightwear')
)
SELECT
  id,
  AI.CLASSIFY(
    ('You are given a description of a product. Your task is to classify the product. ',
     'The product description is as follows: ',
     styles_details.productDisplayName, ' ', styles_details.productDescriptors.description.value),
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
FROM product_selection styles_details
"""


def make_q5_verifier():
    def make_staging(ids):
        id_list = ",".join(str(int(i)) for i in ids)
        return f"""
        CREATE OR REPLACE TABLE {STAGING_TABLE} AS
        SELECT * FROM {DATASET}.STYLES_DETAILS WHERE id IN ({id_list})
        """
    return AiGenerateVerifier(
        verify_sql=_q5_sql_for(STAGING_TABLE),
        make_staging_sql=make_staging,
        id_column="id", value_column="category",
        coerce_id=int,
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
        scenario="ecomm", query_id=5, scale_factor=500,
        params={"tau_high": TAU_HIGH, "categories": CATEGORIES},
        cascade_form=(
            "M-cascade: 5-anchor argmax classification on text emb; AbsoluteBand on top1−top2 "
            "confidence (>TAU_HIGH → confident, else BQ); AiGenerateVerifier wrapping AI.CLASSIFY."
        ),
        extra={"category_anchors": CATEGORY_ANCHORS},
    )

    print("Loading products + computing dase 5-class classification ...")
    pdf_full = pd.read_parquet(PRODUCTS_PARQUET)
    sdf = pd.read_parquet(STYLES_PARQUET)
    def get_typename(x): return x.get("typeName") if isinstance(x, dict) else None
    sdf["m"] = sdf["masterCategory"].apply(get_typename)
    sdf["s"] = sdf["subCategory"].apply(get_typename)
    excluded = {"Saree", "Apparel Set", "Loungewear and Nightwear"}
    in_scope_mask = (sdf["m"] == "Apparel") & (~sdf["s"].isin(excluded))
    valid_ids = set(sdf.loc[in_scope_mask, "id"].astype(int).tolist())
    pdf_scope = pdf_full[pdf_full["Id"].isin(valid_ids)].reset_index(drop=True)
    n_total = len(pdf_scope)
    embeddings = np.stack(pdf_scope["embedding"].tolist()).astype(np.float32)
    gt_map = {int(r["id"]): str(r["s"]) for _, r in sdf[in_scope_mask].iterrows()}

    print(f"  scope (Apparel - excluded): {n_total} products")
    profile["data"] = {
        "n_products_in_scope": n_total,
        "scope_filter": "masterCategory='Apparel' AND subCategory NOT IN (Saree, Apparel Set, Loungewear and Nightwear)",
    }

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
    profile["dase_partition"] = {
        "n_confident": len(confident_idx),
        "n_uncertain": len(uncertain_idx),
        "tau_high": TAU_HIGH,
        "uncertain_ids": uncertain_ids,
    }

    client = bq_client(PROJECT)

    print("\n=== Per-row cost calibration ===")
    sample_texts = [str(pdf_scope.iloc[i]["text"])[:1000] for i in range(min(10, n_total))]
    cal = per_row_cost(
        client,
        prompt="Is this product description meaningful? ",
        sample_texts=sample_texts,
        method_label="AI.GENERATE_BOOL on product text + thinking_budget=0",
        k=10,
    )
    per_row = cal.per_row_cost_usd
    print(f"  per_row=${per_row:.6f}")
    profile["calibration"] = cal.to_dict() | {
        "_caveat": "Q5 uses AI.CLASSIFY (single-token output). Token usage similar to AI.GENERATE_BOOL.",
    }

    if SKIP_BASELINE:
        b_ari=PAPER_BQ_Q5["score_ari"]; bwall=PAPER_BQ_Q5["latency_s"]; bslot=None
        bcost=PAPER_BQ_Q5["cost_usd"]; bcalls=n_total
        bres = {}
        profile["baseline"] = {"_status":"aborted","method":"...",
            "score":{"ari":b_ari,"_source":"paper"},
            "latency_breakdown":{"wall_s":bwall,"slot_ms":None,"_source":"paper"},
            "cost_breakdown":{"n_llm_calls":bcalls,"per_row_cost_usd":per_row,
                              "total_cost_usd":bcost,"_source":"paper"}}
    elif os.path.exists(BASELINE_CACHE):
        print(f"\n=== Baseline (cached from {BASELINE_CACHE}) ===")
        with open(BASELINE_CACHE) as f: cache = json.load(f)
        bres = {int(k): v for k, v in cache["bres"].items()}
        bwall=cache["wall_s"]; bslot=cache["slot_ms"]; b_ari=cache["ari"]; bcalls=cache["n_calls"]
        bcost = per_row * bcalls
        print(f"  cached: returned {len(bres)} (id, category); ARI={b_ari:.4f}")
        profile["baseline"] = {"method":"sembench q5.sql verbatim — CACHED",
            "_cache_source":BASELINE_CACHE,
            "sql":_q5_sql_for(f"{DATASET}.STYLES_DETAILS").strip(),
            "score":{"ari":float(b_ari)},
            "latency_breakdown":{"wall_s":bwall,"slot_ms":bslot},
            "cost_breakdown":{"n_llm_calls":bcalls,"per_row_cost_usd":per_row,
                              "total_cost_usd":bcost}}
    else:
        print("\n=== Baseline (sembench q5.sql verbatim on STYLES_DETAILS) ===")
        bdf, bwall, bslot, bsql = run_query(client, _q5_sql_for(f"{DATASET}.STYLES_DETAILS"))
        bres = {int(row["id"]): str(row["category"]).strip() for _, row in bdf.iterrows()}
        ids_sorted = sorted(bres.keys() & gt_map.keys())
        b_ari = ari_score([bres[i] for i in ids_sorted], [gt_map[i] for i in ids_sorted])
        bcalls = n_total; bcost = per_row * bcalls
        print(f"  returned {len(bres)} (id, category); ARI={b_ari:.4f}; "
              f"wall={bwall:.2f}s slot={bslot}; cost=${bcost:.6f}")
        with open(BASELINE_CACHE, "w") as f:
            json.dump({"bres": {str(k): v for k, v in bres.items()},
                      "wall_s": bwall, "slot_ms": bslot, "ari": float(b_ari),
                      "n_calls": bcalls,
                      "_note": "Cached BQ baseline. Delete to force re-run."}, f, indent=2)
        profile["baseline"] = {"method":"sembench q5.sql verbatim", "sql": bsql,
            "score":{"ari":float(b_ari)},
            "latency_breakdown":{"wall_s":bwall,"slot_ms":bslot},
            "cost_breakdown":{"n_llm_calls":bcalls,"per_row_cost_usd":per_row,
                              "total_cost_usd":bcost}}

    # Cascade verifier
    print(f"\n=== Cascade: AiGenerateVerifier on {len(uncertain_ids)} uncertain ids ===")
    verifier = make_q5_verifier()
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
        "method": "M-cascade: 5-anchor argmax classification + threshold + AiGenerateVerifier on uncertain",
        "verifier": vres.to_dict(),
        "score": {"ari": float(c_ari)},
        "totals": {"wall_s": cascade_total_wall, "slot_ms_bq_total": cascade_total_slot,
                   "cost_usd": vres.cost_usd, "n_llm_calls": vres.n_calls},
    }
    paper_n_calls = round(PAPER_BQ_Q5["cost_usd"] / per_row) if per_row else None
    profile["comparison"] = {
        "score": {"paper_BQ": PAPER_BQ_Q5["score_ari"], "paper_DASE_NN": PAPER_DASE_NN_Q5["score_ari"],
                  "ours_BQ": float(b_ari), "ours_cascade": float(c_ari)},
        "wall_s": {"paper_BQ": PAPER_BQ_Q5["latency_s"], "paper_DASE_NN": PAPER_DASE_NN_Q5["latency_s"],
                   "ours_BQ": bwall, "ours_cascade": cascade_total_wall},
        "cost_usd": {"paper_BQ": PAPER_BQ_Q5["cost_usd"], "paper_DASE_NN": PAPER_DASE_NN_Q5["cost_usd"],
                     "ours_BQ": bcost, "ours_cascade": vres.cost_usd},
        "n_llm_calls": {"paper_BQ": paper_n_calls, "paper_DASE_NN": 0,
                        "ours_BQ": bcalls, "ours_cascade": vres.n_calls},
    }

    write_profile(profile, PROFILE_PATH)

    print_summary(
        f"Ecomm Q5",
        columns=["paper BQ", "DASE+NN", "ours BQ", "ours cascade"],
        rows=[
            ("ARI",        [PAPER_BQ_Q5["score_ari"], PAPER_DASE_NN_Q5["score_ari"], b_ari, c_ari], ".2f"),
            ("wall (s)",   [PAPER_BQ_Q5["latency_s"], PAPER_DASE_NN_Q5["latency_s"], bwall, cascade_total_wall], ".2f"),
            ("cost ($)",   [PAPER_BQ_Q5["cost_usd"], PAPER_DASE_NN_Q5["cost_usd"], bcost, vres.cost_usd], ".4f"),
            ("#LLM calls", [paper_n_calls, 0, bcalls, vres.n_calls], "d"),
        ],
    )


if __name__ == "__main__":
    main()
