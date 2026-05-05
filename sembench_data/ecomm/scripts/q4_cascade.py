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
    ari_score, build_profile, write_profile, print_summary, cosine_sim_batch,
)

ECOMM_DIR = os.path.abspath(os.path.join(_HERE, ".."))
IMAGES_PARQUET = os.path.join(ECOMM_DIR, "data", "products_image.parquet")
STYLES_PARQUET = os.path.join(ECOMM_DIR, "cache", "sf_500", "styles_details.parquet")
PROFILE_PATH = os.path.join(ECOMM_DIR, "outputs", "Q4.json")
BASELINE_CACHE = os.path.join(ECOMM_DIR, "outputs", "Q4_baseline_cache.json")

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
PAPER_BQ_Q4 = {"score_ari": 0.69, "latency_s": 31.0, "cost_usd": 0.37}
SKIP_BASELINE = False


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

    # Baseline
    if SKIP_BASELINE:
        b_ari = PAPER_BQ_Q4["score_ari"]; bwall = PAPER_BQ_Q4["latency_s"]; bslot = None
        bcost = PAPER_BQ_Q4["cost_usd"]; bcalls = n_total
        bres = {}
        profile["baseline"] = {"_status": "aborted", "method": "...",
            "score": {"ari": b_ari, "_source": "paper"},
            "latency_breakdown": {"wall_s": bwall, "slot_ms": None, "_source": "paper"},
            "cost_breakdown": {"n_llm_calls": bcalls, "per_row_cost_usd": per_row,
                               "total_cost_usd": bcost, "_source": "paper"}}
    elif os.path.exists(BASELINE_CACHE):
        print(f"\n=== Baseline (cached from {BASELINE_CACHE}) ===")
        with open(BASELINE_CACHE) as f:
            cache = json.load(f)
        bres = {int(k): v for k, v in cache["bres"].items()}
        bwall, bslot, b_ari, bcalls = cache["wall_s"], cache["slot_ms"], cache["ari"], cache["n_calls"]
        bcost = per_row * bcalls
        print(f"  cached: returned {len(bres)} (id, color); ARI={b_ari:.4f}, "
              f"wall={bwall:.2f}s, cost=${bcost:.6f}")
        profile["baseline"] = {
            "method": "sembench bigquery/q4.sql verbatim — CACHED", "_cache_source": BASELINE_CACHE,
            "sql": _q4_sql_for(f"{DATASET}.STYLES_DETAILS").strip(),
            "score": {"ari": float(b_ari)},
            "latency_breakdown": {"wall_s": bwall, "slot_ms": bslot},
            "cost_breakdown": {"n_llm_calls": bcalls, "per_row_cost_usd": per_row,
                               "total_cost_usd": bcost},
        }
    else:
        print("\n=== Baseline (sembench q4.sql verbatim on STYLES_DETAILS) ===")
        bdf, bwall, bslot, bsql = run_query(client, _q4_sql_for(f"{DATASET}.STYLES_DETAILS"))
        bres = {int(row["id"]): str(row["category"]).strip() for _, row in bdf.iterrows()}
        ids_sorted = sorted(bres.keys() & gt_map.keys())
        b_ari = ari_score([bres[i] for i in ids_sorted], [gt_map[i] for i in ids_sorted])
        bcalls = n_total
        bcost = per_row * bcalls
        print(f"  returned {len(bres)} (id, color); ARI={b_ari:.4f}; "
              f"wall={bwall:.2f}s slot={bslot}; cost=${bcost:.6f}")
        with open(BASELINE_CACHE, "w") as f:
            json.dump({
                "bres": {str(k): v for k, v in bres.items()},
                "wall_s": bwall, "slot_ms": bslot, "ari": float(b_ari), "n_calls": bcalls,
                "_note": "Cached BQ baseline output. Delete to force re-run.",
            }, f, indent=2)
        profile["baseline"] = {
            "method": "sembench bigquery/q4.sql verbatim", "sql": bsql,
            "score": {"ari": float(b_ari)},
            "latency_breakdown": {"wall_s": bwall, "slot_ms": bslot},
            "cost_breakdown": {"n_llm_calls": bcalls, "per_row_cost_usd": per_row,
                               "total_cost_usd": bcost},
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
    paper_n_calls = round(PAPER_BQ_Q4["cost_usd"] / per_row) if per_row else None
    profile["comparison"] = {
        "score": {"paper_BQ": PAPER_BQ_Q4["score_ari"], "paper_DASE_NN": None,
                  "ours_BQ": float(b_ari), "ours_cascade": float(c_ari)},
        "wall_s": {"paper_BQ": PAPER_BQ_Q4["latency_s"], "paper_DASE_NN": None,
                   "ours_BQ": bwall, "ours_cascade": cascade_total_wall},
        "cost_usd": {"paper_BQ": PAPER_BQ_Q4["cost_usd"], "paper_DASE_NN": None,
                     "ours_BQ": bcost, "ours_cascade": vres.cost_usd},
        "n_llm_calls": {"paper_BQ": paper_n_calls, "paper_DASE_NN": 0,
                        "ours_BQ": bcalls, "ours_cascade": vres.n_calls},
    }

    write_profile(profile, PROFILE_PATH)

    print_summary(
        f"Ecomm Q4",
        columns=["paper BQ", "ours BQ", "ours cascade"],
        rows=[
            ("ARI",        [PAPER_BQ_Q4["score_ari"], b_ari, c_ari], ".2f"),
            ("wall (s)",   [PAPER_BQ_Q4["latency_s"], bwall, cascade_total_wall], ".2f"),
            ("cost ($)",   [PAPER_BQ_Q4["cost_usd"], bcost, vres.cost_usd], ".4f"),
            ("#LLM calls", [paper_n_calls, bcalls, vres.n_calls], "d"),
        ],
    )


if __name__ == "__main__":
    main()
