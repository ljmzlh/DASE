#!/usr/bin/env -S python -u
"""
Ecomm Q2 cascade — yellow+silver sports shoes (image F operator with 3 sub-margins).

NL: find product ids of yellow+silver sports shoes (image-based).
GT: 5 ids = [10037, 3312, 10102, 3462, 41825].
Eval: F1 over id sets.

Refactored to use dase_cascade. Operator (paper Table 3): F (decomposed AND).

This Q is a special-case F: it computes 3 *independent* MarginSignals on the
same embeddings (shoes / yellow / silver) and AND-combines their per-row labels.
There's no Signal class for "3 independent margins AND-combined", so we use
3 MarginSignals + 3 AbsoluteBand partitions inline, then construct a partition
manually and run an AiIfVerifier on the resulting uncertain set.

Stage 2 verifier SQL uses a sub-EXTERNAL-TABLE on uncertain GCS URIs.
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import (
    MarginSignal, AbsoluteBand, AiIfVerifier,
    bq_client, per_row_cost, run_query,
    f1_set, build_profile, write_profile,
)

ECOMM_DIR = os.path.abspath(os.path.join(_HERE, ".."))
IMAGES_PARQUET = os.path.join(ECOMM_DIR, "data", "products_image.parquet")
STYLES_PARQUET = os.path.join(ECOMM_DIR, "cache", "sf_500", "styles_details.parquet")
PROFILE_PATH = os.path.join(ECOMM_DIR, "outputs", "Q2.json")

PROJECT = os.environ.get("GCP_PROJECT", "")
DATASET = "fashion_product_images"
GCS_BUCKET = f"{PROJECT}-mmb-fashion-product-images-bucket"
STAGING_TABLE = f"{DATASET}.q2_uncertain"

POS_SHOES = [
    "a pair of sports shoes",
    "athletic sneakers designed for sports",
    "running or training sports footwear",
]
NEG_SHOES = [
    "a product that is not a pair of sports shoes",
    "a fashion item that is not athletic footwear",
    "an apparel product that is not sports shoes",
]
POS_YELLOW = [
    "a product that has yellow in its color",
    "a fashion item featuring the color yellow",
    "a product whose colors include yellow",
]
NEG_YELLOW = [
    "a product that does not have yellow",
    "a product without any yellow color",
    "a product whose colors do not include yellow",
]
POS_SILVER = [
    "a product that has silver in its color",
    "a fashion item featuring the color silver",
    "a product whose colors include silver",
]
NEG_SILVER = [
    "a product that does not have silver",
    "a product without any silver color",
    "a product whose colors do not include silver",
]

TAU_HIGH = 0.10
TAU_LOW = -0.02


Q2_PROMPT = "The image shows a (pair of) sports shoe(s) that feature the colors yellow and silver."


def _q2_sql_for_external(table_ref: str) -> str:
    return f"""
SELECT
  ARRAY_FIRST(SPLIT(ARRAY_LAST(SPLIT(images.uri, '/')), '.')) as id
FROM EXTERNAL_OBJECT_TRANSFORM(TABLE `{table_ref}`, ['SIGNED_URL']) as images
WHERE true
  AND AI.IF(
    ('{Q2_PROMPT}', images.ref),
    connection_id => 'us.connection',
    endpoint => 'gemini-2.5-flash'
  )
"""


def make_q2_verifier():
    """Stage 1 builds an EXTERNAL TABLE over the uncertain GCS URIs;
    Stage 2 runs sembench q2.sql verbatim on it."""
    def make_staging(uris):
        uri_list = ", ".join(f"'{u}'" for u in uris)
        return f"""
        CREATE OR REPLACE EXTERNAL TABLE {STAGING_TABLE}
        WITH CONNECTION `us.connection`
        OPTIONS(
          object_metadata = 'SIMPLE',
          uris = [{uri_list}]
        )
        """
    return AiIfVerifier(
        verify_sql=_q2_sql_for_external(STAGING_TABLE),
        make_staging_sql=make_staging,
        id_column="id", coerce_id=int,
    )


def main():
    profile = build_profile(
        scenario="ecomm", query_id=2, scale_factor=500,
        params={"tau_high": TAU_HIGH, "tau_low": TAU_LOW},
        cascade_form=(
            "F-cascade (decomposed AND): 3× MarginSignal + AbsoluteBand on (shoes / yellow / silver); "
            "AND-combine: 3-yes → cascade YES, any-no → cascade NO, else → uncertain → "
            "AiIfVerifier on EXTERNAL-TABLE(uncertain URIs); cascade_ids = yes ∪ bq_yes."
        ),
        extra={"dase_prompts": {
            "shoes_pos": POS_SHOES, "shoes_neg": NEG_SHOES,
            "yellow_pos": POS_YELLOW, "yellow_neg": NEG_YELLOW,
            "silver_pos": POS_SILVER, "silver_neg": NEG_SILVER,
        }},
    )

    print("Loading 500 image captions + 3 sub-margins ...")
    pdf = pd.read_parquet(IMAGES_PARQUET)
    sdf = pd.read_parquet(STYLES_PARQUET)
    n_total = len(pdf)
    image_emb = np.stack(pdf["embedding"].tolist()).astype(np.float32)
    # GT
    gt_ids = set()
    for _, row in sdf.iterrows():
        at = row.get("articleType") or {}
        tn = at.get("typeName") if isinstance(at, dict) else None
        cs = {row.get("baseColour"), row.get("colour1"), row.get("colour2")}
        if tn == "Sports Shoes" and {"Yellow", "Silver"}.issubset(cs):
            gt_ids.add(int(row["id"]))
    n_gt = len(gt_ids)
    print(f"  {n_total} products, GT positive: {n_gt} ids = {sorted(gt_ids)}")
    profile["data"] = {"n_products": n_total, "n_gt_positive": n_gt,
                       "gt_ids": sorted(list(gt_ids))}

    import time as _t
    t0 = _t.time()
    shoes_margin  = MarginSignal(POS_SHOES,  NEG_SHOES ).compute(image_emb)
    yellow_margin = MarginSignal(POS_YELLOW, NEG_YELLOW).compute(image_emb)
    silver_margin = MarginSignal(POS_SILVER, NEG_SILVER).compute(image_emb)
    band = AbsoluteBand(tau_low=TAU_LOW, tau_high=TAU_HIGH)
    shoes_part  = band.partition(shoes_margin)
    yellow_part = band.partition(yellow_margin)
    silver_part = band.partition(silver_margin)
    t_dase = _t.time() - t0

    print(f"  shoes_margin   range: [{shoes_margin.min():+.3f}, {shoes_margin.max():+.3f}]")
    print(f"  yellow_margin  range: [{yellow_margin.min():+.3f}, {yellow_margin.max():+.3f}]")
    print(f"  silver_margin  range: [{silver_margin.min():+.3f}, {silver_margin.max():+.3f}]")

    # Per-row label per sub-filter: +1 yes, -1 no, 0 uncertain
    def part_to_labels(part):
        out = np.zeros(n_total, dtype=int)
        out[part.confident_pos] = 1
        out[part.confident_neg] = -1
        return out
    shoes_lbl  = part_to_labels(shoes_part)
    yellow_lbl = part_to_labels(yellow_part)
    silver_lbl = part_to_labels(silver_part)

    n_shoes_yes,  n_shoes_no  = int((shoes_lbl ==1).sum()),  int((shoes_lbl ==-1).sum())
    n_yellow_yes, n_yellow_no = int((yellow_lbl==1).sum()), int((yellow_lbl==-1).sum())
    n_silver_yes, n_silver_no = int((silver_lbl==1).sum()), int((silver_lbl==-1).sum())
    print(f"  shoes:  yes={n_shoes_yes}, no={n_shoes_no}, uncertain={n_total-n_shoes_yes-n_shoes_no}")
    print(f"  yellow: yes={n_yellow_yes}, no={n_yellow_no}, uncertain={n_total-n_yellow_yes-n_yellow_no}")
    print(f"  silver: yes={n_silver_yes}, no={n_silver_no}, uncertain={n_total-n_silver_yes-n_silver_no}")

    # AND-combine
    cascade_yes_idx, cascade_no_idx, uncertain_idx = [], [], []
    for i in range(n_total):
        labels = (shoes_lbl[i], yellow_lbl[i], silver_lbl[i])
        if -1 in labels:
            cascade_no_idx.append(i)
        elif all(l == 1 for l in labels):
            cascade_yes_idx.append(i)
        else:
            uncertain_idx.append(i)

    print(f"\n  cascade YES (3-yes): {len(cascade_yes_idx)}")
    print(f"  cascade NO  (any-no): {len(cascade_no_idx)}")
    print(f"  uncertain → BQ: {len(uncertain_idx)}")

    cascade_yes_ids = [int(pdf.iloc[i]["Id"]) for i in cascade_yes_idx]
    uncertain_ids = [int(pdf.iloc[i]["Id"]) for i in uncertain_idx]
    uncertain_uris = [f"gs://{GCS_BUCKET}/{pid}.jpg" for pid in uncertain_ids]

    profile["dase_breakdown"] = {"dase_compute_s": t_dase, "total_s": t_dase}
    profile["dase_partition"] = {
        "shoes":  {"yes": n_shoes_yes,  "no": n_shoes_no,  "uncertain": n_total-n_shoes_yes-n_shoes_no},
        "yellow": {"yes": n_yellow_yes, "no": n_yellow_no, "uncertain": n_total-n_yellow_yes-n_yellow_no},
        "silver": {"yes": n_silver_yes, "no": n_silver_no, "uncertain": n_total-n_silver_yes-n_silver_no},
        "AND_cascade_yes": len(cascade_yes_idx),
        "AND_cascade_no":  len(cascade_no_idx),
        "AND_uncertain_to_bq": len(uncertain_idx),
        "cascade_yes_ids": cascade_yes_ids,
        "uncertain_ids": uncertain_ids,
    }

    client = bq_client(PROJECT)

    print("\n=== Per-row cost calibration ===")
    sample_uris = [f"gs://{GCS_BUCKET}/{int(pdf.iloc[i]['Id'])}.jpg" for i in range(min(10, n_total))]
    cal = per_row_cost(
        client, Q2_PROMPT,
        sample_uris=sample_uris,
        ext_table=f"EXTERNAL_OBJECT_TRANSFORM(TABLE {DATASET}.IMAGES, ['SIGNED_URL']) AS",
        method_label="AI.GENERATE_BOOL on Q2 image prompt + thinking_budget=0",
        k=10,
    )
    per_row = cal.per_row_cost_usd
    print(f"  per_row=${per_row:.6f}, sample_cost=${cal.sample_cost_usd:.6f}, elapsed={cal.elapsed_s:.1f}s")
    profile["calibration"] = cal.to_dict()


    # ── Cascade Stage 1+2 via AiIfVerifier ──
    verifier = make_q2_verifier()
    print(f"\n=== Cascade Stage 1+2: AiIfVerifier on {len(uncertain_uris)} uncertain URIs ===")
    if uncertain_uris:
        vres = verifier.verify(client, uncertain_uris, per_row)
        bq_pos_uncertain_ids = set(int(x) for x in vres.positive_ids)
    else:
        from dase_cascade import VerifierResult
        vres = VerifierResult(positive_ids=set())
        bq_pos_uncertain_ids = set()
    print(f"  BQ returned {len(bq_pos_uncertain_ids)} positives; wall={vres.wall_s:.2f}s "
          f"slot={vres.slot_ms} cost=${vres.cost_usd:.6f}")

    cascade_ids = set(cascade_yes_ids) | bq_pos_uncertain_ids
    cp, cr, c_f1 = f1_set(cascade_ids, gt_ids)
    print(f"\n  cascade {len(cascade_ids)} ids ({len(cascade_yes_ids)} dase yes + "
          f"{len(bq_pos_uncertain_ids)} bq-uncertain-yes); P={cp:.4f} R={cr:.4f} F1={c_f1:.4f}")

    cascade_total_wall = t_dase + vres.ctas_wall_s + vres.wall_s
    cascade_total_slot = vres.ctas_slot_ms + vres.slot_ms
    profile["cascade"] = {
        "method": ("F-cascade (decomposed AND): 3 MarginSignal+AbsoluteBand sub-filters; AND-combine; "
                   "AiIfVerifier with sub-EXTERNAL-TABLE on uncertain URIs."),
        "verifier": vres.to_dict(),
        "cascade_ids": sorted(list(cascade_ids)),
        "score": {"precision": cp, "recall": cr, "f1_score": c_f1},
        "totals": {
            "wall_s": cascade_total_wall,
            "wall_breakdown_s": {"dase": t_dase, "bq_stage1_ctas": vres.ctas_wall_s, "bq_stage2": vres.wall_s},
            "slot_ms_bq_total": cascade_total_slot,
            "cost_usd": vres.cost_usd, "n_llm_calls": vres.n_calls,
        },
    }

    write_profile(profile, PROFILE_PATH)



if __name__ == "__main__":
    main()
