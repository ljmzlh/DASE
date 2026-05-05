#!/usr/bin/env -S python -u
"""
Wildlife Q2 cascade — count elephant audios (single-modal F + COUNT aggregation).

NL: How many sound recordings of elephants do we have in our database?
GT: Animal == 'Elephant' → 5 of 66.
Eval: relative_error_score on COUNT.

Refactored to use dase_cascade unified solver. Operator (paper Table 3): F.
Audio analog of Q1: same Cascade primitive; only the modality (audio) and
the external-table / staging-column names change.
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import (
    Cascade, MarginSignal, AlphaBand, AiIfVerifier,
    bq_client, per_row_cost, run_query,
    relative_error_score, build_profile, write_profile, print_summary,
)

# ─── Paths / scenario constants ──────────────────────────────────────────
WILDLIFE_DIR = os.path.abspath(os.path.join(_HERE, ".."))
AUDIO_CSV    = os.path.join(WILDLIFE_DIR, "cache", "audio_data.csv")
EMB_PATH     = os.path.join(WILDLIFE_DIR, "data", "audio_embeddings.npz")
PROFILE_PATH = os.path.join(WILDLIFE_DIR, "outputs", "Q2.json")

PROJECT      = os.environ.get("GCP_PROJECT", "")
BUCKET       = f"{PROJECT}-animals_dataset"
DATASET      = "animals_dataset"
STAGING      = f"{DATASET}.q2_uncertain_mm"

PROMPT = "Does this audio contain an elephant sound? "

POSITIVE = [
    "a sound recording of an elephant",
    "audio of an elephant trumpeting or vocalizing",
    "elephant call sound clip",
]
NEGATIVE = [
    "a sound recording of an animal that is not an elephant",
    "audio of a non-elephant animal vocalization",
    "animal sound clip without any elephant",
]

ALPHA = 0.2
PAPER_BQ_Q2 = {"score": 0.19, "latency_s": 9.4, "cost_usd": 0.01}
PAPER_DASE_NN_Q2 = {"score": 0.83, "latency_s": 5e-4, "cost_usd": 1e-9}
SKIP_BASELINE = False


def make_elephant_audio_verifier():
    """Build the BQ verifier for Q2 (audio multimodal).

    Stage 1 (CTAS): join uncertain AudioPaths against audio_data_external to
                    get the multimodal `audio` ref column.
    Stage 2 (AI.IF): SELECT AudioPath FROM staging WHERE AI.IF — returns the
                    subset of uncertain URIs that BQ confirmed contain elephant
                    sounds. (Caller takes len(positive_set) for the COUNT.)
    """
    def make_staging(uris):
        items = ",".join(f"'{u}'" for u in uris)
        return f"""
        CREATE OR REPLACE TABLE {STAGING} AS
        SELECT m.AudioPath AS uri, ot.ref AS audio
        FROM {DATASET}.audio_data_files m
        JOIN {DATASET}.audio_data_external ot ON ot.uri = m.AudioPath
        WHERE m.AudioPath IN UNNEST([{items}])
        """

    verify_sql = f"""
    SELECT uri AS id FROM {STAGING}
    WHERE AI.IF(prompt => ('{PROMPT}', audio),
                connection_id => 'us.connection',
                endpoint => 'gemini-2.5-flash')
    """
    return AiIfVerifier(
        verify_sql=verify_sql, make_staging_sql=make_staging,
        id_column="id", coerce_id=str,
    )


def run_baseline(client):
    """Verbatim sembench Q2.sql on the full audio_data_mm — returns elephant count."""
    sql = f"""
    SELECT COUNT(*) AS count
    FROM {DATASET}.audio_data_mm
    WHERE AI.IF(prompt => ('{PROMPT}', audio),
                connection_id => 'us.connection',
                endpoint => 'gemini-2.5-flash')
    """
    return run_query(client, sql)


def main():
    profile = build_profile(
        scenario="wildlife", query_id=2, scale_factor=200,
        prompt=PROMPT, params={"alpha": ALPHA},
        cascade_form="F-cascade: MarginSignal + AlphaBand + AiIfVerifier; client COUNT.",
        extra={"dase_prompts": {"positive": POSITIVE, "negative": NEGATIVE}},
    )

    print("Loading audio data + caption embeddings...")
    df = pd.read_csv(AUDIO_CSV)
    audio_emb = np.load(EMB_PATH)["caption_emb"]
    assert len(df) == audio_emb.shape[0]
    df["GcsUri"] = df["AudioPath"].apply(
        lambda p: f"gs://{BUCKET}/animal_audio/{os.path.basename(p)}")
    n_total = len(df)
    n_gt = int((df["Animal"] == "Elephant").sum())
    print(f"  {n_total} audios, GT elephant count = {n_gt}")
    profile["data"] = {"n_audios": n_total, "n_gt_elephant": n_gt}

    client = bq_client(PROJECT)

    # ── Per-row cost calibration ──
    print("\n=== Per-row cost calibration ===")
    cal = per_row_cost(
        client, PROMPT,
        sample_uris=df["GcsUri"].head(5).tolist(),
        ext_table=f"{DATASET}.audio_data_external",
    )
    per_row = cal.per_row_cost_usd
    print(f"  per_row=${per_row:.6f}, sample_cost=${cal.sample_cost_usd:.6f}, elapsed={cal.elapsed_s:.1f}s")
    profile["calibration"] = cal.to_dict()

    # ── Cascade: Signal+Band on embeddings, Verifier on uncertain URIs ──
    cascade = Cascade(
        embeddings=audio_emb,
        ids=df["GcsUri"].tolist(),  # gs:// — matches m.AudioPath in BQ audio_data_files
        signal=MarginSignal(positive_prompts=POSITIVE, negative_prompts=NEGATIVE),
        band=AlphaBand(alpha=ALPHA),
        verifier=make_elephant_audio_verifier(),
    )
    print("\n=== Cascade (Signal → Band → Verifier) ===")
    cres = cascade.run(client, per_row)

    n_confident_pos = len(cres.confident_pos_ids)
    bq_pos = cres.verifier_result.positive_ids
    n_uncertain = len(cres.uncertain_ids)
    cascade_count = n_confident_pos + len(bq_pos)
    cscore = relative_error_score(cascade_count, n_gt)
    cascade_total_wall = cres.total_wall_s
    cascade_total_slot = cres.verifier_result.ctas_slot_ms + cres.verifier_result.slot_ms
    print(f"  alpha={ALPHA}, uncertain={n_uncertain}, confident_pos={n_confident_pos}, "
          f"bq_yes_on_uncertain={len(bq_pos)}")
    print(f"  cascade_count={cascade_count} (GT={n_gt})  score={cscore:.4f}")
    print(f"  wall={cascade_total_wall:.2f}s  slot={cascade_total_slot}  "
          f"calls={cres.verifier_result.n_calls}  cost=${cres.verifier_result.cost_usd:.6f}")

    profile["dase_partition"] = cres.partition.to_dict() | {
        "n_confident_pos": n_confident_pos,
    }

    # ── Baseline (verbatim sembench Q2.sql) ──
    if SKIP_BASELINE:
        b_score, bwall, bslot = PAPER_BQ_Q2["score"], PAPER_BQ_Q2["latency_s"], None
        bcost, bcount = PAPER_BQ_Q2["cost_usd"], None
        bcalls = round(bcost / per_row)
        profile["baseline"] = {"_status": "aborted",
                               "score": {"score": b_score, "_source": "paper"},
                               "latency_breakdown": {"wall_s": bwall, "_source": "paper"},
                               "cost_breakdown": {"n_llm_calls": bcalls, "total_cost_usd": bcost, "_source": "paper"}}
    else:
        print(f"\n=== Baseline (sembench Q2.sql verbatim) ===")
        bdf, bwall, bslot, bsql = run_baseline(client)
        bcount = int(bdf.iloc[0]["count"])
        bcalls = n_total
        bcost = per_row * bcalls
        b_score = relative_error_score(bcount, n_gt)
        print(f"  count={bcount} (GT={n_gt})")
        print(f"  wall={bwall:.2f}s  slot={bslot}  calls={bcalls}  cost=${bcost:.6f}  score={b_score:.4f}")
        profile["baseline"] = {
            "method": "sembench bigquery/Q2.sql verbatim on audio_data_mm", "sql": bsql,
            "result_count": bcount, "score": {"score": b_score},
            "latency_breakdown": {"wall_s": bwall, "slot_ms": bslot},
            "cost_breakdown": {"n_llm_calls": bcalls, "per_row_cost_usd": per_row, "total_cost_usd": bcost},
        }

    profile["cascade"] = {
        "method": "F-cascade with COUNT aggregation: Cascade(MarginSignal, AlphaBand, AiIfVerifier).run()",
        "verifier": cres.verifier_result.to_dict(),
        "cascade_count": cascade_count,
        "cascade_count_breakdown": {"dase_confident_pos": n_confident_pos, "bq_uncertain_pos": len(bq_pos)},
        "score": {"score": cscore},
        "totals": {"wall_s": cascade_total_wall, "slot_ms_bq_total": cascade_total_slot,
                   "cost_usd": cres.verifier_result.cost_usd,
                   "n_llm_calls": cres.verifier_result.n_calls},
    }

    profile["comparison"] = {
        "score":       {"paper_BQ": PAPER_BQ_Q2["score"],   "paper_DASE_NN": PAPER_DASE_NN_Q2["score"],   "ours_BQ": b_score, "ours_cascade": cscore},
        "wall_s":      {"paper_BQ": PAPER_BQ_Q2["latency_s"], "paper_DASE_NN": PAPER_DASE_NN_Q2["latency_s"], "ours_BQ": bwall,  "ours_cascade": cascade_total_wall},
        "cost_usd":    {"paper_BQ": PAPER_BQ_Q2["cost_usd"], "paper_DASE_NN": PAPER_DASE_NN_Q2["cost_usd"], "ours_BQ": bcost,    "ours_cascade": cres.verifier_result.cost_usd},
        "n_llm_calls": {"paper_BQ": round(PAPER_BQ_Q2["cost_usd"] / per_row), "paper_DASE_NN": 0, "ours_BQ": bcalls, "ours_cascade": cres.verifier_result.n_calls},
    }

    write_profile(profile, PROFILE_PATH)

    print_summary(
        f"Wildlife Q2 (alpha={ALPHA})",
        columns=["paper BQ", "DASE+NN", "ours BQ", "ours cascade"],
        rows=[
            ("score",      [PAPER_BQ_Q2["score"], PAPER_DASE_NN_Q2["score"], b_score, cscore], ".2f"),
            ("count",      [None, None, bcount, cascade_count]),
            ("wall (s)",   [PAPER_BQ_Q2["latency_s"], PAPER_DASE_NN_Q2["latency_s"], bwall, cascade_total_wall], ".2f"),
            ("cost ($)",   [PAPER_BQ_Q2["cost_usd"], PAPER_DASE_NN_Q2["cost_usd"], bcost, cres.verifier_result.cost_usd], ".4f"),
            ("#LLM calls", [round(PAPER_BQ_Q2["cost_usd"] / per_row), 0, bcalls, cres.verifier_result.n_calls], "d"),
        ],
    )


if __name__ == "__main__":
    main()
