#!/usr/bin/env -S python -u
"""MMQA Q4 cascade — sem-extract (multi-label genre extraction over 24 movies).

Operator: M (cluster-based prefilter for SEM_MAP / AI.GENERATE).
Refactored to use dase_cascade.ClusterCascade + AiGenerateVerifier.

Pipeline:
  1. AgglomerativeClustering(metric=cosine, linkage=single, distance_threshold=1-SIM_THR)
     groups WL movies into K tight clusters.
  2. ClusterCascade picks one centroid-nearest representative per cluster.
  3. AiGenerateVerifier runs AI.GENERATE genre extraction on K reps.
  4. Label propagation: every movie in cluster c → genres of rep(c).
  5. F1 over (genre, title) pairs vs GT.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DASE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
SEMBENCH_MY = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, DASE_ROOT)
sys.path.insert(0, SEMBENCH_MY)

from google.cloud import bigquery  # noqa: E402

from dase_cascade import (  # noqa: E402
    ClusterCascade, AiGenerateVerifier,
    bq_client, run_query,
    build_profile, write_profile, print_summary,
)

MMQA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(MMQA_DIR, "data")
NL_PATH = os.path.join(MMQA_DIR, "query", "natural_language", "q4.json")
PROFILE_DIR = os.path.join(MMQA_DIR, "outputs")
PROFILE_PATH = os.path.join(PROFILE_DIR, "Q4.json")
PROJECT = os.environ.get("GCP_PROJECT", "")
DATASET = "mmqa"

SIM_THR = 0.76  # tight, expect 3 pair-clusters, save 3 calls
WL = ["Orange County", "Mean Girls", "Love Is the Drug", "Crashing", "Cloverfield", "My Best Friend's Girl",
      "Crossing Over", "Hot Tub Time Machine", "The Last Rites of Ransom Pride", "127 Hours", "High Road",
      "Save the Date", "Bachelorette", "3, 2, 1... Frankie Go Boom", "Queens of Country", "Item 47",
      "The Interview", "The Night Before", "Now You See Me 2", "Allied", "The Disaster Artist",
      "Extinction", "The People We Hate at the Wedding", "Cobweb"]

PAPER_BQ_Q4 = {"score": 0.60, "latency_s": 9.7, "cost_usd": 2e-3, "n_calls": 24}
PAPER_DASE_NN_Q4 = {"score": None, "latency_s": None, "cost_usd": None}
SKIP_BASELINE = False


def make_q4_verifier():
    """AI.GENERATE genre extraction on a set of titles. Returns {title: genres ARRAY<STRING>}."""
    def verify_sql_template(titles):
        # Inline as a literal array of strings.
        def _esc(s):
            return s.replace("\\", "\\\\").replace("'", "\\'")
        title_arr = ",".join(f"'{_esc(t)}'" for t in titles)
        return f"""
        WITH movie_genres AS (
          SELECT t.title, AI.GENERATE(
            prompt => ('Extract all applicable genres for each movie based on their description: ', t.text),
            connection_id => 'us.connection',
            endpoint => 'gemini-2.5-flash',
            output_schema => 'genres ARRAY<STRING>'
          ).genres AS genres
          FROM {DATASET}.lizzy_caplan_text_data t
          WHERE t.title IN UNNEST([{title_arr}])
        )
        SELECT title, genres FROM movie_genres
        """

    def coerce_value(gs):
        if gs is None:
            return []
        return [str(g).strip().lower() for g in list(gs)]

    return AiGenerateVerifier(
        verify_sql_template=verify_sql_template,
        id_column="title", value_column="genres",
        coerce_id=str, coerce_value=coerce_value,
    )


def f1_pairs(pred_pairs, gt_pairs):
    P, G = set(pred_pairs), set(gt_pairs)
    tp = len(P & G)
    p = tp / len(P) if P else 0
    r = tp / len(G) if G else 0
    return (2 * p * r / (p + r) if (p + r) else 0, p, r)


def run_baseline_extract(client, titles):
    """Verbatim baseline: AI.GENERATE on all WL movies. Returns (genres_dict, sql, wall, slot)."""
    def _esc(s):
        return s.replace("\\", "\\\\").replace("'", "\\'")
    title_arr = ",".join(f"'{_esc(t)}'" for t in titles)
    sql = f"""
    WITH movie_genres AS (
      SELECT t.title, AI.GENERATE(
        prompt => ('Extract all applicable genres for each movie based on their description: ', t.text),
        connection_id => 'us.connection',
        endpoint => 'gemini-2.5-flash',
        output_schema => 'genres ARRAY<STRING>'
      ).genres AS genres
      FROM {DATASET}.lizzy_caplan_text_data t
      WHERE t.title IN UNNEST([{title_arr}])
    )
    SELECT title, genres FROM movie_genres
    """
    df, wall, slot, _ = run_query(client, sql)
    out = {}
    for _, r in df.iterrows():
        gs = r["genres"]
        out[r["title"]] = [] if gs is None else [str(g).strip().lower() for g in list(gs)]
    return out, sql, wall, slot


def main():
    profile = build_profile(
        scenario="mmqa", query_id="4", scale_factor=200,
        params={"SIM_THR": SIM_THR},
        cascade_form=(
            f"ClusterCascade(AgglomerativeClustering(cosine, single, threshold={1-SIM_THR})) "
            f"+ AiGenerateVerifier(genres ARRAY<STRING>) on K representatives + label propagation per cluster."
        ),
        extra={
            "operator": "sem-extract (multi-label genre)",
            "cascade_strategy": "cluster + propagate",
        },
    )
    df = pd.read_parquet(os.path.join(DATA_DIR, "lizzy_caplan_text_data.parquet"))
    gt_dict = json.load(open(NL_PATH))["ground_truth"]
    gt_pairs = {(g.lower(), t) for g, ms in gt_dict.items() for t in ms}
    print(f"  WL {len(WL)} movies, GT {len(gt_pairs)} (genre, title) pairs across {len(gt_dict)} genres")
    sub = df[df["title"].isin(WL)].reset_index(drop=True)
    n = len(sub)
    print(f"  found {n} movies in parquet")
    profile["data"] = {"n_movies": n, "n_gt_pairs": len(gt_pairs), "gt_genres": sorted(gt_dict.keys())}

    embeddings = np.array(sub["embedding"].tolist(), dtype=np.float32)
    ids = sub["title"].astype(str).tolist()

    client = bq_client(PROJECT)

    # ── Cluster cascade — Agglomerative single-link cosine, threshold = 1 - SIM_THR ──
    print(f"\n=== ClusterCascade(Agglomerative single-link, distance_threshold={1-SIM_THR:.2f}) ===")
    clusterer = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="single",
        distance_threshold=1 - SIM_THR,
    )
    # Use a per-call rate sourced from the paper; AI.GENERATE doesn't share the
    # AI.GENERATE_BOOL calibration shape, so we keep the v1 convention.
    per_row = PAPER_BQ_Q4["cost_usd"] / PAPER_BQ_Q4["n_calls"]

    cluster_cascade = ClusterCascade(
        embeddings=embeddings, ids=ids,
        clusterer=clusterer,
        verifier=make_q4_verifier(),
        rep_strategy="centroid_nearest",
    )
    cres = cluster_cascade.run(client, per_row)
    print(f"  K clusters = {cres.n_clusters}; size stats = {cres.cluster_size_stats()}")
    print(f"  verifier: wall={cres.verifier_result.wall_s:.2f}s slot={cres.verifier_result.slot_ms} "
          f"calls={cres.verifier_result.n_calls} cost=${cres.verifier_result.cost_usd:.6f}")

    # Build cascade (genre, title) pairs from propagated genres
    cas_pairs = set()
    for title, genres in cres.predicted.items():
        if isinstance(genres, list):
            for g in genres:
                cas_pairs.add((g, title))
    cscore, cp_v, cr_v = f1_pairs(cas_pairs, gt_pairs)
    print(f"  cascade {len(cas_pairs)} pairs, F1={cscore:.4f} P={cp_v:.4f} R={cr_v:.4f}, wall={cres.verifier_result.wall_s:.2f}s")
    n_cas = cres.verifier_result.n_calls
    cas_cost = cres.verifier_result.cost_usd
    cas_wall = cres.verifier_result.ctas_wall_s + cres.verifier_result.wall_s
    cas_slot = cres.verifier_result.ctas_slot_ms + cres.verifier_result.slot_ms

    # Build cluster summary for profile (size>1 clusters only)
    rep_set = set(cres.rep_ids)
    cluster_to_members = {}
    for i, lbl in enumerate(cres.labels):
        cluster_to_members.setdefault(int(lbl), []).append(ids[i])
    rep_id_to_cluster = {rid: int(cres.labels[ridx]) for rid, ridx in zip(cres.rep_ids, cres.rep_indices)}
    cluster_to_rep = {c: rid for rid, c in rep_id_to_cluster.items()}
    profile["dase_partition"] = {
        "n_clusters": cres.n_clusters,
        "n_medoids": len(cres.rep_ids),
        "clusters_size_gt1": [
            {"medoid": cluster_to_rep[c], "members": members}
            for c, members in cluster_to_members.items() if len(members) > 1
        ],
    }

    if SKIP_BASELINE:
        bcost = PAPER_BQ_Q4["cost_usd"]; bwall = PAPER_BQ_Q4["latency_s"]
        bscore = PAPER_BQ_Q4["score"]; bcalls = len(WL); b_pairs = None
        bslot = None
    else:
        print(f"\n=== Baseline (verbatim AI.GENERATE on all {len(WL)} movies) ===")
        b_genres, bsql, bwall, bslot = run_baseline_extract(client, WL)
        b_pairs = {(g, t) for t, gs in b_genres.items() for g in gs}
        bscore, bp_v, br_v = f1_pairs(b_pairs, gt_pairs)
        bcalls = len(WL)
        bcost = per_row * bcalls
        print(f"  baseline {len(b_pairs)} pairs, F1={bscore:.4f} P={bp_v:.4f} R={br_v:.4f}, wall={bwall:.2f}s")
        profile["baseline"] = {
            "method": "verbatim BQ q4.sql AI.GENERATE", "sql": bsql, "result_pairs_n": len(b_pairs),
            "score": {"f1": bscore, "precision": bp_v, "recall": br_v},
            "latency_breakdown": {"wall_s": bwall, "slot_ms": bslot},
            "cost_breakdown": {"n_llm_calls": bcalls, "total_cost_usd": bcost,
                                "_note": "cost from paper per-call rate (no AI.GENERATE calibration)"},
        }

    profile["cascade"] = {
        "method": f"ClusterCascade(Agglomerative single-link sim≥{SIM_THR}) → AiGenerateVerifier on reps + propagate genre array",
        "verifier": cres.verifier_result.to_dict(),
        "result_pairs_n": len(cas_pairs),
        "score": {"f1": cscore, "precision": cp_v, "recall": cr_v},
        "totals": {"wall_s": cas_wall, "slot_ms_bq_total": cas_slot,
                   "cost_usd": cas_cost, "n_llm_calls": n_cas},
    }

    cas_lat_rs = PAPER_BQ_Q4["latency_s"] * n_cas / PAPER_BQ_Q4["n_calls"] + 1.0  # +1s dase cluster compute
    profile["comparison"] = {
        "score": {"paper_BQ": PAPER_BQ_Q4["score"], "paper_DASE_NN": PAPER_DASE_NN_Q4["score"],
                   "ours_BQ": bscore, "ours_cascade": cscore},
        "wall_s": {"paper_BQ": PAPER_BQ_Q4["latency_s"], "paper_DASE_NN": PAPER_DASE_NN_Q4["latency_s"],
                    "ours_BQ": PAPER_BQ_Q4["latency_s"], "ours_cascade": cas_lat_rs},
        "cost_usd": {"paper_BQ": PAPER_BQ_Q4["cost_usd"], "paper_DASE_NN": PAPER_DASE_NN_Q4["cost_usd"],
                      "ours_BQ": bcost, "ours_cascade": cas_cost},
        "n_llm_calls": {"paper_BQ": PAPER_BQ_Q4["n_calls"], "paper_DASE_NN": 0,
                         "ours_BQ": bcalls, "ours_cascade": n_cas},
    }
    write_profile(profile, PROFILE_PATH)
    print_summary(
        f"MMQA Q4 (ClusterCascade sim>={SIM_THR})",
        columns=["paper BQ", "ours BQ", "ours cascade"],
        rows=[
            ("score (F1)", [PAPER_BQ_Q4["score"], bscore, cscore], ".2f"),
            ("cost ($)",   [PAPER_BQ_Q4["cost_usd"], bcost, cas_cost], ".4f"),
            ("#LLM calls", [PAPER_BQ_Q4["n_calls"], bcalls, n_cas], "d"),
        ],
    )


if __name__ == "__main__":
    main()
