#!/usr/bin/env python3
"""Build movie cascade_profiles/cascade_summary.csv from Q1-Q10.json.

Movie profiles use the older comparison schema:
    comparison.score_f1.{paper, baseline, cascade}
    comparison.latency_s.paper
    comparison.cost_usd.paper
    comparison.n_llm_calls.{paper_implied, baseline_est, cascade}
This builder reads those keys and emits the canonical cascade_summary.csv schema.
"""
import csv
import json
import os

PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
OUT_CSV = os.path.join(PROFILE_DIR, "cascade_summary.csv")

OPERATOR = {
    1: "F+L", 2: "F+L",
    3: "F", 4: "F",
    5: "J+L", 6: "J+L", 7: "J",
    8: "C",
    9: "R", 10: "R",
}

DASE_WALL_DEFAULT = 1.25
STAGE1_CTAS_WALL_DEFAULT = 2.5

METRIC_KEYS = ("score", "score_f1", "score_spearman", "score_ari")


def _first_metric_block(cmp_):
    for k in METRIC_KEYS:
        if k in cmp_ and isinstance(cmp_[k], dict):
            return cmp_[k]
    return {}


def get_score(p):
    s = p.get("cascade", {}).get("score", {}) or {}
    for k in ("f1_score", "f1", "score", "spearman_correlation", "spearman", "ari"):
        if k in s and s[k] is not None:
            return float(s[k])
    cmp_ = p.get("comparison", {})
    block = _first_metric_block(cmp_)
    for k in ("ours_cascade", "cascade"):
        if k in block and block[k] is not None:
            return float(block[k])
    return None


def get_paper_bq(p):
    cmp_ = p.get("comparison", {})
    score_block = _first_metric_block(cmp_)
    wall_block = cmp_.get("wall_s") or cmp_.get("latency_s") or {}
    cost_block = cmp_.get("cost_usd") or {}
    score = score_block.get("paper_BQ", score_block.get("paper"))
    wall = wall_block.get("paper_BQ", wall_block.get("paper"))
    cost = cost_block.get("paper_BQ", cost_block.get("paper"))
    return score, cost, wall


def get_paper_dase(p):
    cmp_ = p.get("comparison", {})
    s = _first_metric_block(cmp_)
    return s.get("paper_DASE_NN", s.get("paper_dase"))


def get_n_calls(p):
    nc = p.get("comparison", {}).get("n_llm_calls", {}) or {}
    paper = (
        nc.get("paper_BQ")
        or nc.get("paper")
        or nc.get("paper_implied")
    )
    cascade = nc.get("ours_cascade") or nc.get("cascade")
    if paper is None:
        paper = nc.get("ours_BQ") or nc.get("baseline") or nc.get("baseline_est")
    if cascade is None:
        cascade = (
            (p.get("cascade", {}).get("totals", {}) or {}).get("n_llm_calls")
            or (p.get("cascade", {}).get("cost_breakdown", {}) or {}).get("n_llm_calls")
        )
    return paper, cascade


def get_walls(p):
    cas = p.get("cascade", {})
    wb = (cas.get("totals", {}) or {}).get("wall_breakdown_s") or {}
    dase_wall = wb.get("dase") or wb.get("dase_stage0")
    src = "measured"
    if dase_wall is None:
        dase_wall = DASE_WALL_DEFAULT
        src = "defaulted_dase"
    return dase_wall, STAGE1_CTAS_WALL_DEFAULT, src


def per_call_lat_from_paper(p):
    _, _, paper_wall = get_paper_bq(p)
    paper_calls, _ = get_n_calls(p)
    if paper_wall is not None and paper_calls:
        return paper_wall / paper_calls
    return None


# Closest-by-operator borrow targets (used when a Q has no paper_BQ wall).
BORROW_FROM = {9: 10}  # Q9 movie (rank R) borrows from Q10 (rank R)


def main():
    profiles = {}
    for q in range(1, 11):
        with open(os.path.join(PROFILE_DIR, f"Q{q}.json")) as f:
            profiles[q] = json.load(f)
    pcl_paper = {q: per_call_lat_from_paper(profiles[q]) for q in profiles}

    rows = []
    for q in range(1, 11):
        p = profiles[q]

        score = get_score(p)
        cas_totals = p.get("cascade", {}).get("totals", {}) or {}
        cost = (
            cas_totals.get("cost_usd")
            or (p.get("cascade", {}).get("cost_breakdown", {}) or {}).get("total_cost_usd")
        )
        paper_score, paper_cost, paper_wall = get_paper_bq(p)
        paper_dase = get_paper_dase(p)
        paper_calls, cascade_calls = get_n_calls(p)
        dase_wall, stage1_wall, src = get_walls(p)
        per_call_lat = pcl_paper[q]

        if per_call_lat is not None and cascade_calls is not None:
            stage2_aiif_wall = per_call_lat * cascade_calls
            cascade_lat = dase_wall + stage1_wall + stage2_aiif_wall
            par_source = f"per_call_lat ({src})"
        elif q in BORROW_FROM and pcl_paper.get(BORROW_FROM[q]) is not None and cascade_calls is not None:
            donor = BORROW_FROM[q]
            per_call_lat = pcl_paper[donor]
            stage2_aiif_wall = per_call_lat * cascade_calls
            cascade_lat = dase_wall + stage1_wall + stage2_aiif_wall
            par_source = f"borrowed: {cascade_calls}×{per_call_lat:.4f}s (Q{donor}, same operator)"
        else:
            cascade_lat = dase_wall + stage1_wall
            per_call_lat = None
            par_source = "no paper_BQ; could not borrow"

        rows.append({
            "q": q,
            "operator": OPERATOR[q],
            "cascade_score": score,
            "cascade_cost_usd": cost,
            "cascade_latency_s": cascade_lat,
            "paper_bq_score": paper_score,
            "paper_bq_cost_usd": paper_cost,
            "paper_bq_latency_s": paper_wall,
            "paper_dase_score": paper_dase,
            "parallelism_used": per_call_lat,
            "parallelism_source": par_source,
        })

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_CSV}")
    for r in rows:
        per_call_str = f"{r['parallelism_used']:.4f}s" if r['parallelism_used'] is not None else "n/a"
        score_str = f"{r['cascade_score']:.2f}" if r['cascade_score'] is not None else "n/a"
        print(f"  Q{r['q']:>2} {r['operator']:<5} "
              f"score={score_str} "
              f"cost=${r['cascade_cost_usd']:.4f} "
              f"lat={r['cascade_latency_s']:.2f}s "
              f"per_call={per_call_str} ({r['parallelism_source']})")


if __name__ == "__main__":
    main()
