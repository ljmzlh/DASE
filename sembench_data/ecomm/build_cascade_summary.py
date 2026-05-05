#!/usr/bin/env python3
"""Build ecomm cascade_profiles/cascade_summary.csv from Q1-Q10.json.

Latency methodology:
    cascade_lat = dase_wall + stage1_ctas_wall + stage2_aiif_wall
    per_call_lat   = paper_wall / paper_n_calls
    stage2_aiif_wall = per_call_lat × cascade_n_calls

Per-call latency uses paper-day BQ × Gemini API rate (stable), not our env's
BQ slot allocation (jitter).

Q10 ecomm has paper_BQ = X (no published number) — fall back to measured
cascade wall (we have no paper-day reference for that Q).
"""
import csv
import json
import os

PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
OUT_CSV = os.path.join(PROFILE_DIR, "cascade_summary.csv")

OPERATOR = {
    1: "F", 2: "F",
    3: "M", 4: "M",
    5: "C", 6: "C",
    7: "J", 8: "J", 9: "J",
    10: "F J",
    11: "F J", 12: "F M", 13: "F", 14: "F J R",
}

DASE_WALL_DEFAULT = 1.25
STAGE1_CTAS_WALL_DEFAULT = 2.5


def get_score(p):
    s = p.get("cascade", {}).get("score", {})
    for k in ("f1_score", "f1", "score", "spearman", "ari"):
        if k in s and s[k] is not None:
            return float(s[k])
    return None


def get_paper_bq(p):
    cmp_ = p.get("comparison", {})
    score = cmp_.get("score", {})
    wall = cmp_.get("wall_s", {})
    cost = cmp_.get("cost_usd", {})
    return (
        score.get("paper_BQ", score.get("paper")),
        cost.get("paper_BQ", cost.get("paper")),
        wall.get("paper_BQ", wall.get("paper")),
    )


def get_paper_dase(p):
    s = p.get("comparison", {}).get("score", {})
    return s.get("paper_DASE_NN", s.get("paper_dase"))


def get_n_calls(p):
    nc = p.get("comparison", {}).get("n_llm_calls", {}) or {}
    paper = nc.get("paper_BQ") or nc.get("paper_implied")
    cascade = nc.get("ours_cascade") or nc.get("cascade")
    if paper is None:
        paper = nc.get("ours_BQ") or nc.get("baseline")
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
    """Compute per_call_lat from a Q's paper_BQ if available; else None."""
    _, _, paper_wall = get_paper_bq(p)
    paper_calls, _ = get_n_calls(p)
    if paper_wall is not None and paper_calls:
        return paper_wall / paper_calls
    return None


def load_all_profiles():
    """Q -> (profile, per_call_lat)."""
    out = {}
    for q in range(1, 15):
        path = os.path.join(PROFILE_DIR, f"Q{q}.json")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            p = json.load(f)
        out[q] = (p, per_call_lat_from_paper(p))
    return out


def estimate_borrowed_stage2_wall(profiles, q):
    """For Qs with no paper_BQ reference (Q10, Q11): split cascade calls by
    modality (unary image / pair image) and borrow per_call_lat from
    closest-modality Q in same scenario.

    Returns (stage2_wall, breakdown_dict) or (None, None) if no breakdown."""
    p = profiles[q][0]
    breakdown = ((p.get("cascade", {}).get("totals", {}) or {})
                 .get("n_llm_calls_breakdown") or {})
    n_unary = breakdown.get("stage1_unary", 0)
    n_pair = breakdown.get("stage3_pair", 0)
    if n_unary == 0 and n_pair == 0:
        return None, None
    pcl_unary = profiles[4][1]   # Q4 ecomm: image unary AI.CLASSIFY
    pcl_pair = profiles[9][1]    # Q9 ecomm: image pair AI.IF
    stage2_wall = n_unary * pcl_unary + n_pair * pcl_pair
    return stage2_wall, {
        "n_unary": n_unary, "pcl_unary": pcl_unary,
        "n_pair": n_pair, "pcl_pair": pcl_pair,
    }


def main():
    profiles = load_all_profiles()
    rows = []
    for q in sorted(profiles.keys()):
        p, per_call_lat = profiles[q]

        score = get_score(p)
        cost = p["cascade"]["totals"]["cost_usd"]
        paper_score, paper_cost, paper_wall = get_paper_bq(p)
        paper_dase = get_paper_dase(p)
        _, cascade_calls = get_n_calls(p)
        dase_wall, stage1_wall, src = get_walls(p)

        if per_call_lat is not None and cascade_calls is not None:
            stage2_aiif_wall = per_call_lat * cascade_calls
            cascade_lat = dase_wall + stage1_wall + stage2_aiif_wall
            par_source = f"per_call_lat ({src})"
        else:
            stage2_wall, breakdown = estimate_borrowed_stage2_wall(profiles, q)
            if stage2_wall is not None:
                cascade_lat = dase_wall + stage1_wall + stage2_wall
                par_source = (
                    f"borrowed: {breakdown['n_unary']}×{breakdown['pcl_unary']:.4f}s (Q4) + "
                    f"{breakdown['n_pair']}×{breakdown['pcl_pair']:.4f}s (Q9)"
                )
                per_call_lat = stage2_wall / cascade_calls if cascade_calls else None
            else:
                cascade_lat = dase_wall + stage1_wall
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
        per_call_str = f"{r['parallelism_used']:.3f}s" if r['parallelism_used'] is not None else "n/a"
        score_str = f"{r['cascade_score']:.2f}" if r['cascade_score'] is not None else "n/a"
        print(f"  Q{r['q']:>2} {r['operator']:<5} "
              f"score={score_str} "
              f"cost=${r['cascade_cost_usd']:.4f} "
              f"lat={r['cascade_latency_s']:.2f}s "
              f"per_call={per_call_str} ({r['parallelism_source']})")


if __name__ == "__main__":
    main()
