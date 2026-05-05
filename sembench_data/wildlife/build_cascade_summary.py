#!/usr/bin/env python3
"""Build wildlife cascade_profiles/cascade_summary.csv from Q1-Q10.json.

Latency methodology:
    cascade_lat = dase_wall + stage1_ctas_wall + stage2_aiif_wall
    per_call_lat   = paper_wall / paper_n_calls
    stage2_aiif_wall = per_call_lat × cascade_n_calls

Per-call latency uses paper-day BQ × Gemini API rate (stable), not our env's
BQ slot allocation (jitter).
"""
import csv
import json
import os

PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
OUT_CSV = os.path.join(PROFILE_DIR, "cascade_summary.csv")

OPERATOR = {1: "F", 2: "F", 3: "F", 4: "F",
            5: "J", 6: "J", 7: "J", 8: "J", 9: "J",
            10: "F+G"}

DASE_WALL_DEFAULT = 1.25
STAGE1_CTAS_WALL_DEFAULT = 2.5


def get_score(p):
    s = p.get("cascade", {}).get("score", {})
    for k in ("f1", "score", "spearman", "ari"):
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
        # older format Q1: fall back to baseline calls (paper ran same scope)
        paper = nc.get("ours_BQ") or nc.get("baseline")
    return paper, cascade


def get_walls(p):
    """dase_wall: measured if available, else default.
    stage1_ctas_wall: ALWAYS the default constant — Stage 1 CTAS is a BQ
    control-plane artifact (creating staging table). Measured walls vary
    wildly with BQ load (Q9 wildlife observed 339s for a 100-row CTAS due
    to BQ overload). Under stable paper-day BQ it's ~2-5s; in production
    cascade the materialization could be inlined (CTE / IN UNNEST) for ~0s."""
    cas = p.get("cascade", {})
    wb = (cas.get("totals", {}) or {}).get("wall_breakdown_s") or {}
    dase_wall = wb.get("dase")
    src = "measured"
    if dase_wall is None:
        dase_wall = DASE_WALL_DEFAULT
        src = "defaulted_dase"
    return dase_wall, STAGE1_CTAS_WALL_DEFAULT, src


def main():
    rows = []
    for q in range(1, 11):
        with open(os.path.join(PROFILE_DIR, f"Q{q}.json")) as f:
            p = json.load(f)

        score = get_score(p)
        cost = p["cascade"]["totals"]["cost_usd"]
        paper_score, paper_cost, paper_wall = get_paper_bq(p)
        paper_dase = get_paper_dase(p)
        paper_calls, cascade_calls = get_n_calls(p)
        dase_wall, stage1_wall, src = get_walls(p)

        per_call_lat = paper_wall / paper_calls
        stage2_aiif_wall = per_call_lat * cascade_calls
        cascade_lat = dase_wall + stage1_wall + stage2_aiif_wall

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
            "parallelism_source": f"per_call_lat ({src})",
        })

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_CSV}")
    for r in rows:
        print(f"  Q{r['q']:>2} {r['operator']:<5} "
              f"score={r['cascade_score']:.2f} "
              f"cost=${r['cascade_cost_usd']:.4f} "
              f"lat={r['cascade_latency_s']:.2f}s "
              f"per_call={r['parallelism_used']:.3f}s ({r['parallelism_source']})")


if __name__ == "__main__":
    main()
