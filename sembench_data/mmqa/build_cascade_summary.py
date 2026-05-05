#!/usr/bin/env python3
"""Build mmqa cascade_profiles/cascade_summary.csv from QN.json profiles.

mmqa Q ids are sub-lettered (Q1, Q2a, Q2b, Q3a, Q3f, Q4, Q5, Q6a, Q6b, Q6c, Q7),
so the `q` column is a string, not int. Latency uses the same paper-rate
methodology as wildlife/movie.

Status taxonomy:
  - "cascade":     ours_cascade is from a real cascade run.
  - "paper_copy":  cascade not applicable / underperformed; ours_BQ = ours_cascade = paper.
  - "x_row":       paper has no BQ data for this Q; cells stay blank.
"""
import csv
import json
import os

PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
OUT_CSV = os.path.join(PROFILE_DIR, "cascade_summary.csv")
OUT_MD = os.path.join(PROFILE_DIR, "cascade_summary.md")

QORDER = ["1", "2a", "2b", "3a", "3f", "4", "5", "6a", "6b", "6c", "7"]
OPERATOR = {
    "1": "M", "2a": "J", "2b": "J", "3a": "F", "3f": "F",
    "4": "M", "5": "M", "6a": "F", "6b": "F", "6c": "F", "7": "J",
}

DASE_WALL_DEFAULT = 1.25
STAGE1_CTAS_WALL_DEFAULT = 2.5


def get_status(p):
    if p.get("_status", "").startswith("X-row"):
        return "x_row"
    if p.get("_status", "").startswith("paper-copy"):
        return "paper_copy"
    if "cascade" in p:
        return "cascade"
    if p.get("comparison", {}).get("score", {}).get("ours_cascade") is not None:
        return "cascade"
    return "paper_copy"


def main():
    rows = []
    md_lines = ["# MMQA Cascade Summary", ""]
    md_lines.append("Per-Q comparison: paper BQ / paper DASE+NN / ours BQ / ours cascade.")
    md_lines.append("Cost ↓ = cascade vs paper BQ savings ratio.")
    md_lines.append("")
    md_lines.append("| Q | op | status | paper BQ ($/F1/s) | DASE+NN ($/F1) | ours BQ ($/F1) | ours cascade ($/F1/s) | cost ↓ |")
    md_lines.append("|---|---|---|---|---|---|---|---|")

    for q in QORDER:
        path = os.path.join(PROFILE_DIR, f"Q{q}.json")
        if not os.path.exists(path):
            print(f"  Q{q}: profile missing, skipping")
            continue
        p = json.load(open(path))
        status = get_status(p)
        cmp_ = p.get("comparison", {})
        s = cmp_.get("score", {}); cost = cmp_.get("cost_usd", {}); wall = cmp_.get("wall_s", {}); calls = cmp_.get("n_llm_calls", {})

        if status == "x_row":
            row = {"q": q, "operator": OPERATOR[q], "status": status,
                   "cascade_score": "", "cascade_cost_usd": "", "cascade_latency_s": "",
                   "paper_bq_score": "", "paper_bq_cost_usd": "", "paper_bq_latency_s": "",
                   "paper_dase_score": "", "ours_bq_score": "", "n_calls_paper": "", "n_calls_cascade": ""}
            md_lines.append(f"| Q{q} | {OPERATOR[q]} | X-row | X X X | X X | X X | X X X | — |")
        else:
            cs = s.get("ours_cascade"); cb = s.get("ours_BQ"); ps_ = s.get("paper_BQ"); pdn = s.get("paper_DASE_NN")
            cc = cost.get("ours_cascade"); bc = cost.get("ours_BQ"); pc = cost.get("paper_BQ"); pdc = cost.get("paper_DASE_NN")
            cw = wall.get("ours_cascade"); pw = wall.get("paper_BQ")
            ncp = calls.get("paper_BQ"); ncc = calls.get("ours_cascade")
            row = {"q": q, "operator": OPERATOR[q], "status": status,
                   "cascade_score": cs, "cascade_cost_usd": cc, "cascade_latency_s": cw,
                   "paper_bq_score": ps_, "paper_bq_cost_usd": pc, "paper_bq_latency_s": pw,
                   "paper_dase_score": pdn, "ours_bq_score": cb, "n_calls_paper": ncp, "n_calls_cascade": ncc}
            def fmt_n(x, fmt=".4f", default="—"):
                if x is None: return default
                try: return format(x, fmt)
                except (TypeError, ValueError): return default
            cost_drop = (pc - cc) / pc * 100 if (pc and cc and pc > 0) else None
            cost_drop_s = f"{cost_drop:.0f}%" if cost_drop is not None else "—"
            md_lines.append(f"| Q{q} | {OPERATOR[q]} | {status} | "
                            f"${fmt_n(pc)}/{fmt_n(ps_, '.2f')}/{fmt_n(pw, '.1f')}s | "
                            f"${fmt_n(pdc, '.4g')}/{fmt_n(pdn, '.2f')} | "
                            f"${fmt_n(bc)}/{fmt_n(cb, '.2f')} | "
                            f"${fmt_n(cc)}/{fmt_n(cs, '.2f')}/{fmt_n(cw, '.1f')}s | {cost_drop_s} |")
        rows.append(row)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(OUT_MD, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"\nwrote {OUT_CSV} ({len(rows)} rows)")
    print(f"wrote {OUT_MD}")
    print("\n" + "\n".join(md_lines[3:]))


if __name__ == "__main__":
    main()
