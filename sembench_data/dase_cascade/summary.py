"""Build paper-facing summaries from measured DASE cascade profiles."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping


FIELDNAMES = (
    "q",
    "operator",
    "score_metric",
    "score",
    "latency_s",
    "cost_usd",
    "n_llm_calls",
    "slot_ms_bq_total",
    "method",
)

SCORE_KEYS = (
    "f1_score",
    "f1",
    "ari",
    "spearman_correlation",
    "spearman",
    "score",
)


def _mapping(value, label: str, profile_path: Path) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{profile_path}: {label} must be an object")
    return value


def _score(cascade: Mapping, profile_path: Path) -> tuple[str, float]:
    scores = _mapping(cascade.get("score"), "cascade.score", profile_path)
    for key in SCORE_KEYS:
        value = scores.get(key)
        if value is not None:
            return key, float(value)
    raise ValueError(
        f"{profile_path}: cascade.score has none of {', '.join(SCORE_KEYS)}"
    )


def _profile_row(profile_path: Path, query_id: str, operator: str) -> dict:
    with profile_path.open(encoding="utf-8") as stream:
        profile = _mapping(json.load(stream), "profile", profile_path)

    cascade = _mapping(profile.get("cascade"), "cascade", profile_path)
    totals = _mapping(cascade.get("totals"), "cascade.totals", profile_path)
    missing = [
        key
        for key in ("wall_s", "cost_usd", "n_llm_calls", "slot_ms_bq_total")
        if key not in totals
    ]
    if missing:
        raise ValueError(
            f"{profile_path}: cascade.totals missing {', '.join(missing)}"
        )

    score_metric, score = _score(cascade, profile_path)
    return {
        "q": query_id,
        "operator": operator,
        "score_metric": score_metric,
        "score": score,
        "latency_s": float(totals["wall_s"]),
        "cost_usd": float(totals["cost_usd"]),
        "n_llm_calls": int(totals["n_llm_calls"]),
        "slot_ms_bq_total": int(totals["slot_ms_bq_total"]),
        "method": str(cascade.get("method", "")),
    }


def build_cascade_summary(
    scenario_dir: str | Path,
    query_ids: Iterable[str | int],
    operators: Mapping[str, str],
) -> list[dict]:
    """Write ``outputs/cascade_summary.csv`` using DASE cascade fields only."""
    output_dir = Path(scenario_dir) / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for raw_query_id in query_ids:
        query_id = str(raw_query_id)
        if query_id not in operators:
            raise ValueError(f"missing operator for Q{query_id}")
        profile_path = output_dir / f"Q{query_id}.json"
        if not profile_path.is_file():
            print(f"WARN: {profile_path} missing, skipping")
            continue
        rows.append(_profile_row(profile_path, query_id, operators[query_id]))

    output_path = output_dir / "cascade_summary.csv"
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output_path} ({len(rows)} DASE rows)")
    return rows