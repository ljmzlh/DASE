"""Compare distribution-aware and round-robin stream selection on W3."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

import ours.sys as system


def answer_ids(answer: list[Any]) -> set[Any]:
    return {row["id"] if isinstance(row, dict) else row[0] for row in answer}


def recall_at_k(ground_truth: dict[str, Any], results: list[dict[str, Any]]) -> float:
    truth = {row["query_id"]: answer_ids(row["answer"]) for row in ground_truth["results"]}
    recalls = []
    for result in results:
        expected = truth[result["query_id"]]
        actual = {row[0] for row in result["answer"]}
        recalls.append(len(expected & actual) / len(expected) if expected else 1.0)
    return sum(recalls) / len(recalls)


def run_pass(
    conn: psycopg.Connection,
    queries: list[dict[str, Any]],
    workload_type: str,
    adaptive: bool,
) -> tuple[list[dict[str, Any]], float]:
    output = []
    started = time.perf_counter()
    for query in queries:
        rows, _ = system.run_query(conn, query, workload_type, adaptive=adaptive, eps=0.0)
        output.append({
            "query_id": query["query_id"],
            "answer": [list(row) for row in rows],
            "K": query.get("K", 20),
        })
        conn.commit()
    return output, time.perf_counter() - started


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("imdb", "molecule"), default="imdb")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/ablate_stream_selection.json"))
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    workload_path = root / f"{args.dataset}_data/workload/w3/w3_queries_100.json"
    truth_path = root / f"{args.dataset}_data/results/w3/w3_queries_100_gt.json"
    workload = json.loads(workload_path.read_text(encoding="utf-8"))
    queries = workload["queries"][: args.limit] if args.limit else workload["queries"]
    ground_truth = json.loads(truth_path.read_text(encoding="utf-8"))
    database_url = args.database_url or system.DATASET_DB_URLS[args.dataset]

    system.DATABASE_URL = database_url
    system.DATASET = args.dataset
    system.ID_COL, system.TEXT_COL = system.DATASET_COLS[args.dataset]
    system.VEC_COLS = system.DATASET_VEC_COLS[args.dataset]
    from ours import fss
    fss.DATASET_VEC_COLS = system.VEC_COLS
    fss._load_table_meta(args.dataset)

    conn = psycopg.connect(database_url, row_factory=dict_row)
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute("SET enable_seqscan = off")
        cur.execute("SET work_mem = '1GB'")
    conn.commit()
    run_pass(conn, queries, workload["workload"], adaptive=True)

    result = {}
    for adaptive in (True, False):
        rows, seconds = run_pass(conn, queries, workload["workload"], adaptive)
        result["adaptive" if adaptive else "round_robin"] = {
            "recall": recall_at_k(ground_truth, rows),
            "qps": len(rows) / seconds,
            "seconds": seconds,
        }
    conn.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())