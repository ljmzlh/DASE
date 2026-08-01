"""Ablate SemJI on IMDB W5 while keeping the DASE seed stream fixed."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

import ours.sys as system


def score_threshold_recall(ground_truth: list[dict[str, Any]], scores: list[float]) -> float:
    if not ground_truth:
        return 1.0
    threshold = max(float(answer["score"]) for answer in ground_truth)
    return sum(score <= threshold + 1e-6 for score in scores) / len(ground_truth)


def run_without_semji(
    conn: psycopg.Connection,
    query: dict[str, Any],
    partner_ef: int,
) -> list[float]:
    result_count = int(query["K"])
    join = query["join"]
    threshold = float(join["distance_threshold"])
    seed_join_field = join["embed_left"]
    partner_join_field = join["embed_right"]
    query_vector = Vector(query["scoring"]["signals"][0]["query_embed"])
    partner_rate = float(query.get("partner_rate", 1.0))
    limit = int(math.ceil(result_count / max(partner_rate, 0.01)))
    scores: list[float] = []
    checked = 0
    with conn.cursor() as cur:
        while len(scores) < result_count and limit <= 200_000:
            cur.execute(f"SET hnsw.ef_search = {min(1000, max(20, limit))}")
            cur.execute(
                f'SELECT id, "{seed_join_field}" AS join_embedding, '
                f'(plot_emb <-> %(query)s) AS score FROM imdb_t1_hnsw '
                f'ORDER BY score LIMIT %(limit)s',
                {"query": query_vector, "limit": limit},
            )
            candidates = cur.fetchall()
            for candidate in candidates[checked:]:
                cur.execute(f"SET hnsw.ef_search = {partner_ef}")
                cur.execute(
                    f'SELECT ("{partner_join_field}" <-> %(embedding)s) AS distance '
                    f'FROM imdb_t2_hnsw ORDER BY distance LIMIT 1',
                    {"embedding": Vector(candidate["join_embedding"])},
                )
                partner = cur.fetchone()
                if partner is not None and float(partner["distance"]) <= threshold:
                    scores.append(float(candidate["score"]))
                    if len(scores) == result_count:
                        break
            checked = len(candidates)
            if checked < limit:
                break
            limit *= 2
    conn.commit()
    return scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default=os.environ.get("IMDB_DATABASE_URL", "postgresql://localhost/imdb"),
    )
    parser.add_argument(
        "--workload", type=Path, default=Path("imdb_data/workload/w5/w5_queries_100.json")
    )
    parser.add_argument(
        "--ground-truth", type=Path, default=Path("imdb_data/results/w5/w5_queries_100_gt.json")
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--partner-ef", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("results/ablate_semji.json"))
    args = parser.parse_args(argv)

    workload = json.loads(args.workload.read_text(encoding="utf-8"))
    queries = workload["queries"][: args.limit] if args.limit else workload["queries"]
    ground_truth = {
        row["query_id"]: row["answer"]
        for row in json.loads(args.ground_truth.read_text(encoding="utf-8"))["results"]
    }
    system.DATABASE_URL = args.database_url
    system.DATASET = "imdb"
    system.ID_COL, system.TEXT_COL = system.DATASET_COLS["imdb"]
    system.VEC_COLS = system.DATASET_VEC_COLS["imdb"]

    conn = psycopg.connect(args.database_url, row_factory=dict_row)
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute("SET enable_seqscan = off")
        cur.execute("SET work_mem = '1GB'")
    conn.commit()

    for query in queries:
        system.run_query(conn, query, "W5")
    conn.commit()

    totals = {"semji": {"seconds": 0.0, "recall": 0.0}, "on_the_fly": {"seconds": 0.0, "recall": 0.0}}
    for query in queries:
        query_id = query["query_id"]
        started = time.perf_counter()
        rows, _ = system.run_query(conn, query, "W5")
        totals["semji"]["seconds"] += time.perf_counter() - started
        conn.commit()
        totals["semji"]["recall"] += score_threshold_recall(
            ground_truth[query_id], [float(row[-1]) for row in rows]
        )

        started = time.perf_counter()
        scores = run_without_semji(conn, query, args.partner_ef)
        totals["on_the_fly"]["seconds"] += time.perf_counter() - started
        totals["on_the_fly"]["recall"] += score_threshold_recall(ground_truth[query_id], scores)
    conn.close()

    count = len(queries)
    result = {
        variant: {
            "recall": values["recall"] / count,
            "qps": count / values["seconds"],
            "seconds": values["seconds"],
        }
        for variant, values in totals.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())