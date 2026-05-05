import os
import random
from typing import List, Sequence, Tuple

import psycopg2

DIMENSION = 128
DSN = dict(host=os.environ.get("PGHOST", "127.0.0.1"), database="molecule", user=os.environ.get("PGUSER", "postgres"), password=os.environ.get("PGPASSWORD", ""))
TOP_K = 10


def output_plan(line: str) -> None:
    if "[" in line and "]" in line:
        import re

        def truncate_vector(match: re.Match[str]) -> str:
            vec_str = match.group(0)
            nums = vec_str.strip("[]").split(",")
            if len(nums) > 10:
                nums = nums[:10]
                return "[" + ",".join(nums) + ", ...]"
            return vec_str

        line = re.sub(r"\[[^\]]+\]", truncate_vector, line)
    print(line)


def _random_vector(dim: int = DIMENSION) -> List[float]:
    return [random.uniform(-1, 1) for _ in range(dim)]


def _format_vector(values: Sequence[float]) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in values) + "]"


def _fetch_top_k_neighbors(conn, query_vector: str) -> Tuple[int, Sequence[Tuple[int, float]], List[str], List[str]]:
    with conn.cursor() as cursor:
        cursor.execute("SET client_min_messages TO INFO")
        cursor.execute("SET enable_seqscan TO off")
        cursor.execute("SET hnsw.ef_search = 32")
        all_notices: List[str] = []

        def _drain_notices():
            if conn.notices:
                all_notices.extend(conn.notices)
                conn.notices.clear()

        conn.notices.clear()
        cursor.execute("SELECT COUNT(*) FROM item")
        _drain_notices()
        (row_count,) = cursor.fetchone()
        cursor.execute(
            """
            EXPLAIN
            SELECT id, embed <-> %s AS distance
            FROM item
            ORDER BY embed <-> %s
            LIMIT %s
            """,
            (query_vector, query_vector, TOP_K),
        )
        _drain_notices()
        plan = [row[0] for row in cursor.fetchall()]
        cursor.execute(
            """
            SELECT id, embed <-> %s AS distance
            FROM item
            ORDER BY embed <-> %s
            LIMIT %s
            """,
            (query_vector, query_vector, TOP_K),
        )
        rows = cursor.fetchall()
        _drain_notices()
    notices = all_notices
    return row_count, rows, notices, plan


def test_query_top_10_neighbors_from_item_table(conn):
    query_vector = _format_vector(_random_vector())
    row_count, rows, notices, plan = _fetch_top_k_neighbors(conn, query_vector)

    print("Query plan:")
    for line in plan:
        output_plan(line)
    print()
    print("Top 10 neighbors (id, distance):", rows)
    if notices:
        print("pgvector elog messages:")
        for notice in notices:
            print(notice.rstrip())
    else:
        print("pgvector elog messages: <none>")


if __name__ == "__main__":
    random.seed(42)
    with psycopg2.connect(**DSN) as conn:
        for _ in range(2):
            test_query_top_10_neighbors_from_item_table(conn)
