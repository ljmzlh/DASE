import os
import random
from typing import List

import psycopg2
from psycopg2.extras import execute_values

DIMENSION = 128
ROW_COUNT = 1000
DSN = dict(host=os.environ.get("PGHOST", "127.0.0.1"), database="molecule", user=os.environ.get("PGUSER", "postgres"), password=os.environ.get("PGPASSWORD", ""))
HNSW_OPS = "vector_l2_ops"
HNSW_M = 8
HNSW_EF_CONSTRUCTION = 16


def _connect():
    return psycopg2.connect(**DSN)


def _ensure_schema(cursor):
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cursor.execute("DROP TABLE IF EXISTS item;")
    cursor.execute("DROP TABLE IF EXISTS items;")
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS item (
            id bigserial PRIMARY KEY,
            embed vector({DIMENSION})
        )
        """
    )


def _random_vector() -> List[float]:
    return [random.uniform(-1, 1) for _ in range(DIMENSION)]


def _format_vector(values: List[float]) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in values) + "]"


def _insert_vectors(cursor):
    vectors = [(_format_vector(_random_vector()),) for _ in range(ROW_COUNT)]
    execute_values(cursor, "INSERT INTO item (embed) VALUES %s", vectors)


def _create_hnsw_index(cursor):
    cursor.execute(
        f"""
        CREATE INDEX item_embed_hnsw_idx
        ON item USING hnsw (embed {HNSW_OPS})
        WITH (m = %s, ef_construction = %s)
        """,
        (HNSW_M, HNSW_EF_CONSTRUCTION),
    )


def main():
    random.seed()
    with _connect() as conn:
        with conn.cursor() as cursor:
            _ensure_schema(cursor)
            _insert_vectors(cursor)
            _create_hnsw_index(cursor)


if __name__ == "__main__":
    main()