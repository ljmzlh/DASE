"""Benchmark the five SemJI construction strategies in paper Table 4."""
from __future__ import annotations

from batch_hnsw.order import locality_order  # first: caps OpenBLAS threads before numpy loads

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql

IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def checked_identifier(value: str) -> str:
    if not IDENTIFIER.fullmatch(value):
        raise ValueError(f"invalid SQL identifier: {value!r}")
    return value


def create_result_table(conn: psycopg.Connection, name: str) -> None:
    with conn.cursor() as cur:
        cur.execute(sql.SQL("DROP TABLE IF EXISTS {} ").format(sql.Identifier(name)))
        cur.execute(
            sql.SQL(
                "CREATE TEMP TABLE {} (left_id bigint, right_id bigint, distance double precision)"
            ).format(sql.Identifier(name))
        )
    conn.commit()


def left_source_sql(table: str, id_column: str, vector_column: str, limit: int | None) -> sql.Composed:
    query = sql.SQL("SELECT {id}, {vector} FROM {table} WHERE {vector} IS NOT NULL ORDER BY {id}").format(
        id=sql.Identifier(id_column),
        vector=sql.Identifier(vector_column),
        table=sql.Identifier(table),
    )
    if limit is not None:
        query += sql.SQL(" LIMIT {} ").format(sql.Literal(int(limit)))
    return query


def build_exact(
    conn: psycopg.Connection,
    result_table: str,
    left_table: str,
    right_table: str,
    left_id: str,
    right_id: str,
    left_vector: str,
    right_vector: str,
    radius: float,
    left_limit: int | None,
) -> None:
    source = left_source_sql(left_table, left_id, left_vector, left_limit)
    self_join = left_table == right_table and left_id == right_id
    self_filter = sql.SQL("AND right_side.{right_id} > left_side.{left_id}").format(
        right_id=sql.Identifier(right_id), left_id=sql.Identifier(left_id)
    ) if self_join else sql.SQL("")
    query = sql.SQL(
        """
        INSERT INTO {result}
        SELECT left_side.{left_id}, right_side.{right_id},
               right_side.{right_vector} <-> left_side.{left_vector} AS distance
        FROM ({source}) AS left_side
        CROSS JOIN {right_table} AS right_side
        WHERE right_side.{right_vector} IS NOT NULL
          AND right_side.{right_vector} <-> left_side.{left_vector} <= %s
          {self_filter}
        """
    ).format(
        result=sql.Identifier(result_table), left_id=sql.Identifier(left_id),
        right_id=sql.Identifier(right_id), left_vector=sql.Identifier(left_vector),
        right_vector=sql.Identifier(right_vector), source=source,
        right_table=sql.Identifier(right_table), self_filter=self_filter,
    )
    with conn.cursor() as cur:
        cur.execute(query, (radius,))
    conn.commit()


def _configure_hnsw(conn: psycopg.Connection, radius: float, ef_search: int) -> None:
    with conn.cursor() as cur:
        cur.execute("SET hnsw.radius = %s", (radius,))
        cur.execute("SET hnsw.ef_search = %s", (ef_search,))
        cur.execute("SET enable_seqscan = off")


def build_iterative(
    conn: psycopg.Connection,
    result_table: str,
    left_table: str,
    right_table: str,
    left_id: str,
    right_id: str,
    left_vector: str,
    right_vector: str,
    radius: float,
    ef_search: int,
    left_limit: int | None,
) -> None:
    _configure_hnsw(conn, radius, ef_search)
    self_join = left_table == right_table and left_id == right_id
    source = left_source_sql(left_table, left_id, left_vector, left_limit)
    with conn.cursor(name="semji_iterative_source") as source_cursor, conn.cursor() as cur:
        source_cursor.itersize = 512
        source_cursor.execute(source)
        for source_id, source_vector in source_cursor:
            self_filter = sql.SQL("AND {id} > %s").format(id=sql.Identifier(right_id)) if self_join else sql.SQL("")
            parameters: tuple[Any, ...]
            if self_join:
                parameters = (
                    source_id, source_vector, source_vector, source_id, source_vector
                )
            else:
                parameters = (source_id, source_vector, source_vector, source_vector)
            query = sql.SQL(
                """
                INSERT INTO {result}
                SELECT %s, {right_id}, {right_vector} <-> %s::vector
                FROM {right_table}
                WHERE {right_vector} <->@ %s::vector {self_filter}
                ORDER BY {right_vector} <-> %s::vector
                """
            ).format(
                result=sql.Identifier(result_table), right_id=sql.Identifier(right_id),
                right_vector=sql.Identifier(right_vector), right_table=sql.Identifier(right_table),
                self_filter=self_filter,
            )
            cur.execute(query, parameters)
    conn.commit()


def build_batched(
    conn: psycopg.Connection,
    result_table: str,
    left_table: str,
    right_table: str,
    left_id: str,
    right_id: str,
    left_vector: str,
    right_vector: str,
    radius: float,
    ef_search: int,
    left_limit: int | None,
    batch_size: int,
) -> None:
    _configure_hnsw(conn, radius, ef_search)
    self_join = left_table == right_table and left_id == right_id
    source = left_source_sql(left_table, left_id, left_vector, left_limit)
    with conn.cursor(name="semji_batched_source") as source_cursor, conn.cursor() as cur:
        source_cursor.itersize = batch_size
        source_cursor.execute(source)
        while True:
            rows = source_cursor.fetchmany(batch_size)
            if not rows:
                break
            values = sql.SQL(", ").join(sql.SQL("(%s, %s::vector)") for _ in rows)
            parameters = [value for row in rows for value in row]
            self_filter = sql.SQL("AND candidate.{right_id} > query.left_id").format(
                right_id=sql.Identifier(right_id)
            ) if self_join else sql.SQL("")
            query = sql.SQL(
                """
                INSERT INTO {result}
                WITH query(left_id, embedding) AS (VALUES {values})
                SELECT query.left_id, candidate.{right_id}, candidate.distance
                FROM query
                CROSS JOIN LATERAL (
                    SELECT {right_id}, {right_vector} <-> query.embedding AS distance
                    FROM {right_table} AS candidate
                    WHERE candidate.{right_vector} <->@ query.embedding {self_filter}
                    ORDER BY candidate.{right_vector} <-> query.embedding
                ) AS candidate
                """
            ).format(
                result=sql.Identifier(result_table), values=values,
                right_id=sql.Identifier(right_id), right_vector=sql.Identifier(right_vector),
                right_table=sql.Identifier(right_table), self_filter=self_filter,
            )
            cur.execute(query, parameters)
    conn.commit()


def build_vectorized(
    conn: psycopg.Connection,
    result_table: str,
    left_table: str,
    right_table: str,
    left_id: str,
    right_id: str,
    left_vector: str,
    right_vector: str,
    radius: float,
    ef_search: int,
    left_limit: int | None,
) -> None:
    _configure_hnsw(conn, radius, ef_search)
    self_join = left_table == right_table and left_id == right_id
    source = left_source_sql(left_table, left_id, left_vector, left_limit)
    self_filter = sql.SQL("AND candidate.{right_id} > left_side.{left_id}").format(
        right_id=sql.Identifier(right_id), left_id=sql.Identifier(left_id)
    ) if self_join else sql.SQL("")
    query = sql.SQL(
        """
        INSERT INTO {result}
        SELECT left_side.{left_id}, candidate.{right_id}, candidate.distance
        FROM ({source}) AS left_side
        CROSS JOIN LATERAL (
            SELECT {right_id}, {right_vector} <-> left_side.{left_vector} AS distance
            FROM {right_table} AS candidate
            WHERE candidate.{right_vector} <->@ left_side.{left_vector} {self_filter}
            ORDER BY candidate.{right_vector} <-> left_side.{left_vector}
        ) AS candidate
        """
    ).format(
        result=sql.Identifier(result_table), left_id=sql.Identifier(left_id),
        right_id=sql.Identifier(right_id), left_vector=sql.Identifier(left_vector),
        right_vector=sql.Identifier(right_vector), source=source,
        right_table=sql.Identifier(right_table), self_filter=self_filter,
    )
    with conn.cursor() as cur:
        cur.execute(query)
    conn.commit()


def _l2_hnsw_index(conn: psycopg.Connection, table: str, vector_column: str) -> str:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT i.indexrelid::regclass::text
            FROM pg_index i
            JOIN pg_class c ON c.oid = i.indexrelid
            JOIN pg_am a ON a.oid = c.relam
            JOIN pg_opclass oc ON oc.oid = i.indclass[0]
            JOIN pg_attribute att ON att.attrelid = i.indrelid AND att.attnum = i.indkey[0]
            WHERE i.indrelid = %s::regclass AND a.amname = 'hnsw'
              AND oc.opcname = 'vector_l2_ops' AND att.attname = %s
            """,
            (table, vector_column),
        )
        row = cur.fetchone()
    if row is None:
        raise SystemExit(f"no hnsw vector_l2_ops index on {table}({vector_column})")
    return row[0]


def build_shared(
    conn: psycopg.Connection,
    result_table: str,
    left_table: str,
    right_table: str,
    left_id: str,
    right_id: str,
    left_vector: str,
    right_vector: str,
    radius: float,
    ef_search: int,
    left_limit: int | None,
    group_size: int,
    seeds: int,
) -> dict[str, Any]:
    """Shared traversal (dase_batch_hnsw extension): all probes run in one
    bh.range_join_all call over the left rows in k-means locality order, and each
    probe after the first in a group starts from the previous probe's `seeds`
    nearest nodes instead of descending the upper layers. The time includes the
    ordering step (reading the left vectors, k-means, writing the order table)."""
    right_index = _l2_hnsw_index(conn, right_table, right_vector)
    self_join = left_table == right_table and left_id == right_id

    started = time.perf_counter()
    # Binary rows (vectors arrive as float4 arrays); ClientCursor cannot fetch binary
    with psycopg.Cursor(conn) as cur:
        cur.execute(left_source_sql(left_table, left_id, left_vector, left_limit), binary=True)
        rows = cur.fetchall()
    order, groups = locality_order(np.stack([row[1] for row in rows]).astype(np.float32), group_size)
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS semji_shared_order")
        cur.execute("CREATE TEMP TABLE semji_shared_order (id bigint PRIMARY KEY, ord int, grp int)")
        with cur.copy("COPY semji_shared_order (id, ord, grp) FROM STDIN") as copy:
            for position, (row_index, group) in enumerate(zip(order, groups)):
                copy.write_row((rows[row_index][0], position, int(group)))
        cur.execute("ANALYZE semji_shared_order")
    order_seconds = time.perf_counter() - started

    left_sql = sql.SQL(
        "SELECT l.{id}::int8, l.{vector}, o.grp FROM {table} l "
        "JOIN semji_shared_order o ON o.id = l.{id} ORDER BY o.ord"
    ).format(
        id=sql.Identifier(left_id), vector=sql.Identifier(left_vector), table=sql.Identifier(left_table)
    ).as_string(conn)
    self_filter = sql.SQL("WHERE candidate.{right_id} > r.left_id").format(
        right_id=sql.Identifier(right_id)
    ) if self_join else sql.SQL("")
    query = sql.SQL(
        """
        INSERT INTO {result}
        SELECT r.left_id, candidate.{right_id}, r.dist
        FROM bh.range_join_all(%s, %s::regclass, %s, %s, 'warm', %s) AS r
        JOIN {right_table} AS candidate ON candidate.ctid = r.right_tid
        {self_filter}
        """
    ).format(
        result=sql.Identifier(result_table), right_id=sql.Identifier(right_id),
        right_table=sql.Identifier(right_table), self_filter=self_filter,
    )
    with conn.cursor() as cur:
        cur.execute(query, (left_sql, right_index, radius, ef_search, seeds))
        cur.execute("SELECT bh.last_stats()")
        stats = json.loads(cur.fetchone()[0])
    conn.commit()
    return {"order_seconds": order_seconds, "descents": stats["n_descents"], "probes": stats["n_queries"]}


def table_count(conn: psycopg.Connection, table: str) -> int:
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table)))
        return int(cur.fetchone()[0])


def recall_against(conn: psycopg.Connection, candidate: str, exact: str) -> float:
    exact_count = table_count(conn, exact)
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT count(*) FROM {candidate} c JOIN {exact} e USING (left_id, right_id)"
            ).format(candidate=sql.Identifier(candidate), exact=sql.Identifier(exact))
        )
        intersection = int(cur.fetchone()[0])
    return intersection / exact_count if exact_count else 1.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url", default=os.environ.get("IMDB_DATABASE_URL", "postgresql://localhost/imdb")
    )
    parser.add_argument("--left-table", default="imdb_t1_hnsw")
    parser.add_argument("--right-table", default="imdb_t2_hnsw")
    parser.add_argument("--left-id", default="id")
    parser.add_argument("--right-id", default="id")
    parser.add_argument("--left-vector", default="actor_director_emb")
    parser.add_argument("--right-vector", default="actor_director_emb")
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--ef-search", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--left-limit", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=256, help="shared: rows per k-means group")
    parser.add_argument("--seeds", type=int, default=4, help="shared: nodes carried to the next probe")
    parser.add_argument(
        "--methods", nargs="+", choices=("exact", "iterative", "batched", "vectorized", "shared"),
        default=["exact", "iterative", "batched", "vectorized", "shared"],
    )
    parser.add_argument("--output", type=Path, default=Path("results/semji_construction.json"))
    args = parser.parse_args(argv)

    for name in (
        args.left_table, args.right_table, args.left_id, args.right_id,
        args.left_vector, args.right_vector,
    ):
        checked_identifier(name)
    builders: dict[str, Callable[..., None]] = {
        "exact": build_exact,
        "iterative": build_iterative,
        "batched": build_batched,
        "vectorized": build_vectorized,
        "shared": build_shared,
    }
    common = {
        "left_table": args.left_table, "right_table": args.right_table,
        "left_id": args.left_id, "right_id": args.right_id,
        "left_vector": args.left_vector, "right_vector": args.right_vector,
        "radius": args.radius, "left_limit": args.left_limit,
    }
    results: dict[str, Any] = {}
    # Client-side binding: `SET hnsw.radius = %s` cannot take a server-side parameter.
    with psycopg.connect(args.database_url, cursor_factory=psycopg.ClientCursor) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            # The pgvector fork writes LOG lines for every HNSW scan it runs; the
            # paper's timings are taken without them.
            cur.execute("SET log_min_messages = fatal")
            if "shared" in args.methods:
                try:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS dase_batch_hnsw")
                except psycopg.Error as error:
                    raise SystemExit(
                        f"CREATE EXTENSION dase_batch_hnsw failed: {error}\n"
                        "Install it from batch_hnsw/ (README §1.2), or leave out --methods shared."
                    )
        conn.commit()
        for method in args.methods:
            result_table = f"semji_benchmark_{method}"
            create_result_table(conn, result_table)
            keyword_args = dict(common)
            if method != "exact":
                keyword_args["ef_search"] = args.ef_search
            if method == "batched":
                keyword_args["batch_size"] = args.batch_size
            if method == "shared":
                keyword_args["group_size"] = args.group_size
                keyword_args["seeds"] = args.seeds
            started = time.perf_counter()
            details = builders[method](conn, result_table=result_table, **keyword_args)
            seconds = time.perf_counter() - started
            results[method] = {"seconds": seconds, "pairs": table_count(conn, result_table)}
            results[method].update(details or {})
        if "exact" in results:
            for method in results:
                results[method]["recall"] = recall_against(
                    conn, f"semji_benchmark_{method}", "semji_benchmark_exact"
                )

    payload = {"config": vars(args) | {"output": str(args.output)}, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())