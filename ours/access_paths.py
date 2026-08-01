"""Shared access-path helpers used by FSS and the W3/W4 executors."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import psycopg
from pgvector import Vector
from psycopg.rows import tuple_row

from ours.utils import METRIC_OP, quote


def compute_ivfflat_probes(
    stream_size: int,
    *,
    dataset: str,
    lists: int = 100,
    probe_factor: float = 1.0,
) -> int:
    """Choose an IVFFlat probe count from the requested stream size."""
    rows_per_list = 500 if dataset == "molecule" else 430
    raw = math.ceil(3 * stream_size / rows_per_list)
    probes = min(lists, max(round(lists ** 0.65), raw))
    return min(lists, max(1, math.ceil(probes * max(probe_factor, 0.0))))


def assert_ivfflat_exists(
    conn: psycopg.Connection, table: str, field: str
) -> None:
    """Raise when no IVFFlat index covers ``<table>_ivf.<field>``."""
    relation = f"{table.lower()}_ivf"
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM pg_index ix
            JOIN pg_class i ON i.oid = ix.indexrelid
            JOIN pg_class t ON t.oid = ix.indrelid
            JOIN pg_am am ON am.oid = i.relam
            JOIN pg_attribute a
              ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
            WHERE t.relname = %s
              AND am.amname = 'ivfflat'
              AND a.attname = %s
            LIMIT 1
            """,
            (relation, field),
        )
        if cur.fetchone() is None:
            raise RuntimeError(f"IVFFlat index on {relation}.{field} not found")


def build_predicate_id_sql(
    table: str,
    predicates: List[Dict[str, Any]],
    *,
    id_column: str,
) -> Tuple[str, Dict[str, Any]]:
    """Build a parameterized predicate-first ID scan on an IVFFlat relation."""
    relation = f"{table.lower()}_ivf"
    where_parts: List[str] = []
    params: Dict[str, Any] = {}
    for index, predicate in enumerate(predicates):
        if predicate["table"].lower() != table.lower():
            continue
        key = f"p{index}"
        operator = predicate["operator"]
        if operator == "in":
            where_parts.append(f'{quote(predicate["attribute"])} = ANY(%({key})s)')
            params[key] = list(predicate["value"])
        else:
            where_parts.append(
                f'{quote(predicate["attribute"])} {operator} %({key})s'
            )
            params[key] = predicate["value"]
    where_sql = " AND ".join(where_parts) if where_parts else "TRUE"
    return (
        f"SELECT {quote(id_column)} FROM {quote(relation)} WHERE {where_sql}",
        params,
    )


def run_predicate_first_w4(
    conn: psycopg.Connection,
    query: Dict[str, Any],
    *,
    id_column: str,
    text_column: str,
) -> Tuple[List[Tuple], int]:
    """Filter W4 rows first, then score the surviving IDs in one SQL query."""
    scoring = query["scoring"]
    signals = scoring["signals"]
    weights = scoring["weights"]
    table = signals[0]["table"]
    relation = f"{table.lower()}_ivf"
    id_sql, id_params = build_predicate_id_sql(
        table,
        query.get("predicates", []),
        id_column=id_column,
    )
    with conn.cursor(row_factory=tuple_row) as cur:
        cur.execute(id_sql, id_params)
        ids = [row[0] for row in cur.fetchall()]
    if not ids:
        return [], 0

    score_parts: List[str] = []
    params: Dict[str, Any] = {
        "ids": ids,
        "limit_k": int(query.get("K", 20)),
    }
    for index, (signal, weight) in enumerate(zip(signals, weights)):
        field = signal["field"]
        metric = signal.get("metric", "jaccard" if field == "mol_ecfp" else "l2")
        key = f"qv{index}"
        query_embedding = signal["query_embed"]
        if metric == "jaccard":
            params[key] = "".join(str(int(bit)) for bit in query_embedding)
            score_parts.append(
                f'{float(weight)} * (t.{quote(field)} <%%> '
                f'%({key})s::bit({len(query_embedding)}))'
            )
        else:
            params[key] = Vector(query_embedding)
            score_parts.append(
                f'{float(weight)} * (t.{quote(field)} {METRIC_OP[metric]} %({key})s)'
            )

    score_expression = " + ".join(score_parts)
    statement = (
        f"SELECT t.{quote(id_column)}, t.{quote(text_column)}, "
        f"({score_expression}) AS score_dist "
        f"FROM {quote(relation)} t "
        f"WHERE t.{quote(id_column)} = ANY(%(ids)s) "
        f"ORDER BY score_dist LIMIT %(limit_k)s"
    )
    with conn.cursor(row_factory=tuple_row) as cur:
        cur.execute("SET LOCAL enable_indexscan = off")
        cur.execute(statement, params)
        return cur.fetchall(), len(ids)