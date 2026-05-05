"""
JIM — Join Indices Manager for DASE.

Converts a join condition J into a predicate P_J (a set of valid entity IDs)
by querying the pre-built TI (Table Index) tables.

Functionality:
    jim_get_valid_ids(conn, join_spec, predicates, side)
        → List[int]   (entity IDs on `side` with at least one valid partner)

Usage:
    from ours.jim import jim_get_valid_ids
"""

from typing import Any, Dict, List

import psycopg

from ours.utils import find_ti_table


def jim_get_seed_ids_for_targets(
    conn: psycopg.Connection,
    join_spec: Dict[str, Any],
    target_ids: List[int],
    side: str,
) -> List[int]:
    """
    Given a set of valid target IDs, return seed IDs that are join-linked
    to at least one of them in the TI table within the distance threshold.

    Args:
        conn: DB connection.
        join_spec: Join condition dict.
        target_ids: IDs from the target table that pass P_target.
        side: "left" or "right" — which side is the seed.
    """
    if not target_ids:
        return []

    t_left = join_spec["table_left"].lower()
    t_right = join_spec["table_right"].lower()
    tau = float(join_spec["distance_threshold"])

    ti_table = find_ti_table(conn, t_left, t_right, tau)

    seed_id_col = f"{t_left}.id" if side == "left" else f"{t_right}.id"
    target_id_col = f"{t_right}.id" if side == "left" else f"{t_left}.id"

    sql = (
        f'SELECT DISTINCT "{seed_id_col}" FROM "{ti_table}" '
        f'WHERE "{target_id_col}" = ANY(%(target_ids)s) '
        f'AND "dis" <= %(tau)s'
    )
    with conn.cursor(row_factory=psycopg.rows.tuple_row) as cur:
        cur.execute(sql, {"target_ids": target_ids, "tau": tau})
        return [r[0] for r in cur.fetchall()]


def jim_get_valid_ids(
    conn: psycopg.Connection,
    join_spec: Dict[str, Any],
    predicates: List[Dict[str, Any]],
    side: str,
) -> List[int]:
    """
    Query the TI table to get valid entity IDs for the given side.

    Converts J → P_J: returns IDs from *side* ("left" or "right") that
    have at least one valid join partner passing all predicates.

    Args:
        conn: DB connection.
        join_spec: Join condition dict with table_left, table_right,
                   distance_threshold, etc.
        predicates: All predicates from the query (each has a "table" key).
        side: "left" or "right" — which side's IDs to return.
    """
    t_left = join_spec["table_left"].lower()
    t_right = join_spec["table_right"].lower()
    tau = float(join_spec["distance_threshold"])

    ti_table = find_ti_table(conn, t_left, t_right, tau)

    where_parts: List[str] = []
    params: Dict[str, Any] = {}

    where_parts.append('"dis" <= %(tau)s')
    params["tau"] = tau

    for i, p in enumerate(predicates):
        tbl = p["table"].lower()
        attr = p["attribute"]
        op = p["operator"]
        val = p["value"]
        ti_col = f"{tbl}.{attr}"
        key = f"jp{i}"
        if op == "in":
            where_parts.append(f'"{ti_col}" = ANY(%({key})s)')
            params[key] = list(val) if not isinstance(val, list) else val
        else:
            where_parts.append(f'"{ti_col}" {op} %({key})s')
            params[key] = val

    where_sql = " AND ".join(where_parts)

    id_col = f"{t_left}.id" if side == "left" else f"{t_right}.id"
    sql = f'SELECT DISTINCT "{id_col}" FROM "{ti_table}" WHERE {where_sql}'

    with conn.cursor(row_factory=psycopg.rows.tuple_row) as cur:
        cur.execute(sql, params)
        return [r[0] for r in cur.fetchall()]
