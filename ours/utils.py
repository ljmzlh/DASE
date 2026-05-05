"""
Shared utilities for DASE operators (FSS, MSA, JIM).
"""

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import psycopg
from psycopg.rows import dict_row, tuple_row
from pgvector import Vector


# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------

METRIC_OP = {"l2": "<->", "ip": "<#>", "cos": "<=>", "jaccard": "<%>"}

SQL_OPS = {">=", ">", "<=", "<", "=", "!=", "@>"}


def quote(name: str) -> str:
    """Double-quote a SQL identifier."""
    return f'"{name}"'


# ---------------------------------------------------------------------------
# Predicate evaluation (Python-side post-filter)
# ---------------------------------------------------------------------------

def check_predicate(row: Dict[str, Any], attr: str, op: str, val) -> bool:
    """Evaluate one predicate against a row dict."""
    v = row.get(attr)
    if v is None:
        return False
    if op == ">=":  return v >= val
    if op == ">":   return v > val
    if op == "<=":  return v <= val
    if op == "<":   return v < val
    if op == "=":   return v == val
    if op == "!=":  return v != val
    if op == "in":  return v in val
    if op == "@>":  return set(val).issubset(set(v))
    return False


# ---------------------------------------------------------------------------
# TI table lookup
# ---------------------------------------------------------------------------

def find_ti_table(
    conn: psycopg.Connection, t_left: str, t_right: str, tau: float
) -> str:
    """
    Find the best TI table for the given join.  Picks the smallest
    existing threshold >= tau so all valid pairs are included.
    Falls back to exact match name if nothing found.
    """
    prefix = f"ti_{t_left.lower()}_{t_right.lower()}_"
    with conn.cursor(row_factory=psycopg.rows.tuple_row) as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name LIKE %s",
            [prefix + "%"],
        )
        candidates = [r[0] for r in cur.fetchall()]

    if not candidates:
        return f"ti_{t_left.lower()}_{t_right.lower()}_{tau}"

    best = None
    best_tau = float("inf")
    for name in candidates:
        suffix = name[len(prefix):]
        try:
            ti_tau = float(suffix)
        except ValueError:
            continue
        if ti_tau >= tau and ti_tau < best_tau:
            best = name
            best_tau = ti_tau

    if best is not None:
        return best

    # 没有 ti_tau >= tau 的表：退回到已存在的最大 tau（结果会漏掉 dis∈(max_ti_tau, tau] 的 pair，
    # 调用方应确保在该 tau 下 graph 已接近全连通，漏掉的 pair 对算法无意义）。
    numeric = []
    for name in candidates:
        try:
            numeric.append((float(name[len(prefix):]), name))
        except ValueError:
            continue
    numeric.sort()
    return numeric[-1][1]


def find_ti_table_ladder(
    conn: psycopg.Connection, t_left: str, t_right: str, query_tau: float
) -> List[Tuple[str, float]]:
    """
    Return all TI tables usable for a join with `query_tau`, as a
    ladder sorted ascending by stored tau.

    The returned ladder is a list of ``(table_name, table_tau)`` such
    that streaming through it in order yields all pairs with
    ``dis <= query_tau``:

      - For each entry with ``table_tau <= query_tau``, consume pairs
        from it in dis-ascending order (they cover dis in
        ``(prev_table_tau, table_tau]``, with the first entry covering
        ``[0, table_tau]``).
      - The last entry is the smallest-tau table with
        ``table_tau >= query_tau``; only pairs with
        ``dis <= query_tau`` are consumed from it.

    Example: with tables ``ti_0.5`` and ``ti_0.6``:

      - query_tau=0.6 → ``[(ti_0.5, 0.5), (ti_0.6, 0.6)]``
        (start on small 2.1M-row table; escalate to 58M-row only if
        unconverged)
      - query_tau=0.5 → ``[(ti_0.5, 0.5)]``
      - query_tau=0.4 → ``[(ti_0.5, 0.5)]`` (no smaller table exists,
        upper is clamped to query_tau)
      - query_tau=0.7 → ``[(ti_0.5, 0.5), (ti_0.6, 0.6)]`` — but note
        no table covers (0.6, 0.7], so pairs beyond 0.6 are absent.

    Returns empty list if no candidate tables exist.
    """
    prefix = f"ti_{t_left.lower()}_{t_right.lower()}_"
    with conn.cursor(row_factory=psycopg.rows.tuple_row) as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name LIKE %s",
            [prefix + "%"],
        )
        candidates = [r[0] for r in cur.fetchall()]

    parsed: List[Tuple[str, float]] = []
    for name in candidates:
        try:
            parsed.append((name, float(name[len(prefix):])))
        except ValueError:
            continue
    parsed.sort(key=lambda x: x[1])

    ladder: List[Tuple[str, float]] = []
    for name, tt in parsed:
        ladder.append((name, tt))
        if tt >= query_tau:
            break
    return ladder


# ---------------------------------------------------------------------------
# Predicate → ID set resolution
# ---------------------------------------------------------------------------

def build_where(
    predicates: List[Dict[str, Any]],
    prefix: str = "p",
) -> Tuple[List[str], Dict[str, Any]]:
    """Build SQL WHERE fragments from a list of predicate dicts.

    Returns (parts, params) where parts is a list of SQL fragments
    and params is a dict of named query parameters.
    """
    parts: List[str] = []
    params: Dict[str, Any] = {}
    for i, p in enumerate(predicates):
        attr, op, val = p["attribute"], p["operator"], p["value"]
        key = f"{prefix}{i}"
        if op == "in":
            parts.append(f'{quote(attr)} = ANY(%({key})s)')
            params[key] = list(val) if not isinstance(val, list) else val
        elif op in SQL_OPS:
            parts.append(f'{quote(attr)} {op} %({key})s')
            params[key] = val
        else:
            raise ValueError(f"Unsupported predicate operator: {op}")
    return parts, params


def resolve_predicate_ids(
    conn: psycopg.Connection,
    table: str,
    predicates: List[Dict[str, Any]],
) -> Set[int]:
    """Resolve all predicates on *table* to a set of valid IDs.

    - Clause predicates (>=, <=, etc.) → SQL query for matching IDs.
    - "in" predicates → use the value set directly.
    - Intersect all results.
    """
    # "id IN (...)" can be used directly; all others must go through SQL
    id_in_preds = [p for p in predicates if p["operator"] == "in" and p["attribute"] == "id"]
    sql_preds = [p for p in predicates if not (p["operator"] == "in" and p["attribute"] == "id")]

    result_set: Optional[Set[int]] = None

    if sql_preds:
        where_parts, params = build_where(sql_preds)
        sql = (
            f'SELECT "id" FROM {quote(table)} '
            f'WHERE {" AND ".join(where_parts)}'
        )
        with conn.cursor(row_factory=tuple_row) as cur:
            cur.execute(sql, params)
            result_set = {r[0] for r in cur.fetchall()}

    for p in id_in_preds:
        vals = set(p["value"])
        if result_set is None:
            result_set = vals
        else:
            result_set &= vals

    return result_set if result_set is not None else set()


def get_max_id(conn: psycopg.Connection, table: str) -> int:
    """Return MAX(id) for a table."""
    with conn.cursor(row_factory=tuple_row) as cur:
        cur.execute(f'SELECT MAX("id") FROM {quote(table)}')
        return cur.fetchone()[0] or 0


# ---------------------------------------------------------------------------
# Bitmap / filtered HNSW search
# ---------------------------------------------------------------------------

def make_bitmap(valid_ids: Iterable[int], max_id: int) -> bytes:
    """Build a bitmap filter: bit[id] = 1 if id is in valid_ids."""
    bitset = bytearray(max_id // 8 + 2)
    for aid in valid_ids:
        if aid is not None and 0 <= aid <= max_id:
            bitset[aid >> 3] |= 1 << (aid & 0x07)
    return bytes(bitset)


def _assert_no_ivfflat(conn: psycopg.Connection, table: str, field: str) -> None:
    """Raise if an IVFFlat index exists on (table, field).

    The planner may pick IVFFlat over HNSW, silently ignoring the
    bitmap filter passed via the <-># operator.
    """
    with conn.cursor(row_factory=tuple_row) as cur:
        cur.execute(
            "SELECT indexname FROM pg_indexes "
            "WHERE tablename = %s AND indexdef ILIKE '%%ivfflat%%' "
            "AND indexdef ILIKE %s",
            [table.lower(), f"%{field}%"],
        )
        row = cur.fetchone()
    if row:
        raise RuntimeError(
            f"IVFFlat index '{row[0]}' exists on {table}.{field}. "
            f"The <-># operator requires HNSW; drop the IVFFlat index first."
        )


def filtered_hnsw_search(
    conn: psycopg.Connection,
    table: str,
    field: str,
    query_vec,
    metric: str,
    bitmap: bytes,
    limit: int,
    offset: int = 0,
    ef_search: int = 400,
) -> List[Tuple[Dict[str, Any], float]]:
    """Execute a filtered HNSW search using the <-># operator.

    Sets the id_map table GUC, disables seqscan, and runs the filtered
    vector search.  Returns list of (row_dict, score).
    """
    _assert_no_ivfflat(conn, table, field)
    dist_op = METRIC_OP[metric]
    id_map_table = f"public.{table}_id_map"
    qv = Vector(query_vec) if not isinstance(query_vec, Vector) else query_vec

    params = {"qv": qv, "bm": bitmap, "lim": limit, "off": offset}

    sql = (
        f"SELECT *, ({quote(field)} {dist_op} %(qv)s) AS _score "
        f"FROM {quote(table)} "
        f"WHERE ({quote(field)} OPERATOR(public.<->#) %(bm)s::bytea) "
        f"ORDER BY {quote(field)} {dist_op} %(qv)s "
        f"LIMIT %(lim)s OFFSET %(off)s"
    )

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(f"SET hnsw.ef_search = {ef_search}")
        cur.execute("SET hnsw.enable_2hop = on")
        cur.execute(f"SET hnsw.id_map_table = '{id_map_table}'")
        cur.execute("SET enable_seqscan = off")
        cur.execute(sql, params)
        rows = cur.fetchall()

    return [(dict(r), float(r["_score"])) for r in rows]


# ---------------------------------------------------------------------------
# id_map table builder (ctid → real id)
# ---------------------------------------------------------------------------

def build_id_map_table(
    conn: psycopg.Connection,
    source_table: str,
) -> str:
    """Build an id_map table mapping (blkno, offno) → real id.

    The HNSW index extension uses this mapping to translate internal
    node positions to real entity IDs for bitmap filtering.

    Returns the created table name.
    """
    map_table = f"{source_table}_id_map"

    with conn.cursor() as cur:
        cur.execute(f'DROP TABLE IF EXISTS {quote(map_table)}')
        cur.execute(
            f'CREATE TABLE {quote(map_table)} ('
            f'  blkno integer NOT NULL,'
            f'  offno integer NOT NULL,'
            f'  realid integer NOT NULL,'
            f'  PRIMARY KEY (blkno, offno)'
            f')'
        )
        cur.execute(f'SELECT ctid, "id" FROM {quote(source_table)}')
        rows = cur.fetchall()

        batch = []
        for ctid, row_id in rows:
            ctid_str = str(ctid).strip("()")
            blkno, offno = map(int, ctid_str.split(","))
            batch.append((blkno, offno, row_id))

        if batch:
            with conn.cursor() as ins_cur:
                ins_cur.executemany(
                    f'INSERT INTO {quote(map_table)} (blkno, offno, realid) '
                    f'VALUES (%s, %s, %s)',
                    batch,
                )
    conn.commit()
    return map_table
