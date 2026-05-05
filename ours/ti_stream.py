"""
TIScoreStream: stream TI-table pairs ordered by join-distance ``dis``.

Split out from FilteredScoreStreamer because TI access is fundamentally
different from entity-table access (composite PK, dis btree, no HNSW).

Design — ladder streaming with keyset-paginated LIMIT doubling:

Given multiple precomputed TI tables with thresholds τ_0 < τ_1 < ... (e.g.
``ti_0.5`` and ``ti_0.6``), and a query-side ``query_tau``, the stream
consumes pairs from the smallest table first.  Phase i fetches from
``ti_{τ_i}`` pairs with ``dis`` in ``(τ_{i-1}, min(τ_i, query_tau)]`` in
ascending dis order, exhausting the phase before moving up.

Within each phase, ``_fetch_more`` issues ``SELECT ... ORDER BY dis
LIMIT K`` with a keyset cursor ``(dis, pk_col_1, pk_col_2, ...) > last``.
``K`` doubles after each fetch up to ``FETCH_K_MAX``, and resets at each
phase boundary.  This avoids Postgres ever materialising the entire
58M-row ``ti_0.6`` for a W7/W8 query that converges inside the 2.1M-row
``ti_0.5``.

Public API is duck-typed to ``FilteredScoreStreamer`` so the W7/W8
threshold loop treats both uniformly.
"""

import time as _time
from typing import Any, Dict, List, Optional, Tuple

import psycopg
from psycopg.rows import dict_row

from ours.utils import quote as _quote


SQL_OPS = {">=", "<=", ">", "<", "=", "!=", "@>"}

FETCH_K_INIT = 200
FETCH_K_MAX = 100_000


class TIScoreStream:
    """
    Stream ``(entry, dis)`` pairs from a ladder of TI tables, ascending
    by ``dis``, up to an inclusive ``tau`` bound.

    next()        -> (entry_dict, dis) | None
    peek_score()  -> float | None
    fetch_topk(k) -> list[(entry_dict, dis)]
    """

    def __init__(
        self,
        conn: psycopg.Connection,
        tables: List[Tuple[str, float]],
        tau: float,
        predicates: Optional[List[Dict[str, Any]]] = None,
        pk_cols: Optional[List[str]] = None,
        field: str = "dis",
        select_cols: Optional[List[str]] = None,
        # Back-compat: old single-table kwarg. Ignored if `tables` is non-empty.
        table: Optional[str] = None,
        chunk_step: Optional[float] = None,  # unused, kept for call-site compat
        init_fetch_K: Optional[int] = None,
    ):
        """
        Args:
            conn: active psycopg connection.
            tables: ladder of ``(table_name, table_tau)`` sorted ascending
                by ``table_tau``.  See ``find_ti_table_ladder``.
            tau: query-side upper inclusive bound on ``dis``.
            predicates: non-dis predicates.  Attribute names should already
                be qualified (e.g. ``"imdb_t1.rating"``).  No ``dis``
                predicate — use ``tau`` instead.
            pk_cols: primary-key columns used as keyset-pagination
                tie-breakers (e.g. ``["imdb_t1.id", "imdb_t2.id"]``).
                Required for deterministic ordering and correct paging
                across duplicate-``dis`` rows.
            field: join-distance column name.
            select_cols: explicit projection.  If None, ``SELECT *``.
            table: back-compat — if ``tables`` is empty and this is set,
                build a 1-entry ladder ``[(table, tau)]``.
            chunk_step: ignored; retained so old call sites don't break.
        """
        self.conn = conn
        if not tables and table is not None:
            tables = [(table, float(tau))]
        self._ladder: List[Tuple[str, float]] = list(tables)
        self.tau = float(tau)
        self.predicates = list(predicates or [])
        self.pk_cols: List[str] = list(pk_cols or [])
        self.field = field
        self.select_cols = list(select_cols) if select_cols else None

        # Phase state
        self._phase_idx: int = 0
        self._phase_prev_tau: float = 0.0  # exclusive lower dis bound (prev table's tau, or 0)
        self._cursor: Optional[Tuple[Any, ...]] = None  # (dis, *pk_vals) last emitted
        self._init_fetch_K: int = int(init_fetch_K) if init_fetch_K else FETCH_K_INIT
        self._fetch_K: int = self._init_fetch_K

        self._buffer: List[Tuple[Dict[str, Any], float]] = []
        self._buf_pos: int = 0
        self._exhausted: bool = not self._ladder

        self.profile: Dict[str, Any] = {
            "init_table_size": 0.0,
            "init_selectivity": 0.0,
            "fetch_db_query": 0.0,
            "fetch_postfilter": 0.0,
            "fetch_n_batches": 0,
            "filtered_stream_length": 0,
            "cursor_on_filtered_stream": 0,
            "n_queries_issued": 0,
            "max_dis_reached": 0.0,
            "phases_entered": 0,
            "ladder_len": len(self._ladder),
        }

    @property
    def strategy(self) -> str:
        return "ti_ladder"

    # ------------------------------------------------------------------
    # Public interface (duck-typed to match FilteredScoreStreamer)
    # ------------------------------------------------------------------

    def next(self) -> Optional[Tuple[Dict[str, Any], float]]:
        if self._buf_pos < len(self._buffer):
            item = self._buffer[self._buf_pos]
            self._buf_pos += 1
            self.profile["cursor_on_filtered_stream"] += 1
            return item
        if self._exhausted:
            return None
        self._fetch_more()
        if not self._buffer:
            return None
        self._buf_pos = 1
        self.profile["cursor_on_filtered_stream"] += 1
        return self._buffer[0]

    def peek_score(self) -> Optional[float]:
        if self._buf_pos < len(self._buffer):
            return self._buffer[self._buf_pos][1]
        if self._exhausted:
            return None
        self._fetch_more()
        if not self._buffer:
            return None
        return self._buffer[0][1]

    def fetch_topk(self, k: int) -> List[Tuple[Dict[str, Any], float]]:
        results: List[Tuple[Dict[str, Any], float]] = []
        while len(results) < k:
            item = self.next()
            if item is None:
                break
            results.append(item)
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_other_where(self) -> Tuple[List[str], Dict[str, Any]]:
        parts: List[str] = []
        params: Dict[str, Any] = {}
        for i, p in enumerate(self.predicates):
            attr, op, val = p["attribute"], p["operator"], p["value"]
            key = f"tp{i}"
            if op == "in":
                parts.append(f'{_quote(attr)} = ANY(%({key})s)')
                params[key] = list(val) if not isinstance(val, list) else val
            elif op in SQL_OPS:
                parts.append(f'{_quote(attr)} {op} %({key})s')
                params[key] = val
            else:
                raise ValueError(f"Unsupported predicate operator: {op}")
        return parts, params

    def _select_expr(self) -> str:
        if self.select_cols:
            return ", ".join(_quote(c) for c in self.select_cols)
        return "*"

    def _order_by_sql(self) -> str:
        keys = [_quote(self.field)] + [_quote(c) for c in self.pk_cols]
        return ", ".join(keys)

    def _advance_phase(self, current_upper: float) -> None:
        self._phase_idx += 1
        self._phase_prev_tau = current_upper
        self._cursor = None
        self._fetch_K = self._init_fetch_K

    def _fetch_more(self) -> None:
        self._buffer = []
        self._buf_pos = 0
        self.profile["fetch_n_batches"] += 1

        other_where, other_params = self._build_other_where()
        select_expr = self._select_expr()
        order_by = self._order_by_sql()

        while self._phase_idx < len(self._ladder):
            table, table_tau = self._ladder[self._phase_idx]
            upper = min(table_tau, self.tau)

            # Skip degenerate phases (upper <= prev lower means no pairs here)
            if upper <= self._phase_prev_tau:
                self._advance_phase(upper)
                continue

            where_parts = list(other_where)
            params = dict(other_params)

            # Lower bound: explicit dis > prev_tau for phases > 0.
            # Phase 0 has no lower bound (picks up dis=0).
            if self._phase_idx > 0:
                where_parts.append(f'{_quote(self.field)} > %(_dl)s')
                params["_dl"] = self._phase_prev_tau
            where_parts.append(f'{_quote(self.field)} <= %(_dh)s')
            params["_dh"] = upper

            # Keyset cursor within the phase.
            if self._cursor is not None:
                cursor_cols = [_quote(self.field)] + [_quote(c) for c in self.pk_cols]
                cursor_placeholders = [f"%(_c{i})s" for i in range(len(cursor_cols))]
                where_parts.append(
                    f'({", ".join(cursor_cols)}) > ({", ".join(cursor_placeholders)})'
                )
                for i, v in enumerate(self._cursor):
                    params[f"_c{i}"] = v

            params["_lim"] = self._fetch_K

            sql = (
                f'SELECT {select_expr}, {_quote(self.field)} AS _score '
                f'FROM {_quote(table)} '
                f'WHERE {" AND ".join(where_parts)} '
                f'ORDER BY {order_by} '
                f'LIMIT %(_lim)s'
            )

            t0 = _time.perf_counter()
            with self.conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
            self.profile["fetch_db_query"] += _time.perf_counter() - t0
            self.profile["n_queries_issued"] += 1

            if self._cursor is None:
                self.profile["phases_entered"] += 1

            if rows:
                self._buffer = [(dict(r), float(r["_score"])) for r in rows]
                self.profile["filtered_stream_length"] += len(rows)
                last = rows[-1]
                last_dis = float(last["_score"])
                self._cursor = tuple([last_dis] + [last[c] for c in self.pk_cols])
                self.profile["max_dis_reached"] = last_dis

                if len(rows) < self._fetch_K:
                    # Phase exhausted (short read).
                    self._advance_phase(upper)
                else:
                    # More may be available in this phase; double K for next call.
                    self._fetch_K = min(self._fetch_K * 2, FETCH_K_MAX)
                return

            # Empty phase: advance.
            self._advance_phase(upper)

        self._exhausted = True
