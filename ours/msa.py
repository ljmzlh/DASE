"""
Multi-Scoring Aggregator (MSA) — top-K operator for DASE.

MSA takes multiple FSS streams (each providing one scoring dimension),
combines them via a monotonic score_f, and returns top-K answer tuples
with early termination (Threshold Algorithm).

Algorithm:
  1. Pull a seed entry from one of the streams (round-robin).
  2. Synthesize the seed into a complete answer tuple by finding the
     best matching partner(s) for the missing table(s).
  3. Compute total score via score_f, insert into top-K heap.
  4. Update upper bound from peek_scores of all streams.
     If K-th best score <= upper_bound, terminate.

Three cases for synthesis:
  Case 1 — 1 signal on T1: seed is t1, find best t2 passing join + P_t2.
  Case 2 — signals on T1 + T2: seed from stream_i, find best partner
            on the other table ranked by that table's signal.
  Case 3 — Case 2 + join distance stream: TI stream produces (t1, t2)
            pairs directly; look up missing signal scores.

Works for both cross-table and self-join cases.

Usage (from /dase/):
    python -m ours.msa [workload_path]
"""

import heapq
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import psycopg
from psycopg.rows import dict_row
from pgvector import Vector
from pgvector.psycopg import register_vector

from ours.fss import FilteredScoreStreamer
from ours.utils import METRIC_OP, quote as _quote, check_predicate as _check_predicate, find_ti_table


def _norm_table_name(name: str) -> str:
    """Normalize table name for role/predicate matching (strip _hnsw suffix)."""
    n = name.lower()
    if n.endswith("_hnsw"):
        n = n[:-5]
    return n


# ---------------------------------------------------------------------------
# Synthesize: complete a seed into an answer tuple
# ---------------------------------------------------------------------------

def synthesize(
    conn: psycopg.Connection,
    seed: Dict[str, Any],
    seed_join_embed: Any,
    partner_table: str,
    partner_join_field: str,
    tau: float,
    metric: str,
    predicates: Optional[List[Dict[str, Any]]] = None,
    order_signal: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[Dict[str, Any], float]]:
    """
    Given a seed entity, find the best matching partner from partner_table.

    The partner must:
      1. Pass join condition: dist(seed_join_embed, target.partner_join_field) <= tau
      2. Pass all predicates on partner_table
      3. If order_signal is provided, be ranked by that signal (best first).
         Otherwise, any valid partner (pick the one with smallest join distance).

    Args:
        conn: DB connection (pgvector registered).
        seed: The seed entity dict (already fetched row).
        seed_join_embed: The seed's join embedding value (numpy-like).
        partner_table: Table to search for the partner.
        partner_join_field: Embedding column in partner_table for join distance.
        tau: Join distance threshold.
        metric: "l2" | "cos" | "ip".
        predicates: Predicates to apply on the partner_table rows.
        order_signal: Optional scoring signal to rank valid partners.
            {"type": "semantic", "field": "plot_emb", "query_embed": [...], "metric": "l2"}
            If None, partners are ordered by join distance (smallest first).

    Returns:
        (partner_entry, partner_score) or None if no valid partner exists.
        partner_score is the order_signal score if provided, else join distance.
    """
    dist_op = METRIC_OP.get(metric, "<->")

    # Build WHERE clause: join condition + predicates
    where_parts = [f'({_quote(partner_join_field)} {dist_op} %(seed_emb)s) <= %(tau)s']
    params: Dict[str, Any] = {
        "seed_emb": Vector(seed_join_embed),
        "tau": tau,
    }

    if predicates:
        for i, p in enumerate(predicates):
            attr, op, val = p["attribute"], p["operator"], p["value"]
            key = f"sp{i}"
            if op == "in":
                where_parts.append(f'{_quote(attr)} = ANY(%({key})s)')
                params[key] = list(val) if not isinstance(val, list) else val
            else:
                where_parts.append(f'{_quote(attr)} {op} %({key})s')
                params[key] = val

    where_sql = " AND ".join(where_parts)

    # Build ORDER BY and score expression
    if order_signal is not None:
        sig = order_signal
        if sig["type"] == "semantic":
            field = sig["field"]
            sig_metric = sig.get("metric", "l2")
            sig_op = METRIC_OP[sig_metric]
            params["order_qv"] = Vector(sig["query_embed"])
            score_expr = f'({_quote(field)} {sig_op} %(order_qv)s)'
        elif sig["type"] == "attribute":
            field = sig["field"]
            direction = sig.get("direction", "asc").upper()
            score_expr = f'(-1 * {_quote(field)})' if direction == "DESC" else _quote(field)
        else:
            # fallback: order by join distance
            score_expr = f'({_quote(partner_join_field)} {dist_op} %(seed_emb)s)'
    else:
        # No order signal: pick partner with smallest join distance
        score_expr = f'({_quote(partner_join_field)} {dist_op} %(seed_emb)s)'

    sql = (
        f"SELECT *, {score_expr} AS _score "
        f"FROM {_quote(partner_table)} "
        f"WHERE {where_sql} "
        f"ORDER BY {score_expr} "
        f"LIMIT 1"
    )

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        row = cur.fetchone()

    if row is None:
        return None
    return dict(row), float(row["_score"])


def random_access_score(
    conn: psycopg.Connection,
    table: str,
    scoring_signal: Dict[str, Any],
    entity_id: Any,
    id_col: str = "id",
) -> Optional[float]:
    """
    Look up a single entity's score for a given scoring signal.

    Used when a TI stream produces a (t1, t2) pair and we need to
    fetch the missing signal scores via random access.
    """
    params: Dict[str, Any] = {"eid": entity_id}

    sig_type = scoring_signal["type"]
    if sig_type == "semantic":
        field = scoring_signal["field"]
        metric = scoring_signal.get("metric", "l2")
        dist_op = METRIC_OP[metric]
        params["qv"] = Vector(scoring_signal["query_embed"])
        score_expr = f'({_quote(field)} {dist_op} %(qv)s)'
    elif sig_type == "attribute":
        field = scoring_signal["field"]
        direction = scoring_signal.get("direction", "asc").upper()
        score_expr = f'(-1 * {_quote(field)})' if direction == "DESC" else _quote(field)
    else:
        raise ValueError(f"random_access_score does not support signal type: {sig_type}")

    sql = (
        f"SELECT {score_expr} AS _score "
        f"FROM {_quote(table)} "
        f"WHERE {_quote(id_col)} = %(eid)s"
    )

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        row = cur.fetchone()

    if row is None:
        return None
    return float(row["_score"])


# ---------------------------------------------------------------------------
# MultiScoringAggregator
# ---------------------------------------------------------------------------

class MultiScoringAggregator:
    """
    MSA({FSS_i}, score_f, K) → top-K answer tuples.

    Each answer tuple is (t_left_entry, t_right_entry, total_score).

    The aggregator pulls seeds from FSS streams, synthesizes complete
    answer tuples, and terminates when the K-th best can't be beaten.
    """

    def __init__(
        self,
        conn: psycopg.Connection,
        streams: List[Dict[str, Any]],
        score_f: Callable[..., float],
        k: int,
        join_spec: Dict[str, Any],
        predicates: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Args:
            conn: DB connection.

            streams: List of stream descriptors. Each descriptor:
                {
                    "fss": FilteredScoreStreamer instance,
                    "signal": the scoring signal dict (for random access),
                    "table": which entity table this stream covers,
                    "role": "left" | "right" | "join",
                    "weight_idx": index into score_f's argument list,
                }
                "role" determines what the stream produces:
                  - "left": entries from the left table
                  - "right": entries from the right table
                  - "join": entries from TI table (already (t1, t2) pairs)

            score_f: Monotonic aggregation function.
                Takes a list of per-signal scores (same order as streams)
                and returns a combined score. Lower is better.
                Example: lambda scores: sum(w * s for w, s in zip(weights, scores))

            k: Number of result tuples to return.

            join_spec: Join condition dict from the query:
                {"table_left", "table_right", "embed_left", "embed_right",
                 "distance_threshold", "metric"}

            predicates: All predicates from the query (with "table" field).
                MSA splits them by table for synthesize calls.
        """
        self.conn = conn
        self.streams = streams
        self.score_f = score_f
        self.k = k
        self.join_spec = join_spec
        self.all_predicates = predicates or []

        # Split predicates by table (lowercase for SQL, original for predicate matching)
        t_left = join_spec["table_left"]
        t_right = join_spec["table_right"]
        self.t_left = t_left.lower()
        self.t_right = t_right.lower()
        self._t_left_orig = t_left
        self._t_right_orig = t_right
        self.embed_left = join_spec["embed_left"]
        self.embed_right = join_spec["embed_right"]
        self.tau = float(join_spec["distance_threshold"])
        self.metric = join_spec.get("metric", "l2")

        self.preds_left = [
            p for p in self.all_predicates
            if _norm_table_name(p["table"]) == _norm_table_name(self.t_left)
        ]
        self.preds_right = [
            p for p in self.all_predicates
            if _norm_table_name(p["table"]) == _norm_table_name(self.t_right)
        ]

        # Index streams by role for easy lookup
        self.left_streams = [s for s in streams if s["role"] == "left"]
        self.right_streams = [s for s in streams if s["role"] == "right"]
        self.join_streams = [s for s in streams if s["role"] == "join"]

        # Physical tables used for row lookup / synthesize SQL.
        # Logical join names (self.t_left/self.t_right) are still used for TI columns.
        self.left_table_phys = (
            self.left_streams[0]["table"].lower() if self.left_streams else self.t_left
        )
        self.right_table_phys = (
            self.right_streams[0]["table"].lower() if self.right_streams else self.t_right
        )

        # Find order signals for each side (used in synthesize)
        # If there's a signal on the partner's side, use it to rank partners
        self._left_order_signal = None
        self._right_order_signal = None
        for s in self.left_streams:
            self._left_order_signal = s["signal"]
        for s in self.right_streams:
            self._right_order_signal = s["signal"]

        # Number of scoring dimensions
        self.n_signals = len(streams)

        # Last-seen score per stream (for upper bound)
        self._last_scores = [0.0] * self.n_signals

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> List[Tuple[Dict[str, Any], Dict[str, Any], float]]:
        """
        Execute the threshold algorithm.

        Returns list of (t_left_entry, t_right_entry, total_score),
        sorted by total_score ascending (best first), up to K entries.
        """
        # Top-K heap: max-heap by (-total_score) so we can pop worst
        # Items: (-total_score, tie_breaker, (t_left, t_right, total_score))
        heap: List[Tuple[float, int, Tuple]] = []
        tie = 0

        # Track seen answer tuples to avoid duplicates
        # Key: (id_left, id_right) — works for both cross-join and self-join
        seen_pairs: set = set()

        # Round-robin stream index
        stream_idx = 0
        all_exhausted = False

        while not all_exhausted:
            # Find next non-exhausted stream (round-robin)
            attempts = 0
            while attempts < self.n_signals:
                sd = self.streams[stream_idx]
                fss = sd["fss"]
                if fss.peek_score() is not None:
                    break
                stream_idx = (stream_idx + 1) % self.n_signals
                attempts += 1
            else:
                # All streams exhausted
                break

            sd = self.streams[stream_idx]
            fss = sd["fss"]
            role = sd["role"]
            weight_idx = sd["weight_idx"]

            # Pull one seed
            item = fss.next()
            if item is None:
                stream_idx = (stream_idx + 1) % self.n_signals
                continue

            entry, score = item
            self._last_scores[weight_idx] = score

            # Synthesize seed into answer tuple(s)
            candidates = self._synthesize_seed(entry, score, sd)

            for t_left, t_right, scores_list in candidates:
                pair_key = (t_left.get("id"), t_right.get("id"))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                total_score = self.score_f(scores_list)

                if len(heap) < self.k:
                    heapq.heappush(heap, (-total_score, tie, (t_left, t_right, total_score)))
                    tie += 1
                elif total_score < -heap[0][0]:
                    heapq.heapreplace(heap, (-total_score, tie, (t_left, t_right, total_score)))
                    tie += 1

            # Check early termination
            if len(heap) >= self.k:
                upper = self._upper_bound()
                kth_best = -heap[0][0]  # worst score in top-K
                if kth_best <= upper:
                    break

            stream_idx = (stream_idx + 1) % self.n_signals

        # Extract and sort results
        results = [item for _, _, item in sorted(heap, key=lambda x: -x[0])]
        return results

    # ------------------------------------------------------------------
    # Synthesize seed → candidate answer tuples
    # ------------------------------------------------------------------

    def _synthesize_seed(
        self, entry: Dict[str, Any], seed_score: float, stream_desc: Dict[str, Any]
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any], List[float]]]:
        """
        Complete a seed entry into full answer tuples.

        Returns list of (t_left, t_right, scores_list) where scores_list
        has one float per stream in the same order as self.streams.
        """
        role = stream_desc["role"]
        weight_idx = stream_desc["weight_idx"]

        if role == "left":
            return self._synthesize_left_seed(entry, seed_score, weight_idx)
        elif role == "right":
            return self._synthesize_right_seed(entry, seed_score, weight_idx)
        elif role == "join":
            return self._synthesize_join_seed(entry, seed_score, weight_idx)
        return []

    def _synthesize_left_seed(
        self, seed: Dict, seed_score: float, weight_idx: int
    ) -> List[Tuple[Dict, Dict, List[float]]]:
        """Seed is a t_left entry. Find best t_right partner."""
        seed_emb = seed.get(self.embed_left)
        if seed_emb is None:
            return []

        result = synthesize(
            self.conn, seed, seed_emb,
            partner_table=self.right_table_phys,
            partner_join_field=self.embed_right,
            tau=self.tau, metric=self.metric,
            predicates=self.preds_right,
            order_signal=self._right_order_signal,
        )
        if result is None:
            return []

        partner, partner_score = result

        # Build scores list
        scores = [0.0] * self.n_signals
        scores[weight_idx] = seed_score

        # Fill in partner's signal score
        for s in self.right_streams:
            scores[s["weight_idx"]] = partner_score

        # Fill in join distance signal scores (if any join streams exist)
        for s in self.join_streams:
            dist_op = METRIC_OP.get(self.metric, "<->")
            join_dist = self._compute_join_distance(seed, partner)
            if join_dist is not None:
                scores[s["weight_idx"]] = join_dist

        # Fill in any other left signals via random access
        for s in self.left_streams:
            if s["weight_idx"] == weight_idx:
                continue  # already have this score
            ra_score = random_access_score(
                self.conn, self.left_table_phys, s["signal"], seed["id"]
            )
            if ra_score is not None:
                scores[s["weight_idx"]] = ra_score

        return [(seed, partner, scores)]

    def _synthesize_right_seed(
        self, seed: Dict, seed_score: float, weight_idx: int
    ) -> List[Tuple[Dict, Dict, List[float]]]:
        """Seed is a t_right entry. Find best t_left partner."""
        seed_emb = seed.get(self.embed_right)
        if seed_emb is None:
            return []

        result = synthesize(
            self.conn, seed, seed_emb,
            partner_table=self.left_table_phys,
            partner_join_field=self.embed_left,
            tau=self.tau, metric=self.metric,
            predicates=self.preds_left,
            order_signal=self._left_order_signal,
        )
        if result is None:
            return []

        partner, partner_score = result

        # partner is t_left, seed is t_right
        t_left = partner
        t_right = seed

        scores = [0.0] * self.n_signals
        scores[weight_idx] = seed_score

        for s in self.left_streams:
            scores[s["weight_idx"]] = partner_score

        for s in self.join_streams:
            join_dist = self._compute_join_distance(t_left, t_right)
            if join_dist is not None:
                scores[s["weight_idx"]] = join_dist

        for s in self.right_streams:
            if s["weight_idx"] == weight_idx:
                continue
            ra_score = random_access_score(
                self.conn, self.right_table_phys, s["signal"], seed["id"]
            )
            if ra_score is not None:
                scores[s["weight_idx"]] = ra_score

        return [(t_left, t_right, scores)]

    def _synthesize_join_seed(
        self, entry: Dict, seed_score: float, weight_idx: int
    ) -> List[Tuple[Dict, Dict, List[float]]]:
        """
        Seed is a TI entry — already a (t_left_id, t_right_id) pair.
        Look up missing signal scores via random access.
        """
        # Extract IDs from TI row (columns like "imdb_t1.id", "imdb_t2.id")
        left_id_col = f"{self.t_left.lower()}.id"
        right_id_col = f"{self.t_right.lower()}.id"
        left_id = entry.get(left_id_col)
        right_id = entry.get(right_id_col)

        if left_id is None or right_id is None:
            return []

        # Check predicates on TI's materialized columns
        for p in self.preds_left:
            ti_col = f"{self.t_left.lower()}.{p['attribute']}"
            v = entry.get(ti_col)
            if v is None:
                return []
            if not _check_predicate(entry, ti_col, p["operator"], p["value"]):
                return []
        for p in self.preds_right:
            ti_col = f"{self.t_right.lower()}.{p['attribute']}"
            v = entry.get(ti_col)
            if v is None:
                return []
            if not _check_predicate(entry, ti_col, p["operator"], p["value"]):
                return []

        scores = [0.0] * self.n_signals
        scores[weight_idx] = seed_score

        # Fetch full rows for t_left and t_right (needed for output)
        t_right = self._fetch_row(self.right_table_phys, right_id)
        t_left = self._fetch_row(self.left_table_phys, left_id)
        if t_left is None or t_right is None:
            return []

        # Fill in missing signal scores via random access
        for s in self.left_streams:
            ra_score = random_access_score(
                self.conn, self.left_table_phys, s["signal"], left_id
            )
            if ra_score is not None:
                scores[s["weight_idx"]] = ra_score

        for s in self.right_streams:
            ra_score = random_access_score(
                self.conn, self.right_table_phys, s["signal"], right_id
            )
            if ra_score is not None:
                scores[s["weight_idx"]] = ra_score

        for s in self.join_streams:
            if s["weight_idx"] == weight_idx:
                continue
            # Multiple join streams shouldn't happen, but handle gracefully
            scores[s["weight_idx"]] = seed_score

        return [(t_left, t_right, scores)]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_join_distance(
        self, t_left: Dict, t_right: Dict
    ) -> Optional[float]:
        """Compute join distance between t_left and t_right entries."""
        a = t_left.get(self.embed_left)
        b = t_right.get(self.embed_right)
        if a is None or b is None:
            return None
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if self.metric == "l2":
            return float(np.linalg.norm(a - b))
        elif self.metric == "cos":
            a = a / (np.linalg.norm(a) + 1e-12)
            b = b / (np.linalg.norm(b) + 1e-12)
            return float(np.linalg.norm(a - b))
        return float(np.linalg.norm(a - b))

    def _fetch_row(self, table: str, entity_id: Any) -> Optional[Dict]:
        """Fetch a full row by id."""
        sql = f'SELECT * FROM {_quote(table)} WHERE "id" = %(eid)s'
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, {"eid": entity_id})
            row = cur.fetchone()
        return dict(row) if row else None

    def _upper_bound(self) -> float:
        """
        Compute the upper bound (threshold) from peek_scores.

        Any unseen answer tuple has each signal score >= the last seen
        score on that stream. Since score_f is monotonic, the best
        possible unseen total score is score_f(last_scores).
        """
        bound_scores = list(self._last_scores)

        # For each stream, update with peek_score if available
        # (peek may be tighter than last_seen when stream advanced)
        for i, sd in enumerate(self.streams):
            ps = sd["fss"].peek_score()
            if ps is not None:
                bound_scores[i] = ps
            # If stream exhausted, use last seen score (already in _last_scores)

        return self.score_f(bound_scores)


# ---------------------------------------------------------------------------
# Query runner: parse workload query → MSA
# ---------------------------------------------------------------------------

def build_msa_from_query(
    conn: psycopg.Connection, query: Dict[str, Any]
) -> MultiScoringAggregator:
    """
    Build an MSA from a workload query dict (W6/W7/W8 format).

    Parses scoring signals, creates FSS streams, and wires them into MSA.
    """
    predicates = query.get("predicates", [])
    join_spec = query.get("join", {})
    scoring = query.get("scoring", {})
    k = int(query.get("K", 20))

    signals = scoring.get("signals", [])
    agg = scoring.get("aggregation", "identity")
    weights = scoring.get("weights", [1.0] * len(signals))

    t_left = join_spec["table_left"]
    t_right = join_spec["table_right"]
    tau = float(join_spec["distance_threshold"])

    # Determine TI table name — find the smallest existing TI with threshold >= tau
    ti_table = find_ti_table(conn, t_left, t_right, tau)

    # Build FSS streams
    streams = []
    for i, sig in enumerate(signals):
        sig_type = sig["type"]

        if sig_type == "semantic":
            table = sig["table"]
            role = "left" if _norm_table_name(table) == _norm_table_name(t_left) else "right"

            # Predicates for this table
            table_preds = [
                p for p in predicates
                if _norm_table_name(p["table"]) == _norm_table_name(table)
            ]

            fss = FilteredScoreStreamer(
                conn=conn,
                table=table.lower(),
                scoring_signal={
                    "type": "semantic",
                    "field": sig["field"],
                    "query_embed": sig["query_embed"],
                    "metric": sig.get("metric", "l2"),
                },
                predicates=table_preds,
            )

            streams.append({
                "fss": fss,
                "signal": sig,
                "table": table,
                "role": role,
                "weight_idx": i,
            })

        elif sig_type == "join_distance":
            # Stream from TI table, sorted by "dis"
            # Apply predicates from both sides on TI's materialized columns
            ti_preds = []
            for p in predicates:
                ti_preds.append({
                    "attribute": f"{_norm_table_name(p['table'])}.{p['attribute']}",
                    "operator": p["operator"],
                    "value": p["value"],
                })
            # If TI was built with a larger threshold, filter by query's tau
            ti_preds.append({
                "attribute": "dis",
                "operator": "<=",
                "value": tau,
            })

            fss = FilteredScoreStreamer(
                conn=conn,
                table=ti_table,
                scoring_signal={"type": "relational", "field": "dis"},
                predicates=ti_preds,
            )

            streams.append({
                "fss": fss,
                "signal": sig,
                "table": ti_table,
                "role": "join",
                "weight_idx": i,
            })

        elif sig_type == "attribute":
            table = sig["table"]
            role = "left" if _norm_table_name(table) == _norm_table_name(t_left) else "right"
            table_preds = [
                p for p in predicates
                if _norm_table_name(p["table"]) == _norm_table_name(table)
            ]

            fss = FilteredScoreStreamer(
                conn=conn,
                table=table.lower(),
                scoring_signal={
                    "type": "attribute",
                    "field": sig["field"],
                    "direction": sig.get("direction", "asc"),
                },
                predicates=table_preds,
            )

            streams.append({
                "fss": fss,
                "signal": sig,
                "table": table,
                "role": role,
                "weight_idx": i,
            })

    # Build score_f from aggregation type and weights
    if agg == "weighted_sum":
        def score_f(scores):
            return sum(w * s for w, s in zip(weights, scores))
    elif agg == "sum":
        def score_f(scores):
            return sum(scores)
    elif agg == "min":
        def score_f(scores):
            return min(scores) if scores else 0.0
    elif agg == "max":
        def score_f(scores):
            return max(scores) if scores else 0.0
    else:  # identity
        def score_f(scores):
            return scores[0] if scores else 0.0

    return MultiScoringAggregator(
        conn=conn,
        streams=streams,
        score_f=score_f,
        k=k,
        join_spec=join_spec,
        predicates=predicates,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import json
    import os
    import sys
    import time
    from imdb_data.workload.load import load_workload, get_queries

    path = sys.argv[1] if len(sys.argv) > 1 else os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "imdb_data", "workload", "w6_queries_100.json")
    )

    data = load_workload(path)
    queries = get_queries(data)
    if not queries:
        print("No queries.", file=sys.stderr)
        return

    db_url = "postgresql://localhost/imdb"
    conn = psycopg.connect(db_url, row_factory=dict_row)
    register_vector(conn)

    results_out = []
    t0 = time.perf_counter()

    for qi, q in enumerate(queries):
        qid = q.get("query_id", qi)
        tq = time.perf_counter()

        msa = build_msa_from_query(conn, q)
        answer = msa.run()

        elapsed_q = time.perf_counter() - tq
        print(f"  {qid}: {len(answer)} results in {elapsed_q:.3f}s", file=sys.stderr)

        rows = []
        for t_left, t_right, total_score in answer:
            rows.append([
                t_left.get("id"), t_right.get("id"),
                t_left.get("title", ""), t_right.get("title", ""),
                round(total_score, 6),
            ])
        results_out.append({
            "query_id": qid,
            "answer": rows,
            "elapsed_sec": round(elapsed_q, 6),
            "n_rows": len(rows),
            "K": q.get("K", 20),
        })

    elapsed = time.perf_counter() - t0
    print(f"--- done in {elapsed:.3f}s ({len(queries)} queries) ---", file=sys.stderr)

    workload_stem = os.path.splitext(os.path.basename(path))[0]
    out_path = f"results_msa_{workload_stem}.json"
    out_data = {
        "method": "msa",
        "workload_path": path,
        "total_elapsed_sec": round(elapsed, 6),
        "n_queries": len(results_out),
        "results": results_out,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {out_path}", file=sys.stderr)

    conn.close()


if __name__ == "__main__":
    main()
