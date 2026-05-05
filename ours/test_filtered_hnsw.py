"""
Test filtered_hnsw_search() against ground truth (brute-force SQL).

For each test case, compares:
  1. Ground truth: sequential scan with predicates, ORDER BY distance
  2. filtered_hnsw_search(): bitmap-filtered HNSW via <-># operator

Checks that all returned IDs are valid (pass predicates) and measures
recall against the ground truth.
"""

import os
import sys
import time

import psycopg
from psycopg.rows import tuple_row
from pgvector.psycopg import register_vector

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ours.utils import (
    build_where,
    filtered_hnsw_search,
    get_max_id,
    make_bitmap,
    quote,
    resolve_predicate_ids,
)

DATABASE_URL = "postgresql://localhost/imdb"
K = 10


def ground_truth(conn, table, field, query_vec, predicates, k):
    """Brute-force sequential scan: push predicates into WHERE, ORDER BY distance."""
    where_parts, params = build_where(predicates)
    where_sql = " AND ".join(where_parts) if where_parts else "TRUE"
    params["qv"] = query_vec
    sql = (
        f'SELECT "id", ({quote(field)} <-> %(qv)s) AS dist '
        f"FROM {quote(table)} "
        f"WHERE {where_sql} "
        f"ORDER BY dist "
        f"LIMIT {k}"
    )
    with conn.cursor(row_factory=tuple_row) as cur:
        # Force sequential scan for exact ground truth
        cur.execute("SET LOCAL enable_indexscan = off")
        cur.execute("SET LOCAL enable_indexonlyscan = off")
        cur.execute("SET LOCAL enable_bitmapscan = off")
        cur.execute("SET LOCAL enable_seqscan = on")
        cur.execute(sql, params)
        return [(r[0], float(r[1])) for r in cur.fetchall()]


def run_test(conn, table, field, query_id, predicates, k=K, ef_search=400):
    """Run one test case and print results."""
    # Get query vector
    with conn.cursor(row_factory=tuple_row) as cur:
        cur.execute(
            f'SELECT {quote(field)} FROM {quote(table)} WHERE "id" = %s',
            [query_id],
        )
        query_vec = cur.fetchone()[0]

    # Ground truth (sequential scan)
    t0 = time.time()
    gt = ground_truth(conn, table, field, query_vec, predicates, k)
    gt_time = time.time() - t0
    gt_ids = {r[0] for r in gt}

    # Reset planner settings before filtered HNSW
    with conn.cursor() as cur:
        cur.execute("RESET enable_indexscan")
        cur.execute("RESET enable_indexonlyscan")
        cur.execute("RESET enable_bitmapscan")
        cur.execute("RESET enable_seqscan")

    # Resolve predicates → bitmap → filtered HNSW
    t0 = time.time()
    valid_ids = resolve_predicate_ids(conn, table, predicates)
    max_id = get_max_id(conn, table)
    bitmap = make_bitmap(valid_ids, max_id)
    results = filtered_hnsw_search(
        conn, table, field, query_vec, "l2", bitmap, k, ef_search=ef_search
    )
    hnsw_time = time.time() - t0

    hnsw_ids = {r[0]["id"] for r in results}
    all_valid = all(r[0]["id"] in valid_ids for r in results)
    recall = len(hnsw_ids & gt_ids) / len(gt_ids) if gt_ids else 1.0

    pred_desc = "; ".join(
        f'{p["attribute"]} {p["operator"]} {p["value"]}'
        if p["operator"] != "in"
        else f'{p["attribute"]} in [{len(p["value"])} ids]'
        for p in predicates
    )
    selectivity = len(valid_ids) / max_id if max_id else 0

    print(f"\n--- query_id={query_id}, table={table}, field={field} ---")
    print(f"  Predicates: {pred_desc}")
    print(f"  Selectivity: {selectivity:.4f} ({len(valid_ids)}/{max_id})")
    print(f"  GT: {len(gt)} results in {gt_time:.3f}s")
    print(f"  HNSW: {len(results)} results in {hnsw_time:.3f}s")
    print(f"  All valid: {all_valid}")
    print(f"  Recall@{k}: {recall:.2f}")

    if recall < 1.0:
        missed = gt_ids - hnsw_ids
        print(f"  Missed IDs: {missed}")

    return recall, all_valid


def main():
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)

    # imdb_t1: years 1950-2006, 42378 rows
    test_cases = [
        # High selectivity (~62%)
        (
            "imdb_t1", "plot_emb", 100,
            [{"attribute": "year", "operator": ">=", "value": 1980}],
        ),
        # Mid selectivity (~43%)
        (
            "imdb_t1", "plot_emb", 100,
            [{"attribute": "year", "operator": ">=", "value": 1990}],
        ),
        # ~21% selectivity
        (
            "imdb_t1", "plot_emb", 100,
            [{"attribute": "year", "operator": ">=", "value": 2000}],
        ),
        # ~10% selectivity (range predicate)
        (
            "imdb_t1", "plot_emb", 100,
            [
                {"attribute": "year", "operator": ">=", "value": 1995},
                {"attribute": "year", "operator": "<=", "value": 2000},
            ],
        ),
        # ~5% selectivity ("in" predicate)
        (
            "imdb_t1", "plot_emb", 100,
            [{"attribute": "id", "operator": "in", "value": list(range(1, 2119))}],
        ),
        # imdb_t2 table
        (
            "imdb_t2", "plot_emb", 200,
            [{"attribute": "year", "operator": ">=", "value": 1990}],
        ),
        # Different embedding field (title_emb)
        (
            "imdb_t1", "title_emb", 100,
            [{"attribute": "year", "operator": ">=", "value": 2000}],
        ),
        # Combined clause + in predicates
        (
            "imdb_t1", "plot_emb", 100,
            [
                {"attribute": "year", "operator": ">=", "value": 1990},
                {"attribute": "id", "operator": "in", "value": list(range(1, 21189))},
            ],
        ),
    ]

    for ef in [10, 20, 40, 80]:
        print(f"\n{'='*60}")
        print(f"  ef_search = {ef}")
        print(f"{'='*60}")
        total = 0
        passed = 0
        recalls = []

        for table, field, qid, preds in test_cases:
            recall, valid = run_test(conn, table, field, qid, preds, ef_search=ef)
            total += 1
            if valid:
                passed += 1
            recalls.append(recall)

        print(f"\n  >> ef={ef}: {passed}/{total} all-valid, avg recall@{K}: {sum(recalls)/len(recalls):.2f}")

    conn.close()


if __name__ == "__main__":
    main()
