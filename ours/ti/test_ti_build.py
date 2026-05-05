"""
Verify directhnsw_range by comparing ANN range-search results against
brute-force ground truth on small sampled tables (1000 rows each).

Usage:
    conda run -n dase python test_ti_build.py --threshold 0.5 --vec_col actor_director_emb
"""

import argparse
import sys
import time

import psycopg
from pgvector.psycopg import register_vector

# ── config ──────────────────────────────────────────────────────
DSN = "postgresql://localhost/imdb"
SAMPLE_SIZE = 1000
SOURCE_LEFT = "imdb_t1"
SOURCE_RIGHT = "imdb_t2"
TEST_LEFT = "imdb_t1_1000"
TEST_RIGHT = "imdb_t2_1000"
ID_COL = "id"


def create_sample_tables(conn, vec_col):
    """Create small sampled tables with HNSW index on vec_col."""
    with conn.cursor() as cur:
        for src, dst in [(SOURCE_LEFT, TEST_LEFT), (SOURCE_RIGHT, TEST_RIGHT)]:
            cur.execute(f"DROP TABLE IF EXISTS {dst}")
            cur.execute(f"""
                CREATE TABLE {dst} AS
                SELECT * FROM {src}
                ORDER BY random()
                LIMIT {SAMPLE_SIZE}
            """)
            # reset id to 1..N for cleaner output
            cur.execute(f"ALTER TABLE {dst} DROP COLUMN {ID_COL}")
            cur.execute(f"ALTER TABLE {dst} ADD COLUMN {ID_COL} serial PRIMARY KEY")
            # HNSW index on the vector column
            cur.execute(f"""
                CREATE INDEX ON {dst}
                USING hnsw ({vec_col} vector_l2_ops)
                WITH (m = 16, ef_construction = 64)
            """)
            cur.execute(f"SELECT count(*) FROM {dst}")
            cnt = cur.fetchone()[0]
            print(f"  {dst}: {cnt} rows, HNSW index on {vec_col} built")
        conn.execute("COMMIT")


def run_ann_range(conn, vec_col, threshold, ef_search):
    """Run direct_HNSWRange style query: JOIN LATERAL with <->@ operator."""
    print(f"Running ANN range search (threshold={threshold}, ef_search={ef_search}) ...")
    t0 = time.time()
    with conn.cursor() as cur:
        cur.execute(f"SET hnsw.ef_search = {ef_search}")
        cur.execute(f"SET hnsw.radius = {threshold}")
        cur.execute("SET enable_seqscan = off")

        q_sql = f"""
            SELECT
              q.{ID_COL}  AS id1,
              nn.{ID_COL} AS id2,
              (nn.{vec_col} <-> q.{vec_col}) AS dist
            FROM {TEST_LEFT} AS q
            JOIN LATERAL (
              SELECT nn.{ID_COL}, nn.{vec_col}
              FROM {TEST_RIGHT} AS nn
              WHERE nn.{vec_col} <->@ q.{vec_col}
            ) AS nn ON TRUE
            WHERE q.{vec_col} IS NOT NULL
        """
        cur.execute(q_sql)
        ann = set()
        for row in cur:
            ann.add((int(row[0]), int(row[1])))

        cur.execute("RESET enable_seqscan")

    elapsed = time.time() - t0
    print(f"  ANN: {len(ann)} pairs  ({elapsed:.1f}s)")
    return ann, elapsed


def compute_ground_truth(conn, vec_col, threshold):
    """Brute-force: all (id1, id2) pairs with L2 distance < threshold."""
    print(f"Computing ground truth (brute-force, threshold={threshold}) ...")
    t0 = time.time()
    with conn.cursor() as cur:
        cur.execute("SET enable_indexscan = off")
        cur.execute("SET enable_bitmapscan = off")
        cur.execute(f"""
            SELECT a.{ID_COL} AS id1, b.{ID_COL} AS id2
            FROM {TEST_LEFT} a, {TEST_RIGHT} b
            WHERE a.{vec_col} IS NOT NULL
              AND b.{vec_col} IS NOT NULL
              AND (a.{vec_col} <-> b.{vec_col}) < %s
        """, (threshold,))
        gt = set()
        for row in cur:
            gt.add((int(row[0]), int(row[1])))
        cur.execute("RESET enable_indexscan")
        cur.execute("RESET enable_bitmapscan")

    elapsed = time.time() - t0
    print(f"  Ground truth: {len(gt)} pairs  ({elapsed:.1f}s)")
    return gt, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--vec_col", type=str, default="actor_director_emb")
    parser.add_argument("--ef_search", type=int, default=40)
    parser.add_argument("--skip_create", action="store_true",
                        help="Skip table creation if already done")
    args = parser.parse_args()

    conn = psycopg.connect(DSN)
    conn.autocommit = True
    register_vector(conn)

    # Step 1: create sample tables
    if not args.skip_create:
        print("Step 1: Creating sample tables ...")
        create_sample_tables(conn, args.vec_col)
    else:
        print("Step 1: Skipped (--skip_create)")

    # Step 2: ANN range search
    print(f"\nStep 2: ANN range search ...")
    ann, ann_time = run_ann_range(conn, args.vec_col, args.threshold, args.ef_search)

    # Step 3: brute-force ground truth
    print(f"\nStep 3: Brute-force ground truth ...")
    gt, gt_time = compute_ground_truth(conn, args.vec_col, args.threshold)

    # Step 4: compare
    print(f"\n{'='*60}")
    print(f"Results  (vec_col={args.vec_col}, threshold={args.threshold})")
    print(f"{'='*60}")
    print(f"  ANN pairs:          {len(ann)}")
    print(f"  Ground truth pairs: {len(gt)}")

    if len(gt) > 0:
        tp = len(ann & gt)
        fp = len(ann - gt)
        fn = len(gt - ann)
        recall = tp / len(gt) * 100
        precision = tp / len(ann) * 100 if len(ann) > 0 else 0
        print(f"  True positives:     {tp}")
        print(f"  False positives:    {fp}  (ANN returned but not in GT)")
        print(f"  False negatives:    {fn}  (in GT but ANN missed)")
        print(f"  Recall:             {recall:.2f}%")
        print(f"  Precision:          {precision:.2f}%")
    else:
        print("  No ground truth pairs found — try a larger threshold.")

    print(f"\n  ANN time:           {ann_time:.1f}s")
    print(f"  GT time:            {gt_time:.1f}s")
    print(f"{'='*60}")

    conn.close()


if __name__ == "__main__":
    main()
