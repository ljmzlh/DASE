# build ti (threshold index) A x B with batched hnsw range probes

'''
Same output as `ti_build.py --method directhnsw_range` (TSV query_id/neighbor_id/dist,
then materialize_ti), but all A x B probes run inside one C function
(batch_hnsw/dase_batch_hnsw.c) over left rows in locality order, so nearby
left rows can share their way into B's graph:

  --mode probe    every row descends from the top (the existing per-row algorithm)
  --mode share    one descent per group; members start layer 0 at the leader's entry
  --mode warm     no descent inside a group; each row starts from the previous row's
                  nearest nodes (--seeds of them)
  --order kmeans  k-means groups of ~--group_size rows, chained nearest-neighbour (default)
  --order id      rows by id, no groups (probe only): the existing row order

l2 only. Needs the dase_batch_hnsw extension installed on the server (README §1.2);
each run does CREATE EXTENSION IF NOT EXISTS dase_batch_hnsw, which needs a superuser.

The TI table defaults to ti_{left}_{right}_{threshold}_bh_{mode}, so the table of the
existing build (ti_{left}_{right}_{threshold}) is left alone; --output_table overrides.

usage (from the repository root; connection from PGHOST/PGUSER/PGPASSWORD like ti_build.py):
python -m ours.ti.build_join_index_A_B --config imdb --threshold 0.5
python -m ours.ti.build_join_index_A_B --config imdb --threshold 0.5 --mode probe --order id
python -m ours.ti.build_join_index_A_B --config w5_molecule_0.6 --threshold 0.6 --mode share --group_size 64
'''

import argparse
import io
import json
import os
import struct
import time

from batch_hnsw.order import locality_order  # first: caps OpenBLAS threads before numpy loads

import numpy as np
import psycopg2
from psycopg2 import sql

from .ti_materialize import load_configs, materialize_ti


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _clean_ident(name: str) -> str:
    """Strip surrounding double quotes for use as an Identifier."""
    return name.strip('"') if (name.startswith('"') and name.endswith('"')) else name


def _hnsw_table(t: str) -> str:
    return t if t.endswith("_hnsw") else f"{t}_hnsw"


def install_functions(conn) -> None:
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS dase_batch_hnsw")
    except psycopg2.Error as e:
        raise SystemExit(f"CREATE EXTENSION dase_batch_hnsw failed: {str(e).strip()}\n"
                         "Install it from batch_hnsw/ and connect as a superuser (README §1.2).")


def find_l2_hnsw_index(conn, table: str, vec_col: str) -> str:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT i.indexrelid::regclass::text
            FROM pg_index i
            JOIN pg_class c ON c.oid = i.indexrelid
            JOIN pg_am a ON a.oid = c.relam
            JOIN pg_opclass oc ON oc.oid = i.indclass[0]
            JOIN pg_attribute att ON att.attrelid = i.indrelid AND att.attnum = i.indkey[0]
            WHERE i.indrelid = %s::regclass AND a.amname = 'hnsw'
              AND oc.opcname = 'vector_l2_ops' AND att.attname = %s
        """, (sql.Identifier(table).as_string(conn), vec_col))
        row = cur.fetchone()
    if row is None:
        raise SystemExit(f"no hnsw vector_l2_ops index on {table}({vec_col})")
    return row[0]


def load_vectors(conn, table: str, id_col: str, vec_col: str):
    """(ids, vecs) of the non-null rows, through binary COPY (~20x faster than text rows)."""
    ident = dict(id=sql.Identifier(id_col), vec=sql.Identifier(vec_col), t=sql.Identifier(table))
    buf = io.BytesIO()
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT vector_dims({vec}) FROM {t} WHERE {vec} IS NOT NULL LIMIT 1").format(**ident))
        dim = cur.fetchone()[0]
        cur.copy_expert(sql.SQL("COPY (SELECT {id}::int8, {vec} FROM {t} WHERE {vec} IS NOT NULL ORDER BY {id}) "
                                "TO STDOUT WITH (FORMAT binary)").format(**ident).as_string(conn), buf)
    data = buf.getbuffer()
    start = 19 + struct.unpack(">I", data[15:19])[0]  # signature, flags, header extension
    # per row: field count, (len, int8 id), (len, vector: int16 dim, int16 unused, float4[dim])
    dt = np.dtype([("nf", ">i2"), ("l1", ">i4"), ("id", ">i8"), ("l2", ">i4"),
                   ("dim", ">i2"), ("unused", ">i2"), ("v", ">f4", (dim,))])
    rec = np.frombuffer(data, dtype=dt, count=(len(data) - start - 2) // dt.itemsize, offset=start)
    if not (rec["dim"] == dim).all():
        raise SystemExit(f"{table}.{vec_col}: vectors of different dimensions")
    return rec["id"].astype(np.int64), rec["v"].astype(np.float32)


def write_locality_order(conn, table: str, id_col: str, vec_col: str, group_size: int) -> int:
    """k-means order of the left table into temp table bh_order(id, ord, grp); returns #groups."""
    ids, vecs = load_vectors(conn, table, id_col, vec_col)
    order, groups = locality_order(vecs, group_size)

    buf = io.StringIO()
    for pos, (i, g) in enumerate(zip(order, groups)):
        buf.write(f"{ids[i]}\t{pos}\t{g}\n")
    buf.seek(0)
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS bh_order")
        cur.execute("CREATE TEMP TABLE bh_order (id int8 PRIMARY KEY, ord int4, grp int4)")
        cur.copy_expert("COPY bh_order (id, ord, grp) FROM STDIN", buf)
        cur.execute("ANALYZE bh_order")
    return len(set(groups.tolist()))


def batch_HNSWRange(conn, table_left, table_right, vec_col_left, vec_col_right, id_col_left, id_col_right,
                    ef_search, ti_threshold, output_file, mode, order, group_size, n_seeds):
    """
    左表 × 右表 HNSW range（dist < tau），A x B 的探测全部在 bh.range_join_all 里跑；
    输出格式同 direct_HNSWRange（self-join 时只保留 id1 < id2）。返回计时与计数。
    """
    is_self_join = (table_left == table_right)
    scan_left, scan_right = _hnsw_table(table_left), _hnsw_table(table_right)
    vl, vr = _clean_ident(vec_col_left), _clean_ident(vec_col_right)
    il, ir = _clean_ident(id_col_left), _clean_ident(id_col_right)
    right_index = find_l2_hnsw_index(conn, scan_right, vr)

    t0 = time.perf_counter()
    if order == "kmeans":
        n_groups = write_locality_order(conn, scan_left, il, vl, group_size)
        left_sql = sql.SQL(
            "SELECT l.{id}::int8, l.{vec}, o.grp FROM {t} l JOIN bh_order o ON o.id = l.{id} ORDER BY o.ord")
    else:
        n_groups = 1
        left_sql = sql.SQL("SELECT l.{id}::int8, l.{vec}, 0::int4 FROM {t} l WHERE l.{vec} IS NOT NULL ORDER BY l.{id}")
    left_sql = left_sql.format(id=sql.Identifier(il), vec=sql.Identifier(vl), t=sql.Identifier(scan_left))
    t_order = time.perf_counter() - t0
    print(f"order={order}: {n_groups} groups in {t_order:.1f}s; probing {right_index} (mode={mode})")

    q_sql = sql.SQL("""
        SELECT r.left_id, b.{id_r}, r.dist
        FROM bh.range_join_all(%s, %s::regclass, %s, %s, %s, %s) AS r
        JOIN {tbl_r} AS b ON b.ctid = r.right_tid
    """).format(id_r=sql.Identifier(ir), tbl_r=sql.Identifier(scan_right))

    t0 = time.perf_counter()
    written = 0
    with conn.cursor(name="bh_scan") as scan, open(output_file, "w", encoding="utf-8") as outf:
        outf.write("query_id\tneighbor_id\tdist\n")
        scan.itersize = 100000
        scan.execute(q_sql, (left_sql.as_string(conn), right_index, ti_threshold, ef_search, mode, n_seeds))
        for id1, id2, d in scan:
            if is_self_join and id2 <= id1:
                continue
            outf.write(f"{id1}\t{id2}\t{float(d)}\n")
            written += 1
    t_search = time.perf_counter() - t0
    with conn.cursor() as cur:
        cur.execute("SELECT bh.last_stats()")
        stats = json.loads(cur.fetchone()[0])
    conn.commit()

    print(f"written={written} | search {t_search:.1f}s (inside module {stats['t_total']:.1f}s, "
          f"{stats['n_descents']} descents for {stats['n_queries']} rows)")
    return dict(mode=mode, order=order, group_size=group_size if order == "kmeans" else None,
                n_seeds=n_seeds if mode == "warm" else None, n_groups=n_groups, ef_search=ef_search,
                threshold=ti_threshold, written=written, t_order=t_order, t_search=t_search, module=stats)


def main():
    CONFIGS = load_configs()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, choices=list(CONFIGS.keys()))
    parser.add_argument("--threshold", type=float, required=True, help="l2 radius tau")
    parser.add_argument("--mode", type=str, default="warm", choices=["probe", "share", "warm"])
    parser.add_argument("--order", type=str, default="kmeans", choices=["kmeans", "id"])
    parser.add_argument("--group_size", type=int, default=256)
    parser.add_argument("--seeds", type=int, default=4, help="warm: nodes carried to the next row")
    parser.add_argument("--ef_search", type=int, default=None,
                        help="HNSW ef_search; overrides config's ef_search if given")
    parser.add_argument("--output_table", type=str, default=None)
    parser.add_argument("--skip_materialize", action="store_true", help="stop after the TSV")
    args = parser.parse_args()
    if args.order == "id" and args.mode != "probe":
        parser.error("--mode share/warm needs groups: use --order kmeans")

    wl = CONFIGS[args.config]
    DSN = dict(host=os.environ.get("PGHOST", "127.0.0.1"), database=wl["db"],
               user=os.environ.get("PGUSER", "postgres"),
               password=os.environ.get("PGPASSWORD", ""))
    TABLE_LEFT = wl["table_left"]
    TABLE_RIGHT = wl["table_right"]
    VEC_COL_LEFT = wl.get("vec_col_left") or wl.get("vec_col")
    VEC_COL_RIGHT = wl.get("vec_col_right") or wl.get("vec_col")
    ID_COL_LEFT = wl.get("id_col_left", wl.get("id_col"))
    ID_COL_RIGHT = wl.get("id_col_right", wl.get("id_col"))
    EF_SEARCH = args.ef_search if args.ef_search is not None else wl.get("ef_search")
    assert EF_SEARCH is not None, "ef_search must be set via --ef_search or config's ef_search"
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "ti_cache", args.config)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    threshold = args.threshold
    tag = f"batchhnsw_{args.mode}_{args.order}" + (f"{args.group_size}" if args.order == "kmeans" else "")
    output_file = f"{OUTPUT_DIR}/ti_{TABLE_LEFT}_{TABLE_RIGHT}_{tag}_l2_{threshold}.tsv"

    conn = psycopg2.connect(**DSN)
    conn.autocommit = True
    install_functions(conn)
    conn.autocommit = False

    timing = batch_HNSWRange(
        conn=conn,
        table_left=TABLE_LEFT,
        table_right=TABLE_RIGHT,
        vec_col_left=VEC_COL_LEFT,
        vec_col_right=VEC_COL_RIGHT,
        id_col_left=ID_COL_LEFT,
        id_col_right=ID_COL_RIGHT,
        ef_search=EF_SEARCH,
        ti_threshold=threshold,
        output_file=output_file,
        mode=args.mode,
        order=args.order,
        group_size=args.group_size,
        n_seeds=args.seeds,
    )

    if not args.skip_materialize:
        conn.autocommit = True
        t0 = time.perf_counter()
        materialize_ti(
            conn=conn,
            input_file=output_file,
            output_table=args.output_table or f"ti_{TABLE_LEFT}_{TABLE_RIGHT}_{threshold}_bh_{args.mode}",
            table_left=TABLE_LEFT,
            table_right=TABLE_RIGHT,
            id_col_left=ID_COL_LEFT,
            id_col_right=ID_COL_RIGHT,
            cols_left=wl["cols_left"],
            cols_right=wl["cols_right"],
        )
        timing["t_materialize"] = time.perf_counter() - t0

    with open(output_file[:-len(".tsv")] + ".timing.json", "w") as f:
        json.dump(timing, f, indent=2)
    conn.close()


if __name__ == "__main__":
    main()
