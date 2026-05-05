'''
Materialize TI from TSV (id1, id2, dist) into a PostgreSQL table,
joining back to source tables to include structured columns.

Output columns are named "tablename.colname", e.g. "imdb_t1.id", "imdb_t2.rating".
The distance column is named "dis".

usage (from /dase/):
python -m ours.ti.ti_materialize --config imdb \
    --input ours/ti/ti_cache/imdb/ti_imdb_t1_imdb_t2_directhnsw_range_l2_0.5.tsv \
    --output_table ti_imdb_t1_imdb_t2_0.5
'''

import json
import os
import argparse
import psycopg2
import psycopg2.extras as extras
from tqdm import tqdm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_configs():
    with open(os.path.join(SCRIPT_DIR, "ti_config.json")) as f:
        return json.load(f)


def materialize_ti(conn, input_file, output_table, table_left, table_right,
                   id_col_left, id_col_right, cols_left, cols_right, batch_size=5000):
    """
    Materialize TI from TSV into a PostgreSQL table.

    Args:
        conn: psycopg2 connection (autocommit=True)
        input_file: path to TSV (header: query_id, neighbor_id, dist)
        output_table: name of the output table
        table_left/table_right: source table names
        id_col_left/id_col_right: id column name on each side (may differ for heterogeneous joins)
        cols_left/cols_right: list of column names to include from each side
        batch_size: insert batch size
    """
    # read TSV
    print(f"Reading {input_file} ...")
    rows = []
    left_ids = set()
    right_ids = set()
    with open(input_file) as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            id1, id2, dis = int(parts[0]), int(parts[1]), float(parts[2])
            rows.append((id1, id2, dis))
            left_ids.add(id1)
            right_ids.add(id2)
    print(f"  {len(rows)} pairs, {len(left_ids)} left IDs, {len(right_ids)} right IDs")

    if not rows:
        print("No data to materialize.")
        return

    # fetch columns from source tables
    def fetch_fields(table, id_col, ids, cols):
        field_map = {}
        id_list = list(ids)
        with conn.cursor() as cur:
            for i in tqdm(range(0, len(id_list), 10000),
                          desc=f"Fetching {table}"):
                batch_ids = id_list[i:i + 10000]
                placeholders = ",".join(["%s"] * len(batch_ids))
                col_sql = ", ".join(f'"{c}"' for c in cols)
                cur.execute(
                    f'SELECT "{id_col}", {col_sql} FROM "{table}" '
                    f'WHERE "{id_col}" IN ({placeholders})',
                    batch_ids,
                )
                for row in cur.fetchall():
                    field_map[row[0]] = row[1:]
        return field_map

    def _hnsw_table(t):
        return t if t.endswith("_hnsw") else f"{t}_hnsw"
    left_map = fetch_fields(_hnsw_table(table_left), id_col_left, left_ids, cols_left)
    right_map = fetch_fields(_hnsw_table(table_right), id_col_right, right_ids, cols_right)

    # build output column definitions (format: "tablename.colname")
    out_cols = []
    for c in cols_left:
        out_cols.append(f"{table_left}.{c}")
    out_cols.append("dis")
    for c in cols_right:
        out_cols.append(f"{table_right}.{c}")

    # detect column types from source tables. For ARRAY, also fetch udt_name
    # (element type with leading underscore, e.g. "_text") to preserve element type.
    def get_col_types(table, cols):
        types = {}
        with conn.cursor() as cur:
            for c in cols:
                cur.execute("""
                    SELECT data_type, udt_name FROM information_schema.columns
                    WHERE table_name = %s AND column_name = %s
                """, (table, c))
                row = cur.fetchone()
                types[c] = (row[0], row[1]) if row else ("text", "text")
        return types

    left_types = get_col_types(_hnsw_table(table_left), cols_left)
    right_types = get_col_types(_hnsw_table(table_right), cols_right)

    def pg_type(info):
        data_type, udt = info
        if data_type == "ARRAY":
            # udt_name looks like "_text" / "_int4" / "_int8" / ... strip leading _ and map.
            elem_udt = udt.lstrip("_")
            elem_map = {
                "text": "text", "varchar": "text", "int2": "smallint",
                "int4": "integer", "int8": "bigint",
                "float4": "real", "float8": "double precision",
                "numeric": "numeric", "bool": "boolean",
            }
            return f"{elem_map.get(elem_udt, 'text')}[]"
        mapping = {
            "integer": "integer", "bigint": "bigint", "smallint": "smallint",
            "double precision": "double precision", "real": "real",
            "numeric": "numeric", "text": "text", "character varying": "text",
            "boolean": "boolean",
        }
        return mapping.get(data_type, "text")

    # create table
    col_defs = []
    for c in cols_left:
        col_defs.append(f'"{table_left}.{c}" {pg_type(left_types[c])}')
    col_defs.append('"dis" double precision')
    for c in cols_right:
        col_defs.append(f'"{table_right}.{c}" {pg_type(right_types[c])}')

    with conn.cursor() as cur:
        cur.execute(f'DROP TABLE IF EXISTS "{output_table}"')
        cur.execute(f'CREATE TABLE "{output_table}" ({", ".join(col_defs)})')
    print(f"Created table {output_table} with columns: {out_cols}")

    # insert data
    n_cols = len(cols_left) + 1 + len(cols_right)
    placeholders = ", ".join(["%s"] * n_cols)
    col_names = ", ".join(f'"{c}"' for c in out_cols)
    insert_sql = f'INSERT INTO "{output_table}" ({col_names}) VALUES ({placeholders})'

    batch = []
    skipped = 0
    with conn.cursor() as cur:
        for id1, id2, dis in tqdm(rows, desc="Inserting"):
            if id1 not in left_map or id2 not in right_map:
                skipped += 1
                continue
            row_data = list(left_map[id1]) + [dis] + list(right_map[id2])
            batch.append(row_data)
            if len(batch) >= batch_size:
                extras.execute_batch(cur, insert_sql, batch, page_size=batch_size)
                batch = []
        if batch:
            extras.execute_batch(cur, insert_sql, batch, page_size=batch_size)

    if skipped:
        print(f"  Skipped {skipped} pairs (missing in source tables)")
    print(f"  Inserted {len(rows) - skipped} rows into {output_table}")

    # add index on every column (B-tree for scalars, GIN for arrays)
    array_cols = set()
    for c in cols_left:
        if left_types[c][0] == "ARRAY":
            array_cols.add(f"{table_left}.{c}")
    for c in cols_right:
        if right_types[c][0] == "ARRAY":
            array_cols.add(f"{table_right}.{c}")
    with conn.cursor() as cur:
        for col in out_cols:
            if col in array_cols:
                cur.execute(f'CREATE INDEX ON "{output_table}" USING gin ("{col}")')
            else:
                cur.execute(f'CREATE INDEX ON "{output_table}" ("{col}")')
    print(f"  Added indexes on all {len(out_cols)} columns (gin for arrays)")

    # Covering index for W7/W8 threshold stream keyset pagination
    # (ORDER BY dis, pk_l, pk_r LIMIT K — see ours/ti_stream.py:_fetch_more).
    # Enables index-only scan; on wide rows this avoids ~all heap I/O.
    pk_l_col = f"{table_left}.{id_col_left}"
    pk_r_col = f"{table_right}.{id_col_right}"
    cov_idx = f"{output_table}_cov_dis_pk_idx"
    with conn.cursor() as cur:
        cur.execute(
            f'CREATE INDEX "{cov_idx}" ON "{output_table}" '
            f'("dis", "{pk_l_col}", "{pk_r_col}")'
        )
    print(f"  Added covering index {cov_idx}")


def main():
    CONFIGS = load_configs()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        choices=list(CONFIGS.keys()))
    parser.add_argument("--input", type=str, required=True,
                        help="Path to TI TSV file (id1, id2, dist)")
    parser.add_argument("--cols_left", type=str, default=None,
                        help="Override comma-separated columns from left table")
    parser.add_argument("--cols_right", type=str, default=None,
                        help="Override comma-separated columns from right table")
    parser.add_argument("--output_table", type=str, required=True,
                        help="Name of the output PostgreSQL table")
    parser.add_argument("--batch_size", type=int, default=5000)
    args = parser.parse_args()

    wl = CONFIGS[args.config]
    DSN = dict(host=os.environ.get("PGHOST", "127.0.0.1"), database=wl["db"],
               user=os.environ.get("PGUSER", "postgres"),
               password=os.environ.get("PGPASSWORD", ""))

    cols_left = [c.strip() for c in args.cols_left.split(",")] if args.cols_left else wl["cols_left"]
    cols_right = [c.strip() for c in args.cols_right.split(",")] if args.cols_right else wl["cols_right"]

    conn = psycopg2.connect(**DSN)
    conn.autocommit = True

    id_col_left = wl.get("id_col_left", wl.get("id_col"))
    id_col_right = wl.get("id_col_right", wl.get("id_col"))
    assert id_col_left and id_col_right, "workload must define id_col or id_col_left+id_col_right"

    materialize_ti(
        conn=conn,
        input_file=args.input,
        output_table=args.output_table,
        table_left=wl["table_left"],
        table_right=wl["table_right"],
        id_col_left=id_col_left,
        id_col_right=id_col_right,
        cols_left=cols_left,
        cols_right=cols_right,
        batch_size=args.batch_size,
    )

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
