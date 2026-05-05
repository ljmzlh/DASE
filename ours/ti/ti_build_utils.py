# build w4_corpus self-join ti (tau=0.1) with batch hnsw query
# implement with "lateral cross join"

from multiprocessing import Value
import os
import sys
import time
from typing import Optional

from pandas.core.generic import T
import psycopg2
import psycopg2.extras as extras
from psycopg2 import sql
from pgvector.psycopg2 import register_vector
from tqdm import tqdm


metric_op={'l2': '<->', 'ip': '<#>', 'cos': '<=>'}


def _clean_ident(name: str) -> str:
    """Strip surrounding double quotes for use as dict key / Identifier."""
    return name.strip('"') if (name.startswith('"') and name.endswith('"')) else name


def output_plan(line):
    # 如果行中包含向量（如'['和']'），则只保留前10个元素
                        if '[' in line and ']' in line:
                            import re
                            def truncate_vector(match):
                                vec_str = match.group(0)
                                nums = vec_str.strip('[]').split(',')
                                if len(nums) > 10:
                                    nums = nums[:10]
                                    return '[' + ','.join(nums) + ', ...]'
                                else:
                                    return vec_str
                            # 用正则查找所有向量并替换
                            line = re.sub(r'\[[^\]]+\]', truncate_vector, line)
                        print(line)


def laterCrossJoin_HNSWTopK(
    conn: psycopg2.extensions.connection,
    table_left: str,
    table_right: str,
    vec_col: str,
    id_col_left: str,
    id_col_right: str,
    source_batch: int,
    topk: int,
    ef_search: int,
    ti_ip_threshold: float,
    output_file: str = "w4_ti_lateral_ann.tsv"
):
    """
    构建 TI：左表 × 右表（或 self-join 当 table_left == table_right）。
    按 inner product distance < tau 过滤；使用 server-side cursor 扫左表，批量 VALUES + CROSS JOIN LATERAL 查右表。
    """
    register_vector(conn)
    if os.path.exists(output_file):
        os.remove(output_file)

    lwb = -1 + 1e-6
    is_self_join = (table_left == table_right)
    vc, idl, idr = _clean_ident(vec_col), _clean_ident(id_col_left), _clean_ident(id_col_right)
    tbl_left = sql.Identifier(table_left)
    tbl_right = sql.Identifier(table_right)
    idf_vec = sql.Identifier(vc)
    idf_id_left = sql.Identifier(idl)
    idf_id_right = sql.Identifier(idr)

    try:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL hnsw.ef_search = %s;", (ef_search,))

        with conn.cursor(name="w4_scan", cursor_factory=extras.DictCursor) as scan, \
             conn.cursor(cursor_factory=extras.DictCursor) as cur, \
             open(output_file, "w", encoding="utf-8") as outf:

            outf.write("query_id\tneighbor_id\tip_dist\n")
            scan.itersize = source_batch
            scan.execute(
                sql.SQL("SELECT {id_col}, {vec_col} FROM {table} ORDER BY {id_col};")
                .format(id_col=idf_id_left, vec_col=idf_vec, table=tbl_left)
            )
            start_time = time.time()
            processed = 0

            while True:
                batch = scan.fetchmany(source_batch)
                if not batch:
                    break
                values_rows = []
                params = []
                for r in batch:
                    v = r[vc]
                    if v is None:
                        continue
                    values_rows.append("(%s, %s::vector)")
                    params.extend([r[idl], v])
                if not values_rows:
                    continue
                values_sql = ", ".join(values_rows)

                # Self-join: exclude same row (nn.id <> q.id); two-table: no such condition
                if is_self_join:
                    lateral_where = sql.SQL("WHERE {id_r} <> q.{id_l} AND {vec_col} IS NOT NULL").format(
                        id_r=idf_id_right, id_l=idf_id_left, vec_col=idf_vec
                    )
                else:
                    lateral_where = sql.SQL("WHERE {vec_col} IS NOT NULL").format(vec_col=idf_vec)

                query_sql = sql.SQL("""
                    WITH q({id_l}, v) AS (
                        VALUES {values_clause}
                    )
                    SELECT
                        q.{id_l}  AS query_id,
                        nn.{id_r} AS neighbor_id,
                        (nn.{vec_col} <#> (q.v::vector)) AS ip_dist
                    FROM q
                    CROSS JOIN LATERAL (
                        SELECT {id_r}, {vec_col}
                        FROM {tbl_r}
                        {lateral_where}
                        ORDER BY {vec_col} <#> (q.v::vector)
                        LIMIT %s
                    ) nn;
                """).format(
                    id_l=idf_id_left,
                    id_r=idf_id_right,
                    vec_col=idf_vec,
                    tbl_r=tbl_right,
                    lateral_where=lateral_where,
                    values_clause=sql.SQL(values_sql),
                )
                cur.execute(query_sql, (*params, topk))

                for row in cur:
                    d = row["ip_dist"]
                    if d is None:
                        continue
                    qid = int(row["query_id"])
                    nid = int(row["neighbor_id"])
                    if lwb < float(d) < ti_ip_threshold:
                        if is_self_join and nid <= qid:
                            continue
                        outf.write(f"{qid}\t{nid}\t{float(d)}\n")

                processed += len(values_rows)
                elapsed = time.time() - start_time
                sys.stdout.write(f"\rprocessed: {processed} | elapsed: {elapsed:.1f}s")
                sys.stdout.flush()

        conn.commit()
    finally:
        sys.stdout.write("\n")
        sys.stdout.flush()


def loop_HNSWTopk(
    conn: psycopg2.extensions.connection,
    table_left: str,
    table_right: str,
    vec_col: str,
    id_col_left: str,
    id_col_right: str,
    source_batch: int,
    topk: int,
    ef_search: int,
    ti_threshold: float,
    output_file: str,
    metric: str
):
    """
    逐条查询：扫左表，对每行在右表做 ANN。两表可相同（self-join）。
    """
    register_vector(conn)
    if os.path.exists(output_file):
        os.remove(output_file)

    lwb = -0.1 if metric == "l2" else (-1.0 + 1e-6)
    is_self_join = (table_left == table_right)
    vc, idl, idr = _clean_ident(vec_col), _clean_ident(id_col_left), _clean_ident(id_col_right)
    tbl_left = sql.Identifier(table_left)
    tbl_right = sql.Identifier(table_right)
    idf_vec = sql.Identifier(vc)
    idf_id_left = sql.Identifier(idl)
    idf_id_right = sql.Identifier(idr)
    op = metric_op[metric]

    try:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL hnsw.ef_search = %s;", (ef_search,))

        with conn.cursor(name="w4_scan", cursor_factory=extras.DictCursor) as scan, \
             conn.cursor(cursor_factory=extras.DictCursor) as cur, \
             open(output_file, "w", encoding="utf-8") as outf:

            outf.write("query_id\tneighbor_id\tdist\n")
            scan.itersize = source_batch
            scan.execute(
                sql.SQL("SELECT {id_col}, {vec_col} FROM {table} ORDER BY {id_col};")
                .format(id_col=idf_id_left, vec_col=idf_vec, table=tbl_left)
            )
            start_time = time.time()
            processed = 0

            while True:
                batch = scan.fetchmany(source_batch)
                if not batch:
                    break
                for r in batch:
                    qid = int(r[idl])
                    qv = r[vc]
                    if qv is None:
                        continue

                    if is_self_join:
                        lateral_where = sql.SQL("WHERE {id_r} <> %s AND {vec_col} IS NOT NULL").format(
                            id_r=idf_id_right, vec_col=idf_vec
                        )
                        params = (qv, qid, qv, topk)
                    else:
                        lateral_where = sql.SQL("WHERE {vec_col} IS NOT NULL").format(vec_col=idf_vec)
                        params = (qv, qv, topk)

                    q_sql = sql.SQL("""
                        SELECT {id_r} AS neighbor_id,
                               ({vec_col} """ + op + """ %s::vector) AS dist
                        FROM {tbl_r}
                        {lateral_where}
                        ORDER BY {vec_col} """ + op + """ %s::vector
                        LIMIT %s;
                    """).format(id_r=idf_id_right, vec_col=idf_vec, tbl_r=tbl_right, lateral_where=lateral_where)
                    cur.execute(q_sql, params)

                    for row in cur:
                        d = row["dist"]
                        if d is None:
                            continue
                        nid = int(row["neighbor_id"])
                        if not (lwb < float(d) < ti_threshold):
                            continue
                        if is_self_join and nid <= qid:
                            continue
                        outf.write(f"{qid}\t{nid}\t{float(d)}\n")

                    processed += 1
                    if processed % 100 == 0:
                        sys.stdout.write(f"\rprocessed: {processed} | elapsed: {time.time() - start_time:.1f}s")
                        sys.stdout.flush()

            sys.stdout.write(f"\rprocessed: {processed} | elapsed: {time.time() - start_time:.1f}s\n")
            sys.stdout.flush()

        conn.commit()
    finally:
        sys.stdout.flush()



def direct_HNSWTopk(
    conn: psycopg2.extensions.connection,
    table_left: str,
    table_right: str,
    vec_col: str,
    id_col_left: str,
    id_col_right: str,
    source_batch: int,
    topk: int,
    ef_search: int,
    ti_threshold: float,
    output_file: str,
    metric: str
):
    """
    左表 × 右表 top-k ANN（dist < tau）；self-join 时只保留 id1 < id2。
    """
    print("begin AnnTopk")
    register_vector(conn)
    if os.path.exists(output_file):
        os.remove(output_file)

    op = metric_op[metric]
    is_self_join = (table_left == table_right)

    try:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL hnsw.ef_search = %s;", (ef_search,))

        with conn.cursor(name="direct_scan", cursor_factory=extras.DictCursor) as scan, \
             open(output_file, "w", encoding="utf-8") as outf:

            outf.write("query_id\tneighbor_id\tdist\n")
            if is_self_join:
                lateral_sql = f"""
                  SELECT {id_col_right}, {vec_col}
                  FROM {table_right}
                  WHERE {id_col_right} <> q.{id_col_left}
                  ORDER BY {vec_col} {op} q.{vec_col}
                  LIMIT %s
                """
            else:
                lateral_sql = f"""
                  SELECT {id_col_right}, {vec_col}
                  FROM {table_right}
                  ORDER BY {vec_col} {op} q.{vec_col}
                  LIMIT %s
                """
            q_sql = f"""
                SELECT
                  q.{id_col_left}  AS id1,
                  nn.{id_col_right} AS id2,
                  (nn.{vec_col} {op} q.{vec_col}) AS dist
                FROM {table_left} AS q
                JOIN LATERAL (
                  {lateral_sql}
                ) AS nn ON TRUE
                WHERE q.{vec_col} IS NOT NULL;
            """
            scan.itersize = 10000
            print("begin execute")
            scan.execute(q_sql, (topk,))
            print("done execute")

            written = 0
            start = time.time()
            for row in tqdm(scan, desc="Streaming top-k join"):
                d = row["dist"]
                if d is None:
                    continue
                id1, id2 = int(row["id1"]), int(row["id2"])
                if is_self_join and id2 <= id1:
                    continue
                if float(d) < ti_threshold:
                    outf.write(f"{id1}\t{id2}\t{float(d)}\n")
                    written += 1

            sys.stdout.write(f"\nwritten={written} | elapsed={time.time()-start:.1f}s\n")
            sys.stdout.flush()

        conn.commit()
        return written
    finally:
        pass



########################################################

def direct_RangeSearch(
    conn: psycopg2.extensions.connection,
    table_left: str,
    table_right: str,
    vec_col: str,
    id_col_left: str,
    id_col_right: str,
    source_batch: int,
    topk: int,
    ef_search: int,
    ti_threshold: float,
    output_file: str,
    metric: str
):
    """
    左表 × 右表 range（dist < tau）；self-join 时 LATERAL 内排除同一行。
    """
    print("begin direct Range")
    register_vector(conn)
    if os.path.exists(output_file):
        os.remove(output_file)

    op = metric_op[metric]
    is_self_join = (table_left == table_right)

    try:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL hnsw.ef_search = %s;", (ef_search,))

        with conn.cursor(name="direct_scan", cursor_factory=extras.DictCursor) as scan, \
             open(output_file, "w", encoding="utf-8") as outf:
            outf.write("query_id\tneighbor_id\tdist\n")
            if is_self_join:
                lateral_where = f"AND ({id_col_right} <> q.{id_col_left})"
            else:
                lateral_where = ""
            q_sql = f"""
    SELECT
      q.{id_col_left}  AS id1,
      nn.{id_col_right} AS id2,
      (nn.{vec_col} {op} q.{vec_col}) AS dist
    FROM {table_left} AS q
    JOIN LATERAL (
      SELECT {id_col_right}, {vec_col}
      FROM {table_right}
      WHERE ({vec_col} {op} q.{vec_col}) < %s
      {lateral_where}
    ) AS nn ON TRUE
    WHERE q.{vec_col} IS NOT NULL;
"""
            scan.itersize = 1
            print("begin execute")
            scan.execute(q_sql, (ti_threshold,))
            print("done execute")

            written = 0
            start = time.time()
            for row in tqdm(scan, desc="Streaming direct range join"):
                d = row["dist"]
                if d is None:
                    continue
                id1, id2 = int(row["id1"]), int(row["id2"])
                if is_self_join and id2 <= id1:
                    continue
                outf.write(f"{id1}\t{id2}\t{float(d)}\n")
                outf.flush()
                written += 1

            sys.stdout.write(f"\nwritten={written} | elapsed={time.time()-start:.1f}s\n")
            sys.stdout.flush()

        conn.commit()
        return written
    finally:
        pass







def loop_HNSWRange(
    conn: psycopg2.extensions.connection,
    table_left: str,
    table_right: str,
    vec_col: str,
    id_col_left: str,
    id_col_right: str,
    source_batch: int,
    topk: int,
    ef_search: int,
    ti_threshold: float,
    output_file: str,
    metric: str
):
    """
    逐条：扫左表，对每行在右表做 range ANN。两表可相同（self-join）。
    """
    register_vector(conn)
    if os.path.exists(output_file):
        os.remove(output_file)

    is_self_join = (table_left == table_right)
    vc = _clean_ident(vec_col)
    idl, idr = _clean_ident(id_col_left), _clean_ident(id_col_right)
    tbl_left = sql.Identifier(table_left)
    tbl_right = sql.Identifier(table_right)
    idf_vec = sql.Identifier(vc)
    idf_id_left = sql.Identifier(idl)
    idf_id_right = sql.Identifier(idr)
    op = metric_op[metric]

    try:
        with conn.cursor(name="w4_scan", cursor_factory=extras.DictCursor) as scan, \
             conn.cursor(cursor_factory=extras.DictCursor) as cur, \
             open(output_file, "w", encoding="utf-8") as outf:

            outf.write("query_id\tneighbor_id\tdist\n")
            scan.itersize = source_batch
            scan.execute(
                sql.SQL("""
                    SELECT {id_col}, {vec_col}
                    FROM {table}
                    WHERE {vec_col} IS NOT NULL
                    ORDER BY {id_col};
                """).format(id_col=idf_id_left, vec_col=idf_vec, table=tbl_left)
            )
            cur.execute("SET hnsw.radius = %s;", (ti_threshold,))
            cur.execute("SET enable_seqscan = off;")
            cur.execute("SET LOCAL hnsw.ef_search = %s;", (ef_search,))

            start_time = time.time()
            processed = 0

            while True:
                batch = scan.fetchmany(source_batch)
                if not batch:
                    break
                for r in batch:
                    qid = int(r[idl])
                    qv = r[vc]
                    if qv is None:
                        continue

                    if is_self_join:
                        lateral_where = sql.SQL("AND {id_r} <> %s").format(id_r=idf_id_right)
                        params = (qv, qid, qv)
                    else:
                        lateral_where = sql.SQL("")
                        params = (qv, qv)

                    q_sql = sql.SQL("""
                        SELECT {id_r} AS neighbor_id,
                               ({vec_col} """ + op + """ %s::vector) AS dist
                        FROM {tbl_r}
                        WHERE ({vec_col} """ + op + """@ %s::vector) {lateral_where}
                        ORDER BY {vec_col} """ + op + """ %s::vector;
                    """).format(
                        id_r=idf_id_right, vec_col=idf_vec, tbl_r=tbl_right, lateral_where=lateral_where
                    )
                    cur.execute(q_sql, params)

                    for row in cur:
                        d = row["dist"]
                        nid = int(row["neighbor_id"])
                        if is_self_join and nid <= qid:
                            continue
                        outf.write(f"{qid}\t{nid}\t{float(d)}\n")

                    processed += 1
                    if processed % 100 == 0:
                        sys.stdout.write(f"\rprocessed: {processed} | elapsed: {time.time() - start_time:.1f}s")
                        sys.stdout.flush()

            sys.stdout.write(f"\rprocessed: {processed} | elapsed: {time.time() - start_time:.1f}s\n")
            sys.stdout.flush()

        conn.commit()
    finally:
        sys.stdout.flush()







def laterCrossJoin_HNSWRange(
    conn: psycopg2.extensions.connection,
    table_left: str,
    table_right: str,
    vec_col: str,
    id_col_left: str,
    id_col_right: str,
    source_batch: int,
    topk: int,
    ef_search: int,
    ti_threshold: float,
    output_file: str,
    metric: str
):
    """
    批量：扫左表，VALUES + CROSS JOIN LATERAL 右表 range。两表可相同（self-join）。
    """
    register_vector(conn)
    if os.path.exists(output_file):
        os.remove(output_file)

    is_self_join = (table_left == table_right)
    vc, idl, idr = _clean_ident(vec_col), _clean_ident(id_col_left), _clean_ident(id_col_right)
    tbl_left = sql.Identifier(table_left)
    tbl_right = sql.Identifier(table_right)
    idf_vec = sql.Identifier(vc)
    idf_id_left = sql.Identifier(idl)
    idf_id_right = sql.Identifier(idr)
    op = metric_op[metric]

    try:
        with conn.cursor(name="w4_scan", cursor_factory=extras.DictCursor) as scan, \
             conn.cursor(cursor_factory=extras.DictCursor) as cur, \
             open(output_file, "w", encoding="utf-8") as outf:

            cur.execute("SET LOCAL hnsw.ef_search = %s;", (ef_search,))
            cur.execute("SET hnsw.radius = %s;", (ti_threshold,))
            cur.execute("SET enable_seqscan = off;")
            outf.write("query_id\tneighbor_id\tdist\n")

            scan.itersize = source_batch
            scan.execute(
                sql.SQL("SELECT {id_col}, {vec_col} FROM {table} ORDER BY {id_col};")
                .format(id_col=idf_id_left, vec_col=idf_vec, table=tbl_left)
            )
            start_time = time.time()
            processed = 0

            while True:
                batch = scan.fetchmany(source_batch)
                if not batch:
                    break
                values_rows = []
                params = []
                for r in batch:
                    v = r[vc]
                    if v is None:
                        continue
                    values_rows.append("(%s, %s::vector)")
                    params.extend([r[idl], v])
                if not values_rows:
                    continue
                values_sql = ", ".join(values_rows)

                if is_self_join:
                    lateral_extra = f"AND nn.{id_col_right} <> q.{id_col_left}"
                else:
                    lateral_extra = ""
                query_sql = f"""
                WITH q({id_col_left}, v) AS (
                    VALUES {values_sql}
                )
                SELECT
                    q.{id_col_left}  AS query_id,
                    nn.{id_col_right} AS neighbor_id,
                    (nn.{vec_col} {op} (q.v::vector)) AS dist
                FROM q
                CROSS JOIN LATERAL (
                    SELECT {id_col_right}, {vec_col}
                    FROM {table_right}
                    WHERE ({vec_col} {op}@ q.v::vector) {lateral_extra}
                    ORDER BY {vec_col} {op} q.v::vector
                ) nn
                """
                cur.execute(query_sql, params)

                for row in cur:
                    d = row["dist"]
                    if d is None:
                        continue
                    qid = int(row["query_id"])
                    nid = int(row["neighbor_id"])
                    if is_self_join and nid <= qid:
                        continue
                    outf.write(f"{qid}\t{nid}\t{float(d)}\n")

                processed += len(values_rows)
                elapsed = time.time() - start_time
                sys.stdout.write(f"\rprocessed: {processed} | elapsed: {elapsed:.1f}s")
                sys.stdout.flush()

        conn.commit()
    finally:
        sys.stdout.write("\n")
        sys.stdout.flush()






def direct_HNSWRange(
    conn: psycopg2.extensions.connection,
    table_left: str,
    table_right: str,
    vec_col_left: str,
    vec_col_right: str,
    id_col_left: str,
    id_col_right: str,
    source_batch: int,
    topk: int,
    ef_search: int,
    ti_threshold: float,
    output_file: str,
    metric: str
):
    """
    左表 × 右表 HNSW range（dist < tau）；self-join 时 LATERAL 内排除同一行，输出只保留 id1 < id2。
    对异构表支持左右不同的 vec_col / id_col；self-join 时两侧传同名即可。
    """
    print("begin direct HNSW range")
    register_vector(conn)
    if os.path.exists(output_file):
        os.remove(output_file)

    op = metric_op[metric]
    is_self_join = (table_left == table_right)

    # Use _hnsw suffix tables for HNSW index scan
    def _hnsw_table(t):
        return t if t.endswith("_hnsw") else f"{t}_hnsw"
    scan_left = _hnsw_table(table_left)
    scan_right = _hnsw_table(table_right)

    try:
        with conn.cursor(name="direct_scan", cursor_factory=extras.DictCursor) as scan, \
             conn.cursor() as cur, \
             open(output_file, "w", encoding="utf-8") as outf:

            cur.execute("SET LOCAL hnsw.ef_search = %s;", (ef_search,))
            cur.execute("SET hnsw.radius = %s;", (ti_threshold,))
            cur.execute("SET enable_seqscan = off;")
            outf.write("query_id\tneighbor_id\tdist\n")

            if is_self_join:
                lateral_where = f"WHERE nn.{vec_col_right} {op}@ q.{vec_col_left} AND nn.{id_col_right} <> q.{id_col_left}"
            else:
                lateral_where = f"WHERE nn.{vec_col_right} {op}@ q.{vec_col_left}"
            q_sql = f"""
                SELECT
                  q.{id_col_left}  AS id1,
                  nn.{id_col_right} AS id2,
                  (nn.{vec_col_right} {op} q.{vec_col_left}) AS dist
                FROM {scan_left} AS q
                JOIN LATERAL (
                  SELECT nn.{id_col_right}, nn.{vec_col_right}
                  FROM {scan_right} AS nn
                  {lateral_where}
                  ORDER BY nn.{vec_col_right} {op} q.{vec_col_left}
                ) AS nn ON TRUE
                WHERE q.{vec_col_left} IS NOT NULL;
            """
            scan.itersize = 100000
            scan.execute(q_sql)

            written = 0
            start = time.time()
            for row in tqdm(scan, desc="Streaming range hnsw join"):
                d = row["dist"]
                if d is None:
                    continue
                id1, id2 = int(row["id1"]), int(row["id2"])
                if is_self_join and id2 <= id1:
                    continue
                outf.write(f"{id1}\t{id2}\t{float(d)}\n")
                written += 1

            sys.stdout.write(f"\nwritten={written} | elapsed={time.time()-start:.1f}s\n")
            sys.stdout.flush()

        conn.commit()
        return written
    finally:
        pass


def materialize_w4_ti(
    conn: psycopg2.extensions.connection,
    input_file: str,
    table_name: str = "w4_ti",
    corpus_table: str = "w4_corpus",
    corpus_table_left: Optional[str] = None,
    corpus_table_right: Optional[str] = None,
    id_col: str = "id",
    field_col: str = "field",
    field_col_left: Optional[str] = None,
    field_col_right: Optional[str] = None,
):
    """
    从 TSV 文件物化 TI 表（格式：query_id\\tneighbor_id\\tdist）。
    - 若提供 corpus_table_left / corpus_table_right：左 id 从 left 表取 field，右 id 从 right 表取（两表 join index）。
    - 否则用 corpus_table 作为单表，query_id 与 neighbor_id 都从该表取（self-join 兼容）。
    field_col 为单表时的列名；field_col_left / field_col_right 为双表时的列名，默认用 field_col。
    """
    left_table = corpus_table_left if corpus_table_left is not None else corpus_table
    right_table = corpus_table_right if corpus_table_right is not None else corpus_table
    fl = field_col_left if field_col_left is not None else field_col
    fr = field_col_right if field_col_right is not None else field_col

    with conn.cursor() as cur:
        cur.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        cur.execute(
            f'CREATE TABLE "{table_name}" (id1 int, id2 int, dis float, field1 text, field2 text)'
        )
        conn.commit()

    batch_size = 1000
    with open(input_file, "r", encoding="utf-8") as f:
        next(f)
        with conn.cursor() as cur:
            batch_data = []
            left_ids = set()
            right_ids = set()
            for line in tqdm(f, desc="Reading and preparing data"):
                line = line.strip()
                if not line:
                    continue
                query_id, neighbor_id, dis = line.split("\t")
                query_id = int(query_id)
                neighbor_id = int(neighbor_id)
                dis = float(dis)
                batch_data.append((query_id, neighbor_id, dis))
                left_ids.add(query_id)
                right_ids.add(neighbor_id)

            if not batch_data:
                print("No data to process")
                return

            field_map = {}
            query_batch_size = 10000
            # 左 id -> field1（来自 left 表）
            id_list_left = list(left_ids)
            for i in tqdm(range(0, len(id_list_left), query_batch_size), desc="Fetching left fields"):
                batch_ids = id_list_left[i : i + query_batch_size]
                placeholders = ",".join(["%s"] * len(batch_ids))
                cur.execute(
                    f'SELECT "{id_col}", "{fl}" FROM "{left_table}" WHERE "{id_col}" IN ({placeholders})',
                    batch_ids,
                )
                for row in cur.fetchall():
                    field_map[("left", row[0])] = row[1]
            # 右 id -> field2（来自 right 表）
            id_list_right = list(right_ids)
            for i in tqdm(range(0, len(id_list_right), query_batch_size), desc="Fetching right fields"):
                batch_ids = id_list_right[i : i + query_batch_size]
                placeholders = ",".join(["%s"] * len(batch_ids))
                cur.execute(
                    f'SELECT "{id_col}", "{fr}" FROM "{right_table}" WHERE "{id_col}" IN ({placeholders})',
                    batch_ids,
                )
                for row in cur.fetchall():
                    field_map[("right", row[0])] = row[1]

            insert_batch = []
            for query_id, neighbor_id, dis in tqdm(batch_data, desc="Inserting data"):
                key_left = ("left", query_id)
                key_right = ("right", neighbor_id)
                if key_left not in field_map or key_right not in field_map:
                    continue
                field1 = field_map[key_left]
                field2 = field_map[key_right]
                insert_batch.append((query_id, neighbor_id, dis, field1, field2))
                if len(insert_batch) >= batch_size:
                    extras.execute_batch(
                        cur,
                        f'INSERT INTO "{table_name}" (id1, id2, dis, field1, field2) VALUES (%s, %s, %s, %s, %s)',
                        insert_batch,
                        page_size=batch_size,
                    )
                    conn.commit()
                    insert_batch = []

            if insert_batch:
                extras.execute_batch(
                    cur,
                    f'INSERT INTO "{table_name}" (id1, id2, dis, field1, field2) VALUES (%s, %s, %s, %s, %s)',
                    insert_batch,
                    page_size=batch_size,
                )
                conn.commit()

    print("Materialized ti")
    with conn.cursor() as cur:
        cur.execute(f'CREATE INDEX "{table_name}_idx_dis" ON "{table_name}"(dis)')
        cur.execute(f'CREATE INDEX "{table_name}_idx_field1" ON "{table_name}"(field1)')
        cur.execute(f'CREATE INDEX "{table_name}_idx_field2" ON "{table_name}"(field2)')
        conn.commit()
    print("Added index on dis, field1 and field2")