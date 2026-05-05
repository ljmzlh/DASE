import os
import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2 import sql
import psycopg2.extras as extras
import re
import time
import sys


metric_op = {'l2': '<->', 'ip': '<#>', 'cos': '<=>'}


def install_notice_logger(conn: psycopg2.extensions.connection):
    if hasattr(conn, "set_notice_processor"):
        def _notice_processor(notice):
            msg = notice.strip()
            if msg:
                print(f"[PG NOTICE] {msg}")

        conn.set_notice_processor(_notice_processor)
    else:
        print("[WARN] 当前 psycopg2 版本不支持 set_notice_processor，将回退到 notices 列表轮询", file=sys.stderr)


def flush_notices(conn: psycopg2.extensions.connection):
    notices = getattr(conn, "notices", None)
    if not notices:
        return
    for notice in notices:
        msg = notice.strip()
        if msg:
            print(f"[PG NOTICE] {msg}")
    notices.clear()


def output_plan(line):
    # 如果行中包含向量（如'['和']'），则只保留前10个元素
    if '[' in line and ']' in line:
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

def test_HnswRangeSearch(
    conn: psycopg2.extensions.connection,
    table: str,
    vec_col: str,
    id_col: str,
    ef_search: int,
    ti_threshold: float,
    metric: str
):
    """
    Fetch one query from the source table and execute the query plan.
    """
    install_notice_logger(conn)
    register_vector(conn)  # 若需要
    tbl = sql.Identifier(table)
    idf_vec = sql.Identifier(vec_col)
    idf_id = sql.Identifier(id_col)

    op = sql.SQL(metric_op[metric])

    with conn.cursor(name="w4_scan", cursor_factory=extras.DictCursor) as scan, conn.cursor(cursor_factory=extras.DictCursor) as cur:
            # 用 cur 执行 SET 语句，设置 hnsw.radius 和 enable_seqscan
            cur.execute("SET hnsw.radius = %s;", (ti_threshold,))
            cur.execute("SET enable_seqscan = off;")
            cur.execute("SET LOCAL hnsw.ef_search = %s;", (ef_search,))
            flush_notices(conn)
            
            # fetch one query from the source table
            scan.execute(sql.SQL("SELECT {id_col}, {vec_col} FROM {table} WHERE {vec_col} IS NOT NULL ORDER BY {id_col};").format(id_col=idf_id, vec_col=idf_vec, table=tbl))
            qv = scan.fetchone()[1]


            q_sql = sql.SQL("""
                EXPLAIN ANALYZE 
                SELECT {id_col} AS neighbor_id,
                       ({vec_col} {op} %s::vector) AS dist
                FROM {table}
                WHERE ({vec_col} {op}@ %s::vector)
                ORDER BY {vec_col} {op} %s::vector;
            """).format(id_col=idf_id, vec_col=idf_vec, table=tbl, op=op)

            cur.execute(q_sql, (qv, qv, qv))
            flush_notices(conn)

            plan = cur.fetchall()
            for row in plan:
                line = row[0]
                output_plan(line)
            print('--------------------------------\n\n')
            
            q_sql = sql.SQL("""
                SELECT {id_col} AS neighbor_id,
                       ({vec_col} {op} %s::vector) AS dist
                FROM {table}
                WHERE ({vec_col} {op}@ %s::vector)
                ORDER BY {vec_col} {op} %s::vector;
            """).format(id_col=idf_id, vec_col=idf_vec, table=tbl, op=op)

            cur.execute(q_sql, (qv, qv, qv))
            flush_notices(conn)
            result = cur.fetchall()
            print('result:', result)


def test_HnswFilterSearch(
    conn: psycopg2.extensions.connection,
    table: str,
    vec_col: str,
    id_col: str,
    ef_search: int,
    ti_threshold: float,
    metric: str
):
    install_notice_logger(conn)
    register_vector(conn)

    tbl = table
    vec_col = vec_col
    id_col = id_col
    ef_search = ef_search
    ti_threshold = ti_threshold
    metric = metric
    op = metric_op[metric]


    # seperate scan and index query, becuase only read 1 row from source table
    cur = conn.cursor(cursor_factory=extras.DictCursor)
    # fetch one query vector（这里直接拼接字符串，假设表名/列名可信）
    scan_sql = f'SELECT {id_col}, "{vec_col}" FROM {tbl} WHERE "{vec_col}" IS NOT NULL ORDER BY {id_col};'
    cur.execute(scan_sql)
    qv = cur.fetchone()[1]

    # set what you need to set
    cur.execute("SET hnsw.radius = %s;", (ti_threshold,))
    cur.execute("SET enable_seqscan = off;")
    cur.execute("SET LOCAL hnsw.ef_search = %s;", (ef_search,))
    flush_notices(conn)


    max_id = 140282
    filter_map = bytearray(max_id // 8 + 1)
    for i in range(1, 100):
        filter_map[i // 8] |= 1 << (i % 8)
    filter_map = psycopg2.Binary(filter_map)


    base_sql = f"""
        SELECT {id_col} AS neighbor_id,
               ("{vec_col}" {op} %s::vector) AS dist
        FROM "{tbl}"
        WHERE ("{vec_col}" OPERATOR(public.<->#) %s::bytea)
        ORDER BY "{vec_col}" {op} %s::vector;
    """

    # index query with qv
    q_sql = f"EXPLAIN {base_sql}"
    cur.execute(q_sql, (qv, filter_map, qv))
    flush_notices(conn)


    plan = cur.fetchall()
    for row in plan:
        line = row[0]
        output_plan(line)

    print('\n--------------------------------\n\n')

    q_sql = base_sql
    cur.execute(q_sql, (qv, filter_map, qv))
    flush_notices(conn)
    result = cur.fetchall()
    print('result:', result)    

if __name__ == "__main__":
    DSN = dict(host=os.environ.get("PGHOST", "127.0.0.1"), database="molecule", user=os.environ.get("PGUSER", "postgres"), password=os.environ.get("PGPASSWORD", "")) #local
    conn = psycopg2.connect(**DSN)
    conn.autocommit = False
    #test_HnswRangeSearch(conn, "w4_corpus", "author_name_qwen3-0.6B_1024",
    #                     "id", 20, 0.1, "l2")
    
    test_HnswFilterSearch(conn, "w4_corpus", "author_name_qwen3-0.6B_1024",
                         "id", 20, 0.1, "l2")
    conn.close()