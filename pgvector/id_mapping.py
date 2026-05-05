#!/usr/bin/env python3
import os
import psycopg2
from psycopg2.extras import execute_values
from tqdm import tqdm

DSN = dict(
    host=os.environ.get("PGHOST", "127.0.0.1"),
    port=5432,
    dbname="molecule",
    user=os.environ.get("PGUSER", "postgres"),
    password=os.environ.get("PGPASSWORD", ""),
)

SOURCE_TABLE = "w4_corpus"
TARGET_TABLE = "w4_corpus_id_map"


def build_tid_id_table():
    conn = psycopg2.connect(**DSN)
    try:
        with conn, conn.cursor() as cur:
            print('begin dropping tid id table...')
            cur.execute(f"DROP TABLE IF EXISTS {TARGET_TABLE}")
            print('begin selecting tid id table...')
            cur.execute(
                f"""
                CREATE TABLE {TARGET_TABLE} (
                    blkno integer NOT NULL,
                    offno integer NOT NULL,
                    realid integer NOT NULL,
                    PRIMARY KEY (blkno, offno)
                )
                """
            )
            print('building tid id table...')
            cur.execute(f"SELECT ctid, id FROM {SOURCE_TABLE}")
            rows = []
            for ctid, row_id in tqdm(cur.fetchall()):
                blkno, offno = map(int, str(ctid).strip("()").split(","))
                rows.append((blkno, offno, row_id))

            if rows:
                execute_values(
                    cur,
                    f"INSERT INTO {TARGET_TABLE} (blkno, offno, realid) VALUES %s",
                    rows,
                )
    finally:
        conn.close()


if __name__ == "__main__":
    build_tid_id_table()