"""Load the three NFCorpus parquet exports required by Figure 8."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet

PAPER_TABLE = "nfcorpus_paper_hnsw"
EDGE_TABLE = "ti_nfcorpus_paper_hnsw_nfcorpus_paper_hnsw_0.35"
OUTCOME_TABLE = "ti_nfcorpus_paper_hnsw_nfcorpus_paper_hnsw_study_outcome_0.35"
EMBEDDING_DIMENSIONS = 3072


def _build_id_map(conn, table: str) -> int:
    map_table = f"{table}_id_map"
    with conn.cursor() as cur:
        cur.execute(f'DROP TABLE IF EXISTS "{map_table}"')
        cur.execute(
            f'CREATE TABLE "{map_table}" ('
            "blkno integer NOT NULL, offno integer NOT NULL, realid bigint NOT NULL, "
            "PRIMARY KEY (blkno, offno))"
        )
        cur.execute(f'SELECT ctid, id FROM "{table}"')
        rows = cur.fetchall()
        values = []
        for ctid, real_id in rows:
            block, offset = map(int, str(ctid).strip("()").split(","))
            values.append((block, offset, real_id))
        cur.executemany(
            f'INSERT INTO "{map_table}" (blkno, offno, realid) VALUES (%s, %s, %s)',
            values,
        )
    conn.commit()
    return len(values)


def _load_papers(conn, path: Path) -> int:
    table = parquet.read_table(path).to_pydict()
    count = len(table["id"])
    with conn.cursor() as cur:
        cur.execute(f'DROP TABLE IF EXISTS "{PAPER_TABLE}" CASCADE')
        cur.execute(
            f'CREATE TABLE "{PAPER_TABLE}" ('
            f"id bigint PRIMARY KEY, title text, title_emb vector({EMBEDDING_DIMENSIONS}), "
            "health_condition text, intervention_type text, outcome_measure text, study_design text)"
        )
        rows = [
            (
                table["id"][index],
                table["title"][index],
                np.asarray(table["title_emb"][index], dtype=np.float32),
                table["health_condition"][index],
                table["intervention_type"][index],
                table["outcome_measure"][index],
                table["study_design"][index],
            )
            for index in range(count)
        ]
        cur.executemany(
            f'INSERT INTO "{PAPER_TABLE}" VALUES (%s, %s, %s, %s, %s, %s, %s)',
            rows,
        )
        for attribute in ("health_condition", "intervention_type", "outcome_measure", "study_design"):
            cur.execute(f'CREATE INDEX "idx_{PAPER_TABLE}_{attribute}" ON "{PAPER_TABLE}" ("{attribute}")')
        cur.execute(
            f'CREATE INDEX "idx_{PAPER_TABLE}_title_emb" ON "{PAPER_TABLE}" '
            "USING hnsw (title_emb vector_l2_ops) WITH (m=16, ef_construction=64)"
        )
        cur.execute(f'DROP VIEW IF EXISTS "{PAPER_TABLE}_ivf"')
        cur.execute(f'CREATE VIEW "{PAPER_TABLE}_ivf" AS SELECT * FROM "{PAPER_TABLE}"')
    conn.commit()
    return count


def _load_edges(conn, path: Path, table_name: str, link_type: str) -> int:
    table = parquet.read_table(path).to_pydict()
    count = len(table["id1"])
    index_prefix = "nfcorpus_ic" if table_name == EDGE_TABLE else "nfcorpus_so"
    with conn.cursor() as cur:
        cur.execute(f'DROP TABLE IF EXISTS "{table_name}" CASCADE')
        cur.execute(
            f'CREATE TABLE "{table_name}" ('
            "id1 integer, id2 integer, dis double precision, field1 text, field2 text)"
        )
        cur.executemany(
            f'INSERT INTO "{table_name}" VALUES (%s, %s, %s, %s, %s)',
            [
                (table["id1"][index], table["id2"][index], table["dis"][index], link_type, link_type)
                for index in range(count)
            ],
        )
        cur.execute(f'CREATE INDEX "idx_{index_prefix}_id1" ON "{table_name}" (id1)')
        cur.execute(f'CREATE INDEX "idx_{index_prefix}_id2" ON "{table_name}" (id2)')
        cur.execute(f'CREATE INDEX "idx_{index_prefix}_dis" ON "{table_name}" (dis)')
    conn.commit()
    return count


def run_load(parquet_dir: Path, database_url: str) -> dict[str, int]:
    import psycopg
    from pgvector.psycopg import register_vector

    required = [
        parquet_dir / f"{PAPER_TABLE}.parquet",
        parquet_dir / f"{EDGE_TABLE}.parquet",
        parquet_dir / f"{OUTCOME_TABLE}.parquet",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing NFCorpus parquet files: " + ", ".join(missing))

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        register_vector(conn)
        papers = _load_papers(conn, required[0])
        id_map = _build_id_map(conn, PAPER_TABLE)
        intervention = _load_edges(
            conn, required[1], EDGE_TABLE, "nfcorpus_wise_shared_intervention_condition"
        )
        outcomes = _load_edges(
            conn, required[2], OUTCOME_TABLE, "nfcorpus_wise_shared_study_outcome"
        )
    return {
        "papers": papers,
        "id_map_rows": id_map,
        "intervention_condition_links": intervention,
        "study_outcome_links": outcomes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument(
        "--database-url", default="postgresql://localhost/nfcorpus"
    )
    args = parser.parse_args(argv)
    print(json.dumps(run_load(args.parquet_dir, args.database_url), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())