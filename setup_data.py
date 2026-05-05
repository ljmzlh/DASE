#!/usr/bin/env python3
"""
setup_data.py — Bootstrap a fresh dase_clean checkout for DASE + cascade runs.

Three phases (each independently skippable):
  1. DOWNLOAD   pull per-scenario sembench files (data/, cache/, ground_truth/,
                query/, bq_data/) AND psql parquet dumps from a configurable
                source (GCS prefix or HTTPS prefix) into the local layout.
  2. PSQL       create `imdb` and `molecule` databases (with pgvector),
                CREATE TABLE base tables, COPY rows from the downloaded
                parquets, build hnsw + btree + gin indexes. Then build the
                SemJI / TI tables (ti_imdb_t1_imdb_t2_{0.5,0.6},
                ti_facts_50k_paper_{0.5,0.6,0.7}) by invoking
                `python -m ours.ti.ti_build` for each (config, threshold).
                Skip with --skip-ti — but W5–W8 workloads need them.
  3. BQ SETUP   for scenarios with setup_bq.py (wildlife, mmqa, cars):
                runs the per-scenario setup using the provided GCP project.
                ecomm and movie cascade scripts handle BQ tables inline on
                first run; nothing centralized to do for them here.

Lives at the dase_clean/ root. Sembench scenario files land under
dase_clean/sembench_data/<scenario>/...; psql parquets land under
dase_clean/psql_dump/ (loaded into postgres then transient).

Usage
-----
    # Default (all phases) — uses HF source baked-in below + your GCP project
    python setup_data.py --project my-gcp-proj

    # Custom source / skip phases
    python setup_data.py --source gs://my-bucket/sembench --project my-gcp-proj
    python setup_data.py --skip-bq                                         # data + psql only
    python setup_data.py --skip-download --project my-gcp-proj --scenarios cars,mmqa

Source layout (--source must mirror this):
    <source>/<scenario>/<relative-path>     for sembench scenarios
    <source>/psql/<db>/<table>.parquet       for psql base tables
e.g. https://huggingface.co/datasets/dasepaper/dase_data
     gs://my-bucket/sembench/cars/data/text_complaints.parquet

Three source schemes are supported:
    https://huggingface.co/datasets/<owner>/<repo>   → huggingface_hub SDK
    gs://<bucket>/<prefix>                            → gsutil
    https://...                                       → curl (no recursive dirs)

Requirements
------------
    Download:    HF source → pip: huggingface_hub
                 gs:// source → `gsutil` on PATH
                 plain https:// source → `curl` (only single files; no dirs)
    PSQL setup:  pgvector extension installed on the postgres server.
                 pip: psycopg, pgvector, pyarrow, numpy.
                 The runner role needs CREATEDB privilege.
    BQ setup:    `gcloud auth application-default login`
                 SEMBENCH_SRC env var → upstream sembench/src/ checkout
                 pip: google-cloud-storage, google-cloud-bigquery
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent          # dase_clean/
SEMBENCH_ROOT = ROOT / "sembench_data"           # sembench scenario tree

DEFAULT_HF_SOURCE = "https://huggingface.co/datasets/dasepaper/dase_data"

# ─── Per-scenario sembench download manifest ─────────────────────────────
# Paths are relative to <scenario>/. Trailing "/" → directory (recursive copy).
MANIFEST = {
    "wildlife": [
        "data/audio_embeddings.npz",
        "data/audio_predictions.csv",
        "data/audio_usage_stats.json",
        "data/audio_with_embeddings.tsv",
        "data/image_embeddings.npz",
        "data/image_predictions.csv",
        "data/image_usage_stats.json",
        "data/image_with_embeddings.tsv",
        "cache/audio_data.csv",
        "cache/image_data.csv",
        "bq_data/",
    ],
    "mmqa": [
        "data/ap_warrior.parquet",
        "data/images.parquet",
        "data/lizzy_caplan_text_data.parquet",
        "data/tampa_international_airport.parquet",
        "cache/embed_checkpoints/embed_usage.json",
        "cache/embed_checkpoints/image_caption_usage.json",
        "ground_truth/",
        "query/",
    ],
    "ecomm": [
        "data/products.parquet",
        "data/products_image.parquet",
        "data/products_text.parquet",
        "cache/sf_500/styles_details.parquet",
        "cache/embed_checkpoints/embed_usage.json",
        "cache/embed_checkpoints/image_caption_usage.json",
        "ground_truth/",
    ],
    "cars": [
        "data/cars.parquet",
        "data/audio_cars.parquet",
        "data/image_cars.parquet",
        "data/text_complaints.parquet",
        "cache/embed_checkpoints/audio_caption_usage.json",
        "cache/embed_checkpoints/embed_usage.json",
        "cache/embed_checkpoints/image_caption_usage.json",
        "ground_truth/",
    ],
    "movie": [
        "data/embed_usage_stats.json",
        "data/review_embeddings.npz",
        "data/Reviews_with_embeddings.tsv",
        "cache/Movies.csv",
        "cache/Reviews.csv",
        "gt_Q10.csv",
        "lotus_Q10.csv",
        "lotus_Q9.csv",
        "Q3.csv",
    ],
}

# Scenarios that ship a setup_bq.py wrapper around upstream sembench's BQ setup.
BQ_SCENARIOS = ("wildlife", "mmqa", "cars")

# Per-db TI tables the engine expects (W5–W8 workloads). Each entry is a
# (ti_config_name, threshold) pair fed to `python -m ours.ti.ti_build`.
# `imdb` config covers ti_imdb_t1_imdb_t2_*; the molecule TI is sharded into
# three configs (one per threshold) because each uses a different ef_search.
# ti_build.py already calls materialize_ti() at the end, so no separate step.
TI_BUILD_PLAN: Dict[str, List[Tuple[str, float]]] = {
    "imdb": [("imdb", 0.5), ("imdb", 0.6)],
    "molecule": [
        ("w5_molecule_0.5", 0.5),
        ("w5_molecule_0.6", 0.6),
        ("w5_molecule_0.7", 0.7),
    ],
}

# ─── PSQL manifest ───────────────────────────────────────────────────────
# (db_kind, table_name, parquet rel path under <source>/<PSQL_PREFIX>/)
PSQL_PREFIX = "psql"

PSQL_MANIFEST: Dict[str, Dict] = {
    "imdb": {
        "url_env": "IMDB_DATABASE_URL",
        "default_url": "postgresql://localhost/imdb",
        "tables": ["imdb_t1_hnsw", "imdb_t2_hnsw"],
    },
    "molecule": {
        "url_env": "MOLECULE_DATABASE_URL",
        "default_url": "postgresql://localhost/molecule",
        "tables": ["facts_50k_hnsw", "paper_hnsw", "molecule"],
    },
}

# CREATE TABLE DDLs (verbatim from pg_dump --schema-only, with vector(...) using
# the public schema-qualified type which the `vector` extension installs there).
PSQL_TABLE_DDL: Dict[str, str] = {
    "imdb_t1_hnsw": """
        CREATE TABLE public.imdb_t1_hnsw (
            id bigint NOT NULL,
            tconst text NOT NULL,
            title text,
            year integer,
            genre text,
            rating double precision,
            plot text,
            actor_director text,
            num_votes bigint,
            title_emb vector(1536),
            plot_emb vector(1536),
            actor_director_emb vector(1536)
        )
    """,
    "imdb_t2_hnsw": """
        CREATE TABLE public.imdb_t2_hnsw (
            id bigint NOT NULL,
            tconst text NOT NULL,
            title text,
            year integer,
            genre text,
            rating double precision,
            plot text,
            actor_director text,
            num_votes bigint,
            title_emb vector(1536),
            plot_emb vector(1536),
            actor_director_emb vector(1536)
        )
    """,
    "facts_50k_hnsw": """
        CREATE TABLE public.facts_50k_hnsw (
            fact_id bigint NOT NULL,
            mol_id integer,
            pmid text,
            mol_name text,
            fact_text text,
            mol_ecfp bit(1024),
            mol_mw double precision,
            mol_logp double precision,
            mol_num_atoms integer,
            mol_num_rings integer,
            mol_tpsa double precision,
            mol_num_hbd integer,
            mol_num_hba integer,
            mol_therapeutic_area text[],
            mol_moa text,
            mol_target_type text,
            mol_source_origin text,
            mol_toxicity_flag text,
            fact_text_emb vector(1536),
            fact_type text,
            evidence_type text,
            claim_polarity text,
            organism text,
            numeric_value double precision,
            numeric_unit text
        )
    """,
    "paper_hnsw": """
        CREATE TABLE public.paper_hnsw (
            pmid text NOT NULL,
            doi text,
            journal text,
            issn text,
            eissn text,
            title text,
            year integer,
            authors text[],
            mesh_terms text[],
            keywords text[],
            pub_type text[],
            language text,
            abstract text,
            abstract_emb vector(1536),
            id bigint NOT NULL
        )
    """,
    "molecule": """
        CREATE TABLE public.molecule (
            mol_id integer NOT NULL,
            smiles text NOT NULL,
            fact_count integer NOT NULL,
            ecfp_1024 bit(1024),
            chemberta_768 vector(768),
            molecular_weight double precision,
            logp double precision,
            num_atoms integer,
            num_rings integer,
            tpsa double precision,
            num_hbd integer,
            num_hba integer,
            earliest_year integer,
            latest_year integer,
            num_papers integer,
            therapeutic_area text[],
            mechanism_of_action text,
            target_type text,
            source_origin text,
            toxicity_flag text,
            description text
        )
    """,
}

# Constraints (PK / UNIQUE) — created right after CREATE TABLE so loads can use
# upserts if needed; before any other index for clearer plans.
PSQL_TABLE_CONSTRAINTS: Dict[str, List[str]] = {
    "imdb_t1_hnsw": [
        'ALTER TABLE public.imdb_t1_hnsw ADD CONSTRAINT imdb_t1_hnsw_pkey PRIMARY KEY (id)',
        'ALTER TABLE public.imdb_t1_hnsw ADD CONSTRAINT imdb_t1_hnsw_tconst_key UNIQUE (tconst)',
    ],
    "imdb_t2_hnsw": [
        'ALTER TABLE public.imdb_t2_hnsw ADD CONSTRAINT imdb_t2_hnsw_pkey PRIMARY KEY (id)',
        'ALTER TABLE public.imdb_t2_hnsw ADD CONSTRAINT imdb_t2_hnsw_tconst_key UNIQUE (tconst)',
    ],
    "facts_50k_hnsw": [
        'ALTER TABLE public.facts_50k_hnsw ADD CONSTRAINT facts_50k_hnsw_pkey PRIMARY KEY (fact_id)',
    ],
    "paper_hnsw": [
        'ALTER TABLE public.paper_hnsw ADD CONSTRAINT paper_hnsw_pkey PRIMARY KEY (id)',
        'ALTER TABLE public.paper_hnsw ADD CONSTRAINT paper_hnsw_pmid_key UNIQUE (pmid)',
    ],
    "molecule": [
        'ALTER TABLE public.molecule ADD CONSTRAINT molecule_pkey PRIMARY KEY (mol_id)',
        'ALTER TABLE public.molecule ADD CONSTRAINT molecule_smiles_key UNIQUE (smiles)',
    ],
}

# Indexes — built AFTER COPY for speed (hnsw last; it's the slowest).
# Tables that need an `<table>_id_map` lookup table (ctid→realid) for the
# DASE-pgvector fork's `<->#` filtered-HNSW operator. ours/utils.py
# build_id_map_table() does the same thing; we replicate it here so a fresh
# checkout doesn't need to import the `ours` package mid-setup. Map: table → PK column.
PSQL_ID_MAP_TABLES: Dict[str, str] = {
    "imdb_t1_hnsw":   "id",
    "imdb_t2_hnsw":   "id",
    "facts_50k_hnsw": "fact_id",
    "paper_hnsw":     "id",
}


PSQL_TABLE_INDEXES: Dict[str, List[str]] = {
    "imdb_t1_hnsw": [
        "CREATE INDEX idx_imdb_t1_hnsw_genre  ON public.imdb_t1_hnsw USING btree (genre)",
        "CREATE INDEX idx_imdb_t1_hnsw_rating ON public.imdb_t1_hnsw USING btree (rating)",
        "CREATE INDEX idx_imdb_t1_hnsw_year   ON public.imdb_t1_hnsw USING btree (year)",
        "CREATE INDEX idx_imdb_t1_hnsw_title_emb         ON public.imdb_t1_hnsw USING hnsw (title_emb vector_l2_ops) WITH (m='16', ef_construction='64')",
        "CREATE INDEX idx_imdb_t1_hnsw_plot_emb          ON public.imdb_t1_hnsw USING hnsw (plot_emb vector_l2_ops) WITH (m='16', ef_construction='64')",
        "CREATE INDEX idx_imdb_t1_hnsw_actor_director_emb ON public.imdb_t1_hnsw USING hnsw (actor_director_emb vector_l2_ops) WITH (m='16', ef_construction='64')",
    ],
    "imdb_t2_hnsw": [
        "CREATE INDEX idx_imdb_t2_hnsw_genre  ON public.imdb_t2_hnsw USING btree (genre)",
        "CREATE INDEX idx_imdb_t2_hnsw_rating ON public.imdb_t2_hnsw USING btree (rating)",
        "CREATE INDEX idx_imdb_t2_hnsw_year   ON public.imdb_t2_hnsw USING btree (year)",
        "CREATE INDEX idx_imdb_t2_hnsw_title_emb         ON public.imdb_t2_hnsw USING hnsw (title_emb vector_l2_ops) WITH (m='16', ef_construction='64')",
        "CREATE INDEX idx_imdb_t2_hnsw_plot_emb          ON public.imdb_t2_hnsw USING hnsw (plot_emb vector_l2_ops) WITH (m='16', ef_construction='64')",
        "CREATE INDEX idx_imdb_t2_hnsw_actor_director_emb ON public.imdb_t2_hnsw USING hnsw (actor_director_emb vector_l2_ops) WITH (m='16', ef_construction='64')",
    ],
    "facts_50k_hnsw": [
        "CREATE INDEX idx_facts_50k_hnsw_claim_polarity   ON public.facts_50k_hnsw USING btree (claim_polarity)",
        "CREATE INDEX idx_facts_50k_hnsw_evidence_type    ON public.facts_50k_hnsw USING btree (evidence_type)",
        "CREATE INDEX idx_facts_50k_hnsw_fact_type        ON public.facts_50k_hnsw USING btree (fact_type)",
        "CREATE INDEX idx_facts_50k_hnsw_mol_id           ON public.facts_50k_hnsw USING btree (mol_id)",
        "CREATE INDEX idx_facts_50k_hnsw_mol_logp         ON public.facts_50k_hnsw USING btree (mol_logp)",
        "CREATE INDEX idx_facts_50k_hnsw_mol_moa          ON public.facts_50k_hnsw USING btree (mol_moa)",
        "CREATE INDEX idx_facts_50k_hnsw_mol_mw           ON public.facts_50k_hnsw USING btree (mol_mw)",
        "CREATE INDEX idx_facts_50k_hnsw_mol_num_atoms    ON public.facts_50k_hnsw USING btree (mol_num_atoms)",
        "CREATE INDEX idx_facts_50k_hnsw_mol_num_hba      ON public.facts_50k_hnsw USING btree (mol_num_hba)",
        "CREATE INDEX idx_facts_50k_hnsw_mol_num_hbd      ON public.facts_50k_hnsw USING btree (mol_num_hbd)",
        "CREATE INDEX idx_facts_50k_hnsw_mol_num_rings    ON public.facts_50k_hnsw USING btree (mol_num_rings)",
        "CREATE INDEX idx_facts_50k_hnsw_mol_source_origin ON public.facts_50k_hnsw USING btree (mol_source_origin)",
        "CREATE INDEX idx_facts_50k_hnsw_mol_target_type  ON public.facts_50k_hnsw USING btree (mol_target_type)",
        "CREATE INDEX idx_facts_50k_hnsw_mol_toxicity_flag ON public.facts_50k_hnsw USING btree (mol_toxicity_flag)",
        "CREATE INDEX idx_facts_50k_hnsw_mol_tpsa         ON public.facts_50k_hnsw USING btree (mol_tpsa)",
        "CREATE INDEX idx_facts_50k_hnsw_organism         ON public.facts_50k_hnsw USING btree (organism)",
        "CREATE INDEX idx_facts_50k_hnsw_pmid             ON public.facts_50k_hnsw USING btree (pmid)",
        "CREATE INDEX idx_facts_50k_hnsw_ta_gin           ON public.facts_50k_hnsw USING gin (mol_therapeutic_area)",
        "CREATE INDEX idx_facts_50k_hnsw_ecfp_hnsw        ON public.facts_50k_hnsw USING hnsw (mol_ecfp bit_jaccard_ops) WITH (m='16', ef_construction='200')",
        "CREATE INDEX idx_facts_50k_hnsw_emb_hnsw         ON public.facts_50k_hnsw USING hnsw (fact_text_emb vector_l2_ops) WITH (m='16', ef_construction='200')",
    ],
    "paper_hnsw": [
        "CREATE INDEX idx_paper_hnsw_journal     ON public.paper_hnsw USING btree (journal)",
        "CREATE INDEX idx_paper_hnsw_year        ON public.paper_hnsw USING btree (year)",
        "CREATE INDEX idx_paper_hnsw_mesh_gin    ON public.paper_hnsw USING gin (mesh_terms)",
        "CREATE INDEX idx_paper_hnsw_pub_type    ON public.paper_hnsw USING gin (pub_type)",
        "CREATE INDEX idx_paper_hnsw_abstract_hnsw ON public.paper_hnsw USING hnsw (abstract_emb vector_l2_ops) WITH (m='32', ef_construction='200')",
    ],
    "molecule": [
        "CREATE INDEX idx_molecule_chemberta_hnsw ON public.molecule USING hnsw (chemberta_768 vector_l2_ops)",
        "CREATE INDEX idx_molecule_ecfp_hnsw      ON public.molecule USING hnsw (ecfp_1024 bit_jaccard_ops)",
    ],
}


# ─── Download helpers ─────────────────────────────────────────────────────
HF_URL_RE = re.compile(r"^https://huggingface\.co/datasets/([^/]+/[^/]+)/?(.*)$")


def _is_gcs(uri: str) -> bool:
    return uri.startswith("gs://")


def _is_hf(uri: str) -> bool:
    return HF_URL_RE.match(uri) is not None


def _hf_parse(uri: str) -> Tuple[str, str]:
    """Return (repo_id, in_repo_path_prefix)."""
    m = HF_URL_RE.match(uri)
    if not m:
        raise ValueError(f"not an HF dataset URL: {uri}")
    return m.group(1), m.group(2).strip("/")


def _gcs_copy(source: str, rel: str, dst: Path, *, recursive: bool, dry: bool) -> None:
    src = f"{source.rstrip('/')}/{rel}"
    cmd = ["gsutil", "-q", "-m", "cp"]
    if recursive:
        cmd.append("-r")
        src = src.rstrip("/") + "/*"
    cmd += [src, str(dst)]
    if dry:
        print(f"  [dry] {' '.join(cmd)}")
        return
    dst.mkdir(parents=True, exist_ok=True) if recursive else dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)


def _https_copy(source: str, rel: str, dst: Path, *, recursive: bool, dry: bool) -> None:
    src = f"{source.rstrip('/')}/{rel}"
    if recursive:
        raise SystemExit(
            f"plain HTTPS source can't recursively download a directory ({src}). "
            "Use a gs:// or HF dataset URL instead, or bundle as tar/zip."
        )
    cmd = ["curl", "-fL", "-o", str(dst), src]
    if dry:
        print(f"  [dry] {' '.join(cmd)}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)


def _hf_copy(source: str, rel: str, dst: Path, *, recursive: bool, dry: bool) -> None:
    """Pull `<rel>` (file) or `<rel>/` (directory tree) from an HF dataset repo."""
    repo_id, prefix = _hf_parse(source)
    in_repo_path = f"{prefix}/{rel}".strip("/") if prefix else rel
    if dry:
        kind = "snapshot" if recursive else "file"
        print(f"  [dry] hf {kind}: {repo_id}:{in_repo_path} → {dst}")
        return
    from huggingface_hub import hf_hub_download, snapshot_download
    if recursive:
        snap_root = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=[f"{in_repo_path}/**"],
        )
        src_dir = Path(snap_root) / in_repo_path
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src_dir, dst)
    else:
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=in_repo_path,
        )
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dst)


def _copy_for(source: str):
    """Return the copy function matching the source URL scheme."""
    if _is_hf(source):
        return _hf_copy
    if _is_gcs(source):
        return _gcs_copy
    return _https_copy


def download_scenario(scenario: str, paths: List[str], source: str, dry: bool) -> None:
    sd = SEMBENCH_ROOT / scenario
    print(f"\n[{scenario}] downloading {len(paths)} item(s) from {source}")
    copy = _copy_for(source)
    for rel in paths:
        is_dir = rel.endswith("/")
        rel_clean = rel.rstrip("/")
        in_source_rel = f"{scenario}/{rel_clean}"
        dst = sd / rel_clean
        copy(source, in_source_rel, dst, recursive=is_dir, dry=dry)


def download_psql_parquets(source: str, local_root: Path, dry: bool) -> None:
    """Pull every <source>/psql/<db>/<table>.parquet → local_root/<db>/<table>.parquet."""
    print(f"\n[psql] downloading parquet dumps from {source}/{PSQL_PREFIX}")
    copy = _copy_for(source)
    for db, spec in PSQL_MANIFEST.items():
        for table in spec["tables"]:
            rel = f"{db}/{table}.parquet"
            in_source_rel = f"{PSQL_PREFIX}/{rel}"
            dst = local_root / rel
            copy(source, in_source_rel, dst, recursive=False, dry=dry)


# ─── PSQL setup ───────────────────────────────────────────────────────────
def _split_db_url(url: str) -> Tuple[str, str]:
    """Return (admin_url_without_dbname, dbname). Used to connect to the
    server without naming a (possibly nonexistent) target db so we can CREATE it."""
    # postgres URL looks like: postgresql://user:pw@host:port/dbname?...params
    # We rewrite the path to /postgres so we can issue CREATE DATABASE.
    head, _, tail = url.rpartition("/")
    db_with_params = tail
    db, _, params = db_with_params.partition("?")
    admin_url = head + "/postgres" + (f"?{params}" if params else "")
    return admin_url, db


def _ensure_database(admin_url: str, dbname: str, dry: bool) -> None:
    import psycopg

    if dry:
        print(f"  [dry] CREATE DATABASE {dbname} (if missing) via {admin_url}")
        return
    conn = psycopg.connect(admin_url, autocommit=True)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
            if cur.fetchone() is None:
                print(f"  CREATE DATABASE {dbname}")
                cur.execute(f'CREATE DATABASE "{dbname}"')
            else:
                print(f"  database {dbname} already exists")
    finally:
        conn.close()


def _load_parquet_into_table(conn, table: str, parquet_path: Path, batch: int = 500) -> None:
    """Stream parquet rows into psql via batched executemany."""
    import numpy as np
    import pyarrow.parquet as pq

    pq_table = pq.read_table(parquet_path)
    schema = pq_table.schema
    n = pq_table.num_rows
    print(f"    loading {n} rows from {parquet_path.name}")

    # Column metadata: psql_udt + psql_char_max (set by dump_psql_to_parquet).
    col_meta = []
    for f in schema:
        meta = dict(f.metadata or {})
        udt = meta.get(b"psql_udt", b"").decode()
        char_max = meta.get(b"psql_char_max")
        col_meta.append((f.name, udt, int(char_max) if char_max else None))

    # Build INSERT SQL with ::bit(N) casts for bit columns.
    placeholders = []
    for _, udt, cmax in col_meta:
        if udt == "bit":
            placeholders.append(f"%s::bit({cmax})")
        else:
            placeholders.append("%s")
    col_list = ", ".join(f'"{c}"' for c, _, _ in col_meta)
    val_list = ", ".join(placeholders)
    insert_sql = f'INSERT INTO public."{table}" ({col_list}) VALUES ({val_list})'

    # Pull rows as dict-of-columns for cheap iteration; convert per row.
    cols_data = pq_table.to_pydict()
    col_names = [c[0] for c in col_meta]

    def _convert(value, udt: str, cmax) -> object:
        if value is None:
            return None
        if udt == "vector":
            return np.asarray(value, dtype=np.float32)
        if udt == "bit":
            # value is bytes (fixed_size_binary). Re-expand to bit string of length cmax.
            bit_str = "".join(f"{b:08b}" for b in value)[:cmax]
            return bit_str
        return value

    t0 = time.perf_counter()
    with conn.cursor() as cur:
        params_buf: List[List] = []
        total = 0
        for i in range(n):
            row = [_convert(cols_data[name][i], udt, cmax)
                   for name, udt, cmax in col_meta]
            params_buf.append(row)
            if len(params_buf) >= batch:
                cur.executemany(insert_sql, params_buf)
                total += len(params_buf)
                params_buf.clear()
                if total % 5000 == 0:
                    print(f"      ...inserted {total}/{n} ({time.perf_counter()-t0:.1f}s)")
        if params_buf:
            cur.executemany(insert_sql, params_buf)
            total += len(params_buf)
    conn.commit()
    print(f"    inserted {total} rows in {time.perf_counter()-t0:.1f}s")


def _build_id_map(conn, table: str, id_col: str) -> None:
    """Create `<table>_id_map` mapping (blkno, offno) → realid for the
    DASE-pgvector fork's filtered-HNSW operator. Mirrors
    ours.utils.build_id_map_table; called inline so setup is self-contained."""
    map_table = f"{table}_id_map"
    print(f"    building id_map → public.{map_table} (id={id_col})")
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(f'DROP TABLE IF EXISTS public."{map_table}"')
        cur.execute(
            f'CREATE TABLE public."{map_table}" ('
            f"  blkno integer NOT NULL,"
            f"  offno integer NOT NULL,"
            f"  realid bigint NOT NULL,"
            f"  PRIMARY KEY (blkno, offno))"
        )
        cur.execute(f'SELECT ctid, "{id_col}" FROM public."{table}"')
        rows = cur.fetchall()
        batch = []
        for ctid, rid in rows:
            ctid_str = str(ctid).strip("()")
            blkno, offno = map(int, ctid_str.split(","))
            batch.append((blkno, offno, rid))
        if batch:
            cur.executemany(
                f'INSERT INTO public."{map_table}" (blkno, offno, realid) VALUES (%s, %s, %s)',
                batch,
            )
    conn.commit()
    print(f"    id_map: {len(batch)} rows in {time.perf_counter()-t0:.1f}s")


def setup_psql_db(db_kind: str, db_url: str, parquet_root: Path, dry: bool) -> None:
    """Create the target db (if missing), CREATE EXTENSION vector, then for each
    table: CREATE TABLE → load parquet → add constraints → build indexes."""
    spec = PSQL_MANIFEST[db_kind]
    admin_url, dbname = _split_db_url(db_url)
    print(f"\n[psql:{db_kind}] target db={dbname} (parquet_root={parquet_root})")

    _ensure_database(admin_url, dbname, dry)
    if dry:
        for table in spec["tables"]:
            print(f"  [dry] would CREATE TABLE + load {parquet_root / db_kind / (table + '.parquet')} + indexes")
        return

    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(db_url)
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        register_vector(conn)

        for table in spec["tables"]:
            parquet_path = parquet_root / db_kind / f"{table}.parquet"
            if not parquet_path.exists():
                raise SystemExit(f"missing parquet: {parquet_path} — run download phase first")

            print(f"  [{table}]")
            with conn.cursor() as cur:
                cur.execute(f'DROP TABLE IF EXISTS public."{table}" CASCADE')
                cur.execute(PSQL_TABLE_DDL[table])
            conn.commit()

            _load_parquet_into_table(conn, table, parquet_path)

            print(f"    adding constraints…")
            with conn.cursor() as cur:
                for ddl in PSQL_TABLE_CONSTRAINTS[table]:
                    cur.execute(ddl)
            conn.commit()

            print(f"    building {len(PSQL_TABLE_INDEXES[table])} index(es)…")
            t0 = time.perf_counter()
            with conn.cursor() as cur:
                for ddl in PSQL_TABLE_INDEXES[table]:
                    name = ddl.split()[2]  # 'CREATE INDEX <name> ON ...'
                    t_i = time.perf_counter()
                    cur.execute(ddl)
                    print(f"      ✓ {name} ({time.perf_counter()-t_i:.1f}s)")
            conn.commit()
            print(f"    {table} done ({time.perf_counter()-t0:.1f}s for indexes)")

            if table in PSQL_ID_MAP_TABLES:
                _build_id_map(conn, table, PSQL_ID_MAP_TABLES[table])
    finally:
        conn.close()


# ─── BigQuery setup ───────────────────────────────────────────────────────
def setup_bq(scenario: str, project: str, dry: bool) -> None:
    setup_script = SEMBENCH_ROOT / scenario / "setup_bq.py"
    if not setup_script.exists():
        print(f"[{scenario}] no setup_bq.py — skipping (cascade scripts handle BQ inline)")
        return
    if not os.environ.get("SEMBENCH_SRC"):
        raise SystemExit(
            "SEMBENCH_SRC env var must point to upstream sembench/src/ checkout. "
            "Run: export SEMBENCH_SRC=/path/to/sembench/src"
        )
    cmd = [sys.executable, str(setup_script)]
    env = {**os.environ, "GCP_PROJECT": project}
    print(f"\n[{scenario}] running setup_bq.py (project={project})")
    if dry:
        print(f"  [dry] GCP_PROJECT={project} {' '.join(cmd)}")
        return
    subprocess.run(cmd, check=True, env=env)


# ─── TI build ─────────────────────────────────────────────────────────────
def build_ti_tables(db_kind: str, dry: bool) -> None:
    """Build all SemJI / TI tables for `db_kind` by shelling out to
    `python -m ours.ti.ti_build` once per (config, threshold) pair.
    ti_build.py materializes inline, so no separate ti_materialize call."""
    plan = TI_BUILD_PLAN.get(db_kind, [])
    if not plan:
        return
    print(f"\n[ti:{db_kind}] building {len(plan)} TI table(s) — large; molecule τ=0.7 ≈ 50 GB")
    for config, threshold in plan:
        cmd = [
            sys.executable, "-m", "ours.ti.ti_build",
            "--config", config,
            "--method", "directhnsw_range",
            "--metric", "l2",
            "--threshold", str(threshold),
        ]
        print(f"  → {config} τ={threshold}")
        if dry:
            print(f"    [dry] {' '.join(cmd)}")
            continue
        subprocess.run(cmd, check=True, cwd=str(ROOT))


# ─── CLI ──────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--source",
        default=os.environ.get("SEMBENCH_DATA_SOURCE", DEFAULT_HF_SOURCE),
        help=f"Data source: HF dataset URL, gs:// URI, or plain HTTPS prefix. "
             f"Default: $SEMBENCH_DATA_SOURCE or {DEFAULT_HF_SOURCE}",
    )
    ap.add_argument(
        "--project",
        default=os.environ.get("GCP_PROJECT"),
        help="GCP project id for BigQuery. Default: $GCP_PROJECT",
    )
    ap.add_argument(
        "--scenarios",
        default=",".join(MANIFEST.keys()),
        help="Comma-separated subset to set up. Default: all 5.",
    )
    ap.add_argument(
        "--imdb-url",
        default=os.environ.get(PSQL_MANIFEST["imdb"]["url_env"], PSQL_MANIFEST["imdb"]["default_url"]),
        help=f"psql URL for imdb db. Default: ${PSQL_MANIFEST['imdb']['url_env']} or {PSQL_MANIFEST['imdb']['default_url']}",
    )
    ap.add_argument(
        "--molecule-url",
        default=os.environ.get(PSQL_MANIFEST["molecule"]["url_env"], PSQL_MANIFEST["molecule"]["default_url"]),
        help=f"psql URL for molecule db. Default: ${PSQL_MANIFEST['molecule']['url_env']} or {PSQL_MANIFEST['molecule']['default_url']}",
    )
    ap.add_argument(
        "--psql-parquet-root",
        type=Path,
        default=ROOT / "psql_dump",
        help=f"Local dir holding downloaded psql parquets. Default: {ROOT / 'psql_dump'}",
    )
    ap.add_argument(
        "--psql-dbs",
        default=",".join(PSQL_MANIFEST.keys()),
        help=f"Comma-separated subset of psql databases. Default: {','.join(PSQL_MANIFEST.keys())}",
    )
    ap.add_argument("--skip-download", action="store_true",
                    help="Skip download phase.")
    ap.add_argument("--skip-psql", action="store_true",
                    help="Skip psql setup phase.")
    ap.add_argument("--skip-ti", action="store_true",
                    help="Skip building SemJI / TI tables after psql setup. "
                         "TI build is slow and large; W5–W8 workloads need them.")
    ap.add_argument("--skip-bq", action="store_true",
                    help="Skip BQ setup phase.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print actions without executing.")
    ap.add_argument("--list-manifest", action="store_true",
                    help="Print the per-scenario file manifest and exit.")
    args = ap.parse_args()

    if args.list_manifest:
        print("# sembench scenarios:")
        for s, paths in MANIFEST.items():
            print(f"{s}:")
            for p in paths:
                print(f"  {p}")
        print("\n# psql tables:")
        for db, spec in PSQL_MANIFEST.items():
            print(f"{db} ({spec['default_url']}):")
            for t in spec["tables"]:
                print(f"  {PSQL_PREFIX}/{db}/{t}.parquet")
        return

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    bad = [s for s in scenarios if s not in MANIFEST]
    if bad:
        ap.error(f"unknown scenarios: {bad}; valid: {list(MANIFEST)}")

    psql_dbs = [s.strip() for s in args.psql_dbs.split(",") if s.strip()]
    bad_dbs = [s for s in psql_dbs if s not in PSQL_MANIFEST]
    if bad_dbs:
        ap.error(f"unknown psql dbs: {bad_dbs}; valid: {list(PSQL_MANIFEST)}")

    # ─── Phase 1: download ──────────────────────────────────────────────
    if not args.skip_download:
        if not args.source:
            ap.error("--source (or $SEMBENCH_DATA_SOURCE) required for download")
        if _is_hf(args.source):
            try:
                import huggingface_hub  # noqa: F401
            except ImportError:
                ap.error("HF source requires `pip install huggingface_hub`")
        elif _is_gcs(args.source):
            if shutil.which("gsutil") is None:
                ap.error("gsutil not found on PATH; needed for gs:// sources")
        else:
            if shutil.which("curl") is None:
                ap.error("curl not found on PATH; needed for https:// sources")
        for s in scenarios:
            download_scenario(s, MANIFEST[s], args.source, args.dry_run)
        if not args.skip_psql and psql_dbs:
            download_psql_parquets(args.source, args.psql_parquet_root, args.dry_run)

    # ─── Phase 2: psql setup (+ TI build) ───────────────────────────────
    if not args.skip_psql:
        urls = {"imdb": args.imdb_url, "molecule": args.molecule_url}
        for db_kind in psql_dbs:
            setup_psql_db(db_kind, urls[db_kind], args.psql_parquet_root, args.dry_run)
            if not args.skip_ti:
                build_ti_tables(db_kind, args.dry_run)

    # ─── Phase 3: BigQuery setup ────────────────────────────────────────
    if not args.skip_bq:
        if not args.project:
            ap.error("--project (or $GCP_PROJECT) required for BQ setup")
        for s in scenarios:
            if s in BQ_SCENARIOS:
                setup_bq(s, args.project, args.dry_run)
            else:
                print(f"[{s}] no centralized BQ setup; cascade scripts create tables on demand")

    print("\nsetup_data.py done.")


if __name__ == "__main__":
    main()
