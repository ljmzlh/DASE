# DASE

**Data Agent Similarity-search Engine** — a top-*k* multi-step reasoning query engine on PostgreSQL.

DASE jointly executes multi-attribute filtering, multi-vector similarity scoring, and cross-table semantic joins, exploiting that semantically-related pairs sit at the **tail** of the embedding-distance distribution. The core artifacts are:

- **SemJI** (Semantic Join Index) — pre-materializes pairs `(id_A, tid_A, id_B, tid_B, dist(A,B))` with `dist ≤ τ`. Realized as the `ti_*` tables.
- **MSA** (Multi-Scoring Aggregator) — TA-style top-*k* over multiple sorted score streams with distribution-aware stream selection.
- **FSS** (Filtered-Score Streamer) — single-signal sorted stream with predicate-aware ANN traversal.
- **Predicate-aware HNSW** — a pgvector fork that adds a filtered-HNSW operator (`<->#`) and 2-hop GUC, used by FSS to walk an HNSW graph under a row-level bitmap.

Paper: *Efficiently Linking Unstructured Data for Multi-step Reasoning* (SIGMOD '27 submission).

This submission repository contains only the DASE implementation, its pgvector
extensions, workloads, ground truth, and DASE-specific experiments and
ablations.

---

## 1. Set up the environment

Download this repository from <https://anonymous.4open.science/r/DASE-2C61/> and unpack it. Then set up the environment in one of two ways:
- **Docker (§1.1).** One image holds PostgreSQL with both DASE extensions, the Python dependencies and the code. You only need Docker.
- **Manual install (§1.2).** You install the Python dependencies, a PostgreSQL server and the two extensions yourself.

The commands in §2 and §3 run from the repository root. With Docker, run them through `docker exec dase`, as shown in §1.1.

### 1.1 Docker

The `Dockerfile` at the repository root builds one image that stands in for the venv, the PostgreSQL server and the two extensions of §1.2. It contains:
- PostgreSQL 18 with the pgvector fork and `dase_batch_hnsw`. The runner is the superuser `postgres`.
- This repository at `/dase`.
- The Python dependencies, pinned in `docker/requirements.txt` to the versions behind the paper's numbers. The image runs Python 3.13; the paper's environment ran 3.11.

The data is not in the image. The container runs the server, and you run DASE commands in it with `docker exec`; §2 then downloads the data and loads it into the volume that holds the database.

```bash
docker build -t dase .

docker run -d --name dase --shm-size=8g -v dase-pg:/var/lib/postgresql dase
docker exec dase python setup_data.py --skip-bq                                  # §2
docker exec dase python -m ours.sys imdb_data/workload/w1/w1_queries_100.json    # §3
```

- **Shared memory.** Keep `--shm-size` above `maintenance_work_mem` (4GB). Parallel HNSW builds keep the graph in `/dev/shm`, and Docker's 64MB default makes them fail.
- **Server settings.** They are in `docker/dase.conf` and match the paper's server except two cache sizes: `shared_buffers` is 4GB (the paper used 20GB) and `effective_cache_size` is 16GB (the paper used 60GB). To use the paper's values:
  ```bash
  docker exec dase psql -h localhost -c "ALTER SYSTEM SET shared_buffers = '20GB'"
  docker restart dase
  ```
- **CPU flags.** The C code is compiled without `-march=native`, so the image runs on any x86-64 CPU. `docker build --build-arg OPTFLAGS=-march=native -t dase .` compiles it for the build machine, as in §1.2.1.
- **Where things are stored.** The database lives in volume `dase-pg` and survives `docker rm`. Downloads, SemJI pair files and results go under `/dase` in the container. Copy results out with `docker cp dase:/dase/imdb_data/results .`, or point `--output` at a bind-mounted directory.
- **Disk.** Plan for about 100 GB under Docker's data root:
  - The imdb database is 19 GB.
  - The molecule database is about 55 GB; its τ = 0.7 SemJI table alone is 50 GB.
  - WAL takes up to 16 GB (`max_wal_size`).
  - Downloads and SemJI pair files take about 10 GB. After §2 you can free 4.5 GB with `docker exec dase rm -rf /dase/psql_dump /root/.cache/huggingface`.
- **Not covered.** The BigQuery cascade (§3.3) needs your GCP credentials and an upstream SemBench checkout, which the image does not contain. Mount both and pass `SEMBENCH_SRC` and `GCP_PROJECT` with `-e`.
- **Connecting from outside.** Publish the port and set a password on the first `docker run`: `-p 5432:5432 -e POSTGRES_PASSWORD=...`. Inside the container, connections need no password.

### 1.2 Manual install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

You also need:
- A local **PostgreSQL 15+** server. The runner role must be a **superuser** (the role a fresh `initdb` creates is one). `setup_data.py` and the SemJI scripts run `CREATE EXTENSION` for two C extensions that are not marked trusted (`vector` and `dase_batch_hnsw`), and the Table 4 script sets `log_min_messages`; both need a superuser.
- The **DASE pgvector fork** installed as the `vector` extension (§1.2.1).
- The **`dase_batch_hnsw`** extension (§1.2.2), for the shared-traversal row of Table 4 and `setup_data.py --ti-method shared`.
- For the SemBench cascade track, a GCP project with BigQuery enabled.

#### 1.2.1 Install the DASE pgvector fork

DASE uses extra HNSW machinery not in upstream pgvector — the `<->#` filtered-search operator, the `hnsw.id_map_table` GUC, the `hnsw.enable_2hop` GUC, and the `hnsw_set_tid_map_table()` SQL function. Build and install the fork **before** running `setup_data.py`:

```bash
# from the dase repo root, the fork lives at pgvector/
cd pgvector
make
sudo make install
```

Then `setup_data.py` will pick it up via `CREATE EXTENSION vector` automatically. To verify, after setup:

```sql
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
-- expect 0.8.5+ (DASE fork). Upstream releases stop at 0.8.0.

SELECT oprname FROM pg_operator
WHERE oprname IN ('<->#', '<->@');
-- both must exist for filtered-HNSW to work.
```

If `<->#` is missing, the standard pgvector got installed instead — uninstall it (`DROP EXTENSION vector CASCADE` on each db), `make uninstall` upstream pgvector, and rebuild the fork.

#### 1.2.2 Install the `dase_batch_hnsw` extension

`batch_hnsw/` implements the shared traversal used to build SemJI (the last row of paper Table 4): one C function runs every HNSW range probe over the left rows in k-means locality order, and each probe after the first in a group starts from the previous probe's nearest nodes instead of descending the upper layers. It compiles the fork's HNSW code from `pgvector/`, so build it from the same checkout and against the same server as §1.2.1:

```bash
cd batch_hnsw
make
sudo make install   # add PG_CONFIG=/path/to/pg_config if the pg_config on PATH belongs to another server
```

There is nothing to create by hand: the scripts that use it run `CREATE EXTENSION IF NOT EXISTS dase_batch_hnsw` themselves. The extension requires `vector` in the same database and puts its functions in schema `bh`. To verify:

```sql
CREATE EXTENSION IF NOT EXISTS dase_batch_hnsw;
SELECT proname FROM pg_proc WHERE pronamespace = 'bh'::regnamespace;
-- expect range_join_all and last_stats
```

---

## 2. Set up data

IMDB, MOLECULE, and SemBench data live at the public HF dataset
[`dasepaper/dase_data`](https://huggingface.co/datasets/dasepaper/dase_data).
`setup_data.py` handles their download, PostgreSQL setup, SemJI construction,
and optional BigQuery setup. NFCorpus uses three separate parquet exports and
is loaded only when `--nfcorpus-parquet-dir` is supplied.

```bash
# (optional) override defaults — env vars also read by ours/sys.py
export IMDB_DATABASE_URL=postgresql://user:pw@host:5432/imdb
export MOLECULE_DATABASE_URL=postgresql://user:pw@host:5432/molecule
export NFCORPUS_DATABASE_URL=postgresql://user:pw@host:5432/nfcorpus

# (BigQuery only) upstream sembench checkout + GCP creds
export SEMBENCH_SRC=/path/to/sembench/src
gcloud auth application-default login

# All three phases (download → psql → bq)
python setup_data.py --project <your-gcp-project>

# Just DASE/psql, no BigQuery
python setup_data.py --skip-bq

# Just download
python setup_data.py --skip-psql --skip-bq

# Add NFCorpus after obtaining its three parquet exports
python setup_data.py --skip-download --skip-bq --psql-dbs '' \
	--nfcorpus-parquet-dir /path/to/nfcorpus-parquets
```

`setup_data.py` does, in order:
1. Pull HF dataset → `psql_dump/` (parquets) + `sembench_data/<scenario>/`.
2. `CREATE DATABASE imdb` and `CREATE DATABASE molecule`; `CREATE EXTENSION vector`; `CREATE TABLE` + `COPY` parquet rows; build btree/gin/HNSW indexes; build `<table>_id_map` lookup tables for the four `*_hnsw` base tables. Then build the SemJI / TI tables (`ti_imdb_t1_imdb_t2_{0.5,0.6}`, `ti_facts_50k_paper_{0.5,0.6,0.7}`) — these back W5–W8 cross-table joins. The molecule τ=0.7 TI is ~50 GB and slow; pass `--skip-ti` to skip if you only need W1–W4. By default the TI tables are built with `ours.ti.ti_build --method directhnsw_range`, the build behind the paper's W5–W8 results. `--ti-method shared` builds them with the shared traversal instead (`ours.ti.build_join_index_A_B`, needs §1.2.2): faster, but its approximate search returns a slightly different set of pairs.
3. If requested, load NFCorpus's paper table and two real link relations.
4. Run scenario `setup_bq.py` for `wildlife / mmqa / cars` (the other two cascade scenarios create BQ tables on first run).

For all flags, see `python setup_data.py --help`.

---

## 3. Usage

### 3.1 IMDB / MOLECULE (DASE workloads)

Eight workload classes (paper Table 2):

| WL | 𝒫 (predicates) | Multi-S | 𝒥 (join) | Pattern | Expression |
|---|---|---|---|---|---|
| W1 |  ✗ | ✗ | ✗ | semantic retrieval | `TopK{FSS(∅, s)}` |
| W2 |  ✓ | ✗ | ✗ | filtered retrieval | `TopK{FSS(P, s)}` |
| W3 |  ✗ | ✓ | ✗ | multi-scoring retrieval | `MSA({FSS(∅, sᵢ)}, scoref, K)` |
| W4 |  ✓ | ✓ | ✗ | filtered multi-scoring | `MSA({FSS(P, sᵢ)}, scoref, K)` |
| W5 |  ✗ | ✗ | ✓ | semantic join | `TopK{FSS(P_𝒥, s)}` |
| W6 |  ✓ | ✗ | ✓ | filtered join | `TopK{FSS(P ∪ P_𝒥, s)}` |
| W7 |  ✗ | ✓ | ✓ | multi-scoring join | `MSA({FSS(P_𝒥, sᵢ)}, scoref, K)` |
| W8 |  ✓ | ✓ | ✓ | filtered multi-scoring join | `MSA({FSS(P ∪ P_𝒥, sᵢ)}, scoref, K)` |

Run a workload on either scenario:

```bash
# IMDB W1 — semantic retrieval, single signal
python -m ours.sys imdb_data/workload/w1/w1_queries_100.json

# IMDB W3 — multi-scoring with custom TA slack
python -m ours.sys imdb_data/workload/w3/w3_queries_100.json --eps 0.02

# MOLECULE W5 — cross-table semantic join (requires TI tables)
python -m ours.sys molecule_data/workload/w5/w5_queries_100.json

# Only first N queries
python -m ours.sys imdb_data/workload/w7/w7_queries_100.json --limit 10
```

**Output paths**:
- Per-run results JSON → `<scenario>_data/results/wX/results_dase_wX_queries_100_<run_id>.json`
- Setting snapshot → `<scenario>_data/logs/<run_id>.setting.json`
- Ground-truth queries: `<scenario>_data/results/wX/wX_queries_100_gt.json`

Generated results and logs are intentionally not tracked.

### 3.2 DASE experiments and ablations

| Paper result | Entry point |
|---|---|
| Table 4, SemJI construction | `python -m experiments.semji_construction --help` |
| Table 5, predicate-aware HNSW | `python -m experiments.filtered_search --help` |
| Table 6, SemJI ablation | `python -m experiments.ablate_semji --help` |
| Table 6, stream selection | `python -m experiments.ablate_stream_selection --help` |
| Figure 7, SemJI evolution | `python -m experiments.semji_evolution --help` |
| Figure 8, DASE depth scaling | `python -m experiments.nfcorpus_depth --help` |
| Table 7, anisotropy robustness | `python -m experiments.semji_robustness --help` |

The Figure 8 entry point contains only the DASE directed-edge top-*n* dynamic
program.

The Table 4 script builds the SemJI pair table of `imdb_t1_hnsw × imdb_t2_hnsw`
on `actor_director_emb` by default (τ = 0.5, ef_search = 40). `--methods` picks
the rows: `exact`, `iterative`, `batched`, `vectorized`, and `shared`, the
shared traversal of §1.2.2, whose time includes its k-means ordering step.
Recall is reported against `exact` when it runs in the same invocation.

```bash
# all five rows with recall; exact alone takes over an hour on the full table
python -m experiments.semji_construction
# time only
python -m experiments.semji_construction --methods vectorized shared
```

`--left-limit N` restricts the left side to its first N rows. That is fine for
the four SQL methods but not for `shared`: its k-means groups need the whole
left table, and on a sparse subset the warm starts land far from the probe (on
the first 500 rows its recall drops to 0.71).

---

### 3.3 SemBench cascade

DASE acts as a high-recall **prefilter** for BigQuery's semantic operators (`AI.IF`, `AI.CLASSIFY`, `AI.SCORE`, `AI.GENERATE`). Five scenarios from the upstream SemBench benchmark: `wildlife`, `mmqa`, `ecomm`, `cars`, `movie`. Each Q has two implementations:

- `qN_nn.py` — DASE / NN only (embedding-distance verdicts, no LLM).
- `qN_cascade.py` — DASE prefilter + BigQuery LLM stage; produces a profile JSON consumed by the per-scenario rollup.

```bash
# wildlife Q1 — DASE-only embedding prefilter
python sembench_data/wildlife/scripts/q1_nn.py

# wildlife Q1 — DASE + BigQuery cascade
python sembench_data/wildlife/scripts/q1_cascade.py

# ecomm Q10 — composed F+J (image-pair join with role classification)
python sembench_data/ecomm/scripts/q10_cascade.py

# mmqa Q3a — sub-question variant
python sembench_data/mmqa/scripts/q3a_cascade.py
```

**Output paths**:
- Per-Q profile JSON → `sembench_data/<scenario>/outputs/QN.json`
- DASE Stage-1 retry caches for E-Commerce Q10/Q11 → `sembench_data/ecomm/outputs/QN_stage1_cache.json`
- Generated profiles, stage caches, and summaries are local artifacts and are ignored by Git.

**Per-scenario rollup** — after all Qs in a scenario are done, build the summary:

```bash
python sembench_data/wildlife/build_cascade_summary.py
python sembench_data/mmqa/build_cascade_summary.py
python sembench_data/ecomm/build_cascade_summary.py
python sembench_data/cars/build_cascade_summary.py
python sembench_data/movie/build_cascade_summary.py
```

Output: `sembench_data/<scenario>/outputs/cascade_summary.csv` — measured DASE score, latency, cost, verifier calls, and slot use per query.
