# DASE

**Data Agent Similarity-search Engine** — a top-*k* multi-step reasoning query engine on PostgreSQL.

DASE jointly executes multi-attribute filtering, multi-vector similarity scoring, and cross-table semantic joins, exploiting that semantically-related pairs sit at the **tail** of the embedding-distance distribution. The core artifacts are:

- **SemJI** (Semantic Join Index) — pre-materializes pairs `(id_A, tid_A, id_B, tid_B, dist(A,B))` with `dist ≤ τ`. Realized as the `ti_*` tables.
- **MSA** (Multi-Scoring Aggregator) — TA-style top-*k* over multiple sorted score streams with distribution-aware stream selection.
- **FSS** (Filtered-Score Streamer) — single-signal sorted stream with predicate-aware ANN traversal.
- **Predicate-aware HNSW** — a pgvector fork that adds a filtered-HNSW operator (`<->#`) and 2-hop GUC, used by FSS to walk an HNSW graph under a row-level bitmap.

Paper: *Efficiently Linking Unstructured Data for Multi-step Reasoning* (VLDB).

---

## 1. Clone & set up env

```bash
git clone https://github.com/ljmzlh/DASE.git dase
cd dase

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

You also need:
- A local **PostgreSQL 15+** server. The runner role must have `CREATEDB`.
- The **DASE pgvector fork** installed as the `vector` extension (see below).
- For the SemBench cascade track, a GCP project with BigQuery enabled.

### 1.1 Install the DASE pgvector fork

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

---

## 2. Set up data

All data lives at the public HF dataset [`dasepaper/dase_data`](https://huggingface.co/datasets/dasepaper/dase_data). One script handles download, psql tables (with HNSW indexes + `<table>_id_map` lookups for filtered HNSW), and BigQuery setup.

```bash
# (optional) override defaults — env vars also read by ours/sys.py
export IMDB_DATABASE_URL=postgresql://user:pw@host:5432/imdb
export MOLECULE_DATABASE_URL=postgresql://user:pw@host:5432/molecule

# (BigQuery only) upstream sembench checkout + GCP creds
export SEMBENCH_SRC=/path/to/sembench/src
gcloud auth application-default login

# All three phases (download → psql → bq)
python setup_data.py --project <your-gcp-project>

# Just DASE/psql, no BigQuery
python setup_data.py --skip-bq

# Just download
python setup_data.py --skip-psql --skip-bq
```

`setup_data.py` does, in order:
1. Pull HF dataset → `psql_dump/` (parquets) + `sembench_data/<scenario>/`.
2. `CREATE DATABASE imdb` and `CREATE DATABASE molecule`; `CREATE EXTENSION vector`; `CREATE TABLE` + `COPY` parquet rows; build btree/gin/HNSW indexes; build `<table>_id_map` lookup tables for the four `*_hnsw` base tables. Then build the SemJI / TI tables (`ti_imdb_t1_imdb_t2_{0.5,0.6}`, `ti_facts_50k_paper_{0.5,0.6,0.7}`) — these back W5–W8 cross-table joins. The molecule τ=0.7 TI is ~50 GB and slow; pass `--skip-ti` to skip if you only need W1–W4.
3. Run scenario `setup_bq.py` for `wildlife / mmqa / cars` (the other two cascade scenarios create BQ tables on first run).

For all flags, see `python setup_data.py --help`.

---

## 3. Usage

### 3.1 IMDB / MOLECULE (DASE workloads)

Eight workload classes (paper Table 1):

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
- Pre-existing run artefacts checked into the repo: `<scenario>_data/results/wX/2026XXXX_XXXXXX.json`
- Ground-truth queries: `<scenario>_data/results/wX/wX_queries_100_gt.json`

**Eval notebooks** — one per (scenario, workload), alongside the result JSONs:

```
imdb_data/results/wX/eval_wX_results.ipynb
molecule_data/results/wX/eval_wX_results.ipynb
```

Each notebook reads the GT file + the latest results JSON in the same directory and reports recall / QPS as in paper Table 2.

---

### 3.2 SemBench cascade

DASE acts as a high-recall **prefilter** for BigQuery's semantic operators (`AI.IF`, `AI.CLASSIFY`, `AI.SCORE`, `AI.GENERATE`). Five scenarios from the upstream SemBench benchmark: `wildlife`, `mmqa`, `ecomm`, `cars`, `movie`. Each Q has two implementations:

- `qN_nn.py` — DASE / NN only (embedding-distance verdicts, no LLM).
- `qN_cascade.py` — DASE prefilter + BigQuery LLM stage; produces a profile JSON consumed by the per-scenario rollup.

```bash
# wildlife Q1 — DASE-only baseline
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
- BQ retry caches → `sembench_data/<scenario>/outputs/QN_baseline_cache.json`, `QN_stage1_cache.json`
- All pre-existing runs are checked into the repo at the same paths.

**Per-scenario rollup** — after all Qs in a scenario are done, build the summary:

```bash
python sembench_data/wildlife/build_cascade_summary.py
python sembench_data/mmqa/build_cascade_summary.py
python sembench_data/ecomm/build_cascade_summary.py
python sembench_data/cars/build_cascade_summary.py
python sembench_data/movie/build_cascade_summary.py
```

Output: `sembench_data/<scenario>/outputs/cascade_summary.csv` — the canonical cost/quality/latency table per Q (matches paper Table 3 columns).
