#!/usr/bin/env -S python -u
"""
MMQA BQ setup — wraps sembench's BigQueryMMQASetup with our PROJECT_ID.

Steps performed:
  1. BQ dataset: mmqa (created if missing, location=US)
  2. Upload 5 CSV tables: ap_warrior, ben_piazza, ben_piazza_text_data,
     lizzy_caplan_text_data, tampa_international_airport
  3. GCS bucket: <YOUR_GCP_PROJECT>-mmqa-images (created if missing)
  4. Upload 200 images -> gs://.../*.jpg
  5. EXTERNAL TABLE: mmqa.images (object_metadata='SIMPLE',
     WITH CONNECTION us.connection)

After this, the same q*.sql templates that sembench mmqa/query/bigquery use work
verbatim (just need us.connection).
"""
import os
import sys

# SEMBENCH_SRC must point to the upstream sembench `src/` directory
# (not bundled in this repo); set via env var.
SEMBENCH_SRC = os.environ.get("SEMBENCH_SRC")
if not SEMBENCH_SRC:
    raise RuntimeError(
        "SEMBENCH_SRC env var must point to upstream sembench/src "
        "(provides scenario.mmqa.setup.bigquery)"
    )
sys.path.insert(0, SEMBENCH_SRC)
import scenario.mmqa.setup.bigquery as bq_setup_mod  # noqa: E402

setup = bq_setup_mod.BigQueryMMQASetup()
setup.bq_client = bq_setup_mod.bigquery.Client(project=os.environ.get("GCP_PROJECT", ""))
print(f"BQ project: {setup.bq_client.project}")

MMQA_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(MMQA_DIR, "cache", "sf_200")
print(f"data_dir = {DATA_DIR}")
print()

setup.setup_data(data_dir=DATA_DIR)

print()
print("Setup complete. Verify with:")
print("  bq ls <YOUR_GCP_PROJECT>:mmqa")
print("  bq head -n 3 <YOUR_GCP_PROJECT>:mmqa.lizzy_caplan_text_data")
