#!/usr/bin/env -S python -u
"""
Wildlife BQ setup — wraps sembench's BigQueryAnimalsSetup with our PROJECT_ID.

Steps performed:
  1. GCS bucket: <YOUR_GCP_PROJECT>-animals_dataset (created if missing)
  2. Upload 200 images -> gs://.../animal_images/
  3. Upload 66 audios  -> gs://.../animal_audio/
  4. BQ dataset: animals_dataset (created if missing, location=US)
  5. BQ tables: image_data_images, audio_data_files (metadata + GCS URI)
  6. EXTERNAL TABLEs: image_data_external, audio_data_external
       (object_metadata='SIMPLE', WITH CONNECTION us.connection)
  7. MM tables: image_data_mm, audio_data_mm (JOIN metadata + ext refs)

After this, the same Q1.sql template that sembench bigquery/Q1.sql uses works
verbatim (just needs us.connection).
"""
import os
import sys

# Patch PROJECT_ID in sembench's animals/setup/bigquery before importing it.
# SEMBENCH_SRC must point to the upstream sembench `src/` directory
# (not bundled in this repo); set via env var.
SEMBENCH_SRC = os.environ.get("SEMBENCH_SRC")
if not SEMBENCH_SRC:
    raise RuntimeError(
        "SEMBENCH_SRC env var must point to upstream sembench/src "
        "(provides scenario.animals.setup.bigquery)"
    )
sys.path.insert(0, SEMBENCH_SRC)
import scenario.animals.setup.bigquery as bq_setup_mod  # noqa: E402

bq_setup_mod.PROJECT_ID = os.environ.get("GCP_PROJECT", "")
print(f"Patched PROJECT_ID = {bq_setup_mod.PROJECT_ID}")

# Use the patched module
setup = bq_setup_mod.BigQueryAnimalsSetup()
print(f"GCS bucket: {setup.gcs_bucket_name}")
print(f"BQ project: {setup.bq_client.project}")

WILDLIFE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WILDLIFE_DIR, "bq_data")
print(f"data_dir = {DATA_DIR} (CSVs sym-linked from wildlife/cache/)")
print()

setup.setup_data(scale_factor=200, data_dir=DATA_DIR)

print()
print("Setup complete. Verify with:")
print("  bq ls <YOUR_GCP_PROJECT>:animals_dataset")
print("  bq head -n 3 <YOUR_GCP_PROJECT>:animals_dataset.image_data_mm")
