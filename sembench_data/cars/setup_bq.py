#!/usr/bin/env -S python -u
"""
Cars BQ setup — wraps sembench's BigQueryCarsSetup with our PROJECT_ID.

Why we override setup_data:
  sembench cars/setup/bigquery.py hardcodes
    bucket="gs://bq-mm-benchmark-cars_dataset/car_images/*"
  in the finalize_image_upload() calls — so external tables would point at
  the original benchmark project's bucket, not ours. We pass the correct
  per-bucket URI via self.gcs_bucket_name (animals does this right; cars
  doesn't).

Steps:
  1. Symlink files/cars/data/all_car_{images,audio} -> cache/<same>
     (CSVs use relative paths; resolves them at upload time)
  2. GCS bucket: <YOUR_GCP_PROJECT>-cars_dataset (created if missing)
  3. Upload images -> gs://.../car_images/    (3765 @ sf=19672)
  4. Upload audios -> gs://.../car_audios/    ( 175 @ sf=19672)
  5. BQ dataset:  cars_dataset
  6. BQ tables:   car_images, cars_images(external), car_mm
                  car_audio,  cars_audios(external), audio_mm
                  cars (native), complaints (native)

Run with a bigquery-enabled python (env must include google-cloud-bigquery):
  SEMBENCH_SRC=<path-to-upstream-sembench-src> \
      python -m sembench_data.cars.setup_bq
"""
import os
import sys

# Patch PROJECT_ID before importing the module's classes.
# SEMBENCH_SRC must point to the upstream sembench `src/` directory
# (not bundled in this repo); set via env var.
SEMBENCH_SRC = os.environ.get("SEMBENCH_SRC")
if not SEMBENCH_SRC:
    raise RuntimeError(
        "SEMBENCH_SRC env var must point to upstream sembench/src "
        "(provides scenario.cars.setup.bigquery)"
    )
sys.path.insert(0, SEMBENCH_SRC)
import scenario.cars.setup.bigquery as bq_setup_mod  # noqa: E402
from google.cloud import bigquery  # noqa: E402

bq_setup_mod.PROJECT_ID = os.environ.get("GCP_PROJECT", "")
print(f"Patched PROJECT_ID = {bq_setup_mod.PROJECT_ID}")

# ── paths ────────────────────────────────────────────────────────────
CARS_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(CARS_DIR, "cache")
SCALE_FACTOR = 19672

# Symlink so CSV's relative `files/cars/data/all_car_*` paths resolve.
# Override via SYM_ROOT env if your CSVs use a different relative root.
SYM_ROOT = os.environ.get(
    "SYM_ROOT", os.path.join(CARS_DIR, "..", "..", "files", "cars", "data")
)
os.makedirs(SYM_ROOT, exist_ok=True)
for name in ("all_car_images", "all_car_audio"):
    src = os.path.join(CACHE_DIR, name)
    dst = os.path.join(SYM_ROOT, name)
    if os.path.islink(dst) or os.path.exists(dst):
        print(f"symlink already present: {dst}")
    else:
        os.symlink(src, dst)
        print(f"symlink {dst} -> {src}")


class CarsSetupFixed(bq_setup_mod.BigQueryCarsSetup):
    """Override setup_data to pass correct bucket URIs."""

    def setup_data(self, scale_factor=SCALE_FACTOR, data_dir=CACHE_DIR):
        actual_data_dir = os.path.join(data_dir, f"sf_{scale_factor}")
        gcs = self.gcs_bucket_name  # <YOUR_GCP_PROJECT>-cars_dataset
        ds_id = bq_setup_mod.BQ_DATASET_ID

        # BQ dataset
        dataset_ref = f"{self.bq_client.project}.{ds_id}"
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        self.bq_client.create_dataset(dataset, exists_ok=True)
        print(f"Dataset {dataset_ref} ready.")

        # ── images ───────────────────────────────────────────────────
        # Upload (native car_images) and finalize (external cars_images + car_mm)
        # are gated separately so a 403 / partial run can be retried granularly.
        image_csv = os.path.join(actual_data_dir, f"image_car_data_{scale_factor}.csv")
        if not self.is_data_synchronized(image_csv, "car_images"):
            print(f"\n=== Uploading car images (SF={scale_factor}) ===")
            self.upload_images(local_path=image_csv, gcs_folder="car_images",
                               path_col="image_path", table_id="car_images")
            self.mark_data_synchronized(image_csv)
        else:
            print(f"car_images: synced, skip upload.")
        if not self.table_exists("cars_images") or not self.table_exists("car_mm"):
            print(f"=== Finalizing cars_images + car_mm ===")
            self.finalize_image_upload(
                table_name="cars_images", table_name_multimodal="car_mm",
                image_url_table="car_images", url_col="image_path",
                bucket=f"gs://{gcs}/car_images/*",
            )
        else:
            print(f"cars_images + car_mm: exist, skip.")

        # ── audio ────────────────────────────────────────────────────
        audio_csv = os.path.join(actual_data_dir, f"audio_car_data_{scale_factor}.csv")
        if not self.is_data_synchronized(audio_csv, "car_audio"):
            print(f"\n=== Uploading car audio (SF={scale_factor}) ===")
            self.upload_images(local_path=audio_csv, gcs_folder="car_audios",
                               path_col="audio_path", table_id="car_audio")
            self.mark_data_synchronized(audio_csv)
        else:
            print(f"car_audio: synced, skip upload.")
        if not self.table_exists("cars_audios") or not self.table_exists("audio_mm"):
            print(f"=== Finalizing cars_audios + audio_mm ===")
            self.finalize_image_upload(
                table_name="cars_audios", table_name_multimodal="audio_mm",
                image_url_table="car_audio", url_col="audio_path",
                bucket=f"gs://{gcs}/car_audios/*",
            )
        else:
            print(f"cars_audios + audio_mm: exist, skip.")

        # ── cars (native) ────────────────────────────────────────────
        cars_csv = os.path.join(actual_data_dir, f"car_data_{scale_factor}.csv")
        if os.path.exists(cars_csv):
            if not self.is_data_synchronized(cars_csv, "cars") or not self.table_exists("cars"):
                print(f"\n=== Uploading cars (SF={scale_factor}) ===")
                self.upload_csv_to_bigquery(ds_id, csv_file_path=cars_csv, table_name="cars")
                self.mark_data_synchronized(cars_csv)
            else:
                print(f"cars: synced, skip.")
        else:
            print(f"WARN: {cars_csv} missing, skipping cars.")

        # ── complaints (native) ──────────────────────────────────────
        comp_csv = os.path.join(actual_data_dir, f"text_complaints_data_{scale_factor}.csv")
        if os.path.exists(comp_csv):
            if not self.is_data_synchronized(comp_csv, "complaints") or not self.table_exists("complaints"):
                print(f"\n=== Uploading complaints (SF={scale_factor}) ===")
                self.upload_csv_to_bigquery(ds_id, csv_file_path=comp_csv, table_name="complaints")
                self.mark_data_synchronized(comp_csv)
            else:
                print(f"complaints: synced, skip.")
        else:
            print(f"WARN: {comp_csv} missing, skipping complaints.")

        print("\nCars BQ setup completed.")


def main():
    setup = CarsSetupFixed()
    print(f"GCS bucket: {setup.gcs_bucket_name}")
    print(f"BQ project: {setup.bq_client.project}")
    print()
    setup.setup_data(scale_factor=SCALE_FACTOR, data_dir=CACHE_DIR)
    print()
    print("Verify with:")
    print("  bq ls <YOUR_GCP_PROJECT>:cars_dataset")
    print("  bq head -n 3 <YOUR_GCP_PROJECT>:cars_dataset.car_mm")
    print("  bq head -n 3 <YOUR_GCP_PROJECT>:cars_dataset.audio_mm")


if __name__ == "__main__":
    main()
