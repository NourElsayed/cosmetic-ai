import csv
import hashlib
import json
import time
from pathlib import Path

import requests


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_PATH = (
    PROJECT_ROOT
    / "raw_datasets"
    / "FFHQ"
    / "ffhq-dataset-v2.json"
)

SELECTED_PATH = (
    PROJECT_ROOT
    / "raw_datasets"
    / "FFHQ"
    / "selected_first_5000.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "raw_datasets"
    / "FFHQ"
    / "images1024x1024"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Settings
# ============================================================

TIMEOUT = 120
MAX_RETRIES = 3

# Wait a little between downloads
SLEEP_SECONDS = 0.5


# ============================================================
# Load metadata
# ============================================================

print("Loading FFHQ metadata...")

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print(f"Total metadata entries: {len(metadata)}")


# ============================================================
# Load selected IDs
# ============================================================

selected_ids = []

with open(SELECTED_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        selected_ids.append(row["ffhq_id"])

print(f"Selected images: {len(selected_ids)}")


# ============================================================
# Download function
# ============================================================

def download_image(image_id, image_info):

    output_path = OUTPUT_DIR / f"{int(image_id):05d}.png"

    expected_md5 = image_info["file_md5"]
    url = image_info["file_url"]

    # --------------------------------------------------------
    # If file already exists, verify it instead of downloading
    # --------------------------------------------------------

    if output_path.exists():

        print(f"[SKIP] {image_id} already exists. Checking MD5...")

        md5 = hashlib.md5()

        with open(output_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                md5.update(chunk)

        if md5.hexdigest() == expected_md5:
            print(f"[OK] {image_id} already valid.")
            return True

        print(f"[WARNING] {image_id} exists but MD5 is incorrect.")
        print("Re-downloading...")


    # --------------------------------------------------------
    # Download with retries
    # --------------------------------------------------------

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            print(
                f"[DOWNLOAD] {image_id} "
                f"(attempt {attempt}/{MAX_RETRIES})"
            )

            response = requests.get(
                url,
                stream=True,
                timeout=TIMEOUT
            )

            response.raise_for_status()

            md5 = hashlib.md5()

            temp_path = output_path.with_suffix(".tmp")

            with open(temp_path, "wb") as f:

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):

                    if chunk:

                        f.write(chunk)
                        md5.update(chunk)


            actual_md5 = md5.hexdigest()


            # ------------------------------------------------
            # Verify MD5
            # ------------------------------------------------

            if actual_md5 != expected_md5:

                print(
                    f"[ERROR] {image_id}: MD5 mismatch!"
                )

                temp_path.unlink(missing_ok=True)

                continue


            # ------------------------------------------------
            # Rename only after successful verification
            # ------------------------------------------------

            temp_path.replace(output_path)

            print(f"[OK] {image_id}")

            return True


        except Exception as e:

            print(
                f"[ERROR] {image_id}: {e}"
            )

            temp_path = output_path.with_suffix(".tmp")
            temp_path.unlink(missing_ok=True)

            if attempt < MAX_RETRIES:
                time.sleep(2)


    print(f"[FAILED] {image_id}")

    return False


# ============================================================
# Start downloading
# ============================================================

successful = 0
failed = []

total = len(selected_ids)

print()
print("=" * 60)
print("Starting FFHQ download")
print(f"Total images: {total}")
print("=" * 60)
print()


for index, image_id in enumerate(selected_ids, start=1):

    print()
    print(f"Progress: {index}/{total}")

    image_info = metadata[image_id]["image"]

    success = download_image(
        image_id,
        image_info
    )

    if success:
        successful += 1
    else:
        failed.append(image_id)

    time.sleep(SLEEP_SECONDS)


# ============================================================
# Final report
# ============================================================

print()
print("=" * 60)
print("DOWNLOAD COMPLETED")
print("=" * 60)

print(f"Successful: {successful}")
print(f"Failed:     {len(failed)}")

if failed:

    print()
    print("Failed IDs:")

    for image_id in failed:
        print(image_id)

    failed_path = (
        PROJECT_ROOT
        / "raw_datasets"
        / "FFHQ"
        / "failed_downloads.txt"
    )

    with open(failed_path, "w", encoding="utf-8") as f:

        for image_id in failed:
            f.write(f"{image_id}\n")

    print()
    print(f"Failed IDs saved to: {failed_path}")

else:

    print()
    print("All 5,000 images downloaded successfully.")