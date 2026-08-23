from pathlib import Path
from collections import Counter
import csv
import random

from PIL import Image, UnidentifiedImageError


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\cosmetic-ai")

RAW_DATASETS_DIR = (
    PROJECT_ROOT / "raw_datasets"
)

INVENTORY_PATH = (
    RAW_DATASETS_DIR / "dataset_inventory.csv"
)


# ============================================================
# DATASETS
# ============================================================

DATASETS = [
    "CelebAMask-HQ",
    "FaceSynthetics",
    "FFHQ",
    "SCUT-FBP5500",
]


# ============================================================
# SETTINGS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

RESOLUTION_SAMPLE_SIZE = 100
RANDOM_SEED = 42


# ============================================================
# DATASET INFORMATION
# ============================================================

DATASET_INFO = {

    "CelebAMask-HQ": {
        "source": (
            "https://mmlab.ie.cuhk.edu.hk/projects/"
            "CelebA/CelebAMask_HQ.html"
        ),
        "purpose": (
            "Face parsing and facial-region segmentation."
        ),
        "annotations": (
            "Facial component segmentation masks, "
            "attribute annotations, pose annotations, "
            "and face-image mapping information."
        ),
    },

    "FaceSynthetics": {
        "source": (
            "https://www.microsoft.com/en-us/research/"
            "publication/high-fidelity-face-synthetics/"
        ),
        "purpose": (
            "Synthetic facial images for face-analysis research."
        ),
        "annotations": (
            "Landmark and segmentation-related annotations."
        ),
    },

    "FFHQ": {
        "source": (
            "https://github.com/NVlabs/ffhq-dataset"
        ),
        "purpose": (
            "Natural face images used as before-face images."
        ),
        "annotations": (
            "Dataset metadata associated with the official distribution."
        ),
    },

    "SCUT-FBP5500": {
        "source": (
            "https://github.com/HCIILAB/"
            "SCUT-FBP5500-Database-Release"
        ),
        "purpose": (
            "Facial landmark and beauty-score research data."
        ),
        "annotations": (
            "Facial landmarks, beauty ratings, "
            "image-source information, and train/test split files."
        ),
    },
}


# ============================================================
# HELPERS
# ============================================================

def is_image(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def count_all_files(dataset_dir: Path) -> int:
    """
    Count every file in the original dataset folder.
    """

    return sum(
        1
        for path in dataset_dir.rglob("*")
        if path.is_file()
    )


def format_extensions(dataset_dir: Path) -> str:
    """
    Record extensions of all files in the dataset.
    """

    counter = Counter()

    for path in dataset_dir.rglob("*"):

        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        if suffix:
            counter[suffix] += 1
        else:
            counter["[no extension]"] += 1

    if not counter:
        return "none"

    return "; ".join(
        f"{ext} ({count})"
        for ext, count
        in counter.most_common()
    )


# ============================================================
# DATASET-SPECIFIC PRIMARY IMAGE FINDERS
# ============================================================

def find_celebamask_images(
    dataset_dir: Path
):
    """
    CelebAMask-HQ:
    primary face images are in CelebA-HQ-img.
    """

    image_dir = (
        dataset_dir / "CelebA-HQ-img"
    )

    if not image_dir.exists():
        return []

    return sorted(
        path
        for path in image_dir.rglob("*")
        if is_image(path)
    )


def find_facesynthetics_images(
    dataset_dir: Path
):
    """
    FaceSynthetics:
    primary face images have numeric filenames.
    Files such as:
        000000_ldmks
        000000_seg
    are annotations / auxiliary outputs and are excluded.
    """

    images = []

    for path in dataset_dir.rglob("*"):

        if not is_image(path):
            continue

        stem = path.stem

        if stem.isdigit():
            images.append(path)

    return sorted(
        set(images)
    )


def find_ffhq_images(
    dataset_dir: Path
):
    """
    FFHQ:
    collect image files while excluding obvious
    metadata / annotation folders.
    """

    excluded_keywords = {
        "json",
        "metadata",
        "thumb",
        "thumbnail",
    }

    images = []

    for path in dataset_dir.rglob("*"):

        if not is_image(path):
            continue

        relative_parts = path.relative_to(
            dataset_dir
        ).parts[:-1]

        excluded = False

        for part in relative_parts:

            lower = part.lower()

            if any(
                key in lower
                for key in excluded_keywords
            ):
                excluded = True
                break

        if not excluded:
            images.append(path)

    return sorted(
        set(images)
    )


def find_scut_images(
    dataset_dir: Path
):
    """
    SCUT-FBP5500:
    primary face images are in the Images folder.
    """

    image_dir = (
        dataset_dir / "Images"
    )

    if not image_dir.exists():

        image_dir = (
            dataset_dir / "images"
        )

    if not image_dir.exists():
        return []

    return sorted(
        path
        for path in image_dir.rglob("*")
        if is_image(path)
    )


def find_primary_images(
    dataset_name: str,
    dataset_dir: Path
):

    if dataset_name == "CelebAMask-HQ":

        return find_celebamask_images(
            dataset_dir
        )

    if dataset_name == "FaceSynthetics":

        return find_facesynthetics_images(
            dataset_dir
        )

    if dataset_name == "FFHQ":

        return find_ffhq_images(
            dataset_dir
        )

    if dataset_name == "SCUT-FBP5500":

        return find_scut_images(
            dataset_dir
        )

    return []


# ============================================================
# RESOLUTION
# ============================================================

def inspect_resolutions(
    image_files
) -> str:

    if not image_files:
        return "unknown"

    rng = random.Random(
        RANDOM_SEED
    )

    sample_count = min(
        RESOLUTION_SAMPLE_SIZE,
        len(image_files)
    )

    selected = rng.sample(
        image_files,
        sample_count
    )

    counter = Counter()

    for image_path in selected:

        try:

            with Image.open(
                image_path
            ) as image:

                counter[
                    image.size
                ] += 1

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
        ):

            continue

    if not counter:
        return "unknown"

    return "; ".join(
        f"{width}x{height} ({count})"
        for (
            width,
            height
        ), count in counter.most_common()
    )


# ============================================================
# INSPECT ONE DATASET
# ============================================================

def inspect_dataset(
    dataset_name: str
):

    dataset_dir = (
        RAW_DATASETS_DIR
        / dataset_name
    )

    print()
    print("=" * 75)
    print(
        f"DATASET: {dataset_name}"
    )
    print("=" * 75)

    if not dataset_dir.exists():

        print(
            "ERROR: dataset folder not found."
        )

        return {
            "dataset_name": dataset_name,
            "source": "",
            "purpose": "",
            "file_count": 0,
            "extensions": "",
            "resolution": "",
            "annotations": "",
        }

    info = (
        DATASET_INFO[
            dataset_name
        ]
    )

    # --------------------------------------------------------
    # File count
    # --------------------------------------------------------

    print(
        "Counting all files..."
    )

    file_count = count_all_files(
        dataset_dir
    )

    # --------------------------------------------------------
    # Primary images
    # --------------------------------------------------------

    print(
        "Finding primary face images..."
    )

    image_files = find_primary_images(
        dataset_name,
        dataset_dir
    )

    image_count = len(
        image_files
    )

    # --------------------------------------------------------
    # Extensions
    # --------------------------------------------------------

    print(
        "Collecting extensions..."
    )

    extensions = format_extensions(
        dataset_dir
    )

    # --------------------------------------------------------
    # Resolution
    # --------------------------------------------------------

    print(
        "Checking image resolution "
        f"from up to {RESOLUTION_SAMPLE_SIZE} samples..."
    )

    resolution = inspect_resolutions(
        image_files
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print(
        f"File count: {file_count:,}"
    )

    print(
        f"Image count: {image_count:,}"
    )

    print(
        f"Extensions: {extensions}"
    )

    print(
        f"Resolution: {resolution}"
    )

    print(
        f"Source: {info['source']}"
    )

    print(
        f"Purpose: {info['purpose']}"
    )

    print(
        f"Annotations: {info['annotations']}"
    )

    return {

        "dataset_name":
            dataset_name,

        "source":
            info[
                "source"
            ],

        "purpose":
            info[
                "purpose"
            ],

        "file_count":
            file_count,

        "extensions":
            extensions,

        "resolution":
            resolution,

        "annotations":
            info[
                "annotations"
            ],
    }


# ============================================================
# SAVE INVENTORY
# ============================================================

def save_inventory(
    rows
):

    fields = [
        "dataset_name",
        "source",
        "purpose",
        "file_count",
        "extensions",
        "resolution",
        "annotations",
    ]

    INVENTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        INVENTORY_PATH,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()

        for row in rows:

            writer.writerow({
                field:
                    row.get(
                        field,
                        ""
                    )
                for field in fields
            })


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 75)
    print(
        "DATASET INVENTORY"
    )
    print("=" * 75)

    print(
        f"Raw datasets:\n"
        f"{RAW_DATASETS_DIR}"
    )

    print(
        f"Output:\n"
        f"{INVENTORY_PATH}"
    )

    print("=" * 75)

    if not RAW_DATASETS_DIR.exists():

        raise FileNotFoundError(
            f"Raw datasets folder not found:\n"
            f"{RAW_DATASETS_DIR}"
        )

    rows = []

    for dataset_name in DATASETS:

        result = inspect_dataset(
            dataset_name
        )

        rows.append(
            result
        )

    print()
    print(
        "Saving inventory..."
    )

    save_inventory(
        rows
    )

    print()
    print("=" * 75)
    print(
        "INVENTORY COMPLETE"
    )
    print("=" * 75)

    print(
        f"Saved to:\n"
        f"{INVENTORY_PATH}"
    )

    print()
    print(
        "Original dataset files were only read."
    )

    print(
        "No raw dataset file was modified."
    )

    print("=" * 75)


if __name__ == "__main__":
    main()

