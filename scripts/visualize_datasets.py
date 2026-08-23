from pathlib import Path
import random

from PIL import (
    Image,
    ImageDraw,
    ImageFont,
    UnidentifiedImageError,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\cosmetic-ai")

RAW_DATASETS_DIR = (
    PROJECT_ROOT / "raw_datasets"
)

SAMPLES_ROOT = (
    RAW_DATASETS_DIR / "dataset_samples"
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

SAMPLES_PER_DATASET = 5

RANDOM_SEED = 42


# ============================================================
# DATASET-SPECIFIC PRIMARY IMAGE FINDERS
# ============================================================

def is_image(path: Path):

    return (
        path.is_file()
        and path.suffix.lower()
        in IMAGE_EXTENSIONS
    )


def find_celebamask_images(
    dataset_dir
):

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
    dataset_dir
):

    images = []

    for path in dataset_dir.rglob("*"):

        if not is_image(path):
            continue

        # Only numeric filenames.
        if path.stem.isdigit():
            images.append(path)

    return sorted(
        set(images)
    )


def find_ffhq_images(
    dataset_dir
):

    excluded_keywords = {
        "metadata",
        "thumb",
        "thumbnail",
    }

    images = []

    for path in dataset_dir.rglob("*"):

        if not is_image(path):
            continue

        relative_parts = (
            path.relative_to(
                dataset_dir
            ).parts[:-1]
        )

        excluded = False

        for part in relative_parts:

            lower = part.lower()

            if any(
                keyword in lower
                for keyword in excluded_keywords
            ):
                excluded = True
                break

        if not excluded:
            images.append(path)

    return sorted(
        set(images)
    )


def find_scut_images(
    dataset_dir
):

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
    dataset_name,
    dataset_dir
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
# SAMPLE SELECTION
# ============================================================

def choose_samples(
    image_files
):

    if not image_files:
        return []

    rng = random.Random(
        RANDOM_SEED
    )

    if len(image_files) <= SAMPLES_PER_DATASET:
        return list(image_files)

    return rng.sample(
        image_files,
        SAMPLES_PER_DATASET
    )


# ============================================================
# CONTACT SHEET
# ============================================================

def create_contact_sheet(
    dataset_name,
    sample_files
):

    output_dir = (
        SAMPLES_ROOT
        /
        dataset_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if not sample_files:
        print(
            "  No face images found."
        )
        return

    thumb_width = 280
    thumb_height = 280
    label_height = 45

    columns = min(
        3,
        len(sample_files)
    )

    rows = (
        len(sample_files)
        +
        columns
        -
        1
    ) // columns

    sheet = Image.new(
        "RGB",
        (
            columns * thumb_width,
            rows * (
                thumb_height
                +
                label_height
            )
        ),
        "white"
    )

    draw = ImageDraw.Draw(
        sheet
    )

    for index, image_path in enumerate(
        sample_files
    ):

        try:

            with Image.open(
                image_path
            ) as source:

                image = source.convert(
                    "RGB"
                )

                image.thumbnail(
                    (
                        thumb_width - 10,
                        thumb_height - 10
                    )
                )

                col = (
                    index
                    %
                    columns
                )

                row = (
                    index
                    //
                    columns
                )

                x = (
                    col
                    *
                    thumb_width
                )

                y = (
                    row
                    *
                    (
                        thumb_height
                        +
                        label_height
                    )
                )

                paste_x = (
                    x
                    +
                    (
                        thumb_width
                        -
                        image.width
                    )
                    //
                    2
                )

                paste_y = (
                    y
                    +
                    (
                        thumb_height
                        -
                        image.height
                    )
                    //
                    2
                )

                sheet.paste(
                    image,
                    (
                        paste_x,
                        paste_y
                    )
                )

                draw.text(
                    (
                        x + 8,
                        y + thumb_height + 8
                    ),
                    (
                        f"{index + 1}. "
                        f"{image_path.name}"
                    ),
                    fill="black"
                )

                # Save individual sample.
                individual_path = (
                    output_dir
                    /
                    f"sample_{index + 1:02d}.jpg"
                )

                image.save(
                    individual_path,
                    quality=95
                )

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
        ) as error:

            print(
                f"  Could not read "
                f"{image_path}: "
                f"{error}"
            )

    sheet_path = (
        output_dir
        /
        "contact_sheet.jpg"
    )

    sheet.save(
        sheet_path,
        quality=95
    )

    print(
        f"  Contact sheet:\n"
        f"  {sheet_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "DATASET SAMPLE VISUALIZATION"
    )
    print("=" * 70)

    print(
        f"Raw datasets:\n"
        f"{RAW_DATASETS_DIR}"
    )

    print(
        f"Samples output:\n"
        f"{SAMPLES_ROOT}"
    )

    print("=" * 70)

    if not RAW_DATASETS_DIR.exists():

        raise FileNotFoundError(
            f"Raw datasets folder not found:\n"
            f"{RAW_DATASETS_DIR}"
        )

    for dataset_name in DATASETS:

        print()
        print(
            f"DATASET: {dataset_name}"
        )

        dataset_dir = (
            RAW_DATASETS_DIR
            /
            dataset_name
        )

        if not dataset_dir.exists():

            print(
                "  Folder not found."
            )

            continue

        print(
            "  Finding primary face images..."
        )

        image_files = find_primary_images(
            dataset_name,
            dataset_dir
        )

        print(
            f"  Found "
            f"{len(image_files):,} "
            f"primary images."
        )

        samples = choose_samples(
            image_files
        )

        print(
            f"  Selected "
            f"{len(samples)} samples."
        )

        create_contact_sheet(
            dataset_name,
            samples
        )

    print()
    print("=" * 70)
    print(
        "SAMPLE VISUALIZATION COMPLETE"
    )
    print("=" * 70)

    print(
        f"Output:\n"
        f"{SAMPLES_ROOT}"
    )

    print()
    print(
        "Original raw dataset files were not modified."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()