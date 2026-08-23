from pathlib import Path
import cv2
import json
import numpy as np

from retinaface import RetinaFace


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    PROJECT_ROOT
    / "raw_datasets"
    / "FFHQ"
    / "images1024x1024"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed"
    / "faces"
)

FAILED_FILE = (
    PROJECT_ROOT
    / "processed"
    / "ffhq_failed.json"
)

# Images manually excluded after reviewing the processing
# results and finding them unsuitable / distorted.
MANUALLY_EXCLUDED_FILE = (
    PROJECT_ROOT
    / "processed"
    / "ffhq_manually_excluded_after_processing.json"
)


# ============================================================
# Settings
# ============================================================

IMAGE_SIZE = 512

# عدد الصور المطلوبة من FFHQ
MAX_IMAGES = 5000

# حجم الـ margin حول الوجه
FACE_MARGIN = 0.25

# المساحة فوق وتحت الوجه
TOP_RATIO = 0.32
BOTTOM_RATIO = 0.42

# أقل confidence مقبول
MIN_CONFIDENCE = 0.50


# ============================================================
# Get eye landmarks
# ============================================================

def get_eyes(landmarks):

    left_eye = np.array(
        landmarks["left_eye"],
        dtype=np.float32
    )

    right_eye = np.array(
        landmarks["right_eye"],
        dtype=np.float32
    )

    # ترتيب العينين حسب X في الصورة
    eyes = sorted(
        [left_eye, right_eye],
        key=lambda p: p[0]
    )

    return eyes[0], eyes[1]


# ============================================================
# Crop face with margin
# ============================================================

def crop_face(image, facial_area):

    x1, y1, x2, y2 = facial_area

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    face_w = x2 - x1
    face_h = y2 - y1

    if face_w <= 0 or face_h <= 0:
        return None

    # Margin جانبي
    side_margin = int(
        face_w * FACE_MARGIN
    )

    # Margin فوق
    top_margin = int(
        face_h * TOP_RATIO
    )

    # Margin تحت
    bottom_margin = int(
        face_h * BOTTOM_RATIO
    )

    crop_x1 = x1 - side_margin
    crop_y1 = y1 - top_margin
    crop_x2 = x2 + side_margin
    crop_y2 = y2 + bottom_margin

    # ممنوع نخرج خارج حدود الصورة
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)

    crop_x2 = min(
        image.shape[1],
        crop_x2
    )

    crop_y2 = min(
        image.shape[0],
        crop_y2
    )

    if crop_x2 <= crop_x1:
        return None

    if crop_y2 <= crop_y1:
        return None

    cropped = image[
        crop_y1:crop_y2,
        crop_x1:crop_x2
    ]

    return cropped


# ============================================================
# Align using EYES ONLY
# ============================================================

def align_using_eyes(
    cropped,
    original_eyes,
    facial_area
):

    x1, y1, _, _ = facial_area

    # تحويل إحداثيات العينين من الصورة الأصلية
    # إلى إحداثيات الـcrop
    left_eye = np.array([
        original_eyes[0][0] - x1,
        original_eyes[0][1] - y1
    ], dtype=np.float32)

    right_eye = np.array([
        original_eyes[1][0] - x1,
        original_eyes[1][1] - y1
    ], dtype=np.float32)

    # مركز العينين
    eye_center = (
        left_eye + right_eye
    ) / 2.0

    dx = (
        right_eye[0]
        - left_eye[0]
    )

    dy = (
        right_eye[1]
        - left_eye[1]
    )

    # زاوية العينين
    angle = np.degrees(
        np.arctan2(dy, dx)
    )

    # المسافة بين العينين
    eye_distance = np.linalg.norm(
        right_eye - left_eye
    )

    if eye_distance < 10:
        return None

    # --------------------------------------------------------
    # Rotation فقط
    # --------------------------------------------------------

    rotation_matrix = cv2.getRotationMatrix2D(
        tuple(eye_center),
        angle,
        1.0
    )

    h, w = cropped.shape[:2]

    rotated = cv2.warpAffine(
        cropped,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


# ============================================================
# Final square crop
# ============================================================

def make_square(image):

    if image is None:
        return None

    h, w = image.shape[:2]

    if h <= 0 or w <= 0:
        return None

    side = min(h, w)

    start_x = max(
        0,
        (w - side) // 2
    )

    start_y = max(
        0,
        (h - side) // 2
    )

    square = image[
        start_y:start_y + side,
        start_x:start_x + side
    ]

    return square


# ============================================================
# Process one image
# ============================================================

def process_image(image_path):

    # --------------------------------------------------------
    # 1. Read
    # --------------------------------------------------------

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        return False, "cannot_read_image"

    # --------------------------------------------------------
    # 2. RetinaFace detection
    # --------------------------------------------------------

    detections = RetinaFace.detect_faces(
        str(image_path)
    )

    if not isinstance(detections, dict):
        return False, "no_valid_face"

    if len(detections) == 0:
        return False, "no_valid_face"

    # --------------------------------------------------------
    # 3. Choose strongest face
    # --------------------------------------------------------

    best_face = max(
        detections.values(),
        key=lambda face: face["score"]
    )

    score = float(
        best_face["score"]
    )

    if score < MIN_CONFIDENCE:
        return False, (
            f"low_confidence_{score:.3f}"
        )

    # --------------------------------------------------------
    # 4. Get face box
    # --------------------------------------------------------

    facial_area = best_face.get(
        "facial_area"
    )

    if facial_area is None:
        return False, "missing_face_box"

    # --------------------------------------------------------
    # 5. Get EYES ONLY
    # --------------------------------------------------------

    landmarks = best_face.get(
        "landmarks"
    )

    if landmarks is None:
        return False, "missing_landmarks"

    if (
        "left_eye" not in landmarks
        or
        "right_eye" not in landmarks
    ):
        return False, "missing_eyes"

    left_eye, right_eye = get_eyes(
        landmarks
    )

    # --------------------------------------------------------
    # 6. Crop BEFORE alignment
    # --------------------------------------------------------

    cropped = crop_face(
        image,
        facial_area
    )

    if cropped is None:
        return False, "crop_failed"

    # --------------------------------------------------------
    # 7. Align using EYES ONLY
    # --------------------------------------------------------

    aligned = align_using_eyes(
        cropped,
        (left_eye, right_eye),
        facial_area
    )

    if aligned is None:
        return False, "alignment_failed"

    # --------------------------------------------------------
    # 8. Make square
    # --------------------------------------------------------

    square = make_square(
        aligned
    )

    if square is None:
        return False, "square_crop_failed"

    # --------------------------------------------------------
    # 9. Resize to 512x512
    # --------------------------------------------------------

    final_image = cv2.resize(
        square,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # 10. Basic sanity check
    # --------------------------------------------------------

    if (
        final_image is None
        or
        final_image.shape[0] != IMAGE_SIZE
        or
        final_image.shape[1] != IMAGE_SIZE
    ):
        return False, "invalid_output"

    # --------------------------------------------------------
    # 11. Save
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / image_path.name
    )

    success = cv2.imwrite(
        str(output_path),
        final_image
    )

    if not success:
        return False, "save_failed"

    return True, None


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Get first 5000 input images
    # --------------------------------------------------------

    all_input_files = sorted(
        INPUT_DIR.glob("*.png")
    )

    selected_files = all_input_files[
        :MAX_IMAGES
    ]

    # --------------------------------------------------------
    # IMPORTANT:
    # Remove duplicate filenames from the input list
    # --------------------------------------------------------

    unique_files = {}

    for image_path in selected_files:
        unique_files[image_path.name] = image_path

    image_files = list(
        unique_files.values()
    )

    print(
        f"Selected from input: "
        f"{len(image_files)}"
    )

    print(
        f"Unique input files: "
        f"{len(set(p.name for p in image_files))}"
    )

    # --------------------------------------------------------
    # Get files that REALLY exist now in processed/faces
    # --------------------------------------------------------

    processed_names = {
        p.name
        for p in OUTPUT_DIR.glob("*.png")
    }

    # --------------------------------------------------------
    # Get images manually excluded after reviewing the
    # processing results.
    #
    # These images must NOT be processed again, even if:
    # - the original image still exists in the input folder
    # - the processed image was manually deleted
    #
    # We intentionally keep this separate from ffhq_failed.json
    # because these images were manually judged unsuitable
    # after reviewing the processing results.
    # --------------------------------------------------------

    if MANUALLY_EXCLUDED_FILE.exists():

        with open(
            MANUALLY_EXCLUDED_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            manually_excluded_data = json.load(f)

        manually_excluded_ids = set(
            manually_excluded_data.get(
                "excluded_images",
                []
            )
        )

    else:

        manually_excluded_ids = set()

    print(
        f"Manually excluded after processing: "
        f"{len(manually_excluded_ids)}"
    )

    # --------------------------------------------------------
    # Only files that:
    #
    # 1. Are NOT already processed
    # 2. Are NOT manually excluded
    #
    # will be processed.
    #
    # We intentionally DO NOT use ffhq_failed.json
    # to skip files.
    # --------------------------------------------------------

    to_process = [
        image_path
        for image_path in image_files
        if (
            image_path.name not in processed_names
            and image_path.stem not in manually_excluded_ids
        )
    ]

    already_processed = sum(
        1
        for image_path in image_files
        if image_path.name in processed_names
    )

    manually_excluded_count = sum(
        1
        for image_path in image_files
        if image_path.stem in manually_excluded_ids
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    expected_remaining = (
        len(image_files)
        - already_processed
        - manually_excluded_count
    )

    if len(to_process) != expected_remaining:
        raise RuntimeError(
            "Processing list count mismatch."
        )

    print(
        f"Already processed: "
        f"{already_processed}"
    )

    print(
        f"Manually excluded: "
        f"{manually_excluded_count}"
    )

    print(
        f"Remaining to process: "
        f"{len(to_process)}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Process ONLY remaining files
    # --------------------------------------------------------

    successful = 0
    failed = []

    for i, image_path in enumerate(
        to_process,
        start=1
    ):

        print(
            f"[{i}/{len(to_process)}] "
            f"{image_path.name}",
            end=" ... "
        )

        try:

            success, reason = process_image(
                image_path
            )

            if success:

                print("OK")
                successful += 1

            else:

                print(
                    f"FAILED ({reason})"
                )

                failed.append({
                    "id": image_path.stem,
                    "file": image_path.name,
                    "reason": reason
                })

        except Exception as e:

            print(
                f"ERROR ({e})"
            )

            failed.append({
                "id": image_path.stem,
                "file": image_path.name,
                "reason": str(e)
            })

    # --------------------------------------------------------
    # Save failed report
    # --------------------------------------------------------

    with open(
        FAILED_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            failed,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)

    print(
        f"Selected from input: "
        f"{len(image_files)}"
    )

    print(
        f"Already processed: "
        f"{already_processed}"
    )

    print(
        f"Manually excluded: "
        f"{manually_excluded_count}"
    )

    print(
        f"Processed this run: "
        f"{len(to_process)}"
    )

    print(
        f"Successful this run: "
        f"{successful}"
    )

    print(
        f"Failed this run: "
        f"{len(failed)}"
    )

    print(
        f"Output: "
        f"{OUTPUT_DIR}"
    )

    print(
        f"Failed report: "
        f"{FAILED_FILE}"
    )

    print(
        f"Manual exclusion list: "
        f"{MANUALLY_EXCLUDED_FILE}"
    )

    print("=" * 60)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()