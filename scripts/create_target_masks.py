from pathlib import Path
import json

import cv2
import numpy as np


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FACES_DIR = (
    PROJECT_ROOT
    / "processed"
    / "faces"
)

RAW_MASKS_DIR = (
    PROJECT_ROOT
    / "processed"
    / "masks"
    / "raw"
)

LANDMARKS_DIR = (
    PROJECT_ROOT
    / "processed"
    / "landmarks"
)

TARGET_MASKS_DIR = (
    PROJECT_ROOT
    / "processed"
    / "masks"
    / "targets"
)

PREVIEW_DIR = (
    PROJECT_ROOT
    / "processed"
    / "masks"
    / "previews"
)

FAILED_FILE = (
    PROJECT_ROOT
    / "processed"
    / "target_masks_failed.json"
)


# ============================================================
# Test settings
# ============================================================

TEST_MODE = False
TEST_LIMIT = 50


# ============================================================
# Operations
# ============================================================

OPERATIONS = [
    "rhinoplasty",
    "chin_augmentation",
    "jawline_contouring",
    "facelift",
    "blepharoplasty",
    "lip_enhancement",
]


# ============================================================
# BiSeNet / CelebAMask-HQ class IDs
# ============================================================

BACKGROUND = 0
SKIN = 1

LEFT_BROW = 2
RIGHT_BROW = 3

LEFT_EYE = 4
RIGHT_EYE = 5

EYE_GLASSES = 6

LEFT_EAR = 7
RIGHT_EAR = 8
EAR_RING = 9

NOSE = 10
MOUTH = 11
UPPER_LIP = 12
LOWER_LIP = 13

NECK = 14
NECK_L = 15
CLOTH = 16
HAIR = 17
HAT = 18


# ============================================================
# Geometry settings
# ============================================================

FEATURE_DILATION = 1

# Thin jawline target.
# We intentionally keep this close to the landmark line.
JAWLINE_THICKNESS = 6

# No extra dilation for the jawline.
JAWLINE_FINAL_DILATION = 0


# ============================================================
# Facelift crop / face-region settings
# ============================================================

# Facelift does NOT use MediaPipe landmarks.
#
# Instead:
#   1. Start from the full image.
#   2. Use BiSeNet SKIN.
#   3. Estimate the facial area from the skin distribution.
#   4. Remove hair / ears / neck / non-face semantics.
#
# The crop is estimated directly from the parsing mask.
#
# These margins keep the target from touching the extreme
# image borders.
FACE_REGION_MARGIN_X = 0.04
FACE_REGION_MARGIN_Y_TOP = 0.03
FACE_REGION_MARGIN_Y_BOTTOM = 0.08


# ============================================================
# Utility
# ============================================================

def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def load_image(path):

    return cv2.imread(
        str(path)
    )


def dilate_mask(
    mask,
    iterations=1
):
    """
    Small morphological expansion.
    """

    if iterations <= 0:
        return mask

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    return cv2.dilate(
        mask,
        kernel,
        iterations=iterations
    )


# ============================================================
# Landmark helpers
# ============================================================

def get_group_points(
    landmark_data,
    group_name
):
    """
    Read existing MediaPipe landmark pixel coordinates.
    """

    groups = (
        landmark_data
        .get("landmarks", {})
        .get("groups", {})
    )

    points = groups.get(
        group_name,
        []
    )

    result = []

    for point in points:

        if (
            "pixel_x" not in point
            or
            "pixel_y" not in point
        ):
            continue

        result.append([
            float(point["pixel_x"]),
            float(point["pixel_y"])
        ])

    return result


# ============================================================
# Geometry helpers
# ============================================================

def polygon_mask(
    points,
    height,
    width
):
    """
    Filled convex region from landmark points.
    """

    result = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    if len(points) < 3:
        return result

    pts = np.array(
        points,
        dtype=np.int32
    )

    hull = cv2.convexHull(
        pts
    )

    cv2.fillConvexPoly(
        result,
        hull,
        255
    )

    return result


def polyline_band_mask(
    points,
    height,
    width,
    thickness
):
    """
    Thin band around a landmark line.
    """

    result = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    if len(points) < 2:
        return result

    pts = np.array(
        points,
        dtype=np.int32
    )

    cv2.polylines(
        result,
        [pts],
        False,
        255,
        thickness,
        cv2.LINE_AA
    )

    return result


# ============================================================
# BiSeNet masks
# ============================================================

def class_mask(
    parsing_mask,
    class_ids
):
    """
    Binary mask for selected BiSeNet classes.
    """

    result = np.zeros(
        parsing_mask.shape,
        dtype=np.uint8
    )

    for class_id in class_ids:

        result[
            parsing_mask == class_id
        ] = 255

    return result


# ============================================================
# 1. Rhinoplasty
# ============================================================

def create_rhinoplasty_mask(
    parsing_mask,
    landmark_data
):
    """
    BiSeNet ONLY.

    No MediaPipe landmarks.
    """

    return dilate_mask(
        class_mask(
            parsing_mask,
            [NOSE]
        ),
        FEATURE_DILATION
    )


# ============================================================
# 2. Lip Enhancement
# ============================================================

def create_lip_mask(
    parsing_mask,
    landmark_data
):
    """
    BiSeNet ONLY.

    No MediaPipe landmarks.
    """

    return dilate_mask(
        class_mask(
            parsing_mask,
            [
                UPPER_LIP,
                LOWER_LIP
            ]
        ),
        FEATURE_DILATION
    )


# ============================================================
# 3. Blepharoplasty
# ============================================================

def create_blepharoplasty_mask(
    parsing_mask,
    landmark_data
):
    """
    MediaPipe eye landmarks + BiSeNet.

    IMPORTANT:

    Landmarks determine WHERE the eyes are.

    BiSeNet contributes eye pixels inside those regions.

    EYE_GLASSES does NOT remove the eye anymore.

    This is intentional for glasses cases.
    """

    height, width = parsing_mask.shape

    # --------------------------------------------------------
    # Eye landmark geometry
    # --------------------------------------------------------

    left_points = get_group_points(
        landmark_data,
        "left_eye"
    )

    right_points = get_group_points(
        landmark_data,
        "right_eye"
    )

    left_geometry = polygon_mask(
        left_points,
        height,
        width
    )

    right_geometry = polygon_mask(
        right_points,
        height,
        width
    )

    geometry = cv2.bitwise_or(
        left_geometry,
        right_geometry
    )

    # Slight expansion
    geometry = dilate_mask(
        geometry,
        FEATURE_DILATION
    )

    # --------------------------------------------------------
    # BiSeNet eye classes
    # --------------------------------------------------------

    semantic = class_mask(
        parsing_mask,
        [
            LEFT_EYE,
            RIGHT_EYE
        ]
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT subtract EYE_GLASSES.
    #
    # If glasses cover the eye, MediaPipe geometry
    # still keeps the target in the correct eye region.
    # --------------------------------------------------------

    # First preference: BiSeNet eye pixels inside geometry.
    semantic_inside_eye = cv2.bitwise_and(
        geometry,
        semantic
    )

    # --------------------------------------------------------
    # Fallback:
    #
    # If BiSeNet sees little/no eye because of glasses,
    # keep the landmark-defined region itself.
    #
    # This prevents glasses from deleting the target.
    # --------------------------------------------------------

    semantic_pixels = cv2.countNonZero(
        semantic_inside_eye
    )

    geometry_pixels = cv2.countNonZero(
        geometry
    )

    if geometry_pixels == 0:

        return geometry

    coverage = (
        semantic_pixels
        /
        geometry_pixels
    )

    # If BiSeNet found at least some eye pixels,
    # combine them with the landmark geometry.
    #
    # The geometry remains authoritative.
    if coverage > 0.05:

        final = cv2.bitwise_or(
            geometry,
            semantic_inside_eye
        )

    else:

        final = geometry

    return final


# ============================================================
# 4. Chin Augmentation
# ============================================================

def create_chin_mask(
    parsing_mask,
    landmark_data
):
    """
    Existing approach:
        MediaPipe chin + BiSeNet SKIN.
    """

    height, width = parsing_mask.shape

    chin_points = get_group_points(
        landmark_data,
        "chin"
    )

    if len(chin_points) < 3:

        return np.zeros(
            (height, width),
            dtype=np.uint8
        )

    geometry = polygon_mask(
        chin_points,
        height,
        width
    )

    geometry = dilate_mask(
        geometry,
        FEATURE_DILATION
    )

    semantic = class_mask(
        parsing_mask,
        [SKIN]
    )

    return cv2.bitwise_and(
        geometry,
        semantic
    )


# ============================================================
# 5. Jawline Contouring
# ============================================================

def create_jawline_mask(
    parsing_mask,
    landmark_data
):
    """
    MediaPipe jawline + BiSeNet SKIN.

    The landmark line is intentionally THIN.
    """

    height, width = parsing_mask.shape

    jaw_points = get_group_points(
        landmark_data,
        "jaw"
    )

    if len(jaw_points) < 2:

        return np.zeros(
            (height, width),
            dtype=np.uint8
        )

    # --------------------------------------------------------
    # Thin geometry line
    # --------------------------------------------------------

    geometry = polyline_band_mask(
        jaw_points,
        height,
        width,
        JAWLINE_THICKNESS
    )

    # --------------------------------------------------------
    # BiSeNet skin
    # --------------------------------------------------------

    semantic = class_mask(
        parsing_mask,
        [SKIN]
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    final = cv2.bitwise_and(
        geometry,
        semantic
    )

    return dilate_mask(
        final,
        JAWLINE_FINAL_DILATION
    )


# ============================================================
# Facelift helper:
# estimate face crop WITHOUT landmarks
# ============================================================

def estimate_face_region_from_parsing(
    parsing_mask
):
    """
    Estimate a facial region using BiSeNet semantics only.

    No MediaPipe landmarks are used.

    Strategy:

        SKIN
          +
        semantic exclusions
          +
        connected components
          +
        bounding region

    This is NOT a clinical face boundary.

    It is a project-defined image-space facial region.
    """

    height, width = (
        parsing_mask.shape
    )

    # --------------------------------------------------------
    # Candidate facial skin
    # --------------------------------------------------------

    skin = class_mask(
        parsing_mask,
        [SKIN]
    )

    # --------------------------------------------------------
    # Remove areas that are definitely not the target
    # --------------------------------------------------------

    non_face = class_mask(
        parsing_mask,
        [
            HAIR,
            HAT,
            CLOTH,
            NECK,
            NECK_L,
            LEFT_EAR,
            RIGHT_EAR,
            EAR_RING,
        ]
    )

    candidate = skin.copy()

    candidate[
        non_face > 0
    ] = 0

    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            candidate,
            connectivity=8
        )
    )

    if num_labels <= 1:

        return candidate

    # --------------------------------------------------------
    # Choose the component closest to image center.
    #
    # This avoids using isolated skin-like regions
    # elsewhere in the image.
    # --------------------------------------------------------

    image_center_x = width / 2.0
    image_center_y = height * 0.45

    best_label = 0
    best_score = -1.0

    for label in range(
        1,
        num_labels
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area <= 0:
            continue

        cx, cy = centroids[
            label
        ]

        # Normalize distance
        dx = (
            cx
            -
            image_center_x
        ) / width

        dy = (
            cy
            -
            image_center_y
        ) / height

        distance = (
            dx * dx
            +
            dy * dy
        )

        # Larger area is good.
        # Smaller distance to face center is good.

        score = (
            float(area)
            /
            float(height * width)
        ) * 10.0

        score -= (
            distance * 5.0
        )

        if score > best_score:

            best_score = score
            best_label = label

    # --------------------------------------------------------
    # Selected component
    # --------------------------------------------------------

    result = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    if best_label > 0:

        result[
            labels == best_label
        ] = 255

    # --------------------------------------------------------
    # Crop margins
    #
    # We keep this image-space crop conservative.
    # --------------------------------------------------------

    ys, xs = np.where(
        result > 0
    )

    if len(xs) == 0:

        return result

    x_min = int(
        xs.min()
    )

    x_max = int(
        xs.max()
    )

    y_min = int(
        ys.min()
    )

    y_max = int(
        ys.max()
    )

    margin_x = int(
        width
        *
        FACE_REGION_MARGIN_X
    )

    margin_top = int(
        height
        *
        FACE_REGION_MARGIN_Y_TOP
    )

    margin_bottom = int(
        height
        *
        FACE_REGION_MARGIN_Y_BOTTOM
    )

    x_min = max(
        0,
        x_min - margin_x
    )

    x_max = min(
        width - 1,
        x_max + margin_x
    )

    y_min = max(
        0,
        y_min - margin_top
    )

    y_max = min(
        height - 1,
        y_max + margin_bottom
    )

    crop_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    crop_mask[
        y_min:y_max + 1,
        x_min:x_max + 1
    ] = 255

    # Final face region:
    # estimated image-space crop AND candidate skin.
    final_region = cv2.bitwise_and(
        crop_mask,
        result
    )

    return final_region


# ============================================================
# 6. Facelift
# ============================================================

def create_facelift_mask(
    parsing_mask,
    landmark_data
):
    """
    Facelift target:

    BiSeNet SKIN only.

    MediaPipe landmarks are NOT used.

    We start from facial skin as classified by BiSeNet,
    then exclude semantic regions that should not be
    modified by the facelift target.
    """

    # --------------------------------------------------------
    # Start with BiSeNet SKIN
    # --------------------------------------------------------

    mask = class_mask(
        parsing_mask,
        [SKIN]
    )

    # --------------------------------------------------------
    # Exclusions
    #
    # Keep:
    #   facial skin
    #
    # Exclude:
    #   eyes
    #   eyebrows
    #   nose
    #   mouth
    #   lips
    #   glasses
    #   ears
    #   ear rings
    #   hair
    #   hat
    #   neck
    #   clothes
    # --------------------------------------------------------

    excluded = class_mask(
        parsing_mask,
        [
            LEFT_EYE,
            RIGHT_EYE,

            LEFT_BROW,
            RIGHT_BROW,

            NOSE,

            MOUTH,
            UPPER_LIP,
            LOWER_LIP,

            EYE_GLASSES,

            LEFT_EAR,
            RIGHT_EAR,

            EAR_RING,

            HAIR,
            HAT,

            NECK,
            NECK_L,
            CLOTH,
        ]
    )

    # --------------------------------------------------------
    # Remove excluded semantic classes
    # --------------------------------------------------------

    mask[
        excluded > 0
    ] = 0

    return mask



# ============================================================
# Create all target masks
# ============================================================

def create_operation_masks(
    parsing_mask,
    landmark_data
):

    return {

        "rhinoplasty":
            create_rhinoplasty_mask(
                parsing_mask,
                landmark_data
            ),

        "chin_augmentation":
            create_chin_mask(
                parsing_mask,
                landmark_data
            ),

        "jawline_contouring":
            create_jawline_mask(
                parsing_mask,
                landmark_data
            ),

        "facelift":
            create_facelift_mask(
                parsing_mask,
                landmark_data
            ),

        "blepharoplasty":
            create_blepharoplasty_mask(
                parsing_mask,
                landmark_data
            ),

        "lip_enhancement":
            create_lip_mask(
                parsing_mask,
                landmark_data
            ),
    }


# ============================================================
# BiSeNet visualization
# ============================================================

def colorize_parsing(
    parsing_mask
):

    colors = {

        BACKGROUND: (0, 0, 0),

        SKIN: (170, 170, 170),

        LEFT_BROW: (0, 140, 255),
        RIGHT_BROW: (0, 140, 255),

        LEFT_EYE: (0, 255, 0),
        RIGHT_EYE: (0, 255, 0),

        EYE_GLASSES: (255, 0, 255),

        LEFT_EAR: (100, 180, 100),
        RIGHT_EAR: (100, 180, 100),

        EAR_RING: (255, 180, 0),

        NOSE: (255, 100, 100),

        MOUTH: (120, 80, 255),

        UPPER_LIP: (0, 100, 255),
        LOWER_LIP: (0, 100, 255),

        NECK: (120, 120, 120),
        NECK_L: (120, 120, 120),

        CLOTH: (80, 80, 80),

        HAIR: (40, 40, 40),

        HAT: (160, 80, 0),
    }

    output = np.zeros(
        (
            parsing_mask.shape[0],
            parsing_mask.shape[1],
            3
        ),
        dtype=np.uint8
    )

    for class_id, color in colors.items():

        output[
            parsing_mask == class_id
        ] = color

    return output


# ============================================================
# Draw geometry
# ============================================================

def draw_points(
    image,
    points,
    color=(0, 255, 255),
    radius=3
):

    for x, y in points:

        cv2.circle(
            image,
            (
                int(round(x)),
                int(round(y))
            ),
            radius,
            color,
            -1
        )


def draw_line(
    image,
    points,
    color=(0, 255, 255),
    thickness=2
):

    if len(points) < 2:
        return

    pts = np.array(
        [
            [
                int(round(x)),
                int(round(y))
            ]
            for x, y in points
        ],
        dtype=np.int32
    )

    cv2.polylines(
        image,
        [pts],
        False,
        color,
        thickness,
        cv2.LINE_AA
    )


def draw_operation_geometry(
    image,
    operation,
    landmark_data
):
    """
    Only draw landmarks for operations that use them.
    """

    output = image.copy()

    if operation == "blepharoplasty":

        left = get_group_points(
            landmark_data,
            "left_eye"
        )

        right = get_group_points(
            landmark_data,
            "right_eye"
        )

        draw_points(
            output,
            left,
            color=(0, 255, 0)
        )

        draw_points(
            output,
            right,
            color=(0, 255, 0)
        )

        draw_line(
            output,
            left,
            color=(0, 255, 0)
        )

        draw_line(
            output,
            right,
            color=(0, 255, 0)
        )

    elif operation == "jawline_contouring":

        jaw = get_group_points(
            landmark_data,
            "jaw"
        )

        draw_points(
            output,
            jaw,
            color=(0, 255, 255)
        )

        draw_line(
            output,
            jaw,
            color=(0, 255, 255)
        )

    # No landmarks for:
    #
    # rhinoplasty
    # lip_enhancement
    # facelift
    #
    # Chin is intentionally kept as before and can be shown
    # if desired.

    elif operation == "chin_augmentation":

        chin = get_group_points(
            landmark_data,
            "chin"
        )

        draw_points(
            output,
            chin,
            color=(255, 0, 255)
        )

        draw_line(
            output,
            chin,
            color=(255, 0, 255)
        )

    return output


# ============================================================
# Diagnostic preview
# ============================================================

def add_label(
    image,
    text
):

    output = image.copy()

    cv2.rectangle(
        output,
        (0, 0),
        (260, 34),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        output,
        text,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return output


def create_diagnostic_preview(
    image,
    geometry_preview,
    parsing_visual,
    final_mask,
    operation
):
    """
    ORIGINAL | LANDMARKS | BISENET | FINAL TARGET
    """

    original = add_label(
        image,
        "ORIGINAL"
    )

    geometry = add_label(
        geometry_preview,
        "LANDMARKS / REGION"
    )

    parsing = add_label(
        parsing_visual,
        "BISENET"
    )

    overlay = image.copy()

    overlay[
        final_mask > 0
    ] = (
        0,
        255,
        0
    )

    overlay = cv2.addWeighted(
        image,
        0.70,
        overlay,
        0.30,
        0
    )

    overlay = add_label(
        overlay,
        "FINAL TARGET"
    )

    for panel in [
        original,
        geometry,
        parsing,
        overlay
    ]:

        cv2.putText(
            panel,
            operation.upper(),
            (
                8,
                panel.shape[0] - 10
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return np.hstack([
        original,
        geometry,
        parsing,
        overlay
    ])


# ============================================================
# Process one image
# ============================================================

def process_image(
    image_path
):

    image_id = image_path.stem

    image = load_image(
        image_path
    )

    if image is None:

        return (
            False,
            "cannot_read_image"
        )

    # --------------------------------------------------------
    # Raw BiSeNet mask
    # --------------------------------------------------------

    raw_mask_path = (
        RAW_MASKS_DIR
        / f"{image_id}_mask.png"
    )

    if not raw_mask_path.exists():

        return (
            False,
            "raw_mask_not_found"
        )

    parsing_mask = cv2.imread(
        str(raw_mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if parsing_mask is None:

        return (
            False,
            "cannot_read_raw_mask"
        )

    if (
        parsing_mask.shape[:2]
        != image.shape[:2]
    ):

        return (
            False,
            "image_mask_size_mismatch"
        )

    # --------------------------------------------------------
    # Landmarks JSON
    #
    # Still loaded because:
    #   - Blepharoplasty
    #   - Jawline
    #   - Chin
    #
    # But it is not used for the other operations.
    # --------------------------------------------------------

    landmark_path = (
        LANDMARKS_DIR
        / f"{image_id}.json"
    )

    if not landmark_path.exists():

        return (
            False,
            "landmarks_not_found"
        )

    try:

        landmark_data = load_json(
            landmark_path
        )

    except Exception as e:

        return (
            False,
            f"invalid_landmark_json:{e}"
        )

    # --------------------------------------------------------
    # Generate target masks
    # --------------------------------------------------------

    target_masks = (
        create_operation_masks(
            parsing_mask,
            landmark_data
        )
    )

    # --------------------------------------------------------
    # BiSeNet visualization
    # --------------------------------------------------------

    parsing_visual = colorize_parsing(
        parsing_mask
    )

    # --------------------------------------------------------
    # Save each operation
    # --------------------------------------------------------

    for operation, final_mask in (
        target_masks.items()
    ):

        target_dir = (
            TARGET_MASKS_DIR
            / operation
        )

        preview_dir = (
            PREVIEW_DIR
            / operation
        )

        target_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        preview_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Save final binary mask
        # ----------------------------------------------------

        target_path = (
            target_dir
            / f"{image_id}.png"
        )

        if not cv2.imwrite(
            str(target_path),
            final_mask
        ):

            return (
                False,
                f"target_save_failed_{operation}"
            )

        # ----------------------------------------------------
        # Diagnostic preview
        # ----------------------------------------------------

        geometry_preview = (
            draw_operation_geometry(
                image,
                operation,
                landmark_data
            )
        )

        preview = (
            create_diagnostic_preview(
                image,
                geometry_preview,
                parsing_visual,
                final_mask,
                operation
            )
        )

        preview_path = (
            preview_dir
            / f"{image_id}.png"
        )

        if not cv2.imwrite(
            str(preview_path),
            preview
        ):

            return (
                False,
                f"preview_save_failed_{operation}"
            )

    return (
        True,
        None
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 75)
    print(
        "TARGET MASKS - SAMPLE REVISION"
    )
    print("=" * 75)

    print(
        f"Faces:\n{FACES_DIR}"
    )

    print(
        f"Raw masks:\n{RAW_MASKS_DIR}"
    )

    print(
        f"Landmarks:\n{LANDMARKS_DIR}"
    )

    print(
        f"Target masks:\n{TARGET_MASKS_DIR}"
    )

    print(
        f"Previews:\n{PREVIEW_DIR}"
    )

    print("=" * 75)

    if not FACES_DIR.exists():

        print(
            "ERROR: faces directory not found."
        )

        return

    if not RAW_MASKS_DIR.exists():

        print(
            "ERROR: raw masks directory not found."
        )

        return

    if not LANDMARKS_DIR.exists():

        print(
            "ERROR: landmarks directory not found."
        )

        return

    image_files = sorted(
        FACES_DIR.glob("*.png")
    )

    if TEST_MODE:

        image_files = (
            image_files[
                :TEST_LIMIT
            ]
        )

        print(
            f"TEST MODE: first "
            f"{TEST_LIMIT} images"
        )

    else:

        print(
            "FULL MODE: all images"
        )

    total = len(
        image_files
    )

    print(
        f"Images selected: {total}"
    )

    if total == 0:

        print(
            "ERROR: no face images found."
        )

        return

    # --------------------------------------------------------
    # Make output directories
    # --------------------------------------------------------

    for operation in OPERATIONS:

        (
            TARGET_MASKS_DIR
            / operation
        ).mkdir(
            parents=True,
            exist_ok=True
        )

        (
            PREVIEW_DIR
            / operation
        ).mkdir(
            parents=True,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    successful = 0
    failed = []

    for i, image_path in enumerate(
        image_files,
        start=1
    ):

        print(
            f"[{i}/{total}] "
            f"{image_path.name}",
            end=" ... "
        )

        try:

            success, reason = process_image(
                image_path
            )

        except Exception as e:

            success = False
            reason = str(e)

        if success:

            successful += 1

            print(
                "OK"
            )

        else:

            print(
                f"FAILED ({reason})"
            )

            failed.append({
                "id": image_path.stem,
                "file": image_path.name,
                "reason": reason
            })

    # --------------------------------------------------------
    # Failed report
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

    rate = (
        successful
        /
        total
        *
        100.0
        if total > 0
        else 0.0
    )

    print()
    print("=" * 75)
    print(
        "TARGET REGION SUMMARY"
    )
    print("=" * 75)

    print(
        f"Faces selected: {total}"
    )

    print(
        f"Successful:     {successful}"
    )

    print(
        f"Failed:          {len(failed)}"
    )

    print(
        f"Success rate:    {rate:.2f}%"
    )

    print()

    print(
        f"Target masks:\n"
        f"{TARGET_MASKS_DIR}"
    )

    print()

    print(
        f"Previews:\n"
        f"{PREVIEW_DIR}"
    )

    print()

    print(
        f"Failed report:\n"
        f"{FAILED_FILE}"
    )

    print("=" * 75)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()

