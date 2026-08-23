from pathlib import Path
import json

import cv2
import numpy as np
import mediapipe as mp


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FACES_DIR = (
    PROJECT_ROOT
    / "processed"
    / "faces"
)

LANDMARKS_DIR = (
    PROJECT_ROOT
    / "processed"
    / "landmarks"
)

PREVIEW_DIR = (
    LANDMARKS_DIR
    / "previews"
)

FAILED_FILE = (
    PROJECT_ROOT
    / "processed"
    / "landmarks_failed.json"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "face_landmarker.task"
)


# ============================================================
# Processing settings
# ============================================================

# مهم:
# False = يعالج كل الصور الموجودة في faces
# True  = يعالج أول TEST_LIMIT فقط
TEST_MODE = False

TEST_LIMIT = 50

MIN_SUCCESS_RATE = 95.0


# ============================================================
# Preview settings
# ============================================================

# True = يحفظ previews
# False = لا يحفظ previews
SAVE_GROUP_PREVIEWS = True

# لأننا نريد preview لكل الصور
PREVIEW_ALL_IMAGES = True

PREVIEW_GROUPS = [
    "jaw",
    "nose",
    "eyes",
    "lips",
    "chin",
    "brows",
]


# ============================================================
# MediaPipe
# ============================================================

BaseOptions = mp.tasks.BaseOptions

FaceLandmarker = (
    mp.tasks.vision.FaceLandmarker
)

FaceLandmarkerOptions = (
    mp.tasks.vision.FaceLandmarkerOptions
)

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)


# ============================================================
# Landmark groups
# ============================================================

# ------------------------------------------------------------
# Complete MediaPipe Face Oval
# ------------------------------------------------------------

FACE_OVAL = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
]


# ============================================================
# JAW
# ============================================================

JAW_LEFT = [
    152,
    148,
    176,
    149,
    150,
    136,
    172,

]


JAW_RIGHT = [
    397,
    365,
    379,
    378,
    400,
    377,
    152,
]


JAW = [
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
]


# ============================================================
# CHIN
# ============================================================

CHIN = [
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
]


# ============================================================
# Nose
# ============================================================

NOSE = [
    1,
    2,
    98,
    327,
    168,
    197,
    5,
    4,
    195,
    48,
    278,
    64,
    294,
    275,
    440,
    344,
    220,
    115,
]


# ============================================================
# Eyes
# ============================================================

LEFT_EYE = [
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    173,
    157,
    158,
    159,
    160,
    161,
    246,
]


RIGHT_EYE = [
    362,
    382,
    381,
    380,
    374,
    373,
    390,
    249,
    263,
    466,
    388,
    387,
    386,
    385,
    384,
    398,
]


# ============================================================
# Lips
# ============================================================

LIPS = [
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    308,
    324,
    318,
    402,
    317,
    14,
    87,
    178,
    88,
    95,
    78,
    191,
    80,
    81,
    82,
    13,
    312,
    311,
    310,
    415,
]


# ============================================================
# Brows
# ============================================================

LEFT_BROW = [
    46,
    53,
    52,
    65,
    55,
    70,
    63,
    105,
    66,
    107,
]


RIGHT_BROW = [
    276,
    283,
    282,
    295,
    285,
    300,
    293,
    334,
    296,
    336,
]


# ============================================================
# Final groups
# ============================================================

LANDMARK_GROUPS = {

    "jaw": JAW,

    "jaw_left": JAW_LEFT,

    "jaw_right": JAW_RIGHT,

    "chin": CHIN,

    "nose": NOSE,

    "left_eye": LEFT_EYE,

    "right_eye": RIGHT_EYE,

    "eyes": LEFT_EYE + RIGHT_EYE,

    "lips": LIPS,

    "left_brow": LEFT_BROW,

    "right_brow": RIGHT_BROW,

    "brows": LEFT_BROW + RIGHT_BROW,
}


# ============================================================
# Utility
# ============================================================

def ensure_directories():

    LANDMARKS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PREVIEW_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for group_name in PREVIEW_GROUPS:

        (
            PREVIEW_DIR
            / group_name
        ).mkdir(
            parents=True,
            exist_ok=True
        )


def load_image(path):

    image = cv2.imread(
        str(path)
    )

    if image is None:
        return None

    return image


def create_mp_image(image):

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


# ============================================================
# Landmark serialization
# ============================================================

def landmark_to_dict(
    landmark,
    index,
    width,
    height
):

    x = float(
        landmark.x
    )

    y = float(
        landmark.y
    )

    pixel_x = x * width

    pixel_y = y * height

    result = {

        "index": int(index),

        "x": x,

        "y": y,

        "z": float(
            landmark.z
        ),

        "pixel_x": float(
            pixel_x
        ),

        "pixel_y": float(
            pixel_y
        ),
    }

    visibility = getattr(
        landmark,
        "visibility",
        None
    )

    if visibility is not None:

        try:

            result["visibility"] = float(
                visibility
            )

        except (
            TypeError,
            ValueError
        ):
            pass

    presence = getattr(
        landmark,
        "presence",
        None
    )

    if presence is not None:

        try:

            result["presence"] = float(
                presence
            )

        except (
            TypeError,
            ValueError
        ):
            pass

    return result


# ============================================================
# Group creation
# ============================================================

def create_groups(all_landmarks):

    by_index = {
        point["index"]: point
        for point in all_landmarks
    }

    groups = {}

    for group_name, indices in (
        LANDMARK_GROUPS.items()
    ):

        group_points = []

        for index in indices:

            if index not in by_index:
                continue

            group_points.append(
                by_index[index]
            )

        groups[group_name] = (
            group_points
        )

    return groups


# ============================================================
# Validation
# ============================================================

def validate_landmarks(
    all_landmarks,
    groups
):

    if not all_landmarks:

        return (
            False,
            "no_face_landmarks"
        )

    if len(all_landmarks) < 400:

        return (
            False,
            f"too_few_landmarks_{len(all_landmarks)}"
        )

    required_groups = [
        "jaw",
        "chin",
        "nose",
        "eyes",
        "lips",
    ]

    for group_name in required_groups:

        points = groups.get(
            group_name,
            []
        )

        if len(points) < 3:

            return (
                False,
                f"invalid_group_{group_name}"
            )

    for point in all_landmarks:

        x = point["x"]
        y = point["y"]
        z = point["z"]

        if not (
            np.isfinite(x)
            and
            np.isfinite(y)
            and
            np.isfinite(z)
        ):

            return (
                False,
                "non_finite_landmark"
            )

        if not (
            -0.5 <= x <= 1.5
            and
            -0.5 <= y <= 1.5
        ):

            return (
                False,
                "invalid_landmark_coordinates"
            )

    return (
        True,
        None
    )


# ============================================================
# Save JSON
# ============================================================

def save_landmarks(
    output_path,
    image_id,
    image_width,
    image_height,
    all_landmarks,
    groups
):

    data = {

        "image_id": image_id,

        "image": {
            "width": int(
                image_width
            ),

            "height": int(
                image_height
            ),
        },

        "model": {
            "name": (
                "MediaPipe Face Landmarker"
            ),

            "mode": "IMAGE",
        },

        "landmarks": {

            "count": len(
                all_landmarks
            ),

            # Raw MediaPipe landmarks
            "points": all_landmarks,

            # Groups based on the same raw points
            "groups": groups,
        },
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# Preview drawing
# ============================================================

def draw_points(
    image,
    points,
    radius=3,
    color=(0, 255, 0)
):

    for point in points:

        x = int(
            round(
                point["pixel_x"]
            )
        )

        y = int(
            round(
                point["pixel_y"]
            )
        )

        cv2.circle(
            image,
            (x, y),
            radius,
            color,
            -1
        )


def draw_polyline(
    image,
    points,
    color=(0, 255, 255),
    thickness=1
):

    if len(points) < 2:
        return

    pts = np.array(
        [
            [
                int(
                    round(
                        point["pixel_x"]
                    )
                ),
                int(
                    round(
                        point["pixel_y"]
                    )
                )
            ]
            for point in points
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


# ============================================================
# Create ONE group preview
# ============================================================

def create_group_preview(
    image,
    group_name,
    groups
):

    preview = image.copy()

    points = groups.get(
        group_name,
        []
    )

    if len(points) == 0:
        return preview

    # --------------------------------------------------------
    # Jaw
    #
    # نقاط + توصيل فقط
    # لا يتم تعديل أي coordinate.
    # --------------------------------------------------------

    if group_name == "jaw":

        draw_polyline(
            preview,
            points,
            color=(0, 255, 255),
            thickness=1
        )

        draw_points(
            preview,
            points,
            radius=3,
            color=(0, 255, 255)
        )

    # --------------------------------------------------------
    # Eyes
    # --------------------------------------------------------

    elif group_name == "eyes":

        draw_points(
            preview,
            points,
            radius=3,
            color=(0, 255, 0)
        )

        draw_polyline(
            preview,
            points,
            color=(0, 255, 0),
            thickness=1
        )

    # --------------------------------------------------------
    # Nose
    # --------------------------------------------------------

    elif group_name == "nose":

        draw_points(
            preview,
            points,
            radius=3,
            color=(255, 0, 0)
        )

        draw_polyline(
            preview,
            points,
            color=(255, 0, 0),
            thickness=1
        )

    # --------------------------------------------------------
    # Lips
    # --------------------------------------------------------

    elif group_name == "lips":

        draw_points(
            preview,
            points,
            radius=3,
            color=(0, 0, 255)
        )

        draw_polyline(
            preview,
            points,
            color=(0, 0, 255),
            thickness=1
        )

    # --------------------------------------------------------
    # Chin
    # --------------------------------------------------------

    elif group_name == "chin":

        draw_polyline(
            preview,
            points,
            color=(255, 0, 255),
            thickness=1
        )

        draw_points(
            preview,
            points,
            radius=3,
            color=(255, 0, 255)
        )

    # --------------------------------------------------------
    # Brows
    # --------------------------------------------------------

    elif group_name == "brows":

        draw_points(
            preview,
            points,
            radius=3,
            color=(255, 255, 0)
        )

        draw_polyline(
            preview,
            points,
            color=(255, 255, 0),
            thickness=1
        )

    # --------------------------------------------------------
    # Default
    # --------------------------------------------------------

    else:

        draw_points(
            preview,
            points,
            radius=3
        )

    return preview


# ============================================================
# Save separate group previews
# ============================================================

def save_group_previews(
    image,
    image_id,
    groups
):

    if not SAVE_GROUP_PREVIEWS:
        return True, None

    for group_name in PREVIEW_GROUPS:

        points = groups.get(
            group_name,
            []
        )

        if len(points) == 0:

            return (
                False,
                f"empty_preview_group_{group_name}"
            )

        preview = create_group_preview(
            image,
            group_name,
            groups
        )

        group_dir = (
            PREVIEW_DIR
            / group_name
        )

        group_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        preview_path = (
            group_dir
            / f"{image_id}.png"
        )

        success = cv2.imwrite(
            str(preview_path),
            preview
        )

        if not success:

            return (
                False,
                f"preview_save_failed_{group_name}"
            )

    return (
        True,
        None
    )


# ============================================================
# Process one image
# ============================================================

def process_image(
    image_path,
    landmarker,
    preview_enabled=True
):

    image_id = image_path.stem

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = load_image(
        image_path
    )

    if image is None:

        return (
            False,
            "cannot_read_image"
        )

    height, width = (
        image.shape[:2]
    )

    # --------------------------------------------------------
    # Create MediaPipe image
    # --------------------------------------------------------

    try:

        mp_image = create_mp_image(
            image
        )

    except Exception as e:

        return (
            False,
            f"mediapipe_image_error: {e}"
        )

    # --------------------------------------------------------
    # Detect
    # --------------------------------------------------------

    try:

        result = landmarker.detect(
            mp_image
        )

    except Exception as e:

        return (
            False,
            f"landmarker_error: {e}"
        )

    # --------------------------------------------------------
    # No face
    # --------------------------------------------------------

    if (
        result.face_landmarks is None
        or
        len(result.face_landmarks) == 0
    ):

        return (
            False,
            "no_face_detected"
        )

    # --------------------------------------------------------
    # Multiple faces
    # --------------------------------------------------------

    if len(result.face_landmarks) > 1:

        return (
            False,
            f"multiple_faces_detected_"
            f"{len(result.face_landmarks)}"
        )

    # --------------------------------------------------------
    # Face landmarks
    # --------------------------------------------------------

    face_landmarks = (
        result.face_landmarks[0]
    )

    # --------------------------------------------------------
    # Serialize original MediaPipe landmarks
    # --------------------------------------------------------

    all_landmarks = []

    for index, landmark in enumerate(
        face_landmarks
    ):

        all_landmarks.append(
            landmark_to_dict(
                landmark,
                index,
                width,
                height
            )
        )

    # --------------------------------------------------------
    # Groups
    #
    # لا يوجد أي تعديل على landmarks.
    # --------------------------------------------------------

    groups = create_groups(
        all_landmarks
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    valid, reason = validate_landmarks(
        all_landmarks,
        groups
    )

    if not valid:

        return (
            False,
            reason
        )

    # --------------------------------------------------------
    # JSON output
    # --------------------------------------------------------

    output_path = (
        LANDMARKS_DIR
        / f"{image_id}.json"
    )

    try:

        save_landmarks(
            output_path,
            image_id,
            width,
            height,
            all_landmarks,
            groups
        )

    except Exception as e:

        return (
            False,
            f"json_save_error: {e}"
        )

    # --------------------------------------------------------
    # Preview
    #
    # الرسم فقط.
    # لا يؤثر على landmarks.
    # --------------------------------------------------------

    if preview_enabled:

        try:

            success, reason = (
                save_group_previews(
                    image,
                    image_id,
                    groups
                )
            )

            if not success:

                return (
                    False,
                    reason
                )

        except Exception as e:

            return (
                False,
                f"preview_error: {e}"
            )

    return (
        True,
        None
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)

    print(
        "MEDIAPIPE FACE LANDMARK PIPELINE"
    )

    print("=" * 70)

    print(
        f"Faces:\n{FACES_DIR}"
    )

    print(
        f"Model:\n{MODEL_PATH}"
    )

    print(
        f"Landmarks:\n{LANDMARKS_DIR}"
    )

    print(
        f"Previews:\n{PREVIEW_DIR}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Check input directory
    # --------------------------------------------------------

    if not FACES_DIR.exists():

        print(
            "ERROR: faces directory not found."
        )

        return

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        print(
            "ERROR: face_landmarker.task not found."
        )

        print(
            f"Expected:\n{MODEL_PATH}"
        )

        return

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    ensure_directories()

    # --------------------------------------------------------
    # Find all PNG images
    # --------------------------------------------------------

    image_files = sorted(
        FACES_DIR.glob("*.png")
    )

    # --------------------------------------------------------
    # TEST MODE
    # --------------------------------------------------------

    if TEST_MODE:

        image_files = (
            image_files[
                :TEST_LIMIT
            ]
        )

        print(
            f"TEST MODE: first {TEST_LIMIT} images"
        )

    else:

        print(
            "FULL MODE: all images in faces"
        )

    total = len(
        image_files
    )

    print(
        f"Images to process: {total}"
    )

    # --------------------------------------------------------
    # No images
    # --------------------------------------------------------

    if total == 0:

        print(
            "ERROR: no PNG images found."
        )

        return

    # ========================================================
    # Face Landmarker options
    # ========================================================

    options = FaceLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=str(
                MODEL_PATH
            )
        ),

        running_mode=VisionRunningMode.IMAGE,

        num_faces=1,

        min_face_detection_confidence=0.5,

        min_face_presence_confidence=0.5,

        min_tracking_confidence=0.5,
    )

    failed = []

    successful = 0

    # ========================================================
    # Run MediaPipe
    # ========================================================

    try:

        with FaceLandmarker.create_from_options(
            options
        ) as landmarker:

            for i, image_path in enumerate(
                image_files,
                start=1
            ):

                print(
                    f"[{i}/{total}] "
                    f"{image_path.name}",
                    end=" ... "
                )

                # ------------------------------------------------
                # Preview لكل الصور
                # ------------------------------------------------

                if (
                    SAVE_GROUP_PREVIEWS
                    and
                    PREVIEW_ALL_IMAGES
                ):

                    preview_enabled = True

                elif SAVE_GROUP_PREVIEWS:

                    preview_enabled = (
                        i <= 10
                    )

                else:

                    preview_enabled = False

                # ------------------------------------------------
                # Process
                # ------------------------------------------------

                try:

                    success, reason = (
                        process_image(
                            image_path,
                            landmarker,
                            preview_enabled
                        )
                    )

                except Exception as e:

                    success = False

                    reason = str(e)

                # ------------------------------------------------
                # Result
                # ------------------------------------------------

                if success:

                    successful += 1

                    if preview_enabled:

                        print(
                            "OK + PREVIEWS"
                        )

                    else:

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

                        "reason": reason,

                    })

    except Exception as e:

        print()

        print(
            "FATAL ERROR while creating "
            "Face Landmarker:"
        )

        print(e)

        return

    # ========================================================
    # Save failed report
    # ========================================================

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

    # ========================================================
    # Summary
    # ========================================================

    failed_count = len(
        failed
    )

    success_rate = (
        successful / total * 100
        if total > 0
        else 0
    )

    print()

    print("=" * 70)

    print(
        "LANDMARK SUMMARY"
    )

    print("=" * 70)

    print(
        f"Images processed: {total}"
    )

    print(
        f"Successful:       {successful}"
    )

    print(
        f"Failed:            {failed_count}"
    )

    print(
        f"Success rate:      {success_rate:.2f}%"
    )

    print(
        f"JSON output:\n{LANDMARKS_DIR}"
    )

    print(
        f"Separate previews:\n{PREVIEW_DIR}"
    )

    print(
        f"Failed report:\n{FAILED_FILE}"
    )

    print("=" * 70)

    print()

    print(
        "PREVIEW GROUPS:"
    )

    for group_name in PREVIEW_GROUPS:

        print(
            f"  {group_name}: "
            f"{PREVIEW_DIR / group_name}"
        )

    print()

    print(
        f"Preview all images: "
        f"{PREVIEW_ALL_IMAGES}"
    )

    print()

    # ========================================================
    # Final result
    # ========================================================

    if success_rate >= MIN_SUCCESS_RATE:

        print(
            "RESULT: PASS"
        )

        print(
            f"Landmark success rate "
            f"{success_rate:.2f}% >= "
            f"{MIN_SUCCESS_RATE:.2f}%"
        )

    else:

        print(
            "RESULT: REVIEW"
        )

        print(
            f"Landmark success rate "
            f"{success_rate:.2f}% < "
            f"{MIN_SUCCESS_RATE:.2f}%"
        )

    print("=" * 70)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()