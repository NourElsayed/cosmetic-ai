from pathlib import Path
import json
import sys
import importlib
import importlib.util

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

FACES_DIR = (
    PROJECT_ROOT
    / "processed"
    / "faces"
)

RAW_MASKS_DIR = (
    PROJECT_ROOT
    / "processed"
    / "masks"
    /"raw"
)

FACE_PARSING_DIR = (
    PROJECT_ROOT
    / "third_party"
    / "face_parsing"
)

MODEL_PY = (
    FACE_PARSING_DIR
    / "model.py"
)

RESNET_PY = (
    FACE_PARSING_DIR
    / "resnet.py"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "face_parsing"
    / "79999_iter.pth"
)

FAILED_FILE = (
    PROJECT_ROOT
    / "processed"
    / "face_parsing_raw_failed.json"
)


# ============================================================
# SETTINGS
# ============================================================

TEST_MODE = False
TEST_LIMIT = 50

NUM_CLASSES = 19
INPUT_SIZE = 512


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# TRANSFORM
# ============================================================

TRANSFORM = transforms.Compose([

    transforms.Resize(
        (
            INPUT_SIZE,
            INPUT_SIZE
        )
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# CHECK FILES
# ============================================================

def check_required_files():

    required = {
        "model.py": MODEL_PY,
        "resnet.py": RESNET_PY,
        "79999_iter.pth": MODEL_PATH,
    }

    missing = []

    for name, path in required.items():

        if not path.exists():

            missing.append(
                f"{name}: {path}"
            )

    if missing:

        raise FileNotFoundError(
            "Missing BiSeNet files:\n"
            +
            "\n".join(missing)
        )


# ============================================================
# IMPORT RESNET
# ============================================================

def import_resnet_module():

    face_parsing_dir = str(
        FACE_PARSING_DIR
    )

    if face_parsing_dir not in sys.path:

        sys.path.insert(
            0,
            face_parsing_dir
        )

    importlib.invalidate_caches()

    if "resnet" in sys.modules:

        del sys.modules["resnet"]

    spec = (
        importlib.util
        .spec_from_file_location(
            "resnet",
            str(RESNET_PY)
        )
    )

    if spec is None or spec.loader is None:

        raise ImportError(
            "Could not create loader for:\n"
            f"{RESNET_PY}"
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    sys.modules["resnet"] = module

    try:

        spec.loader.exec_module(
            module
        )

    except Exception as e:

        raise ImportError(
            "Failed to load resnet.py.\n\n"
            f"{RESNET_PY}\n\n"
            f"{e}"
        )

    return module


# ============================================================
# IMPORT BISENET
# ============================================================

def import_bisenet_class():

    import_resnet_module()

    if "face_parsing_model" in sys.modules:

        del sys.modules[
            "face_parsing_model"
        ]

    spec = (
        importlib.util
        .spec_from_file_location(
            "face_parsing_model",
            str(MODEL_PY)
        )
    )

    if spec is None or spec.loader is None:

        raise ImportError(
            "Could not create loader for:\n"
            f"{MODEL_PY}"
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    sys.modules[
        "face_parsing_model"
    ] = module

    try:

        spec.loader.exec_module(
            module
        )

    except Exception as e:

        raise ImportError(
            "Failed to load model.py.\n\n"
            f"{MODEL_PY}\n\n"
            f"{e}"
        )

    if not hasattr(
        module,
        "BiSeNet"
    ):

        raise ImportError(
            "BiSeNet not found in model.py."
        )

    return module.BiSeNet


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("=" * 70)
    print("LOADING BISENET")
    print("=" * 70)

    print(
        f"model.py:\n{MODEL_PY}"
    )

    print(
        f"resnet.py:\n{RESNET_PY}"
    )

    print(
        f"checkpoint:\n{MODEL_PATH}"
    )

    print(
        f"device:\n{DEVICE}"
    )

    check_required_files()

    BiSeNet = (
        import_bisenet_class()
    )

    print(
        "BiSeNet class imported successfully."
    )

    model = BiSeNet(
        n_classes=NUM_CLASSES
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if (
        isinstance(checkpoint, dict)
        and
        "state_dict" in checkpoint
    ):

        state_dict = (
            checkpoint["state_dict"]
        )

    elif (
        isinstance(checkpoint, dict)
        and
        "model_state_dict" in checkpoint
    ):

        state_dict = (
            checkpoint["model_state_dict"]
        )

    elif (
        isinstance(checkpoint, dict)
        and
        "model" in checkpoint
    ):

        state_dict = (
            checkpoint["model"]
        )

    else:

        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[
                len("module.") :
            ]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    model.to(
        DEVICE
    )

    model.eval()

    print(
        "Checkpoint loaded successfully."
    )

    print(
        "BiSeNet ready."
    )

    print("=" * 70)

    return model


# ============================================================
# PARSE ONE IMAGE
# ============================================================

@torch.no_grad()
def parse_face(
    model,
    image_path
):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        return (
            False,
            None,
            "cannot_read_image"
        )

    height, width = (
        image.shape[:2]
    )

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(
        rgb
    )

    tensor = TRANSFORM(
        pil_image
    )

    tensor = tensor.unsqueeze(
        0
    )

    tensor = tensor.to(
        DEVICE
    )

    output = model(
        tensor
    )

    if isinstance(
        output,
        (tuple, list)
    ):

        if len(output) == 0:

            return (
                False,
                None,
                "empty_model_output"
            )

        output = output[0]

    if not torch.is_tensor(output):

        return (
            False,
            None,
            "model_output_not_tensor"
        )

    if output.ndim != 4:

        return (
            False,
            None,
            f"unexpected_output_shape_"
            f"{tuple(output.shape)}"
        )

    if output.shape[1] != NUM_CLASSES:

        return (
            False,
            None,
            f"unexpected_class_count_"
            f"{output.shape[1]}"
        )

    prediction = torch.argmax(
        output,
        dim=1
    )

    prediction = (
        prediction[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    prediction = cv2.resize(
        prediction,
        (
            width,
            height
        ),
        interpolation=cv2.INTER_NEAREST
    )

    return (
        True,
        prediction,
        None
    )


# ============================================================
# SAVE MASK
# ============================================================

def save_mask(
    mask,
    image_id
):

    RAW_MASKS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        RAW_MASKS_DIR
        / f"{image_id}_mask.png"
    )

    success = cv2.imwrite(
        str(output_path),
        mask
    )

    if not success:

        return (
            False,
            str(output_path)
        )

    return (
        True,
        None
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "RAW FACE PARSING MASK GENERATION"
    )
    print("=" * 70)

    print(
        f"Faces:\n{FACES_DIR}"
    )

    print(
        f"Output:\n{RAW_MASKS_DIR}"
    )

    print("=" * 70)

    if not FACES_DIR.exists():

        print(
            "ERROR: faces directory not found."
        )

        return

    try:

        model = load_model()

    except Exception as e:

        print()
        print(
            "FATAL MODEL ERROR:"
        )

        print(e)

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

            success, mask, reason = (
                parse_face(
                    model,
                    image_path
                )
            )

            if not success:

                print(
                    f"FAILED ({reason})"
                )

                failed.append({
                    "id": image_path.stem,
                    "file": image_path.name,
                    "reason": reason
                })

                continue

            save_success, save_reason = (
                save_mask(
                    mask,
                    image_path.stem
                )
            )

            if not save_success:

                print(
                    f"FAILED "
                    f"(save:{save_reason})"
                )

                failed.append({
                    "id": image_path.stem,
                    "file": image_path.name,
                    "reason": (
                        f"save_failed:"
                        f"{save_reason}"
                    )
                })

                continue

            classes = sorted(
                np.unique(mask).tolist()
            )

            print(
                f"OK | classes={classes}"
            )

            successful += 1

        except Exception as e:

            print(
                f"ERROR ({e})"
            )

            failed.append({
                "id": image_path.stem,
                "file": image_path.name,
                "reason": str(e)
            })

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

    print()
    print("=" * 70)
    print(
        "RAW FACE PARSING SUMMARY"
    )
    print("=" * 70)

    print(
        f"Faces selected: {total}"
    )

    print(
        f"Successful:     {successful}"
    )

    print(
        f"Failed:          {len(failed)}"
    )

    rate = (
        successful / total * 100
        if total
        else 0
    )

    print(
        f"Success rate:    {rate:.2f}%"
    )

    print(
        f"Raw masks:\n{RAW_MASKS_DIR}"
    )

    print(
        f"Failed report:\n{FAILED_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()

