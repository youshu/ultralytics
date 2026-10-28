# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Portable classification preprocessing contracts shared by datasets and deployment consumers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    import torchvision.transforms as T

import numpy as np

PREPROCESS_VERSION = 1


def classification_preprocess(
    imgsz: int | list[int] | tuple[int, int] = 224, classify_resize: str = "default", letterbox_scaleup: bool = True
) -> dict:
    """Build a JSON-serializable, deterministic classification preprocessing description."""
    if classify_resize not in {"default", "letterbox"}:
        raise ValueError("classify_resize must resolve to default or letterbox")
    size = list(imgsz) if isinstance(imgsz, (list, tuple)) else [imgsz, imgsz]
    if len(size) == 1:
        size *= 2
    if len(size) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in size):
        raise ValueError("classification imgsz must contain positive integer dimensions")
    if classify_resize == "letterbox" and (size[0] != size[1] or size[0] % 32):
        raise ValueError("classification letterbox requires a square imgsz divisible by 32")
    if not isinstance(letterbox_scaleup, bool):
        raise ValueError("letterbox_scaleup must be a boolean")
    result = {
        "schema_version": PREPROCESS_VERSION,
        "kind": "ultralytics-classify",
        "classify_resize": classify_resize,
        "imgsz": size,
        "color_order": "RGB",
        "layout": "CHW",
        "pixel_scale": 1 / 255,
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
    }
    if classify_resize == "letterbox":
        result.update(
            letterbox_scaleup=letterbox_scaleup,
            interpolation="opencv-linear",
            padding_value=114,
            center=True,
            auto=False,
        )
    else:
        result.update(interpolation="torchvision-bilinear", crop="center")
    return result


def validate_classification_preprocess(spec: dict) -> dict:
    """Validate a supported contract without silently accepting different preprocessing semantics."""
    if not isinstance(spec, dict) or spec.get("schema_version") != PREPROCESS_VERSION:
        raise ValueError("Missing or unsupported classification preprocessing metadata; re-train/export the model")
    expected = classification_preprocess(
        spec.get("imgsz"), spec.get("classify_resize"), spec.get("letterbox_scaleup", True)
    )
    if spec != expected:
        raise ValueError("Invalid classification preprocessing metadata")
    return expected


def resolve_classification_preprocess(
    model: Any,
    imgsz: int | list[int] | tuple[int, int] | None = None,
    classify_resize: str | None = None,
    letterbox_scaleup: bool | None = None,
    *,
    require_metadata: bool = False,
) -> dict:
    """Resolve explicit overrides against a model, backend, or exported model path.

    Args:
        model (Any): PyTorch model, backend, or exported model path.
        imgsz (int | list[int] | tuple[int, int] | None): Explicit target size, or None to follow the model.
        classify_resize (str | None): Spatial policy: auto, default, or letterbox.
        letterbox_scaleup (bool | None): Explicit enlargement policy, or None to follow the model.
        require_metadata (bool): Reject models without a recorded preprocessing contract.

    Returns:
        (dict): Validated effective preprocessing, including explicit overrides.

    Raises:
        ValueError: The contract, sidecar hash, or static export size is inconsistent.
    """
    metadata = {}
    exported = isinstance(model, (str, Path)) and Path(model).suffix.lower() != ".pt"
    if exported:
        from ultralytics.nn.backends.base import BaseBackend

        metadata = BaseBackend.read_metadata(model)
        saved = metadata.get("classify_preprocess")
    else:
        saved = getattr(model, "classify_preprocess", None)
        if saved is None:
            saved = getattr(getattr(model, "model", None), "classify_preprocess", None)
        metadata = getattr(model, "metadata", {}) or {}
        saved = saved or metadata.get("classify_preprocess")
        exported = getattr(model, "format", "pt") != "pt"
    if isinstance(saved, str):
        saved = json.loads(saved)
    if saved is not None:
        saved = validate_classification_preprocess(saved)
    elif require_metadata:
        raise ValueError("Business YOLO models require classification preprocessing metadata; re-train/export")
    mode = classify_resize if classify_resize not in (None, "auto") else (saved or {}).get("classify_resize", "default")
    size = imgsz if imgsz is not None else (saved or {}).get("imgsz")
    if size is None:
        size = metadata.get("imgsz") or (getattr(model, "args", {}) or {}).get("imgsz", 224)
    scaleup = letterbox_scaleup if letterbox_scaleup is not None else (saved or {}).get("letterbox_scaleup", True)
    resolved = classification_preprocess(size, mode, scaleup)
    if exported and saved:
        import ast

        metadata_size = metadata.get("imgsz")
        if isinstance(metadata_size, str):
            metadata_size = ast.literal_eval(metadata_size)
        if metadata_size is not None and list(metadata_size) != saved["imgsz"]:
            raise ValueError("Classification metadata differs from the exported input shape")
        args = metadata.get("args", {})
        if isinstance(args, str):
            args = ast.literal_eval(args)
        if not args.get("dynamic", False) and resolved["imgsz"] != saved["imgsz"]:
            raise ValueError("Requested classification imgsz differs from the static exported input shape")
    source = model if isinstance(model, (str, Path)) else getattr(model, "weight", None)
    if isinstance(source, (str, Path)):
        sidecar = Path(str(source) + ".preprocess.json")
        if sidecar.exists():
            manifest = json.loads(sidecar.read_text())
            if manifest.get("model_sha256") != _file_hash(source) or manifest.get("preprocess") != saved:
                raise ValueError("Classification preprocessing sidecar does not match the model")
    return resolved


def classification_transforms_from_spec(spec: dict) -> T.Compose:
    """Restore the deterministic PIL-to-tensor transform recorded in a contract."""
    from ultralytics.data.augment import classify_transforms

    spec = validate_classification_preprocess(spec)
    return classify_transforms(
        size=spec["imgsz"],
        classify_resize=spec["classify_resize"],
        letterbox_scaleup=spec.get("letterbox_scaleup", True),
    )


def classification_tensor(image_bgr: np.ndarray, spec: dict) -> torch.Tensor:
    """Preprocess a decoded uint8 BGR image into a float32 RGB BCHW tensor exactly once."""
    from PIL import Image

    if (
        not isinstance(image_bgr, np.ndarray)
        or image_bgr.dtype != np.uint8
        or image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
        or not image_bgr.size
    ):
        raise ValueError("Classification preprocessing requires a nonempty uint8 BGR image")
    rgb = Image.fromarray(np.ascontiguousarray(image_bgr[:, :, ::-1]))
    return classification_transforms_from_spec(spec)(rgb).unsqueeze(0)


def _file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_classification_sidecar(path: str | Path, spec: dict) -> Path:
    """Write preprocessing metadata bound to the completed exported model bytes."""
    path = Path(path)
    manifest = {"preprocess": validate_classification_preprocess(spec), "model_sha256": _file_hash(path)}
    sidecar = Path(str(path) + ".preprocess.json")
    temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    temporary.replace(sidecar)
    return sidecar
