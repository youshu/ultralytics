# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Regression checks for classification geometry, checkpoints and exported contracts."""

import json

import cv2
import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms as T

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.augment import ClassifyLetterBox, classify_augmentations, classify_transforms
from ultralytics.data.classification import (
    classification_preprocess,
    classification_tensor,
    classification_transforms_from_spec,
    resolve_classification_preprocess,
    write_classification_sidecar,
)
from ultralytics.data.dataset import ClassificationDataset


def test_default_matches_native_transforms():
    image = Image.fromarray(np.random.default_rng(4).integers(0, 256, (41, 101, 3), dtype=np.uint8))
    native = T.Compose([T.Resize(64), T.CenterCrop(64), T.ToTensor(), T.Normalize([0] * 3, [1] * 3)])
    torch.testing.assert_close(classify_transforms(64)(image), native(image), rtol=0, atol=0)
    assert isinstance(classify_augmentations(64).transforms[0], T.RandomResizedCrop)


@pytest.mark.parametrize("shape", [(40, 80), (80, 40), (50, 50), (1, 2000), (2000, 1), (13, 27)])
@pytest.mark.parametrize("scaleup", [True, False])
@pytest.mark.parametrize("size", [64, 224])
def test_geometry_and_color(shape, scaleup, size):
    bgr = np.full((*shape, 3), (17, 81, 203), dtype=np.uint8)
    spec = classification_preprocess(size, "letterbox", scaleup)
    tensor = classification_tensor(bgr, spec)
    assert tensor.shape == (1, 3, size, size)
    pixels = tensor[0].permute(1, 2, 0).reshape(-1, 3)
    assert ((pixels == torch.tensor([203, 81, 17]) / 255).all(1)).any()
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    expected = ClassifyLetterBox(size, scaleup=scaleup)(rgb)
    np.testing.assert_array_equal((tensor[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8), expected)


def test_scale_does_not_crop_letterbox():
    image = Image.fromarray(np.full((40, 80, 3), 22, dtype=np.uint8))
    kw = {
        "size": 64,
        "hflip": 0,
        "vflip": 0,
        "hsv_h": 0,
        "hsv_s": 0,
        "hsv_v": 0,
        "erasing": 0,
        "classify_resize": "letterbox",
    }
    torch.manual_seed(7)
    a = classify_augmentations(scale=(0.1, 1), **kw)(image)
    torch.manual_seed(7)
    b = classify_augmentations(scale=(1, 1), **kw)(image)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_roundtrip_prediction_and_size_switch(tmp_path):
    torch.set_num_threads(2)
    model = YOLO("yolo11n-cls.yaml")
    spec = classification_preprocess(64, "letterbox", False)
    model.model.classify_preprocess = spec
    model.model.transforms = classification_transforms_from_spec(spec)
    model.model.args.update(imgsz=64, classify_resize="letterbox", letterbox_scaleup=False)
    path = tmp_path / "model.pt"
    model.save(path)
    restored = YOLO(path)
    image = tmp_path / "wide.png"
    cv2.imwrite(str(image), np.full((20, 80, 3), 33, dtype=np.uint8))
    result = restored.predict(str(image), device="cpu", verbose=False)[0]
    assert result.path == str(image)
    assert result.orig_img.shape == (20, 80, 3)
    assert restored.predictor.classify_preprocess == spec
    restored.predict(str(image), device="cpu", verbose=False, imgsz=96)
    assert restored.predictor.classify_preprocess["imgsz"] == [96, 96]
    restored.predict(str(image), device="cpu", verbose=False, classify_resize="default")
    assert restored.predictor.classify_preprocess["classify_resize"] == "default"
    restored.predict(str(image), device="cpu", verbose=False)
    assert restored.predictor.classify_preprocess == spec


def test_dataset_train_eval_parity(tmp_path):
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        cv2.imwrite(str(tmp_path / name / "x.png"), np.full((20, 80, 3), 100, dtype=np.uint8))
    args = get_cfg(
        overrides={
            "imgsz": 64,
            "classify_resize": "letterbox",
            "letterbox_scaleup": False,
            "scale": 0.2,
            "auto_augment": None,
            "hsv_h": 0,
            "hsv_s": 0,
            "hsv_v": 0,
            "fliplr": 0,
            "flipud": 0,
            "erasing": 0,
            "cache": False,
        }
    )
    train = ClassificationDataset(str(tmp_path), args, augment=True)
    val = ClassificationDataset(str(tmp_path), args, augment=False)
    torch.testing.assert_close(train[0]["img"], val[0]["img"], rtol=0, atol=0)


def test_export_metadata_and_conflicts(tmp_path):
    # A real TensorRT metadata header can be validated without installing a GPU runtime.
    spec = classification_preprocess(64, "letterbox", False)
    header = json.dumps({"classify_preprocess": spec, "imgsz": [64, 64], "args": {"dynamic": False}}).encode()
    path = tmp_path / "model.engine"
    path.write_bytes(len(header).to_bytes(4, "little") + header + b"engine-body")
    write_classification_sidecar(path, spec)
    assert resolve_classification_preprocess(path, require_metadata=True) == spec
    with pytest.raises(ValueError, match="static"):
        resolve_classification_preprocess(path, imgsz=96)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="sidecar"):
        resolve_classification_preprocess(path)
