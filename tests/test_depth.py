

def test_depth_configs_disable_scale_changing_augmentation():
    """H-5: no zoom/rotation/shear/perspective for any depth experiment; identical augmentation everywhere."""
    from pathlib import Path

    from src.utils.config import load_config

    seen = set()
    for f in sorted(Path("configs/depth").glob("*.yaml")):
        if f.name.startswith("_"):
            continue
        t = load_config(f)["train"]
        assert t["scale"] == t["degrees"] == t["shear"] == t["perspective"] == 0.0, f
        seen.add(tuple(t[k] for k in ("hsv_h", "hsv_s", "hsv_v", "fliplr", "flipud", "translate")))
    assert len(seen) == 1


def test_calibration_split_is_validation_and_test_is_refused():
    import pytest

    from src.training.train_depth import calibrate_experiment
    from src.utils.config import apply_overrides, load_config

    cfg = load_config("configs/depth/full_kd.yaml")
    assert cfg["calibration"]["split"] == "val"
    with pytest.raises(ValueError, match="not allowed"):
        calibrate_experiment(apply_overrides(cfg, ["calibration.split=test"]), [], "cpu")


def test_letterbox_geometry_matches_ultralytics_sizes():
    """H-7: long side -> imgsz (aspect kept), centred padding; the four SUN RGB-D sensor sizes."""
    from models.depth.yolo26_depth import letterbox_geometry

    assert letterbox_geometry((530, 730), 640) == (465, 640, 87, 0)
    assert letterbox_geometry((441, 591), 640) == (478, 640, 81, 0)
    assert letterbox_geometry((427, 561), 640) == (488, 640, 76, 0)
    assert letterbox_geometry((480, 360), 640) == (640, 480, 0, 80)  # portrait
    assert letterbox_geometry((640, 640), 640) == (640, 640, 0, 0)


def test_preprocess_letterboxes_and_unletterbox_removes_padding():
    import numpy as np
    import torch

    from models.depth.yolo26_depth import PAD_VALUE, letterbox_geometry, preprocess, unletterbox

    bgr = np.zeros((530, 730, 3), np.uint8)
    t = preprocess(bgr, 640, "cpu")
    nh, nw, top, left = letterbox_geometry(bgr.shape[:2], 640)
    assert t.shape == (1, 3, 640, 640)
    assert torch.allclose(t[0, :, :top], torch.full_like(t[0, :, :top], PAD_VALUE / 255.0))
    assert torch.allclose(t[0, :, top + nh:], torch.full_like(t[0, :, top + nh:], PAD_VALUE / 255.0))
    assert float(t[0, :, top:top + nh].max()) == 0.0
    # a row-index ramp on the letterboxed grid: the crop must start at the first content row and keep every row
    rows = torch.arange(640, dtype=torch.float32).view(1, 1, 640, 1).expand(1, 1, 640, 640)
    out = unletterbox(rows, bgr.shape[:2], 640, out_size=(nh, nw))
    assert out.shape[-2:] == (nh, nw) and out[0, 0, 0, 0] == top and out[0, 0, -1, 0] == top + nh - 1
    # default size: content at the model's 1/4 output stride
    assert unletterbox(torch.ones(1, 1, 160, 160), bgr.shape[:2], 640).shape[-2:] == (116, 160)


def test_validation_letterbox_pads_depth_like_the_image():
    """The val transform pads the depth target with 0 (invalid) exactly where the image is padded."""
    import numpy as np

    from models.depth.yolo26_depth import PAD_VALUE, depth_letterbox_transform

    img = np.full((465, 640, 3), 7, np.uint8)  # as loaded by Ultralytics (long side already 640)
    depth = np.full((465, 640), 2.5, np.float32)
    out = depth_letterbox_transform(640)({"img": img, "depth": depth, "cls": np.zeros((0, 1))})
    assert out["img"].shape[:2] == out["depth"].shape == (640, 640)
    assert (out["img"][:87] == PAD_VALUE).all() and (out["depth"][:87] == 0).all()
    assert (out["img"][87:87 + 465] == 7).all() and (out["depth"][87:87 + 465] == 2.5).all()
    assert (out["depth"][87 + 465:] == 0).all()


def test_depth_trainer_validates_with_letterbox(synthetic_dataset, make_args, depth_pair):
    """ResearchDepthTrainer's val set uses the depth-aware letterbox; the train set keeps the Ultralytics pipeline."""
    from ultralytics.data.augment import LetterBox

    from src.training.train_depth import ResearchDepthTrainer

    tr = ResearchDepthTrainer.__new__(ResearchDepthTrainer)
    tr.args = make_args(imgsz=64, task="depth", rect=False, cache=False, single_cls=False, classes=None, fraction=1.0)
    tr.data = {"nc": 1, "names": {0: "depth"}, "channels": 3, "depth_scale": 1000, "max_depth": 10.0}
    tr.model = depth_pair[1]
    root = synthetic_dataset["root"]
    val = ResearchDepthTrainer.build_dataset(tr, str(root / "images/val"), mode="val", batch=2)
    lbs = [t for t in val.transforms.transforms if isinstance(t, LetterBox)]
    assert len(lbs) == 1 and type(lbs[0]).__name__ == "DepthLetterBox" and not lbs[0].scale_fill
    s = val[0]
    assert s["img"].shape[-2:] == s["depth"].shape[-2:] == (64, 64)
