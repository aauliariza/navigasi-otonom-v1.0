

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
