"""The scratch scenario must differ from the pretrained one only where intended, and stay inside its own folder.

Run from the repository root::

    CUDA_VISIBLE_DEVICES="" python -m pytest scenarios/scratch/tests
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import models  # noqa: E402,F401
from src.utils.checkpoint import experiment_dir  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402

SCENARIO = ROOT / "scenarios" / "scratch"
TASKS = ("detection", "depth")
ALLOWED_DIFF = {"model.weights", "train.epochs", "train.patience", "train.lr0", "output.root", "kd.teacher_config", "_source_"}


def experiments(task: str) -> list[str]:
    return [p.name for p in sorted((ROOT / "configs" / task).glob("*.yaml")) if not p.name.startswith("_")]


ALL = [(t, f) for t in TASKS for f in experiments(t)]


def flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        out.update(flat(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
    return out


def scratch(task: str, name: str) -> dict:
    return load_config(SCENARIO / "configs" / task / name)


def test_every_experiment_is_mirrored():
    assert len(ALL) == 24
    for task in TASKS:
        assert sorted(p.name for p in (SCENARIO / "configs" / task).glob("*.yaml")) == experiments(task)


@pytest.mark.parametrize(("task", "name"), ALL)
def test_config_differs_only_where_intended(task, name):
    a, b = flat(load_config(ROOT / "configs" / task / name)), flat(scratch(task, name))
    assert {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)} <= ALLOWED_DIFF
    assert b["model.weights"] is None and b["experiment.id"] == a["experiment.id"]
    if task == "depth":  # scratch keeps the Tabel 3.8 LR; the smaller fine-tuning LR (H-8) is pretrained-only
        assert b["train.lr0"] == (0.0005 if name.startswith("teacher_x") else 0.001)
    else:
        assert b["train.lr0"] == a["train.lr0"]


@pytest.mark.parametrize(("task", "name"), ALL)
def test_outputs_and_teacher_stay_inside_scenario(task, name):
    cfg = scratch(task, name)
    assert experiment_dir(cfg).is_relative_to(SCENARIO / "outputs")
    if cfg.get("kd", {}).get("enabled"):
        teacher = load_config(cfg["kd"]["teacher_config"])
        assert experiment_dir(teacher).is_relative_to(SCENARIO / "outputs") and teacher["model"]["weights"] is None
        assert teacher["experiment"]["id"] == load_config(load_config(ROOT / "configs" / task / name)["kd"]["teacher_config"])["experiment"]["id"]


@pytest.mark.parametrize(("task", "name"), ALL)
def test_ultralytics_gets_no_pretrained_weights(task, name):
    from src.training.trainer_utils import build_ultralytics_args

    assert build_ultralytics_args(scratch(task, name), task, "m.yaml", False)["pretrained"] is False


@pytest.mark.parametrize("task", TASKS)
def test_training_script_runs_all_teachers_first(task):
    rows = re.findall(r"^run_experiment \S+ (\S+) (\S+)", (SCENARIO / f"train_all_{task}.sh").read_text(), re.M)
    assert sorted(Path(c).name for c, _ in rows) == experiments(task)
    done = set()
    for config, exp_id in rows:
        cfg = load_config(SCENARIO / "configs" / config)
        assert cfg["experiment"]["id"] == exp_id
        if cfg.get("kd", {}).get("enabled"):
            assert load_config(cfg["kd"]["teacher_config"])["experiment"]["id"] in done
        done.add(exp_id)


def test_evaluation_and_selection_stay_inside_scenario():
    for task in TASKS:
        ev = load_config(SCENARIO / "configs/evaluation" / f"{task}.yaml")
        assert resolve_path(ev["output_dir"]).is_relative_to(SCENARIO / "outputs")
        assert all(experiment_dir(load_config(m["config"])).is_relative_to(SCENARIO / "outputs") for m in ev["models"])
        assert len(ev["models"]) == len(experiments(task))
    sel = load_config(SCENARIO / "configs/evaluation/selection.yaml")
    original = load_config(ROOT / "configs/evaluation/selection.yaml")
    assert resolve_path(sel["selected_models_path"]).is_relative_to(SCENARIO / "outputs")
    for task in TASKS:
        assert resolve_path(sel[task]["results"]).is_relative_to(SCENARIO / "outputs")
        assert resolve_path(sel[task]["output_dir"]).is_relative_to(SCENARIO / "outputs")
        assert sel[task]["criteria"] == original[task]["criteria"]


@pytest.mark.slow
def test_real_training_starts_from_random_weights(tmp_path, monkeypatch):
    """Real 1-epoch runs (plain, attention, KD) receive no weights; artefacts land in tmp only."""
    from scripts_helpers import build_synthetic_dataset
    from src.training import train_depth, train_depth_kd, train_detection, train_detection_kd
    from src.training.trainer_utils import ResearchTrainerMixin

    seen = []
    original = ResearchTrainerMixin.get_model

    def spy(self, cfg=None, weights=None, verbose=True):
        seen.append(weights)
        return original(self, cfg, weights, verbose)

    monkeypatch.setattr(ResearchTrainerMixin, "get_model", spy)
    d = build_synthetic_dataset(tmp_path, extra=["split.mode=proposal"])
    common = [f"data.integrity_report={tmp_path / 'report/integrity_report.json'}", "train.epochs=1", "train.batch=3", "train.imgsz=96",
              "train.workers=0", "train.device=cpu", "train.amp=false", "train.plots=false", f"output.root={tmp_path / 'runs'}"]
    det = [*common, f"data.yaml={d['tmp'] / 'data/detect.yaml'}"]
    dep = [*common, f"data.yaml={d['tmp'] / 'data/depth.yaml'}"]
    c = SCENARIO / "configs"

    assert train_detection.main(["--config", str(c / "detection/baseline.yaml"), "--set", *det]) == 0
    assert train_detection.main(["--config", str(c / "detection/student_n_ca.yaml"), "--set", *det]) == 0
    teacher = tmp_path / "runs/detection/DET-02/weights/best.pt"
    assert train_detection_kd.main(["--config", str(c / "detection/feature_prediction_kd.yaml"), "--set", *det, f"kd.teacher_weights={teacher}"]) == 0
    assert train_depth.main(["--config", str(c / "depth/baseline.yaml"), "--set", *dep]) == 0
    assert train_depth.main(["--config", str(c / "depth/student_n_cbam.yaml"), "--set", *dep]) == 0
    teacher = tmp_path / "runs/depth/DEP-02/weights/best.pt"
    assert train_depth_kd.main(["--config", str(c / "depth/full_kd.yaml"), "--set", *dep, f"kd.teacher_weights={teacher}"]) == 0

    assert seen and all(w is None for w in seen)
    for exp in ("detection/DET-02", "detection/DET-07", "detection/DET-05", "depth/DEP-02", "depth/DEP-09", "depth/DEP-07"):
        args = yaml.safe_load((tmp_path / "runs" / exp / "args.yaml").read_text())
        assert args["pretrained"] is False and (tmp_path / "runs" / exp / "weights/best.pt").is_file()
    assert not (SCENARIO / "outputs").exists()
