"""End-to-end integration test on the synthetic dataset (CPU, ~2-4 min).

Covers: dataset prep/validation gate -> teacher/baseline/KD training (detection + depth) -> checkpoint save,
interrupt + resume of a KD run -> log-affine calibration -> evaluation -> EWM-SAW selection -> integrated
system -> distance and navigation evaluation. Numbers produced here are meaningless (synthetic data, 1-2 epochs);
the test only asserts that every stage runs and produces well-formed, finite outputs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def env(synthetic_dataset):
    tmp = synthetic_dataset["tmp"]
    common = [
        f"data.integrity_report={tmp / 'report/integrity_report.json'}",
        "model.weights=null", "train.epochs=1", "train.batch=3", "train.imgsz=96", "train.workers=0", "train.device=cpu",
        "train.amp=false", "train.plots=false", f"output.root={tmp / 'runs'}",
    ]
    return {"tmp": tmp, "det": common + [f"data.yaml={tmp / 'data/detect.yaml'}"], "dep": common + [f"data.yaml={tmp / 'data/depth.yaml'}"]}


def _best(tmp, group, exp_id):
    return tmp / "runs" / group / exp_id / "weights" / "best.pt"


def test_training_requires_integrity_gate(env):
    from src.training import train_detection

    bad = [s for s in env["det"] if not s.startswith("data.integrity_report")] + [f"data.integrity_report={env['tmp'] / 'nope.json'}"]
    with pytest.raises(RuntimeError):
        train_detection.main(["--config", "configs/detection/baseline.yaml", "--set", *bad])


def test_detection_training_kd_and_resume(env):
    from src.training import train_detection, train_detection_kd
    from src.training.trainer_utils import run_training
    from src.utils.checkpoint import load_ultralytics_ckpt

    tmp = env["tmp"]
    assert train_detection.main(["--config", "configs/detection/teacher_x_ca.yaml", "--set", *env["det"], "model.arch=yolo26n.yaml"]) == 0
    assert train_detection.main(["--config", "configs/detection/baseline.yaml", "--set", *env["det"]]) == 0
    teacher = _best(tmp, "detection", "TEACHER-CA")
    kd_set = [*env["det"], f"kd.teacher_weights={teacher}", "train.epochs=2"]

    # interrupt after the first epoch's checkpoint, then resume
    import models.detection.detection_teacher_student as dts
    from src.training.train_detection import ResearchDetectionTrainer
    from src.utils.config import apply_overrides, load_config

    class Interrupt(ResearchDetectionTrainer):
        def research_callbacks(self):
            super().research_callbacks()

            def stop(trainer):
                if trainer.epoch == 0:
                    raise KeyboardInterrupt

            self.add_callback("on_fit_epoch_end", stop)

    cfg = apply_overrides(load_config("configs/detection/ca_full_kd.yaml"), kd_set)
    dts.DetectionKDModel.kd_config = {k: cfg["kd"][k] for k in ("feature", "prediction")}
    from models.detection.yolo26_detector import model_yaml_for
    from src.utils.checkpoint import experiment_dir

    run_dir = experiment_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(KeyboardInterrupt):
        run_training(cfg, Interrupt, "detect", model_yaml_for(cfg["model"], "detect", run_dir), kd_cls=dts.DetectionKDModel)
    last = run_dir / "weights" / "last.pt"
    assert load_ultralytics_ckpt(last)["epoch"] == 0
    assert train_detection_kd.main(["--config", "configs/detection/ca_full_kd.yaml", "--resume", "--set", *kd_set]) == 0
    best = load_ultralytics_ckpt(run_dir / "weights" / "best.pt")
    assert type(best["model"]).__name__ == "DetectionModel"  # student only, no teacher/projector
    lines = (run_dir / "results.csv").read_text().strip().splitlines()
    assert len(lines) == 3 and "train/pkd_loss" in lines[0] and "train/dis_loss" in lines[0]

    assert train_detection_kd.main(["--config", "configs/detection/feature_kd.yaml", "--set", *env["det"], f"kd.teacher_weights={teacher}"]) == 0


def test_depth_training_kd_and_calibration(env):
    from src.training import train_depth, train_depth_kd
    from src.utils.checkpoint import load_ultralytics_ckpt

    tmp = env["tmp"]
    assert train_depth.main(["--config", "configs/depth/teacher_x_cbam.yaml", "--set", *env["dep"], "model.arch=yolo26n-depth.yaml"]) == 0
    assert train_depth.main(["--config", "configs/depth/baseline.yaml", "--set", *env["dep"]]) == 0
    teacher = _best(tmp, "depth", "TEACHER-CBAM")
    assert train_depth_kd.main(["--config", "configs/depth/cbam_full_kd.yaml", "--set", *env["dep"], f"kd.teacher_weights={teacher}"]) == 0
    ck = load_ultralytics_ckpt(_best(tmp, "depth", "DEP-09"))
    cal = ck["log_affine_calibration"]
    assert cal["split"] == "val" and math.isfinite(cal["a"]) and cal["a"] > 0
    head = ck["model"].model[-1]
    assert head.cal_a.item() == pytest.approx(cal["a"]) and head.cal_b.item() == pytest.approx(cal["b"])
    header = (tmp / "runs/depth/DEP-09/results.csv").read_text().splitlines()[0]
    for c in ("kd_output", "kd_feature", "kd_boundary", "kd_relative"):
        assert f"train/{c}" in header


def test_evaluation_selection_integration(env):
    from src.evaluation import evaluate_depth, evaluate_detection, evaluate_distance, evaluate_navigation
    from src.integration.detector_depth_fusion import main as run_system
    from src.optimization import model_selection
    from src.utils.io import load_csv, load_json

    tmp = env["tmp"]
    ov = f"experiment_overrides=[output.root={tmp / 'runs'}]"
    small = ["device=cpu", "imgsz=96", "benchmark.warmup=1", "benchmark.iters=2"]
    assert evaluate_detection.main(["--set", f"data_yaml={tmp / 'data/detect.yaml'}", f"output_dir={tmp / 'eval/det'}", "batch=3", ov, *small,
                                    "models=[{name: YOLO26n, config: configs/detection/baseline.yaml}, {name: YOLO26n+FeatureKD, config: configs/detection/feature_kd.yaml}, {name: YOLO26n+Feat+PredKD(CA-teacher), config: configs/detection/ca_full_kd.yaml}]"]) == 0
    assert evaluate_depth.main(["--set", f"data_yaml={tmp / 'data/depth.yaml'}", f"output_dir={tmp / 'eval/dep'}", ov, *small,
                                "models=[{name: YOLO26n-Depth, config: configs/depth/baseline.yaml}, {name: YOLO26n-Depth+FullKD(CBAM-teacher), config: configs/depth/cbam_full_kd.yaml}]"]) == 0
    rows = load_csv(tmp / "eval/dep/depth_results.csv")
    assert all(np.isfinite(float(r[k])) for r in rows for k in ("delta1", "abs_rel", "sq_rel", "rmse", "rmse_log", "fps", "gflops"))
    assert model_selection.main(["--set", f"detection.results={tmp / 'eval/det/detection_results.csv'}", f"detection.output_dir={tmp / 'sel/det'}",
                                 f"depth.results={tmp / 'eval/dep/depth_results.csv'}", f"depth.output_dir={tmp / 'sel/dep'}",
                                 f"selected_models_path={tmp / 'sel/selected.json'}"]) == 0
    sel = load_json(tmp / "sel/selected.json")
    assert "KD" in sel["detection"]["model"] and "KD" in sel["depth"]["model"]

    sys_ov = [f"models.selected_models={tmp / 'sel/selected.json'}", f"output_dir={tmp / 'system'}", "detection.imgsz=96", "depth.imgsz=96", "detection.conf=0.001"]
    assert run_system(["--source", str(tmp / "data/images/test"), "--set", *sys_ov]) == 0
    dec = load_csv(tmp / "system/decisions.csv")
    assert len(dec) == 3 and all(d["direction"] in {"FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"} for d in dec)
    # image and video front-ends with the real selected checkpoints
    import cv2

    from src.inference import predict_image, predict_video

    assert predict_image.main(["--source", str(tmp / "data/images/test"), "--output", str(tmp / "inf/img"), "--set", *sys_ov]) == 0
    assert len(load_csv(tmp / "inf/img/decisions.csv")) == 3
    frames = [cv2.imread(str(p)) for p in sorted((tmp / "data/images/test").glob("*.jpg"))]
    vw = cv2.VideoWriter(str(tmp / "clip.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (frames[0].shape[1], frames[0].shape[0]))
    for f in frames * 2:
        vw.write(f)
    vw.release()
    assert predict_video.main(["--source", str(tmp / "clip.avi"), "--output", str(tmp / "inf/vid"), "--set", *sys_ov]) == 0
    assert load_json(tmp / "inf/vid/clip_summary.json")["frames_processed"] == 6

    so = "system_overrides=[" + ",".join(sys_ov) + "]"
    assert evaluate_distance.main(["--set", so, f"data_yaml={tmp / 'data/depth.yaml'}", f"output_dir={tmp / 'eval/dist'}"]) == 0
    s = load_json(tmp / "eval/dist/distance_summary.json")
    assert s["objects_total"] >= s["objects_valid"] > 0
    assert all(math.isfinite(s[k]) for k in ("mae", "rmse", "mape"))
    assert evaluate_navigation.main(["--set", so, "gt_source=sunrgbd", f"data_yaml={tmp / 'data/depth.yaml'}", f"output_dir={tmp / 'eval/nav'}"]) == 0
    n = load_json(tmp / "eval/nav/navigation_summary.json")
    assert n["n_total"] == 3 and 0 <= n["accuracy_pct"] <= 100


def test_measured_and_scenario_modes(env, tmp_path):
    """Manual-measurement and scenario CSV formats (values here are placeholders written by the test itself)."""
    import shutil

    from src.evaluation import evaluate_distance, evaluate_navigation
    from src.utils.io import load_json, save_csv

    tmp = env["tmp"]
    img = sorted((tmp / "data/images/val").glob("*.jpg"))[0]
    shutil.copy(img, tmp_path / "scene1.jpg")
    save_csv([{"image": "scene1.jpg", "x1": 0, "y1": 0, "x2": 50, "y2": 50, "gt_distance_m": 2.0}], tmp_path / "meas.csv")
    save_csv([{"scenario_id": "S1", "image": "scene1.jpg", "gt_direction": "FORWARD", "description": "test"}], tmp_path / "scen.csv")
    so = f"system_overrides=[models.selected_models={tmp / 'sel/selected.json'},detection.imgsz=96,depth.imgsz=96]"
    assert evaluate_distance.main(["--set", so, "gt_source=measured", f"measurements_csv={tmp_path / 'meas.csv'}", f"output_dir={tmp_path / 'd'}"]) == 0
    assert load_json(tmp_path / "d/distance_summary.json")["objects_total"] == 1
    assert evaluate_navigation.main(["--set", so, "gt_source=scenarios", f"scenarios_csv={tmp_path / 'scen.csv'}", f"output_dir={tmp_path / 'n'}"]) == 0
    assert load_json(tmp_path / "n/navigation_summary.json")["n_total"] == 1
