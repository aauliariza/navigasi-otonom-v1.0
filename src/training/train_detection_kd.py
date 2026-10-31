"""Train YOLO26n WITH knowledge distillation from a frozen YOLO26x (or YOLO26x+CA) teacher.

The teacher checkpoint is taken from ``kd.teacher_config`` (its ``weights/best.pt``) and must exist; its
SHA-256 is recorded in ``metadata.json``. Active KD terms come from ``kd.feature`` / ``kd.prediction``.

Usage::

    python -m src.training.train_detection_kd --config configs/detection/feature_kd.yaml
    python -m src.training.train_detection_kd --config configs/detection/feature_kd.yaml --resume
"""

from __future__ import annotations

import argparse
import copy

import models  # noqa: F401
from models.detection.detection_teacher_student import DetectionKDModel
from models.detection.yolo26_detector import model_yaml_for
from src.training.train_detection import ResearchDetectionTrainer
from src.training.trainer_utils import run_training
from src.utils.checkpoint import experiment_dir
from src.utils.config import apply_overrides, load_config


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Train YOLO26n detection with KD")
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args.set)
    kd = cfg.get("kd", {})
    if not kd.get("enabled", False):
        raise SystemExit("kd.enabled is false; use python -m src.training.train_detection")
    if str(cfg["model"].get("attention", "none")) != "none":
        raise SystemExit("The student must stay plain YOLO26n (attention is only allowed on the teacher)")
    DetectionKDModel.kd_config = {k: copy.deepcopy(kd[k]) for k in ("feature", "prediction") if k in kd}
    run_dir = experiment_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    model_yaml = None if args.resume else model_yaml_for(cfg["model"], "detect", run_dir, nc=1)
    run_training(cfg, ResearchDetectionTrainer, "detect", model_yaml, resume=args.resume, kd_cls=DetectionKDModel,
                 extra_meta={"kd_config": DetectionKDModel.kd_config})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
