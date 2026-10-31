"""Train a detection model WITHOUT knowledge distillation: YOLO26x teacher, YOLO26x+CA teacher, YOLO26n baseline.

Usage::

    python -m src.training.train_detection --config configs/detection/teacher_x.yaml
    python -m src.training.train_detection --config configs/detection/baseline.yaml
    python -m src.training.train_detection --config configs/detection/baseline.yaml --resume

Outputs (``outputs/detection/<experiment_id>/``): ``weights/{best,last}.pt``, ``results.csv`` (loss + val metrics
per epoch), ``results.png``, ``loss_curves.png``, ``args.yaml``, ``config_used.yaml``, ``metadata.json``,
``gpu_memory.csv``, ``training_summary.json``.
"""

from __future__ import annotations

import argparse

from ultralytics.models.yolo.detect import DetectionTrainer

import models  # noqa: F401  (registers CoordAtt/CBAM)
from models.detection.yolo26_detector import model_yaml_for
from src.training.trainer_utils import ResearchTrainerMixin, run_training
from src.utils.checkpoint import experiment_dir
from src.utils.config import apply_overrides, load_config


class ResearchDetectionTrainer(ResearchTrainerMixin, DetectionTrainer):
    """Ultralytics DetectionTrainer + attention-aware model building (see trainer_utils)."""


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Train YOLO26 detection (no KD)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true", help="Resume from weights/last.pt of this experiment")
    ap.add_argument("--set", nargs="*", default=[], help="Override config values, e.g. train.epochs=1")
    args = ap.parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args.set)
    if cfg.get("kd", {}).get("enabled", False):
        raise SystemExit("This config enables KD; use python -m src.training.train_detection_kd")
    run_dir = experiment_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    model_yaml = None if args.resume else model_yaml_for(cfg["model"], "detect", run_dir, nc=1)
    run_training(cfg, ResearchDetectionTrainer, "detect", model_yaml, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
