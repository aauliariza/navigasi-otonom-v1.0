"""Train YOLO26n-Depth WITH knowledge distillation from a frozen YOLO26x-Depth (or +CBAM) teacher.

Active components come from ``kd.output`` / ``kd.feature`` / ``kd.boundary`` / ``kd.relative`` of the experiment
YAML; each component's weighted value is logged per epoch in ``results.csv`` (``train/kd_*``). The student is
calibrated after training exactly like the baseline (src/training/train_depth.py).

Usage::

    python -m src.training.train_depth_kd --config configs/depth/full_kd.yaml [--resume]
"""

from __future__ import annotations

import argparse
import copy

import models  # noqa: F401
from models.depth.depth_teacher_student import DepthKDModel
from models.detection.yolo26_detector import model_yaml_for
from src.losses.combined_loss import COMPONENTS, merge_depth_kd_config
from src.training.train_depth import ResearchDepthTrainer
from src.training.trainer_utils import run_training
from src.utils.checkpoint import experiment_dir
from src.utils.config import apply_overrides, load_config


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Train YOLO26n-Depth with KD")
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args.set)
    kd = cfg.get("kd", {})
    if not kd.get("enabled", False):
        raise SystemExit("kd.enabled is false; use python -m src.training.train_depth")
    if str(cfg["model"].get("attention", "none")) != "none":
        raise SystemExit("The student must stay plain YOLO26n-Depth (attention is only allowed on the teacher)")
    kd_cfg = {k: copy.deepcopy(v) for k, v in kd.items() if k in (*COMPONENTS, "teacher_space", "min_depth", "max_depth")}
    DepthKDModel.kd_config = kd_cfg
    run_dir = experiment_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    model_yaml = None if args.resume else model_yaml_for(cfg["model"], "depth", run_dir, nc=1)
    run_training(cfg, ResearchDepthTrainer, "depth", model_yaml, resume=args.resume, kd_cls=DepthKDModel,
                 extra_meta={"kd_config": merge_depth_kd_config(kd_cfg)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
