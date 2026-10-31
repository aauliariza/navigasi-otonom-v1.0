"""Train a YOLO26-Depth model WITHOUT KD: YOLO26x-Depth teacher, YOLO26x-Depth+CBAM teacher, YOLO26n-Depth baseline.

After training, the Ultralytics automatic (scale-only) calibration is replaced by the two-parameter
log-affine calibration of models/depth/yolo26_depth.py fitted on ``calibration.split`` (default: ``val``, the
internal validation split carved out of the official alltrain; the official test split is refused), which
is written into ``weights/best.pt`` and ``weights/last.pt`` and recorded in ``calibration.json``.

Usage::

    python -m src.training.train_depth --config configs/depth/teacher_x.yaml
    python -m src.training.train_depth --config configs/depth/baseline.yaml [--resume]
    python -m src.training.train_depth --config configs/depth/baseline.yaml --calibrate-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics.models.yolo.depth import DepthTrainer
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import RANK

import models  # noqa: F401
from models.depth.yolo26_depth import calibrate_checkpoint
from models.detection.yolo26_detector import model_yaml_for
from src.training.trainer_utils import LOG, ResearchTrainerMixin, run_training
from src.utils.checkpoint import experiment_dir
from src.utils.config import apply_overrides, load_config, resolve_path


def calibrate_experiment(cfg: dict, weights: list[Path], device) -> dict | None:
    """Fit/write log-affine calibration for the given checkpoints (best first). Returns the best.pt record."""
    cal = cfg.get("calibration", {})
    if not cal.get("enabled", True):
        LOG.warning("Log-affine calibration disabled by config; checkpoints keep identity calibration.")
        return None
    if str(cal.get("split", "val")) == "test":
        raise ValueError("calibration.split=test is not allowed: calibration parameters must come from validation data")
    data = load_json_yaml(resolve_path(cfg["data"]["yaml"]))
    root = Path(data["path"])
    record = None
    for w in weights:
        if w.is_file():
            r = calibrate_checkpoint(
                w, root, cal.get("split", "val"), int(cfg["train"]["imgsz"]), float(data.get("min_depth", 0.001)),
                float(data.get("max_depth", 10.0)), int(cal.get("max_images", 1000)), device=device,
                seed=int(cfg["train"].get("seed", 42)),
                report_path=experiment_dir(cfg) / f"calibration_{w.stem}.json",
            )
            LOG.info("Log-affine calibration %s: a=%.5f b=%.5f (split=%s, %d images)", w.name, r["a"], r["b"], r["split"], r["num_images"])
            record = record or r
    return record


def load_json_yaml(path: Path) -> dict:
    """Read the dataset YAML (Ultralytics format)."""
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


class ResearchDepthTrainer(ResearchTrainerMixin, DepthTrainer):
    """Ultralytics DepthTrainer + attention-aware model building + research log-affine calibration."""

    def final_eval(self) -> None:
        """Standard final validation of best.pt, then the research calibration (no validation-split fitting)."""
        DetectionTrainer.final_eval(self)
        if RANK in {-1, 0}:
            calibrate_experiment(self.research_cfg, [self.best, self.last], self.device)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Train YOLO26-Depth (no KD)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--calibrate-only", action="store_true", help="Only (re)fit log-affine calibration of existing weights")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args.set)
    if cfg.get("kd", {}).get("enabled", False):
        raise SystemExit("This config enables KD; use python -m src.training.train_depth_kd")
    run_dir = experiment_dir(cfg)
    if args.calibrate_only:
        calibrate_experiment(cfg, [run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt"], cfg["train"].get("device", "cpu") or "cpu")
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    model_yaml = None if args.resume else model_yaml_for(cfg["model"], "depth", run_dir, nc=1)
    run_training(cfg, ResearchDepthTrainer, "depth", model_yaml, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
