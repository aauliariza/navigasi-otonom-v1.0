"""Shared training machinery for every detection / depth experiment.

* :class:`ResearchTrainerMixin` -- overrides ``get_model`` so that (i) attention-augmented YAMLs are built with
  the custom modules registered and (ii) pretrained weights are transferred across the inserted attention layers;
  records peak GPU memory per epoch; writes loss-curve plots at the end of training.
* :func:`distillation_class` -- substitutes Ultralytics' ``DistillationModel`` class *inside the trainer module*
  with a research subclass for the duration of a run. The trainer's own logic (teacher freezing, EMA without
  teacher, student-only checkpoints, resume, NaN recovery) then applies unchanged to the subclass.
* :func:`run_training` -- builds the Ultralytics argument dict from an experiment YAML, enforces the dataset
  integrity gate, saves the exact configuration + metadata next to the checkpoints, and launches training.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import torch

from src.utils.checkpoint import experiment_dir, last_weights, require_file
from src.utils.config import REPO_ROOT, resolve_path
from src.utils.io import load_csv, save_csv, save_json, save_yaml
from src.utils.logger import get_logger
from src.utils.seed import set_seed

LOG = get_logger("training")

# Keys of the ``train`` section that are passed verbatim to Ultralytics.
ULTRALYTICS_TRAIN_KEYS = {
    "epochs", "batch", "imgsz", "optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
    "warmup_momentum", "warmup_bias_lr", "cos_lr", "patience", "amp", "seed", "deterministic", "workers",
    "device", "cache", "plots", "close_mosaic", "save_period", "fraction", "val", "box", "cls", "dfl",
    "dlog", "dgrad", "dlam", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "fliplr", "flipud",
    "mosaic", "mixup", "rect", "nbs", "multi_scale", "time", "max_det", "conf", "iou", "shear", "perspective",
    "cutmix", "copy_paste", "bgr", "erasing", "freeze",
}


class ResearchTrainerMixin:
    """Mixin for Ultralytics trainers used by this repository (see module docstring)."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Build the (possibly attention-augmented) model and transfer ``weights`` into it."""
        from models.common.model_utils import build_from_yaml, register_custom_modules, transfer_weights

        register_custom_modules()
        task = "depth" if self.args.task == "depth" else "detect"
        model = build_from_yaml(cfg, task=task, nc=self.data["nc"], verbose=verbose)
        if task == "detect":
            model = self.set_model_names_for_load(model)
        if weights:
            n = transfer_weights(model, weights, verbose=verbose)
            LOG.info("Transferred %d tensors from pretrained weights", n)
        return model

    def research_callbacks(self) -> None:
        """Register per-epoch GPU-memory logging."""
        self.gpu_memory_log: list[dict] = []

        def on_epoch_end(trainer):
            if torch.cuda.is_available():
                trainer.gpu_memory_log.append(
                    {
                        "epoch": trainer.epoch + 1,
                        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                        "max_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
                    }
                )
                save_csv(trainer.gpu_memory_log, Path(trainer.save_dir) / "gpu_memory.csv")

        def on_train_end(trainer):
            plot_history(Path(trainer.save_dir) / "results.csv", Path(trainer.save_dir) / "loss_curves.png")

        self.add_callback("on_train_epoch_end", on_epoch_end)
        self.add_callback("on_train_end", on_train_end)


@contextmanager
def distillation_class(cls):
    """Temporarily make the Ultralytics trainer instantiate ``cls`` instead of ``DistillationModel``."""
    import ultralytics.engine.trainer as trainer_module

    original = trainer_module.DistillationModel
    trainer_module.DistillationModel = cls
    try:
        yield
    finally:
        trainer_module.DistillationModel = original


def plot_history(results_csv: Path, out_png: Path) -> None:
    """Plot every loss / metric column of an Ultralytics ``results.csv`` (one panel per column)."""
    if not results_csv.is_file():
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_csv(results_csv)
    if not rows:
        return
    cols = [c for c in rows[0] if c.strip() not in {"epoch", "time"} and not c.startswith("lr/")]
    epochs = [float(r["epoch"]) for r in rows]
    n = len(cols)
    ncol = 4
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), tight_layout=True, squeeze=False)
    for ax, c in zip(axes.ravel(), cols):
        ax.plot(epochs, [float(r[c] or "nan") for r in rows], marker=".")
        ax.set_title(c.strip(), fontsize=9)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file (used to pin the teacher checkpoint a student was distilled from)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def environment_info() -> dict:
    """Software / hardware description stored with every experiment."""
    import ultralytics

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    return info


def teacher_checkpoint(cfg: dict) -> Path:
    """Teacher ``best.pt`` resolved from ``kd.teacher_config`` (or an explicit ``kd.teacher_weights``)."""
    from src.utils.config import load_config

    kd = cfg["kd"]
    if kd.get("teacher_weights"):
        path = resolve_path(kd["teacher_weights"])
    else:
        tcfg = load_config(kd["teacher_config"])
        path = experiment_dir(tcfg) / "weights" / "best.pt"
    path = require_file(path, "Teacher checkpoint", f"Train the teacher first: {kd.get('teacher_config', '')}")
    check_teacher_version(path, bool(kd.get("allow_version_mismatch", False)))
    return path


def check_teacher_version(path: Path, allow_mismatch: bool = False) -> None:
    """Refuse a teacher trained with another Ultralytics release (experiments must share one version)."""
    import ultralytics

    from src.utils.checkpoint import load_ultralytics_ckpt
    from src.utils.versions import checkpoint_ultralytics_version

    tv = checkpoint_ultralytics_version(load_ultralytics_ckpt(path))
    if tv and tv != ultralytics.__version__:
        msg = (f"[training] Teacher {path} dilatih dengan Ultralytics {tv}, sedangkan versi terpasang {ultralytics.__version__}.\n"
               f"  Solusi : latih ulang teacher dengan versi terpasang agar teacher dan student memakai versi yang sama, "
               f"atau (tidak disarankan, harus dicatat) --set kd.allow_version_mismatch=true")
        if not allow_mismatch:
            raise RuntimeError(msg)
        LOG.warning(msg)


def dataset_split_record(cfg: dict) -> dict:
    """Which split file every experiment used (path, SHA-256, sizes), so all runs can be checked to share it."""
    import yaml

    data_yaml = resolve_path(cfg["data"]["yaml"])
    rec = {"data_yaml": str(data_yaml)}
    if data_yaml.is_file():
        root = Path(yaml.safe_load(data_yaml.read_text(encoding="utf-8")).get("path", data_yaml.parent))
        split_json = root / "meta" / "splits.json"
        if split_json.is_file():
            from src.utils.io import load_json

            sp = load_json(split_json)
            rec.update(splits_json=str(split_json), splits_sha256=sha256(split_json),
                       sizes={k: len(sp[k]) for k in ("train", "val", "test") if k in sp})
    return rec


def build_ultralytics_args(cfg: dict, task: str, model_yaml: str, resume: bool) -> dict:
    """Translate an experiment YAML into the Ultralytics override dict."""
    run_dir = experiment_dir(cfg)
    if resume:
        last = require_file(last_weights(cfg), "last.pt", "Nothing to resume: the experiment has not been started.")
        return {"model": str(last), "resume": str(last), "data": str(resolve_path(cfg["data"]["yaml"]))}
    args = {k: v for k, v in cfg["train"].items() if k in ULTRALYTICS_TRAIN_KEYS}
    unknown = set(cfg["train"]) - ULTRALYTICS_TRAIN_KEYS
    if unknown:
        raise KeyError(f"Unknown train keys {sorted(unknown)}; add them to ULTRALYTICS_TRAIN_KEYS if intended")
    weights = cfg["model"].get("weights")
    args.update(
        task=task,
        model=model_yaml,
        pretrained=str(resolve_path(weights)) if weights and Path(str(weights)).suffix == ".pt" and resolve_path(weights).is_file() else (weights or False),
        data=str(resolve_path(cfg["data"]["yaml"])),
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
        verbose=True,
    )
    return args


def write_experiment_record(cfg: dict, extra: dict | None = None) -> Path:
    """Save the resolved config and metadata (``config_used.yaml``, ``metadata.json``) in the run folder."""
    run_dir = experiment_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, run_dir / "config_used.yaml")
    meta = {
        "experiment_id": cfg["experiment"]["id"],
        "description": cfg["experiment"].get("description", ""),
        "config_source": cfg.get("_source_"),
        "started_at": datetime.now().astimezone().isoformat(),
        "git_commit": _git_commit(),
        "environment": environment_info(),
        **(extra or {}),
    }
    save_json(meta, run_dir / "metadata.json")
    return run_dir


def run_training(cfg: dict, trainer_cls, task: str, model_yaml: str | None = None, resume: bool = False, kd_cls=None, extra_meta=None):
    """Common launcher (see module docstring). Returns the trainer after training."""
    from src.data.validate_dataset import require_dataset_ok

    if cfg["data"].get("require_integrity", True):
        require_dataset_ok(cfg["data"].get("integrity_report", "outputs/dataset/integrity_report.json"))
    set_seed(int(cfg["train"].get("seed", 42)), bool(cfg["train"].get("deterministic", True)))
    args = build_ultralytics_args(cfg, task, model_yaml, resume)
    if kd_cls is not None and not resume:
        args["distill_model"] = str(teacher_checkpoint(cfg))
        if "feature" in cfg["kd"] and "weight" in cfg["kd"]["feature"]:
            args["dis"] = float(cfg["kd"]["feature"]["weight"])
    meta = {"ultralytics_args": args, "dataset_split": dataset_split_record(cfg), "loss_and_kd": cfg.get("kd", {"enabled": False}),
            "calibration": cfg.get("calibration"), **(extra_meta or {})}
    if "distill_model" in args:
        meta["teacher_checkpoint"] = args["distill_model"]
        meta["teacher_sha256"] = sha256(args["distill_model"])
    write_experiment_record(cfg, meta)
    t0 = time.time()
    trainer = trainer_cls(overrides=args)
    trainer.research_cfg = cfg
    trainer.research_callbacks()
    if kd_cls is not None:
        with distillation_class(kd_cls):
            trainer.train()
    else:
        trainer.train()
    save_json(
        {"finished_at": datetime.now().astimezone().isoformat(), "train_time_h": (time.time() - t0) / 3600, "save_dir": str(trainer.save_dir),
         "best": str(trainer.best), "last": str(trainer.last), "best_fitness": trainer.best_fitness},
        experiment_dir(cfg) / "training_summary.json",
    )
    return trainer
