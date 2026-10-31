"""Model construction utilities shared by detection and depth.

* :func:`register_custom_modules` exposes ``CoordAtt`` and ``CBAM`` to the Ultralytics YAML parser
  (``ultralytics.nn.tasks.parse_model`` resolves module names via the ``tasks`` module globals).
* :func:`attention_yaml` derives a YOLO26(-Depth) YAML with an attention block inserted on each of the
  three pyramid features (P3/P4/P5, layers 16/19/22) right before the head::

      ... 22 (P5)
      23  CoordAtt|CBAM  from 16      (P3, same channels)
      24  CoordAtt|CBAM  from 19      (P4)
      25  CoordAtt|CBAM  from 22      (P5)
      26  Detect|Depth   from [23, 24, 25]

  Layers 0-22 keep their indices, so official pretrained weights load unchanged; the head moves from 23 to 26
  and is transferred by :func:`transfer_weights`.
* :func:`transfer_weights` loads a source model into a target whose head index differs (Identity padding).
* :func:`model_stats` reports total / trainable parameters and GFLOPs.
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn

ATTENTION_MODULES = {"ca": "CoordAtt", "cbam": "CBAM"}
PYRAMID_TAPS = (16, 19, 22)


def register_custom_modules() -> None:
    """Make CoordAtt/CBAM resolvable by name in Ultralytics model YAMLs (idempotent)."""
    import ultralytics.nn.tasks as tasks

    from models.depth.cbam import CBAM
    from models.detection.coordinate_attention import CoordAtt

    tasks.CoordAtt = CoordAtt
    tasks.CBAM = CBAM


def base_yaml(arch: str) -> dict:
    """Load an official Ultralytics model YAML (e.g. ``yolo26x.yaml``, ``yolo26n-depth.yaml``) as a dict."""
    from ultralytics.nn.tasks import yaml_model_load

    return yaml_model_load(arch)


def _tap_channels(cfg: dict, taps: tuple[int, ...], task: str) -> list[int]:
    """Output channels of the tapped layers for the cfg's scale, via a tiny dummy forward."""
    model = build_from_yaml(copy.deepcopy(cfg), task=task, verbose=False)
    chans: dict[int, int] = {}
    hooks = [model.model[i].register_forward_hook(lambda m, i_, o, k=i: chans.__setitem__(k, o.shape[1])) for i in taps]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, 64, 64))
    for h in hooks:
        h.remove()
    return [chans[i] for i in taps]


def attention_yaml(arch: str, attention: str, args: dict | None = None, task: str = "detect", nc: int = 1) -> dict:
    """Return a model YAML dict with ``attention`` ('ca' | 'cbam') inserted before the head (see module doc)."""
    if attention not in ATTENTION_MODULES:
        raise ValueError(f"attention must be one of {sorted(ATTENTION_MODULES)}, got '{attention}'")
    register_custom_modules()
    cfg = base_yaml(arch)
    cfg["nc"] = nc
    head_row = cfg["head"][-1]
    if list(head_row[0]) != list(PYRAMID_TAPS):
        raise ValueError(f"{arch}: expected head inputs {PYRAMID_TAPS}, found {head_row[0]}")
    chans = _tap_channels(cfg, PYRAMID_TAPS, task)
    args = args or {}
    name = ATTENTION_MODULES[attention]
    if attention == "ca":
        rows = [[t, 1, name, [c, int(args.get("reduction", 32))]] for t, c in zip(PYRAMID_TAPS, chans)]
    else:
        rows = [[t, 1, name, [c, int(args.get("reduction", 16)), int(args.get("kernel_size", 7))]] for t, c in zip(PYRAMID_TAPS, chans)]
    first = len(cfg["backbone"]) + len(cfg["head"]) - 1  # index the head had (23)
    cfg["head"] = cfg["head"][:-1] + rows + [[[first, first + 1, first + 2], *head_row[1:]]]
    cfg["attention"] = {"type": attention, "module": name, "taps": list(PYRAMID_TAPS), "channels": chans, **args}
    return cfg


def build_from_yaml(cfg: dict | str, task: str = "detect", nc: int | None = None, verbose: bool = False) -> nn.Module:
    """Instantiate an Ultralytics ``DetectionModel`` or ``DepthModel`` from a YAML path/dict."""
    from ultralytics.nn.tasks import DepthModel, DetectionModel

    register_custom_modules()
    cls = DepthModel if task == "depth" else DetectionModel
    kw = {"nc": nc} if nc is not None else {}
    return cls(cfg, ch=3, verbose=verbose, **kw)


def _as_module(weights) -> nn.Module:
    """Accept a checkpoint path, an Ultralytics checkpoint dict, or a module."""
    if isinstance(weights, (str, Path)):
        from ultralytics.nn.tasks import load_checkpoint

        weights = load_checkpoint(str(weights))[0]
    if isinstance(weights, dict):
        weights = weights.get("ema") or weights["model"]
    return weights


def transfer_weights(model: nn.Module, weights, verbose: bool = True) -> int:
    """Load ``weights`` into ``model``; tolerate attention layers inserted before the head.

    When the target has ``k`` more layers than the source (attention blocks) the source is padded with ``k``
    ``nn.Identity`` layers before its head so that head parameter names line up (``model.23.*`` -> ``model.26.*``),
    then Ultralytics' own ``BaseModel.load`` (shape-matched intersection) performs the copy.

    Returns:
        (int): Number of transferred tensors.
    """
    src = copy.deepcopy(_as_module(weights)).float()
    k = len(model.model) - len(src.model)
    if k < 0:
        raise ValueError(f"Source has more layers ({len(src.model)}) than target ({len(model.model)})")
    if k:
        layers = list(src.model)
        src.model = nn.Sequential(*layers[:-1], *[nn.Identity() for _ in range(k)], layers[-1])
    before = {n: p.detach().clone() for n, p in model.state_dict().items()}
    model.load(src, verbose=verbose)
    after = model.state_dict()
    return sum(1 for n in after if n in before and not torch.equal(before[n], after[n]))


def model_stats(model: nn.Module, imgsz: int = 640) -> dict:
    """Total/trainable parameters and GFLOPs (2 x MACs, Eq. 3.25) of an Ultralytics model at ``imgsz``.

    FLOPs are counted by ``ultralytics-thop`` (``thop.profile``) on a full ``imgsz x imgsz`` input. Ultralytics'
    own ``get_flops`` returns 0.0 on ANY profiling error (e.g. the unrelated legacy ``thop`` package shadowing
    ``ultralytics-thop``); that silent 0 would enter EWM-SAW as "zero cost", so the profiler is called directly here
    and a failure raises an informative error instead.
    """
    import copy as _copy

    try:
        import thop
    except ImportError as e:
        raise RuntimeError("[model_stats] FLOPs tidak dapat dihitung: paket 'ultralytics-thop' tidak terpasang. "
                           "Solusi: pip install ultralytics-thop") from e
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    p0 = next(model.parameters())
    im = torch.zeros((1, 3, imgsz, imgsz), device=p0.device, dtype=p0.dtype)
    m = _copy.deepcopy(model).eval()
    try:
        macs = thop.profile(m, inputs=[im], verbose=False)[0]
    except TypeError as e:  # wrong 'thop' distribution installed
        raise RuntimeError(f"[model_stats] thop.profile gagal ({e}); kemungkinan paket lama 'thop' menimpa 'ultralytics-thop'. "
                           "Solusi: pip uninstall -y thop && pip install --force-reinstall ultralytics-thop") from e
    if not macs or macs <= 0:
        raise RuntimeError("[model_stats] thop.profile menghasilkan 0 MAC; FLOPs tidak valid")
    return {"params_total": int(total), "params_trainable": int(trainable), "gflops": float(macs) * 2 / 1e9, "flops_imgsz": imgsz}
