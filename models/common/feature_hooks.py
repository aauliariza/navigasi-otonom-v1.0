"""Forward-hook feature capture used to inspect / verify the KD tap layers.

Training-time feature capture is done by Ultralytics' ``DistillationModel`` hooks (``FeatureHook``); this
helper exposes the same mechanism for tests and for ``scripts``-level inspection, so the layer indices that
KD reads (e.g. 16/19/22 for the student, 23/24/25 for an attention teacher) can be checked explicitly.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn


def head_input_layers(model: nn.Module) -> list[int]:
    """Indices of the layers feeding the model's last layer (Detect or Depth head)."""
    head = model.model[-1]
    return list(head.f) if isinstance(head.f, (list, tuple)) else [head.f]


@contextmanager
def capture_features(model: nn.Module, layers: list[int]):
    """Context manager yielding a dict ``{layer_idx: output}`` filled during forwards inside the block."""
    store: dict[int, torch.Tensor] = {}
    handles = [model.model[i].register_forward_hook(lambda m, inp, out, k=i: store.__setitem__(k, out)) for i in layers]
    try:
        yield store
    finally:
        for h in handles:
            h.remove()
