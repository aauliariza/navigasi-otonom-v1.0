"""Ultralytics version guard.

The repository subclasses Ultralytics internals (``nn/distill_model.py``) and relies on the YOLO26 / YOLO26-Depth
behaviour of a specific release. Older 8.4.x releases differ in exactly those internals, e.g. Ultralytics 8.4.138
stores distillation features under *layer indices* and has no ``teacher_feats_idx`` (attention teachers with a
shifted head are unsupported; the KD wrappers fail with ``KeyError: 3``), and it still uses the ``end2end``
argument instead of ``nms`` for the Detect-head mode. Every experiment must therefore run on the verified release
(and on the SAME release, so the experiments stay comparable).
"""

from __future__ import annotations

MIN_ULTRALYTICS = (8, 4, 194)
MAX_ULTRALYTICS_EXCLUSIVE = (9, 0, 0)


def _parse(v: str) -> tuple[int, ...]:
    parts = []
    for x in str(v).split(".")[:3]:
        digits = "".join(ch for ch in x if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts + [0] * (3 - len(parts)))


def check_ultralytics_version(version: str | None = None) -> str:
    """Raise an informative RuntimeError unless MIN_ULTRALYTICS <= version < 9.0.0. Returns the version string."""
    if version is None:
        import ultralytics

        version = ultralytics.__version__
    v = _parse(version)
    if not (MIN_ULTRALYTICS <= v < MAX_ULTRALYTICS_EXCLUSIVE):
        need = ".".join(map(str, MIN_ULTRALYTICS))
        raise RuntimeError(
            f"[versions] Ultralytics {version} terpasang, sedangkan repository ini memerlukan >={need},<9.0.0 "
            f"(API DistillationModel dan mode head YOLO26 berbeda antar-versi).\n"
            f"  Solusi : pip install \"ultralytics=={need}\"  (atau pip install -r requirements.txt), lalu latih ulang "
            f"eksperimen yang sudah dilatih dengan versi lain agar seluruh eksperimen memakai versi yang sama."
        )
    return str(version)


def checkpoint_ultralytics_version(ckpt: dict) -> str | None:
    """Ultralytics version stored in a checkpoint dict (``version`` key), if any."""
    return ckpt.get("version") if isinstance(ckpt, dict) else None
