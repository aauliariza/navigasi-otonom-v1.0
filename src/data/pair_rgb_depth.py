"""Discover SUN RGB-D scenes and pair each RGB image with its depth map (proposal Sub-bab 3.2.3.2).

Raw SUN RGB-D layout (``SUNRGBD.zip``)::

    SUNRGBD/<sensor>/<subset>/.../<scene>/
        image/<name>.jpg          RGB
        depth/<name>.png          raw depth (bit-rotated uint16)
        depth_bfx/<name>.png      refined (hole-filled) depth, bit-rotated uint16
        intrinsics.txt            3x3 camera matrix (row-major)

The *identity* of a sample is its scene path relative to the ``SUNRGBD`` root, e.g.
``kv1/NYUdata/NYU0001``. RGB and depth are paired by that identity, never by sorting file lists across
scenes, so a pair can only come from the same scene folder.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

SENSOR_ROOTS = ("kv1", "kv2", "realsense", "xtion")
IMAGE_EXT = (".jpg", ".jpeg", ".png")


@dataclass
class PairRecord:
    """One RGB-depth pair and the result of its integrity checks."""

    scene_id: str
    sample_id: str
    rgb_path: str = ""
    depth_path: str = ""
    intrinsics_path: str = ""
    rgb_shape: list[int] = field(default_factory=list)
    depth_shape: list[int] = field(default_factory=list)
    valid_depth_ratio: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when the pair passed every check."""
        return not self.errors

    def to_dict(self) -> dict:
        """Serialisable view including the ``ok`` flag."""
        d = asdict(self)
        d["ok"] = self.ok
        return d


def scene_relpath(path: str) -> str:
    """Portable scene identity from any absolute/relative SUN RGB-D scene path.

    ``/n/fs/sun3d/data/SUNRGBD/kv1/NYUdata/NYU0001`` -> ``kv1/NYUdata/NYU0001``. Paths in ``allsplit.mat`` and
    ``SUNRGBDMeta2DBB_v2.mat`` are rooted at the authors' machine, so the part after ``SUNRGBD/`` (or from the
    first sensor-root directory) is the only portable key.
    """
    p = str(path).replace("\\", "/").strip().rstrip("/")
    if "SUNRGBD/" in p:
        p = p.split("SUNRGBD/", 1)[1]
    parts = [x for x in p.split("/") if x]
    for i, part in enumerate(parts):
        if part in SENSOR_ROOTS:
            return "/".join(parts[i:])
    return "/".join(parts)


def sample_id_from_scene(scene_id: str) -> str:
    """Filesystem-safe sample identifier used for every output file name."""
    return scene_id.replace("/", "__")


def discover_scenes(sunrgbd_root: str | Path, depth_dir: str = "depth_bfx") -> dict[str, Path]:
    """Map scene identity -> scene directory for every folder containing ``image/`` and ``<depth_dir>/``."""
    root = Path(sunrgbd_root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"SUN RGB-D root not found: {root}\n  Extract SUNRGBD.zip and pass the folder containing kv1/ kv2/ realsense/ xtion/."
        )
    scenes = {}
    for d in sorted(root.rglob(depth_dir)):
        if d.is_dir() and (d.parent / "image").is_dir():
            scenes[d.parent.relative_to(root).as_posix()] = d.parent
    if not scenes:
        raise FileNotFoundError(f"No scene with image/ and {depth_dir}/ under {root}")
    return scenes


def _single_file(folder: Path, exts: tuple[str, ...]) -> tuple[Path | None, str | None]:
    files = sorted(p for p in folder.glob("*") if p.suffix.lower() in exts) if folder.is_dir() else []
    if not files:
        return None, f"missing file in {folder.name}/"
    if len(files) > 1:
        return files[0], f"{len(files)} files in {folder.name}/ (expected 1)"
    return files[0], None


def check_pair(scene_id: str, scene_dir: Path, depth_dir: str = "depth_bfx", min_valid_ratio: float = 0.0) -> PairRecord:
    """Locate and verify the RGB/depth pair of one scene.

    Checks: both files exist and are unique, both decode, RGB is 3-channel, depth is single-channel uint16,
    resolutions are identical (same orientation, H x W), and depth has at least ``min_valid_ratio`` valid pixels.
    """
    from src.data.prepare_depth import read_sunrgbd_depth, valid_depth_mask

    rec = PairRecord(scene_id=scene_id, sample_id=sample_id_from_scene(scene_id))
    rgb, err = _single_file(scene_dir / "image", IMAGE_EXT)
    if err:
        rec.errors.append(f"rgb: {err}")
    dep, err = _single_file(scene_dir / depth_dir, (".png",))
    if err:
        rec.errors.append(f"depth: {err}")
    intr = scene_dir / "intrinsics.txt"
    rec.intrinsics_path = str(intr) if intr.is_file() else ""
    if rgb is None or dep is None:
        return rec
    rec.rgb_path, rec.depth_path = str(rgb), str(dep)

    im = cv2.imread(str(rgb), cv2.IMREAD_COLOR)
    if im is None:
        rec.errors.append("rgb: corrupt/unreadable")
    else:
        rec.rgb_shape = list(im.shape[:2])
    try:
        d = read_sunrgbd_depth(dep)
        rec.depth_shape = list(d.shape[:2])
        rec.valid_depth_ratio = float(valid_depth_mask(d).mean())
    except (OSError, ValueError) as e:
        rec.errors.append(f"depth: {e}")
        return rec
    if rec.rgb_shape and rec.rgb_shape != rec.depth_shape:
        rec.errors.append(f"resolution mismatch rgb={rec.rgb_shape} depth={rec.depth_shape}")
    if rec.valid_depth_ratio <= min_valid_ratio:
        rec.errors.append(f"valid depth ratio {rec.valid_depth_ratio:.4f} <= {min_valid_ratio}")
    return rec


def read_intrinsics(path: str | Path) -> np.ndarray | None:
    """Read SUN RGB-D ``intrinsics.txt`` (3x3, row-major) or return None if absent/malformed."""
    p = Path(path) if path else None
    if p is None or not p.is_file():
        return None
    vals = [float(v) for v in p.read_text().split()]
    if len(vals) != 9:
        return None
    return np.array(vals, dtype=np.float64).reshape(3, 3)
