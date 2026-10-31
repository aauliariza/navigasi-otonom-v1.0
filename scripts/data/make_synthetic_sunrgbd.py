#!/usr/bin/env python
"""Build a tiny SYNTHETIC dataset with the raw SUN RGB-D layout, for smoke tests only.

It reproduces the file formats the real pipeline consumes (bit-rotated uint16 depth in ``depth_bfx/``,
``intrinsics.txt``, ``allsplit.mat`` with ``alltrain``/``alltest`` and ``SUNRGBDMeta2DBB_v2.mat`` with
``gtBb2D``), so preparation -> validation -> training -> evaluation can be exercised without the real data.
The scenes are random boxes in front of a planar wall; NO result computed on them is a research result.

Usage::

    python scripts/data/make_synthetic_sunrgbd.py --out outputs/smoke/raw --train 8 --val 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.prepare_depth import encode_sunrgbd_raw  # noqa: E402


def make_scene(rng: np.random.Generator, h: int = 240, w: int = 320):
    """Return (rgb, depth_m, boxes_xywh): a wall at 3-6 m with 1-4 nearer rectangular objects."""
    wall = rng.uniform(3.0, 6.0)
    depth = np.full((h, w), wall, np.float32) + np.linspace(0, 0.5, h, dtype=np.float32)[:, None]
    rgb = np.full((h, w, 3), 90, np.uint8)
    rgb[:] = (rng.integers(60, 120), rng.integers(60, 120), rng.integers(60, 120))
    boxes = []
    for _ in range(int(rng.integers(1, 5))):
        bw, bh = int(rng.integers(30, 120)), int(rng.integers(30, 120))
        x, y = int(rng.integers(0, w - bw)), int(rng.integers(0, h - bh))
        z = rng.uniform(0.6, wall - 0.3)
        depth[y : y + bh, x : x + bw] = z
        rgb[y : y + bh, x : x + bw] = rng.integers(130, 255, 3)
        boxes.append([x, y, bw, bh])
    holes = rng.random((h, w)) < 0.02
    depth[holes] = 0.0
    return rgb, depth, boxes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="outputs/smoke/raw")
    ap.add_argument("--train", type=int, default=8)
    ap.add_argument("--val", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    from scipy.io import savemat

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    root = out / "SUNRGBD"
    names = {"train": [], "val": []}
    meta = []
    for split, n in (("train", args.train), ("val", args.val)):
        for i in range(n):
            scene_rel = f"kv1/NYUdata/SYN{split}{i:04d}"
            sd = root / scene_rel
            for sub in ("image", "depth_bfx", "depth"):
                (sd / sub).mkdir(parents=True, exist_ok=True)
            rgb, depth, boxes = make_scene(rng)
            cv2.imwrite(str(sd / "image" / f"{i:06d}.jpg"), rgb)
            raw = encode_sunrgbd_raw(np.rint(depth * 1000).astype(np.uint16))
            cv2.imwrite(str(sd / "depth_bfx" / f"{i:06d}.png"), raw)
            cv2.imwrite(str(sd / "depth" / f"{i:06d}.png"), raw)
            (sd / "intrinsics.txt").write_text("260 0 160\n0 260 120\n0 0 1\n")
            names[split].append(f"/n/fs/sun3d/data/SUNRGBD/{scene_rel}")
            gt = np.zeros((len(boxes),), dtype=[("gtBb2D", "O"), ("classname", "O")])
            for k, b in enumerate(boxes):
                gt[k] = (np.array(b, dtype=np.float64), "chair")
            meta.append((f"SUNRGBD/{scene_rel}", gt))
    tb = out / "SUNRGBDtoolbox" / "traintestSUNRGBD"
    tb.mkdir(parents=True, exist_ok=True)
    savemat(str(tb / "allsplit.mat"), {"alltrain": np.array(names["train"], dtype=object), "alltest": np.array(names["val"], dtype=object)})
    arr = np.zeros((len(meta),), dtype=[("sequenceName", "O"), ("groundtruth2DBB", "O")])
    for k, (seq, gt) in enumerate(meta):
        arr[k] = (seq, gt)
    savemat(str(out / "SUNRGBDMeta2DBB_v2.mat"), {"SUNRGBDMeta2DBB": arr})
    print(f"Synthetic raw SUN RGB-D layout written to {out} (train={args.train}, val={args.val})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
