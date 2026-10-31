"""Image testing (prompt section 14.1): JPG / JPEG / PNG / BMP / TIFF / WEBP files or a folder of them.

For every image: read -> preprocessing of both models -> YOLO26n detection -> YOLO26n-Depth metric depth ->
alignment + inner ROI distance -> five sectors free/blocked -> recommended direction -> visualisation.

Usage::

    python -m src.inference.predict_image --source path/to/image.jpg
    python -m src.inference.predict_image --source path/to/folder --output outputs/inference/images \\
        --detection-weights outputs/detection/DET-04/weights/best.pt --depth-weights outputs/depth/DEP-07/weights/best.pt

Outputs (``--output``, default ``outputs/inference/images``): ``<name>_result.jpg`` (RGB with boxes, confidence,
distance, ROI, sectors, direction | depth map), ``<name>_depth.npy`` (metric depth, optional ``--save-depth``),
``objects.csv``, ``decisions.csv`` and ``results.json``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.inference.common import build_arg_parser, load_cfg, load_pipeline, output_dir, result_rows
from src.integration.detector_depth_fusion import InferenceError
from src.utils.config import resolve_path
from src.utils.io import save_csv, save_json
from src.utils.logger import get_logger

LOG = get_logger("predict_image")
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(source: str | Path) -> list[Path]:
    """Image files of a file or folder (sorted). Raises InferenceError when nothing usable is found."""
    p = resolve_path(source)
    if p.is_dir():
        files = sorted(x for x in p.iterdir() if x.suffix.lower() in IMAGE_EXT)
    elif p.is_file():
        files = [p]
    else:
        raise InferenceError("predict_image", "sumber gambar tidak ditemukan", p, "periksa path --source")
    if not files:
        raise InferenceError("predict_image", f"tidak ada berkas gambar ({', '.join(sorted(IMAGE_EXT))})", p)
    return files


def read_image(path: Path) -> np.ndarray:
    """BGR image; InferenceError for unreadable files (OpenCV returns None instead of raising)."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise InferenceError("predict_image", "gambar tidak dapat dibaca OpenCV (rusak atau format tidak didukung)", path)
    return bgr


def run(pipe, images: list[Path], out: Path, max_depth: float, save_depth: bool = False) -> dict:
    """Process every image and write all outputs. Unreadable images are reported and skipped."""
    objs, decs, full, skipped = [], [], [], []
    for p in images:
        try:
            bgr = read_image(p)
        except InferenceError as e:
            LOG.error("%s", e)
            skipped.append({"image": str(p), "error": e.cause})
            continue
        res = pipe.process(bgr)
        cv2.imwrite(str(out / f"{p.stem}_result.jpg"), pipe.render(bgr, res, max_depth, [p.name]))
        if save_depth:
            np.save(out / f"{p.stem}_depth.npy", res["depth"].astype(np.float32))
        o, d = result_rows(p.name, res)
        objs += o
        decs.append(d)
        full.append({"image": p.name, **{k: v for k, v in res.items() if k != "depth"}, "depth_shape": list(res["depth"].shape)})
        LOG.info("%s: %d objek -> %s", p.name, len(res["objects"]), res["direction"])
    save_csv(objs, out / "objects.csv")
    save_csv(decs, out / "decisions.csv")
    summary = {"weights": pipe.weights, "processed": len(full), "skipped": skipped, "results": full}
    save_json(summary, out / "results.json")
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = build_arg_parser("Integrated system on image files", "outputs/inference/images")
    ap.add_argument("--source", required=True, help="Image file or folder")
    ap.add_argument("--save-depth", action="store_true", help="Also save the metric depth map as .npy")
    args = ap.parse_args(argv)
    cfg = load_cfg(args)
    images = list_images(args.source)
    pipe = load_pipeline(cfg)
    summary = run(pipe, images, output_dir(args.output), cfg["distance"]["max_depth"], args.save_depth)
    return 0 if summary["processed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
