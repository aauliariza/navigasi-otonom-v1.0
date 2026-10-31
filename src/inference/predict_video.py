"""Video testing (prompt section 14.2): runs the integrated system on every frame of a local video file.

Each output frame shows boxes + confidence + distance (m), the global navigation ROI and the five sectors with
free/blocked status, the recommended direction, the loop FPS, and the depth map next to the RGB frame.

Frames are processed one at a time (read -> process -> write -> release): nothing but the per-frame CSV rows is
kept in memory, so long videos do not accumulate frames or depth maps.

Usage::

    python -m src.inference.predict_video --source path/to/video.mp4
    python -m src.inference.predict_video --source video.mp4 --stride 2 --max-frames 300 --display

Outputs (``--output``, default ``outputs/inference/videos``): ``<name>_result.mp4``, ``<name>_objects.csv``,
``<name>_decisions.csv`` (direction and sector status per frame) and ``<name>_summary.json``.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from src.inference.common import FpsMeter, build_arg_parser, load_cfg, load_pipeline, output_dir, result_rows
from src.integration.detector_depth_fusion import InferenceError
from src.utils.config import resolve_path
from src.utils.io import save_csv, save_json
from src.utils.logger import get_logger

LOG = get_logger("predict_video")


def open_video(path: str | Path) -> cv2.VideoCapture:
    """Open a video file or raise InferenceError (missing file, unsupported codec)."""
    p = resolve_path(path)
    if not p.is_file():
        raise InferenceError("predict_video", "berkas video tidak ditemukan", p, "periksa path --source")
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        cap.release()
        raise InferenceError("predict_video", "video tidak dapat dibuka (codec/format tidak didukung OpenCV)", p,
                             "konversi ke MP4 (H.264) atau AVI (MJPG), mis. `ffmpeg -i in.mov -c:v libx264 out.mp4`")
    return cap


def make_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """MP4 (mp4v) writer; raises InferenceError if OpenCV cannot create it."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not w.isOpened():
        raise InferenceError("predict_video", "VideoWriter gagal dibuat", path, "periksa izin tulis folder output / codec mp4v")
    return w


def process_video(pipe, source: str | Path, out: Path, max_depth: float, stride: int = 1, max_frames: int | None = None,
                  display: bool = False, save_video: bool = True) -> dict:
    """Process a video and write the annotated video and CSV logs. Returns the summary dict."""
    if stride < 1:
        raise ValueError("stride must be >= 1")
    cap = open_video(source)
    name = Path(str(source)).stem
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    src_fps = src_fps if 0 < src_fps < 1000 else 30.0
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    writer, objs, decs = None, [], []
    meter = FpsMeter()
    index = processed = 0
    t0 = time.perf_counter()
    directions: dict[str, int] = {}
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            index += 1
            if (index - 1) % stride:
                continue
            res = pipe.process(frame)
            fps = meter.tick()
            vis = pipe.render(frame, res, max_depth, [f"frame {index}/{n_src or '?'}", f"FPS {fps:.1f}"])
            if save_video:
                if writer is None:
                    writer = make_writer(out / f"{name}_result.mp4", src_fps / stride, (vis.shape[1], vis.shape[0]))
                writer.write(vis)
            o, d = result_rows(f"{name}#{index}", res, {"frame": index, "loop_fps": round(fps, 3)})
            objs += o
            decs.append(d)
            directions[res["direction"]] = directions.get(res["direction"], 0) + 1
            processed += 1
            if display:
                cv2.imshow("Navigasi otonom - video (q = berhenti)", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if max_frames is not None and processed >= max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
    if processed == 0:
        raise InferenceError("predict_video", "tidak ada frame yang dapat dibaca dari video", source)
    elapsed = time.perf_counter() - t0
    save_csv(objs, out / f"{name}_objects.csv")
    save_csv(decs, out / f"{name}_decisions.csv")
    summary = {"source": str(source), "weights": pipe.weights, "frames_read": index, "frames_processed": processed, "stride": stride,
               "source_fps": src_fps, "processing_fps": processed / elapsed if elapsed > 0 else None,
               "direction_counts": directions, "video": str(out / f"{name}_result.mp4") if save_video else None}
    save_json(summary, out / f"{name}_summary.json")
    LOG.info("%s: %d frame diproses, %.2f FPS, arah=%s", name, processed, summary["processing_fps"] or 0, directions)
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = build_arg_parser("Integrated system on a video file", "outputs/inference/videos")
    ap.add_argument("--source", required=True, help="Video file (mp4, avi, mov, mkv, ...)")
    ap.add_argument("--stride", type=int, default=1, help="Process every n-th frame")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--display", action="store_true", help="Show the annotated frames while processing")
    ap.add_argument("--no-save-video", action="store_true", help="Only write CSV/JSON logs")
    args = ap.parse_args(argv)
    cfg = load_cfg(args)
    open_video(args.source).release()  # fail fast before loading models
    pipe = load_pipeline(cfg)
    process_video(pipe, args.source, output_dir(args.output), cfg["distance"]["max_depth"], args.stride, args.max_frames,
                  args.display, not args.no_save_video)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
