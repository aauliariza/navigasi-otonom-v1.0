"""Realtime testing with ONE monocular RGB webcam (prompt section 14.3).

The camera is used only as an RGB image source (no depth stream of any RGB-D device is opened or read).
Keys in the window: ``q``/``Esc`` quit, ``s`` save the current annotated frame + result JSON, ``r`` start/stop
recording the annotated stream to MP4.

Handled failure cases (each reported with module, cause and suggested fix, never silently ignored):
* camera not detected / cannot be opened (``--camera`` index or device path);
* empty / failed frames: tolerated up to ``--max-failed-frames`` consecutive failures, then the loop stops;
* camera resolution different from the requested ``--width``/``--height``: the actual resolution is used and
  logged (both models resize internally, coordinates are mapped back to the actual frame size);
* model or checkpoint missing (raised by :class:`PerceptionPipeline`); GPU unavailable -> CPU with a warning.

Usage::

    python -m src.inference.webcam                    # device 0, 1280x720 requested (Logitech C270 = 720p)
    python -m src.inference.webcam --camera 1 --width 640 --height 480
    python -m src.inference.webcam --headless --max-frames 100     # no window, log + recordings only

Outputs (``--output``, default ``outputs/inference/webcam``): ``snapshot_<time>.jpg|json``, ``record_<time>.mp4``,
``webcam_decisions.csv`` (direction, sector status and FPS per processed frame), ``webcam_session.json``.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import cv2

from src.inference.common import FpsMeter, build_arg_parser, load_cfg, load_pipeline, output_dir, result_rows
from src.integration.detector_depth_fusion import InferenceError
from src.utils.io import save_csv, save_json
from src.utils.logger import get_logger

LOG = get_logger("webcam")


def parse_camera(value: str) -> int | str:
    """Camera index (``"0"`` -> 0) or a device path / URL string."""
    return int(value) if str(value).isdigit() else str(value)


def open_camera(camera: int | str, width: int | None = None, height: int | None = None, capture_factory=cv2.VideoCapture):
    """Open the camera, request a resolution and return ``(cap, (actual_w, actual_h))``.

    Raises:
        InferenceError: camera not found / cannot be opened, or it opens but delivers no frame.
    """
    cap = capture_factory(camera)
    if cap is None or not cap.isOpened():
        if cap is not None:
            cap.release()
        raise InferenceError("webcam", f"kamera '{camera}' tidak terdeteksi atau gagal dibuka",
                             solution="periksa sambungan USB, izin akses kamera, aplikasi lain yang memakai kamera, "
                                      "atau coba indeks lain (--camera 1)")
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    ok, frame = cap.read()
    if not ok or frame is None or frame.size == 0:
        cap.release()
        raise InferenceError("webcam", f"kamera '{camera}' terbuka tetapi tidak mengirim frame", solution="cabut-pasang kamera atau coba indeks lain")
    actual = (int(frame.shape[1]), int(frame.shape[0]))
    if (width and actual[0] != width) or (height and actual[1] != height):
        LOG.warning("[webcam] resolusi diminta %sx%s, kamera memberi %dx%d; resolusi aktual yang dipakai.", width, height, *actual)
    if frame.ndim != 3 or frame.shape[2] != 3:
        cap.release()
        raise InferenceError("webcam", f"frame kamera bukan citra RGB 3 kanal (shape={frame.shape})",
                             solution="pilih perangkat kamera RGB biasa; stream depth/IR tidak digunakan penelitian ini")
    return cap, actual


def run_webcam(pipe, camera: int | str, out: Path, max_depth: float, width: int | None = None, height: int | None = None,
               headless: bool = False, max_frames: int | None = None, max_failed_frames: int = 30,
               capture_factory=cv2.VideoCapture) -> dict:
    """Realtime loop. Returns a session summary (also saved as JSON)."""
    cap, actual = open_camera(camera, width, height, capture_factory)
    LOG.info("Kamera %s aktif, resolusi %dx%d", camera, *actual)
    meter = FpsMeter()
    decs: list[dict] = []
    writer = None
    processed = failed = consecutive_failed = 0
    stop_reason = "max_frames" if max_frames else "user"
    t_start = time.perf_counter()
    try:
        while max_frames is None or processed < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                failed += 1
                consecutive_failed += 1
                LOG.warning("[webcam] frame kosong/gagal dibaca (%d berturut-turut)", consecutive_failed)
                if consecutive_failed >= max_failed_frames:
                    stop_reason = "camera_lost"
                    LOG.error("[webcam] %d frame gagal berturut-turut; kamera dianggap terputus.", consecutive_failed)
                    break
                continue
            consecutive_failed = 0
            res = pipe.process(frame)
            fps = meter.tick()
            status = [f"CAM {camera}: ON {frame.shape[1]}x{frame.shape[0]}", f"FPS {fps:.1f}", "REC" if writer is not None else ""]
            vis = pipe.render(frame, res, max_depth, [s for s in status if s])
            processed += 1
            _, d = result_rows(f"frame{processed}", res, {"frame": processed, "time_s": round(time.perf_counter() - t_start, 3),
                                                          "loop_fps": round(fps, 3)})
            decs.append(d)
            if writer is not None:
                writer.write(vis)
            if headless:
                continue
            cv2.imshow("Navigasi otonom - webcam (q keluar, s simpan, r rekam)", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                stop_reason = "user"
                break
            if key == ord("s"):
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                cv2.imwrite(str(out / f"snapshot_{stamp}.jpg"), vis)
                save_json({k: v for k, v in res.items() if k != "depth"}, out / f"snapshot_{stamp}.json")
                LOG.info("Snapshot disimpan: snapshot_%s.jpg", stamp)
            if key == ord("r"):
                if writer is None:
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    writer = cv2.VideoWriter(str(out / f"record_{stamp}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (vis.shape[1], vis.shape[0]))
                    LOG.info("Rekam dimulai: record_%s.mp4", stamp)
                else:
                    writer.release()
                    writer = None
                    LOG.info("Rekam dihentikan")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not headless:
            cv2.destroyAllWindows()
    elapsed = time.perf_counter() - t_start
    save_csv(decs, out / "webcam_decisions.csv")
    summary = {"camera": str(camera), "resolution": list(actual), "weights": pipe.weights, "frames_processed": processed,
               "frames_failed": failed, "mean_loop_fps": processed / elapsed if elapsed > 0 else None, "stop_reason": stop_reason}
    save_json(summary, out / "webcam_session.json")
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = build_arg_parser("Integrated system on a realtime monocular RGB webcam", "outputs/inference/webcam")
    ap.add_argument("--camera", default="0", help="Camera index (default 0) or device path")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--headless", action="store_true", help="No display window")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--max-failed-frames", type=int, default=30)
    args = ap.parse_args(argv)
    cfg = load_cfg(args)
    camera = parse_camera(args.camera)
    open_camera(camera, args.width, args.height)[0].release()  # fail fast before loading models
    pipe = load_pipeline(cfg)
    s = run_webcam(pipe, camera, output_dir(args.output), cfg["distance"]["max_depth"], args.width, args.height, args.headless,
                   args.max_frames, args.max_failed_frames)
    return 0 if s["stop_reason"] != "camera_lost" else 2


if __name__ == "__main__":
    raise SystemExit(main())
