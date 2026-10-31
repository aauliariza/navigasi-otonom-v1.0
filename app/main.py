"""Testing interface of the integrated system (prompt section 14.4).

Input: an RGB image file, a local video file, or a webcam (monocular RGB only -- no depth camera, LiDAR, stereo or
range sensor). The page shows which detection and depth checkpoints are loaded, and every tab has a
"Simpan hasil" button that writes the annotated image, depth map (.npy) and result JSON/CSV to
``outputs/inference/interface/``.
Output: detection boxes with confidence and distance, depth map, the five navigation sectors with free/blocked
status, and the recommended direction FORWARD / TURN_LEFT / TURN_RIGHT / STOP. The interface only displays the
decision; it does not control any wheelchair, motor or actuator.

Usage::

    python -m app.main --config configs/system/inference.yaml                 # Gradio web UI (image + webcam)
    python -m app.main --config configs/system/inference.yaml --mode opencv --camera 0   # local webcam window
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from app.components.panels import decision_text, depth_panel, detection_panel, objects_table, sectors_table
from src.inference.common import result_rows
from src.integration.detector_depth_fusion import PerceptionPipeline
from src.utils.config import apply_overrides, load_config, resolve_path
from src.utils.io import save_csv, save_json

SAVE_DIR = "outputs/inference/interface"


def save_result(bgr: np.ndarray, res: dict, pipe: PerceptionPipeline, max_depth: float, out_dir: str | Path = SAVE_DIR, tag: str = "image") -> str:
    """Write annotated image, depth (.npy), result JSON and object/decision CSV; returns the file stem."""
    out = resolve_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    cv2.imwrite(str(out / f"{stem}_result.jpg"), pipe.render(bgr, res, max_depth))
    np.save(out / f"{stem}_depth.npy", res["depth"].astype(np.float32))
    objs, dec = result_rows(stem, res)
    save_csv(objs, out / f"{stem}_objects.csv")
    save_json({"weights": pipe.weights, "decision": dec, **{k: v for k, v in res.items() if k != "depth"}}, out / f"{stem}.json")
    return str(out / stem)


def build_gradio(pipe: PerceptionPipeline, max_depth: float):
    """Gradio Blocks app with image, video and webcam tabs (+ save buttons)."""
    import gradio as gr

    def run(rgb, state):
        if rgb is None:
            return None, None, [], [], "", state
        bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
        res = pipe.process(bgr)
        return (detection_panel(bgr, res), depth_panel(res, bgr.shape[:2], max_depth), objects_table(res), sectors_table(res),
                decision_text(res), {"bgr": bgr, "res": res})

    def save(state, tag):
        if not state:
            return "Belum ada hasil untuk disimpan."
        return "Tersimpan: " + save_result(state["bgr"], state["res"], pipe, max_depth, tag=tag) + "_*"

    def run_video(path):
        from src.inference.predict_video import process_video

        if not path:
            return None, "Pilih berkas video terlebih dahulu."
        out = resolve_path(SAVE_DIR) / f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out.mkdir(parents=True, exist_ok=True)
        s = process_video(pipe, path, out, max_depth)
        return s["video"], (f"{s['frames_processed']} frame diproses, {s['processing_fps']:.2f} FPS. Arah per frame: {s['direction_counts']}\n"
                            f"Hasil disimpan di {out}")

    obj_head = ["ID", "Bounding box (x1,y1,x2,y2)", "Confidence", "Jarak (m)", "R_valid", "Status"]
    sec_head = ["Sektor", "Status", "Objek penghalang"]
    with gr.Blocks(title="Navigasi Otonom - YOLO26n + YOLO26n-Depth (KD)") as demo:
        gr.Markdown("## Deteksi objek, estimasi jarak, dan pemilihan arah gerak dari satu citra RGB\n"
                    f"Model deteksi: `{pipe.weights['detection']}`  \nModel depth: `{pipe.weights['depth']}`  \n"
                    "Keluaran hanya rekomendasi arah pada antarmuka (tidak ada kendali motor).")
        for tab, sources, streaming, tag in (("Gambar RGB", ["upload"], False, "image"), ("Webcam realtime", ["webcam"], True, "webcam")):
            with gr.Tab(tab):
                state = gr.State(None)
                with gr.Row():
                    inp = gr.Image(sources=sources, type="numpy", label="Input RGB", streaming=streaming)
                    out_det = gr.Image(label="Bounding box, confidence, jarak, ROI, sektor, arah")
                    out_dep = gr.Image(label="Depth map metrik (m)")
                decision = gr.Textbox(label="Arah gerak yang direkomendasikan")
                with gr.Row():
                    objs = gr.Dataframe(headers=obj_head, label="Objek dan jarak")
                    secs = gr.Dataframe(headers=sec_head, label="Status sektor navigasi")
                outs = [out_det, out_dep, objs, secs, decision, state]
                if streaming:
                    inp.stream(run, inputs=[inp, state], outputs=outs)
                else:
                    gr.Button("Proses").click(run, inputs=[inp, state], outputs=outs)
                saved = gr.Textbox(label="Status penyimpanan")
                gr.Button("Simpan hasil").click(lambda st, t=tag: save(st, t), inputs=state, outputs=saved)
        with gr.Tab("Video"):
            vin = gr.Video(label="Input video RGB", sources=["upload"])
            vout = gr.Video(label="Hasil (disimpan otomatis)")
            vinfo = gr.Textbox(label="Ringkasan")
            gr.Button("Proses video").click(run_video, inputs=vin, outputs=[vout, vinfo])
    return demo


def run_opencv(pipe: PerceptionPipeline, camera: int, max_depth: float) -> None:
    """Live webcam window (q quit, s save, r record) -- delegates to src.inference.webcam."""
    from src.inference.webcam import run_webcam

    out = resolve_path("outputs/inference/webcam")
    out.mkdir(parents=True, exist_ok=True)
    run_webcam(pipe, camera, out, max_depth)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Interface demonstrasi sistem navigasi")
    ap.add_argument("--config", default="configs/system/inference.yaml")
    ap.add_argument("--mode", choices=["gradio", "opencv"], default="gradio")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args.set)
    pipe = PerceptionPipeline(cfg)
    max_depth = cfg["distance"]["max_depth"]
    if args.mode == "opencv":
        run_opencv(pipe, args.camera, max_depth)
    else:
        build_gradio(pipe, max_depth).launch(server_name=args.host, server_port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
