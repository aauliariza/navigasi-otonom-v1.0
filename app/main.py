"""Demonstration interface of the integrated system (prompt section 12).

Input: an RGB image file or a webcam (monocular RGB only -- no depth camera, LiDAR, stereo or range sensor).
Output: detection boxes with confidence and distance, depth map, the five navigation sectors with free/blocked
status, and the recommended direction FORWARD / TURN_LEFT / TURN_RIGHT / STOP. The interface only displays the
decision; it does not control any wheelchair, motor or actuator.

Usage::

    python -m app.main --config configs/system/inference.yaml                 # Gradio web UI (image + webcam)
    python -m app.main --config configs/system/inference.yaml --mode opencv --camera 0   # local webcam window
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from app.components.panels import decision_text, depth_panel, detection_panel, objects_table, sectors_table
from src.integration.detector_depth_fusion import PerceptionPipeline
from src.utils.config import apply_overrides, load_config


def build_gradio(pipe: PerceptionPipeline, max_depth: float):
    """Gradio Blocks app with an image tab and a webcam tab."""
    import gradio as gr

    def run(rgb):
        if rgb is None:
            return None, None, [], [], ""
        bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
        res = pipe.process(bgr)
        return detection_panel(bgr, res), depth_panel(res, bgr.shape[:2], max_depth), objects_table(res), sectors_table(res), decision_text(res)

    obj_head = ["ID", "Bounding box (x1,y1,x2,y2)", "Confidence", "Jarak (m)", "R_valid", "Status"]
    sec_head = ["Sektor", "Status", "Objek penghalang"]
    with gr.Blocks(title="Navigasi Otonom - YOLO26n + YOLO26n-Depth (KD)") as demo:
        gr.Markdown("## Deteksi objek, estimasi jarak, dan pemilihan arah gerak dari satu citra RGB\n"
                    f"Deteksi: `{pipe.weights['detection']}`  \nDepth: `{pipe.weights['depth']}`")
        for tab, sources, streaming in (("Gambar RGB", ["upload"], False), ("Webcam", ["webcam"], True)):
            with gr.Tab(tab):
                with gr.Row():
                    inp = gr.Image(sources=sources, type="numpy", label="Input RGB", streaming=streaming)
                    out_det = gr.Image(label="Bounding box, jarak, sektor, arah")
                    out_dep = gr.Image(label="Depth map (m)")
                decision = gr.Textbox(label="Arah gerak yang direkomendasikan")
                with gr.Row():
                    objs = gr.Dataframe(headers=obj_head, label="Objek")
                    secs = gr.Dataframe(headers=sec_head, label="Sektor navigasi")
                outs = [out_det, out_dep, objs, secs, decision]
                if streaming:
                    inp.stream(run, inputs=inp, outputs=outs)
                else:
                    gr.Button("Proses").click(run, inputs=inp, outputs=outs)
    return demo


def run_opencv(pipe: PerceptionPipeline, camera: int, max_depth: float) -> None:
    """Live webcam window (press q to quit)."""
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam {camera}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = pipe.process(frame)
            cv2.imshow("Navigasi otonom (q = keluar)", pipe.render(frame, res, max_depth))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


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
