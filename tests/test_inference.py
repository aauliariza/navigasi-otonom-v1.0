"""Image / video / webcam front-ends (src/inference) with a stub pipeline and a fake camera.

The stub replaces only the two neural networks (it returns one object at a fixed distance); rendering, CSV/JSON
export, frame handling, FPS metering and every camera failure path run the real code. The real models are
exercised end to end in tests/test_pipeline_integration.py.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.integration.detector_depth_fusion import InferenceError, PerceptionPipeline
from src.navigation.direction_decision import decide_direction
from src.navigation.navigation_roi import global_navigation_roi
from src.navigation.obstacle_analysis import BlockingRules, analyse_sectors
from src.navigation.sector_division import divide_sectors


class StubPipeline:
    """Same interface as PerceptionPipeline; one object 0.8 m in front of the centre sector."""

    weights = {"detection": "stub_det.pt", "depth": "stub_depth.pt"}
    render = staticmethod(PerceptionPipeline.render)

    def __init__(self):
        self.calls = 0

    def process(self, bgr):
        self.calls += 1
        h, w = bgr.shape[:2]
        obj = {"id": 0, "bbox": [0.45 * w, 0.3 * h, 0.55 * w, 0.9 * h], "confidence": 0.9, "fusion_confidence": 0.9, "distance_m": 0.8,
               "valid": True, "valid_ratio": 1.0, "n_valid": 16, "n_total": 16, "reason": "ok"}
        roi = global_navigation_roi(w, h)
        sectors = analyse_sectors([obj], divide_sectors(roi), roi, BlockingRules())
        cmd, chosen = decide_direction(sectors)
        return {"objects": [obj], "depth": np.full((h // 4, w // 4), 2.0, np.float32), "roi": list(roi), "sectors": sectors,
                "direction": cmd, "chosen_sector": chosen, "timing_ms": {"detection": 1.0, "depth": 1.0, "fusion_navigation": 0.1, "total": 2.1}}


def _frame(i=0, h=120, w=160):
    f = np.full((h, w, 3), 80, np.uint8)
    cv2.rectangle(f, (10 + i, 10), (60 + i, 60), (200, 200, 200), -1)
    return f


def test_stub_matches_decision_rule():
    res = StubPipeline().process(_frame())
    assert res["direction"] == "TURN_LEFT"  # CTR blocked by the 0.8 m object, L free


def test_predict_image(tmp_path):
    from src.inference.predict_image import list_images, run

    src = tmp_path / "imgs"
    src.mkdir()
    cv2.imwrite(str(src / "a.jpg"), _frame())
    cv2.imwrite(str(src / "b.png"), _frame(5))
    (src / "broken.jpg").write_bytes(b"not an image")
    (src / "notes.txt").write_text("ignored")
    files = list_images(src)
    assert [p.name for p in files] == ["a.jpg", "b.png", "broken.jpg"]
    out = tmp_path / "out"
    out.mkdir()
    s = run(StubPipeline(), files, out, 10.0, save_depth=True)
    assert s["processed"] == 2 and len(s["skipped"]) == 1
    assert (out / "a_result.jpg").is_file() and (out / "b_depth.npy").is_file()
    assert "TURN_LEFT" in (out / "decisions.csv").read_text()
    with pytest.raises(InferenceError):
        list_images(tmp_path / "missing")


def _write_video(path, n=6, fps=10.0):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (160, 120))
    assert w.isOpened()
    for i in range(n):
        w.write(_frame(i * 3))
    w.release()


def test_predict_video(tmp_path):
    from src.inference.predict_video import open_video, process_video

    vid = tmp_path / "clip.avi"
    _write_video(vid, n=6)
    out = tmp_path / "out"
    out.mkdir()
    pipe = StubPipeline()
    s = process_video(pipe, vid, out, 10.0, stride=2)
    assert s["frames_processed"] == 3 and pipe.calls == 3
    assert s["direction_counts"] == {"TURN_LEFT": 3}
    cap = cv2.VideoCapture(str(out / "clip_result.mp4"))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    cap.release()
    assert len((out / "clip_decisions.csv").read_text().strip().splitlines()) == 4  # header + 3 frames
    with pytest.raises(InferenceError):
        open_video(tmp_path / "nope.mp4")
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"\x00" * 100)
    with pytest.raises(InferenceError):
        open_video(bad)


class FakeCapture:
    """cv2.VideoCapture stand-in: a scripted list of frames (None = failed read)."""

    def __init__(self, frames, opened=True):
        self.frames, self.opened, self.released, self.props = list(frames), opened, False, {}

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def read(self):
        if not self.frames:
            return False, None
        f = self.frames.pop(0)
        return (f is not None), f

    def release(self):
        self.released = True


def test_webcam_not_detected(tmp_path):
    from src.inference.webcam import open_camera

    cap = FakeCapture([], opened=False)
    with pytest.raises(InferenceError, match="tidak terdeteksi"):
        open_camera(0, capture_factory=lambda _: cap)
    assert cap.released
    with pytest.raises(InferenceError, match="tidak mengirim frame"):
        open_camera(0, capture_factory=lambda _: FakeCapture([None]))
    with pytest.raises(InferenceError, match="3 kanal"):
        open_camera(0, capture_factory=lambda _: FakeCapture([np.zeros((10, 10), np.uint8)]))


def test_webcam_resolution_mismatch_and_frames(tmp_path):
    from src.inference.webcam import open_camera, parse_camera, run_webcam

    assert parse_camera("0") == 0 and parse_camera("/dev/video2") == "/dev/video2"
    cap, actual = open_camera(0, 1280, 720, capture_factory=lambda _: FakeCapture([_frame()]))
    assert actual == (160, 120)  # camera delivers 160x120 although 1280x720 was requested -> actual size is used
    frames = [_frame()] + [_frame(1), None, _frame(2), np.zeros((0, 0, 3), np.uint8), _frame(3)]
    pipe = StubPipeline()
    s = run_webcam(pipe, 0, tmp_path, 10.0, headless=True, max_frames=3, capture_factory=lambda _: FakeCapture(frames))
    assert s["frames_processed"] == 3 and s["frames_failed"] == 2 and s["resolution"] == [160, 120]
    assert (tmp_path / "webcam_decisions.csv").is_file() and (tmp_path / "webcam_session.json").is_file()


def test_webcam_camera_lost(tmp_path):
    from src.inference.webcam import run_webcam

    frames = [_frame(), _frame(1)] + [None] * 5
    s = run_webcam(StubPipeline(), 0, tmp_path, 10.0, headless=True, max_failed_frames=3, capture_factory=lambda _: FakeCapture(frames))
    assert s["stop_reason"] == "camera_lost" and s["frames_processed"] == 1


def test_pipeline_rejects_missing_checkpoint(tmp_path):
    from src.integration.detector_depth_fusion import resolve_model_paths

    cfg = {"models": {"detection_weights": str(tmp_path / "det.pt"), "depth_weights": str(tmp_path / "dep.pt"), "selected_models": None}}
    with pytest.raises(InferenceError, match="checkpoint model deteksi"):
        resolve_model_paths(cfg)
    cfg = {"models": {"detection_weights": None, "depth_weights": None, "selected_models": str(tmp_path / "sel.json")}}
    with pytest.raises(InferenceError, match="EWM-SAW"):
        resolve_model_paths(cfg)


def test_invalid_frame_rejected():
    pipe = PerceptionPipeline.__new__(PerceptionPipeline)
    with pytest.raises(InferenceError):
        pipe.process(np.zeros((10, 10), np.uint8))
