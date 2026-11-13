"""Pipeline integration test with stubbed detectors (no model files, GPU or camera needed).

Landmark stubs follow the MediaPipe Tasks result shapes; this checks our parsing, feature
extraction, classification and overlay wiring - not the accuracy of the real models.
"""
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

vision = pytest.importorskip("mediapipe.tasks.python.vision")

from src import face_mouth, hands_pose, mp_models
import src.pipeline as P
from src.config import load_config
from src.detector import Detection
from src.geometry import axis_angle_deg
from src.overlay import draw

W, H = 640, 480


def L(x, y, v=1.0):
    return types.SimpleNamespace(x=x / W, y=y / H, z=0, visibility=v, presence=1.0)


class FakeFace:
    def detect_for_video(self, img, ts):
        pts = [L(320, 240)] * 478
        pts = list(pts)
        pts[1], pts[152], pts[33], pts[263] = L(320, 240), L(320, 330), L(270, 200), L(370, 200)
        pts[61], pts[291], pts[13], pts[14] = L(292, 290), L(348, 290), L(320, 286), L(320, 296)
        for k, i in enumerate(face_mouth.OUTER_LIP + face_mouth.INNER_LIP):
            if i not in (61, 291, 13, 14):
                a = 2 * math.pi * k / 40
                pts[i] = L(320 + 30 * math.cos(a), 290 + 12 * math.sin(a))
        return types.SimpleNamespace(face_landmarks=[pts])


class FakeHand:
    def detect_for_video(self, img, ts):
        h = [L(470 + 3 * i, 250 + 2 * i) for i in range(21)]
        return types.SimpleNamespace(hand_landmarks=[h],
                                     handedness=[[types.SimpleNamespace(score=0.95, category_name="Right")]])


class FakePose:
    def detect_for_video(self, img, ts):
        p = list([L(320, 200, 0.9)] * 33)
        p[11], p[12] = L(250, 400, 0.9), L(390, 404, 0.9)
        return types.SimpleNamespace(pose_landmarks=[p])


class FakeDet:
    def __init__(self, cfg):
        pass

    def reset(self):
        pass

    def detect(self, frame):
        a, b = (390.0, 270.0), (500.0, 300.0)
        return [Detection("inhaler", 0.9, (390, 262, 500, 308), 1, None, (a, b), 117.0,
                          axis_angle_deg(a, b), "obb")]


@pytest.fixture
def pipe(monkeypatch):
    monkeypatch.setattr(face_mouth, "model_path", lambda cfg, key: "x")
    monkeypatch.setattr(hands_pose, "model_path", lambda cfg, key: "x")
    monkeypatch.setattr(vision.FaceLandmarker, "create_from_options", staticmethod(lambda o: FakeFace()))
    monkeypatch.setattr(vision.HandLandmarker, "create_from_options", staticmethod(lambda o: FakeHand()))
    monkeypatch.setattr(vision.PoseLandmarker, "create_from_options", staticmethod(lambda o: FakePose()))
    monkeypatch.setattr(P, "InhalerDetector", FakeDet)
    cfg = load_config()
    cfg["classifier"]["model_path"] = "models/__does_not_exist__.joblib"   # force rule-based fallback
    return P.InhalerPipeline(cfg)


def test_pipeline_end_to_end(pipe):
    img = np.full((H, W, 3), 70, np.uint8)
    for _ in range(25):
        out = pipe.process(img)
    f = out.features
    assert out.face.detected and abs(out.face.face_scale - 100.0) < 1e-6
    assert 0.6 < f["mp_mouth_dist"] < 0.9
    assert [f[k] for k in ("has_inhaler", "has_face", "has_mouth", "has_hand")] == [1.0] * 4
    assert out.hands.holding_idx == 0
    assert out.prediction is not None and out.prediction.source == "rule-based"
    assert out.prediction.label == "Correct"
    assert set(out.timings) >= {"yolo", "face", "hands", "pose", "features", "classifier", "total"}
    vis = draw(img, out, dict(fps=20.0, latency_ms=50.0, cam_state="ok"))
    assert vis.shape == img.shape and (vis != img).any()
