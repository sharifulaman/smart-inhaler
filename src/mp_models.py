"""MediaPipe Tasks helpers: model-file lookup and image conversion.

Recent MediaPipe releases removed the legacy `mp.solutions` API; the Tasks API used here works on
all current versions. Tasks need .task model bundles, fetched once by scripts/download_models.py.
"""
from __future__ import annotations

import numpy as np

from .config import resolve

MODEL_URLS = {
    "face_model": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "hand_model": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_model": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}


def model_path(cfg: dict, key: str) -> str:
    p = resolve(cfg["mediapipe"][key])
    if not p.exists():
        raise FileNotFoundError(
            f"MediaPipe model file missing: {p}\nRun `python scripts/download_models.py` once "
            f"(or download it from {MODEL_URLS[key]}).")
    return str(p)


def mp_image(rgb: np.ndarray):
    import mediapipe as mp
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


class MonotonicClock:
    """VIDEO running mode needs strictly increasing millisecond timestamps."""

    def __init__(self, step_ms: int = 33):
        self.step, self.t = step_ms, 0

    def next(self) -> int:
        self.t += self.step
        return self.t
