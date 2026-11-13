"""Face and mouth landmarks (MediaPipe Face Landmarker) with head pose and a reliability estimate."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .geometry import NAN, clip01, dist, wrap_axis_deg
from .mp_models import MonotonicClock, model_path, mp_image

OUTER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
INNER_LIP = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
L_EYE_IMG_LEFT, L_EYE_IMG_RIGHT = 33, 263      # outer eye corners (as seen in the image)
L_MOUTH_IMG_LEFT, L_MOUTH_IMG_RIGHT = 61, 291
L_LIP_UP, L_LIP_LO = 13, 14

# Generic 3-D face model (mm) in camera-style axes: x right, y down, z away from camera.
_PNP_IDX = [1, 152, 33, 263, 61, 291]          # nose tip, chin, eye corners, mouth corners
_MODEL_3D = np.array([[0.0, 0.0, 0.0], [0.0, 63.6, 12.5], [-43.3, -32.7, 26.0],
                      [43.3, -32.7, 26.0], [-28.9, 28.9, 24.1], [28.9, 28.9, 24.1]], dtype=np.float64)


@dataclass
class FaceResult:
    detected: bool = False
    landmarks: Optional[np.ndarray] = None        # (478, 2) pixels
    mouth_center: Optional[Tuple[float, float]] = None
    mouth_outer: Optional[np.ndarray] = None      # (n, 2) outer-lip polygon
    mouth_corners: Optional[np.ndarray] = None    # (2, 2)
    mouth_open_ratio: float = NAN                 # inner-lip gap / mouth width
    face_scale: float = NAN                       # outer-eye-corner distance in px (normaliser)
    yaw: float = NAN                              # head angles in degrees (signed proxies)
    pitch: float = NAN
    roll: float = NAN
    visibility: float = 0.0                       # 0..1 face reliability
    mouth_visibility: float = 0.0                 # 0..1 mouth reliability


def _head_pose(pts: np.ndarray, w: int, h: int) -> Tuple[float, float]:
    img = pts[_PNP_IDX].astype(np.float64)
    K = np.array([[w, 0, w / 2.0], [0, w, h / 2.0], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(_MODEL_3D, img, K, np.zeros(4), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return NAN, NAN
    R, _ = cv2.Rodrigues(rvec)
    ang = cv2.RQDecomp3x3(R)[0]
    wrap = lambda a: ((a + 90.0) % 180.0) - 90.0
    return wrap(float(ang[0])), wrap(float(ang[1]))      # pitch, yaw


class FaceMouthDetector:
    """MediaPipe Face Landmarker (same 478-point topology as Face Mesh with iris)."""

    def __init__(self, cfg: dict):
        from mediapipe.tasks.python import BaseOptions, vision  # heavy import kept local

        f = cfg["face"]
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path(cfg, "face_model")),
            running_mode=vision.RunningMode.VIDEO, num_faces=f["max_faces"],
            min_face_detection_confidence=f["min_detection_conf"],
            min_face_presence_confidence=f.get("min_presence_conf", 0.5),
            min_tracking_confidence=f["min_tracking_conf"])
        self.mesh = vision.FaceLandmarker.create_from_options(opts)
        self._clock = MonotonicClock()
        self.min_face_px = float(f.get("min_face_px", 60))
        self._prev_mc: Optional[Tuple[float, float]] = None
        self._jitter = 0.0

    def reset(self) -> None:
        self._prev_mc, self._jitter = None, 0.0

    def process(self, rgb: np.ndarray) -> FaceResult:
        h, w = rgb.shape[:2]
        res = self.mesh.detect_for_video(mp_image(rgb), self._clock.next())
        if not res.face_landmarks:
            self._prev_mc = None
            return FaceResult()
        lm = res.face_landmarks[0]
        pts = np.array([(p.x, p.y) for p in lm], dtype=np.float64) * np.array([w, h])

        scale = dist(pts[L_EYE_IMG_LEFT], pts[L_EYE_IMG_RIGHT])
        c_corner = (pts[L_MOUTH_IMG_LEFT] + pts[L_MOUTH_IMG_RIGHT]) / 2.0
        c_inner = (pts[L_LIP_UP] + pts[L_LIP_LO]) / 2.0
        mc = tuple(((c_corner + c_inner) / 2.0).tolist())
        width = dist(pts[L_MOUTH_IMG_LEFT], pts[L_MOUTH_IMG_RIGHT])
        open_ratio = dist(pts[L_LIP_UP], pts[L_LIP_LO]) / width if width > 1e-6 else NAN

        pitch, yaw = _head_pose(pts, w, h)
        e = pts[L_EYE_IMG_RIGHT] - pts[L_EYE_IMG_LEFT]
        roll = wrap_axis_deg(math.degrees(math.atan2(e[1], e[0])))

        # Reliability heuristics: landmarks inside the frame, face large enough, temporally stable.
        mouth_idx = OUTER_LIP + INNER_LIP
        mp_ = pts[mouth_idx]
        inside = float(np.mean((mp_[:, 0] >= 0) & (mp_[:, 0] < w) & (mp_[:, 1] >= 0) & (mp_[:, 1] < h)))
        face_inside = float(np.mean((pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)))
        size_score = clip01(scale / self.min_face_px)
        if self._prev_mc is not None and scale > 1:
            jump = dist(mc, self._prev_mc) / scale
            self._jitter = 0.7 * self._jitter + 0.3 * jump
        self._prev_mc = mc
        stability = clip01(1.0 - self._jitter / 0.15)
        vis = face_inside * size_score
        mouth_vis = inside * size_score * stability

        return FaceResult(
            detected=True, landmarks=pts, mouth_center=mc, mouth_outer=pts[OUTER_LIP],
            mouth_corners=pts[[L_MOUTH_IMG_LEFT, L_MOUTH_IMG_RIGHT]], mouth_open_ratio=open_ratio,
            face_scale=scale, yaw=yaw, pitch=pitch, roll=roll,
            visibility=vis, mouth_visibility=mouth_vis)
