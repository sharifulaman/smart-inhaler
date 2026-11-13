"""Hand landmarks (MediaPipe Hands) and upper-body pose (MediaPipe Pose)."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .geometry import NAN, axis_angle_deg, points_box_min_dist
from .mp_models import MonotonicClock, model_path, mp_image

PALM_IDX = [0, 5, 9, 13, 17]  # wrist + finger bases


@dataclass
class HandsPoseResult:
    hands: List[np.ndarray] = field(default_factory=list)   # each (21, 2) pixels
    scores: List[float] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    holding_idx: Optional[int] = None                       # hand judged to hold the inhaler
    hand_center: Optional[Tuple[float, float]] = None       # palm centre of that hand
    hand_conf: float = NAN
    pose_points: Optional[np.ndarray] = None                # (33, 2) pixels
    pose_vis: float = NAN
    shoulder_tilt: float = NAN                              # deg, 0 = level shoulders
    torso_lean: float = NAN                                 # deg, 0 = head centred over shoulders


class HandPoseDetector:
    """MediaPipe Hand Landmarker + Pose Landmarker (Tasks API, VIDEO mode)."""

    def __init__(self, cfg: dict):
        from mediapipe.tasks.python import BaseOptions, vision

        h, p = cfg["hands"], cfg["pose"]
        self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path(cfg, "hand_model")),
            running_mode=vision.RunningMode.VIDEO, num_hands=h["max_hands"],
            min_hand_detection_confidence=h["min_detection_conf"],
            min_hand_presence_confidence=h.get("min_presence_conf", 0.5),
            min_tracking_confidence=h["min_tracking_conf"]))
        self.pose = None
        if p.get("enabled", True):
            self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path(cfg, "pose_model")),
                running_mode=vision.RunningMode.VIDEO, num_poses=1,
                min_pose_detection_confidence=p["min_detection_conf"],
                min_pose_presence_confidence=p.get("min_presence_conf", 0.5),
                min_tracking_confidence=p["min_tracking_conf"]))
        self._hand_clock, self._pose_clock = MonotonicClock(), MonotonicClock()
        self.every_n = max(1, int(p.get("every_n_frames", 1)))
        self.hold_frac = float(h.get("hold_dist_frac", 0.75))
        self._pose_cache: Optional[tuple] = None
        self.last_timings: Dict[str, float] = {"hands": 0.0, "pose": 0.0}

    def reset(self) -> None:
        self._pose_cache = None

    def process(self, rgb: np.ndarray, ref_box: Optional[tuple], frame_idx: int) -> HandsPoseResult:
        h, w = rgb.shape[:2]
        res = HandsPoseResult()

        t0 = time.perf_counter()
        hr = self.hands.detect_for_video(mp_image(rgb), self._hand_clock.next())
        if hr.hand_landmarks:
            for lms, hd in zip(hr.hand_landmarks, hr.handedness):
                res.hands.append(np.array([(p.x, p.y) for p in lms]) * np.array([w, h]))
                res.scores.append(float(hd[0].score))
                res.labels.append(str(hd[0].category_name))
            if ref_box is not None:
                self._select_holding_hand(res, ref_box, w, h)
        self.last_timings["hands"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        if self.pose is not None:
            if self._pose_cache is None or frame_idx % self.every_n == 0:
                self._pose_cache = self._run_pose(rgb, w, h)
            if self._pose_cache is not None:
                res.pose_points, res.pose_vis, res.shoulder_tilt, res.torso_lean = self._pose_cache
        self.last_timings["pose"] = (time.perf_counter() - t0) * 1000.0
        return res

    # ------------------------------------------------------------------ internals
    def _select_holding_hand(self, res: HandsPoseResult, box: tuple, w: int, h: int) -> None:
        diag = math.hypot(box[2] - box[0], box[3] - box[1])
        d = [points_box_min_dist(p, box) for p in res.hands]
        i = int(np.argmin(d))
        if d[i] > self.hold_frac * diag:
            return
        pts = res.hands[i]
        inside = float(np.mean((pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)))
        palm = pts[PALM_IDX].mean(axis=0)
        res.holding_idx = i
        res.hand_center = (float(palm[0]), float(palm[1]))
        res.hand_conf = res.scores[i] * inside

    def _run_pose(self, rgb: np.ndarray, w: int, h: int) -> Optional[tuple]:
        pr = self.pose.detect_for_video(mp_image(rgb), self._pose_clock.next())
        if not pr.pose_landmarks:
            return None
        lm = pr.pose_landmarks[0]
        pts = np.array([(p.x, p.y) for p in lm]) * np.array([w, h])
        vis = np.array([p.visibility if p.visibility is not None else 0.0 for p in lm])
        pose_vis = float(np.mean(vis[[0, 11, 12]]))
        tilt = axis_angle_deg(pts[11], pts[12])
        mid = (pts[11] + pts[12]) / 2.0
        lean = math.degrees(math.atan2(mid[0] - pts[0][0], mid[1] - pts[0][1]))
        return pts, pose_vis, tilt, lean
