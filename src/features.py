"""Geometric feature extraction, temporal smoothing, sliding windows and the visibility gate.

Per-frame features are normalised for resolution/distance: positions by image size, and every
distance by the face scale (outer-eye-corner distance, "face-width" units). They are smoothed
with a One-Euro filter. The classifier works on windowed statistics of these per-frame features.
"""
from __future__ import annotations

import math
import warnings
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import NAN, axis_diff_deg, dist, unwrap_axis, wrap_axis_deg

# --------------------------------------------------------------------------- schema
FRAME_FEATURE_NAMES: List[str] = [
    "inhaler_cx", "inhaler_cy",            # inhaler centre / image size
    "mouth_cx", "mouth_cy",                # mouth centre / image size
    "mouthpiece_x", "mouthpiece_y",        # mouthpiece end / image size
    "mp_rel_x", "mp_rel_y",                # mouthpiece minus mouth centre, face-widths
    "mp_mouth_dist",                       # |mp_rel|, face-widths
    "inhaler_size",                        # inhaler long-axis length, face-widths
    "inhaler_angle",                       # inhaler own long-axis angle (deg), image frame
    "inhaler_angle_rel_head",              # inhaler angle minus head roll (deg)
    "hand_inhaler_dist", "hand_mouth_dist",  # palm centre to inhaler centre / mouth, face-widths
    "hand_rel_x", "hand_rel_y",            # palm centre minus mouth centre, face-widths
    "head_yaw", "head_pitch", "head_roll",
    "mouth_open_ratio",
    "torso_lean", "shoulder_tilt",
    "inhaler_conf", "face_vis", "mouth_vis", "hand_vis", "pose_vis",
]
VISIBILITY_NAMES = ["inhaler_conf", "face_vis", "mouth_vis", "hand_vis", "pose_vis"]
FLAG_NAMES = ["has_inhaler", "has_face", "has_mouth", "has_hand"]
META_COLS = ["person_id", "label", "error_type", "angle", "lighting", "distance", "source"]
ALL_COLS = FRAME_FEATURE_NAMES + FLAG_NAMES

SMOOTHED = ["inhaler_cx", "inhaler_cy", "mouth_cx", "mouth_cy", "mouthpiece_x", "mouthpiece_y",
            "mp_rel_x", "mp_rel_y", "inhaler_size", "hand_inhaler_dist", "hand_mouth_dist",
            "hand_rel_x", "hand_rel_y", "head_yaw", "head_pitch", "head_roll",
            "mouth_open_ratio", "torso_lean", "shoulder_tilt"]

STATS = ("mean", "std", "min", "max", "slope")
AXIS_FEATURES = {"inhaler_angle", "inhaler_angle_rel_head"}
WINDOW_BASE = [n for n in FRAME_FEATURE_NAMES if n not in VISIBILITY_NAMES]
CORE_BASE = ["mp_rel_x", "mp_rel_y", "mp_mouth_dist", "inhaler_angle", "inhaler_angle_rel_head",
             "mouth_open_ratio", "hand_inhaler_dist", "hand_mouth_dist", "head_yaw",
             "head_pitch", "head_roll", "inhaler_size"]
COVERAGE_NAMES = ["cov_inhaler", "cov_face", "cov_mouth", "cov_hand"]
_COL_IDX = {n: i for i, n in enumerate(ALL_COLS)}


def frame_csv_columns() -> List[str]:
    return ["frame_idx", "t"] + ALL_COLS


# --------------------------------------------------------------------------- smoothing
class OneEuroFilter:
    """One-Euro filter (Casiez et al. 2012): low jitter at rest, low lag when moving."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x: Optional[float] = None
        self._dx = 0.0
        self._t: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self._x is None or self._t is None:
            self._x, self._dx, self._t = x, 0.0, t
            return x
        dt = max(t - self._t, 1e-3)
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * (x - self._x) / dt + (1 - a_d) * self._dx
        a = self._alpha(self.min_cutoff + self.beta * abs(self._dx), dt)
        self._x = a * x + (1 - a) * self._x
        self._t = t
        return self._x


# --------------------------------------------------------------------------- per-frame
def _isnan(v: float) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


class FeatureExtractor:
    def __init__(self, cfg: dict):
        s = cfg["features"]["smoothing"]
        v = cfg["features"]["valid_thresholds"]
        self._smooth_cfg = s
        self.v_face, self.v_mouth, self.v_hand = v["face_vis"], v["mouth_vis"], v["hand_vis"]
        self.reset()

    def reset(self) -> None:
        s = self._smooth_cfg
        self._filters = {n: OneEuroFilter(s["min_cutoff"], s["beta"], s["d_cutoff"]) for n in SMOOTHED}
        self._angle_filter = OneEuroFilter(s["min_cutoff"], s["beta"], s["d_cutoff"])
        self._prev_angle: Optional[float] = None

    def update(self, shape: Tuple[int, ...], device, face, hands, t: float) -> Dict[str, float]:
        """device: Detection|None, face: FaceResult, hands: HandsPoseResult, t: seconds."""
        h, w = shape[:2]
        raw = {n: NAN for n in FRAME_FEATURE_NAMES}

        face_ok = bool(face.detected and face.visibility >= self.v_face)
        mouth_ok = bool(face.detected and face.mouth_visibility >= self.v_mouth)
        scale = face.face_scale if (face_ok and face.face_scale > 1.0) else NAN
        scale_ok = not _isnan(scale)
        mc = face.mouth_center if mouth_ok else None

        if face.detected:
            raw["face_vis"] = face.visibility
            raw["mouth_vis"] = face.mouth_visibility
        if face_ok:
            raw["head_yaw"], raw["head_pitch"], raw["head_roll"] = face.yaw, face.pitch, face.roll
        if mouth_ok:
            raw["mouth_cx"], raw["mouth_cy"] = mc[0] / w, mc[1] / h
            raw["mouth_open_ratio"] = face.mouth_open_ratio

        if device is not None:
            cx, cy = device.center
            raw["inhaler_cx"], raw["inhaler_cy"] = cx / w, cy / h
            raw["inhaler_conf"] = device.conf
            if device.angle_deg is not None:
                raw["inhaler_angle"] = device.angle_deg
            if device.axis is not None and scale_ok:
                raw["inhaler_size"] = device.length / scale
            if device.axis is not None and mc is not None and scale_ok:
                a, b = device.axis
                mp = a if dist(a, mc) <= dist(b, mc) else b     # axis end nearest the mouth
                raw["mouthpiece_x"], raw["mouthpiece_y"] = mp[0] / w, mp[1] / h
                raw["mp_rel_x"], raw["mp_rel_y"] = (mp[0] - mc[0]) / scale, (mp[1] - mc[1]) / scale

        hand_ok = hands.holding_idx is not None and not _isnan(hands.hand_conf) \
            and hands.hand_conf >= self.v_hand
        if hands.holding_idx is not None:
            raw["hand_vis"] = hands.hand_conf
        if hand_ok and scale_ok:
            hc = hands.hand_center
            if device is not None:
                raw["hand_inhaler_dist"] = dist(hc, device.center) / scale
            if mc is not None:
                raw["hand_mouth_dist"] = dist(hc, mc) / scale
                raw["hand_rel_x"], raw["hand_rel_y"] = (hc[0] - mc[0]) / scale, (hc[1] - mc[1]) / scale

        raw["pose_vis"] = hands.pose_vis
        raw["torso_lean"], raw["shoulder_tilt"] = hands.torso_lean, hands.shoulder_tilt

        out = dict(raw)
        # ---- temporal smoothing (invalid samples reset the filter: no stale carry-over)
        for n in SMOOTHED:
            v = raw[n]
            if _isnan(v):
                self._filters[n].reset()
            else:
                out[n] = self._filters[n](v, t)
        v = raw["inhaler_angle"]
        if _isnan(v):
            self._angle_filter.reset()
            self._prev_angle = None
        else:
            if self._prev_angle is not None:
                v = unwrap_axis(self._prev_angle, v)
            self._prev_angle = v
            out["inhaler_angle"] = wrap_axis_deg(self._angle_filter(v, t))

        # ---- quantities derived from smoothed values (keeps them mutually consistent)
        if not _isnan(out["mp_rel_x"]) and not _isnan(out["mp_rel_y"]):
            out["mp_mouth_dist"] = math.hypot(out["mp_rel_x"], out["mp_rel_y"])
        if not _isnan(out["inhaler_angle"]) and not _isnan(out["head_roll"]):
            out["inhaler_angle_rel_head"] = axis_diff_deg(out["inhaler_angle"], out["head_roll"])

        out["has_inhaler"] = 1.0 if device is not None else 0.0
        out["has_face"] = 1.0 if face_ok else 0.0
        out["has_mouth"] = 1.0 if mouth_ok else 0.0
        out["has_hand"] = 1.0 if hand_ok else 0.0
        return out


# --------------------------------------------------------------------------- windows
def rows_to_array(rows: Sequence[Dict[str, float]]) -> np.ndarray:
    return np.array([[r.get(c, NAN) for c in ALL_COLS] for r in rows], dtype=float)


def window_feature_names() -> List[str]:
    return [f"{b}_{s}" for b in WINDOW_BASE for s in STATS] + COVERAGE_NAMES


def select_window_features(mode: str = "core", exclude_prefixes: Sequence[str] = ()) -> List[str]:
    """Feature subset used by the classifier. Coverage/visibility are never classifier inputs
    (they belong to the gate and would leak recording conditions into the model)."""
    base = CORE_BASE if mode == "core" else WINDOW_BASE
    stats = ("mean", "std", "min", "max") if mode == "core" else STATS
    names = [f"{b}_{s}" for b in base for s in stats]
    return [n for n in names if not any(n.startswith(p) for p in exclude_prefixes)]


def _stats(v: np.ndarray, axis_feature: bool) -> Tuple[float, float, float, float, float]:
    ok = ~np.isnan(v)
    if ok.sum() < 3:
        return (NAN,) * 5
    idx = np.nonzero(ok)[0].astype(float)
    x = v[ok]
    if axis_feature:  # undirected angles: unwrap in the doubled-angle domain to avoid +-90 jumps
        x = np.degrees(np.unwrap(np.radians(2.0 * x))) / 2.0
    slope = float(np.polyfit(idx, x, 1)[0]) if len(x) >= 3 else NAN
    mean = float(np.mean(x))
    if axis_feature:
        mean = wrap_axis_deg(mean)
    return mean, float(np.std(x)), float(np.min(x)), float(np.max(x)), slope


def window_features_from_array(arr: np.ndarray) -> Dict[str, float]:
    """arr: (n_frames, len(ALL_COLS)) in ALL_COLS order."""
    out: Dict[str, float] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for b in WINDOW_BASE:
            vals = _stats(arr[:, _COL_IDX[b]], b in AXIS_FEATURES)
            for s, v in zip(STATS, vals):
                out[f"{b}_{s}"] = v
    out["cov_inhaler"] = float(np.mean(arr[:, _COL_IDX["has_inhaler"]]))
    out["cov_face"] = float(np.mean(arr[:, _COL_IDX["has_face"]]))
    out["cov_mouth"] = float(np.mean(arr[:, _COL_IDX["has_mouth"]]))
    out["cov_hand"] = float(np.mean(arr[:, _COL_IDX["has_hand"]]))
    return out


def window_features(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    return window_features_from_array(rows_to_array(rows))


def gate_check(wf: Dict[str, float], cfg: dict) -> Tuple[bool, List[str]]:
    """Visibility gate: refuse to classify when the required objects are not reliably visible."""
    need = cfg["features"]["min_valid_fraction"]
    reasons = []
    for key, cov in (("inhaler", "cov_inhaler"), ("face", "cov_face"), ("mouth", "cov_mouth")):
        if wf.get(cov, 0.0) < need[key]:
            reasons.append(f"{key} reliably visible in only {wf.get(cov, 0.0):.0%} of recent frames")
    return (not reasons), reasons


class SlidingWindow:
    def __init__(self, size: int):
        self.size = size
        self._rows: Deque[np.ndarray] = deque(maxlen=size)

    def push(self, feats: Dict[str, float]) -> None:
        self._rows.append(np.array([feats.get(c, NAN) for c in ALL_COLS], dtype=float))

    def ready(self) -> bool:
        return len(self._rows) == self.size

    def clear(self) -> None:
        self._rows.clear()

    def features(self) -> Dict[str, float]:
        return window_features_from_array(np.vstack(self._rows))
