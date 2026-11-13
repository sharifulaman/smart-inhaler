import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classifier import CORRECT, UNABLE, WRONG_INHALER, WRONG_MOUTH, LabelStabiliser, RuleBasedClassifier
from src.config import load_config
from src.detector import Detection
from src.face_mouth import FaceResult
from src.features import (ALL_COLS, FeatureExtractor, OneEuroFilter, SlidingWindow, gate_check,
                          select_window_features, window_feature_names, window_features)
from src.geometry import (axis_angle_deg, axis_diff_deg, box_long_axis, obb_long_axis, unwrap_axis,
                          wrap_axis_deg)
from src.hands_pose import HandsPoseResult

CFG = load_config()


def test_obb_long_axis_horizontal_and_vertical():
    poly = np.array([[0, 0], [100, 0], [100, 20], [0, 20]], float)
    a, b, length = obb_long_axis(poly)
    assert length == 100 and abs(axis_angle_deg(a, b)) < 1e-6
    poly_v = np.array([[0, 0], [20, 0], [20, 100], [0, 100]], float)
    a, b, length = obb_long_axis(poly_v)
    assert length == 100 and abs(abs(axis_angle_deg(a, b)) - 90) < 1e-6


def test_obb_long_axis_rotated_45():
    c, s = math.cos(math.radians(45)), math.sin(math.radians(45))
    base = np.array([[-50, -10], [50, -10], [50, 10], [-50, 10]], float)
    poly = base @ np.array([[c, -s], [s, c]]).T
    a, b, length = obb_long_axis(poly)
    assert abs(length - 100) < 1e-6 and abs(axis_angle_deg(a, b) - 45) < 1e-6


def test_axis_wrapping():
    assert wrap_axis_deg(95) == -85 and wrap_axis_deg(-90) == -90
    assert abs(axis_diff_deg(88, -88) - (-4)) < 1e-9
    assert abs(unwrap_axis(89, -89) - 91) < 1e-9
    assert box_long_axis((0, 0, 10, 50))[2] == 50


def test_one_euro_reduces_jitter():
    rng = np.random.default_rng(0)
    noisy = 1.0 + rng.normal(0, 0.05, 200)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    sm = np.array([f(float(v), i / 30) for i, v in enumerate(noisy)])
    assert sm[30:].std() < 0.5 * noisy[30:].std()


def _det(a, b, conf=0.9):
    box = (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]) + 1, max(a[1], b[1]) + 10)
    return Detection("inhaler", conf, box, axis=(a, b), length=math.dist(a, b),
                     angle_deg=axis_angle_deg(a, b), angle_source="obb")


def _face(mc=(320, 240), scale=100.0):
    return FaceResult(detected=True, mouth_center=mc, mouth_open_ratio=0.2, face_scale=scale,
                      yaw=0.0, pitch=0.0, roll=0.0, visibility=0.95, mouth_visibility=0.9)


def test_extractor_distance_and_angle_are_normalised():
    ex = FeatureExtractor(CFG)
    # mouthpiece end at (400,240): 80 px from the mouth, face scale 100 px -> 0.8 face-widths
    f = ex.update((480, 640), _det((400, 240), (500, 240)), _face(), HandsPoseResult(), t=0.0)
    assert abs(f["mp_mouth_dist"] - 0.8) < 1e-6
    assert abs(f["inhaler_angle"]) < 1e-6 and abs(f["inhaler_angle_rel_head"]) < 1e-6
    assert f["has_inhaler"] == f["has_face"] == f["has_mouth"] == 1.0 and f["has_hand"] == 0.0
    # same scene at 2x resolution & 2x distances -> identical normalised distance
    ex2 = FeatureExtractor(CFG)
    f2 = ex2.update((960, 1280), _det((800, 480), (1000, 480)), _face((640, 480), 200.0), HandsPoseResult(), 0.0)
    assert abs(f2["mp_mouth_dist"] - f["mp_mouth_dist"]) < 1e-6


def test_extractor_masks_unreliable_mouth():
    ex = FeatureExtractor(CFG)
    face = _face()
    face.mouth_visibility = 0.1
    f = ex.update((480, 640), _det((400, 240), (500, 240)), face, HandsPoseResult(), 0.0)
    assert f["has_mouth"] == 0.0 and math.isnan(f["mp_mouth_dist"]) and math.isnan(f["mouth_open_ratio"])


def _frames(n, **over):
    ex = FeatureExtractor(CFG)
    rows = []
    for i in range(n):
        rows.append(ex.update((480, 640), _det((400, 240), (500, 240)), _face(), HandsPoseResult(), i / 30))
    for r in rows:
        r.update(over)
    return rows


def test_window_features_and_gate():
    rows = _frames(20)
    wf = window_features(rows)
    assert set(window_feature_names()) == set(wf)
    assert abs(wf["mp_mouth_dist_mean"] - 0.8) < 1e-3 and wf["cov_inhaler"] == 1.0
    ok, _ = gate_check(wf, CFG)
    assert ok
    bad = _frames(20, has_inhaler=0.0)
    ok, reasons = gate_check(window_features(bad), CFG)
    assert not ok and "inhaler" in reasons[0]


def test_axis_window_stats_do_not_jump_at_wrap():
    rows = _frames(20)
    for i, r in enumerate(rows):
        r["inhaler_angle"] = 89.0 if i % 2 else -89.0   # same physical axis +-2 deg around vertical
    wf = window_features(rows)
    assert wf["inhaler_angle_std"] < 5.0


def test_select_features_never_contains_coverage_or_visibility():
    for mode in ("core", "all"):
        names = select_window_features(mode)
        assert not any(n.startswith(("cov_", "face_vis", "mouth_vis", "inhaler_conf")) for n in names)
    assert all(not n.startswith("hand_") for n in select_window_features("core", ["hand_"]))


def test_rule_based_and_stabiliser():
    rb = RuleBasedClassifier(CFG)
    base = dict(mp_mouth_dist_mean=0.4, inhaler_angle_rel_head_mean=5.0, mouth_open_ratio_mean=0.2)
    assert rb.predict(base).label == CORRECT
    assert rb.predict({**base, "mp_mouth_dist_mean": 1.5}).label == WRONG_MOUTH
    assert rb.predict({**base, "inhaler_angle_rel_head_mean": 70.0}).label == WRONG_INHALER
    assert rb.predict({**base, "mp_mouth_dist_mean": float("nan")}).label == UNABLE
    st = LabelStabiliser(3)
    seq = [st.update(x) for x in [CORRECT, CORRECT, CORRECT, WRONG_MOUTH, CORRECT, WRONG_MOUTH, WRONG_MOUTH, WRONG_MOUTH]]
    assert seq[2] == CORRECT and seq[3] == CORRECT and seq[-1] == WRONG_MOUTH


def test_sliding_window_ready():
    w = SlidingWindow(5)
    for r in _frames(4):
        w.push(r)
    assert not w.ready()
    w.push(_frames(1)[0])
    assert w.ready() and len(w.features()) == len(window_feature_names())
