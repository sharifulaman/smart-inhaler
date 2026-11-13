"""Live overlay: detections, landmarks, status panel, feedback and warnings."""
from __future__ import annotations

import math
from typing import Dict, Optional

import cv2
import numpy as np

from .classifier import CORRECT, UNABLE, WRONG_INHALER, WRONG_MOUTH
from .pipeline import PipelineOutput

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
                    (18, 19), (19, 20), (0, 17)]
GREEN, RED, AMBER = (110, 200, 90), (70, 70, 240), (0, 180, 255)
WHITE, CYAN, MAGENTA, GREY, DARK = (245, 245, 245), (255, 210, 60), (220, 90, 220), (170, 170, 170), (25, 25, 25)
LABEL_COLOURS = {CORRECT: GREEN, WRONG_MOUTH: RED, WRONG_INHALER: RED, UNABLE: AMBER}
FEEDBACK = {
    CORRECT: "Visible positioning looks correct.",
    WRONG_MOUTH: "Bring your mouth to the mouthpiece and close your lips around it.",
    WRONG_INHALER: "Hold the inhaler upright with the mouthpiece pointing into your mouth.",
    UNABLE: "Face the camera so your face, mouth and inhaler are fully visible.",
}
DET_COLOURS = {"inhaler": CYAN, "spacer": MAGENTA}


def _nan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _text(img, s, org, scale, colour=WHITE, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, DARK, thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def _pt(p):
    return int(round(p[0])), int(round(p[1]))


def draw(frame: np.ndarray, out: PipelineOutput, hud: Optional[Dict] = None) -> np.ndarray:
    hud = hud or {}
    img = frame.copy()
    s = max(0.5, min(img.shape[:2]) / 720.0)
    _draw_detections(img, out, s)
    _draw_face(img, out, s)
    _draw_hands(img, out, s)
    _draw_link(img, out, s)
    _draw_hud(img, out, hud, s)
    return img


def _draw_detections(img, out, s):
    for d in out.detections:
        col = DET_COLOURS.get(d.cls_name, WHITE)
        if d.held:
            col = GREY
        if d.polygon is not None:
            cv2.polylines(img, [d.polygon.astype(np.int32).reshape(-1, 1, 2)], True, col, 2, cv2.LINE_AA)
        else:
            cv2.rectangle(img, _pt(d.box[:2]), _pt(d.box[2:]), col, 2, cv2.LINE_AA)
        if d.axis is not None:
            cv2.line(img, _pt(d.axis[0]), _pt(d.axis[1]), col, 1, cv2.LINE_AA)
        tag = f"{d.cls_name} {d.conf:.2f}" + (f" #{d.track_id}" if d.track_id is not None else "") \
            + (" (held)" if d.held else "")
        _text(img, tag, (int(d.box[0]), max(int(d.box[1]) - 6, 14)), 0.5 * s, col)


def _draw_face(img, out, s):
    f = out.face
    if not f.detected:
        return
    if f.mouth_outer is not None:
        cv2.polylines(img, [f.mouth_outer.astype(np.int32).reshape(-1, 1, 2)], True, GREEN, 1, cv2.LINE_AA)
    if f.mouth_corners is not None:
        for c in f.mouth_corners:
            cv2.circle(img, _pt(c), max(2, int(3 * s)), AMBER, -1, cv2.LINE_AA)
    if f.mouth_center is not None:
        cv2.drawMarker(img, _pt(f.mouth_center), GREEN, cv2.MARKER_CROSS, int(14 * s), 2, cv2.LINE_AA)
    if f.landmarks is not None:
        for i in (33, 263, 1, 152):   # eye corners, nose tip, chin
            cv2.circle(img, _pt(f.landmarks[i]), max(2, int(2 * s)), WHITE, -1, cv2.LINE_AA)


def _draw_hands(img, out, s):
    h = out.hands
    for i, pts in enumerate(h.hands):
        col = GREEN if i == h.holding_idx else GREY
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, _pt(pts[a]), _pt(pts[b]), col, 1, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, _pt(p), max(2, int(2 * s)), col, -1, cv2.LINE_AA)
    if h.pose_points is not None:
        for a, b in ((11, 12), (0, 11), (0, 12)):
            cv2.line(img, _pt(h.pose_points[a]), _pt(h.pose_points[b]), CYAN, 1, cv2.LINE_AA)


def _draw_link(img, out, s):
    f = out.features
    if _nan(f.get("mp_mouth_dist")) or out.device is None or out.face.mouth_center is None:
        return
    h, w = img.shape[:2]
    mp = (f["mouthpiece_x"] * w, f["mouthpiece_y"] * h)
    cv2.circle(img, _pt(mp), max(4, int(6 * s)), RED, 2, cv2.LINE_AA)
    cv2.line(img, _pt(mp), _pt(out.face.mouth_center), AMBER, 2, cv2.LINE_AA)


def _draw_hud(img, out, hud, s):
    f, dev, face = out.features, out.device, out.face
    pred = out.prediction
    rows = []
    rows.append(("Inhaler", (f"{dev.cls_name} {dev.conf:.2f}" + (" (held)" if dev.held else "")) if dev else "not found",
                 GREEN if dev and not dev.held else (AMBER if dev else RED)))
    rows.append(("Face", f"ok {face.visibility:.2f}" if face.detected else "not found",
                 GREEN if f.get("has_face") else (AMBER if face.detected else RED)))
    rows.append(("Mouth", f"ok {face.mouth_visibility:.2f}" if f.get("has_mouth") else
                 ("low reliability" if face.detected else "not found"),
                 GREEN if f.get("has_mouth") else RED))
    if f.get("has_hand"):
        rows.append(("Hand", "holding", GREEN))
    elif out.hands.hands:   # hand landmarks found, but no inhaler box to tie them to
        rows.append(("Hand", f"detected ({len(out.hands.hands)}), not holding inhaler", AMBER))
    else:
        rows.append(("Hand", "not found", AMBER))
    d, a = f.get("mp_mouth_dist"), f.get("inhaler_angle")
    rows.append(("Mouthpiece-mouth", "--" if _nan(d) else f"{d:.2f} face-widths", WHITE))
    rows.append(("Inhaler angle", "--" if _nan(a) else f"{a:+.0f} deg ({dev.angle_source})", WHITE))
    vis = [v for v in (f.get("face_vis"), f.get("mouth_vis"), f.get("inhaler_conf")) if not _nan(v)]
    rows.append(("Visibility (min)", f"{min(vis):.2f}" if vis else "--", WHITE))
    label = (pred.display_label or pred.label) if pred else "n/a"
    conf = f" ({pred.confidence:.0%})" if pred and pred.confidence > 0 else ""
    rows.append(("Technique", label + conf, LABEL_COLOURS.get(label, WHITE)))

    fs, lh = 0.55 * s, int(26 * s)
    pw, ph = int(380 * s), lh * (len(rows) + 2) + int(8 * s)
    overlay = img.copy()
    cv2.rectangle(overlay, (8, 8), (8 + pw, 8 + ph), DARK, -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    y = 8 + lh
    for name, val, col in rows:
        _text(img, f"{name}:", (16, y), fs, GREY)
        _text(img, val, (16 + int(165 * s), y), fs, col)
        y += lh
    fps, lat = hud.get("fps"), hud.get("latency_ms")
    perf = []
    if fps:
        perf.append(f"{fps:.1f} FPS")
    if lat:
        perf.append(f"{lat:.0f} ms latency")
    if pred is not None:
        perf.append(f"model: {pred.source}")
    if hud.get("recording"):
        perf.append("REC")
    _text(img, "  |  ".join(perf), (16, y), 0.5 * s, GREY)

    h, w = img.shape[:2]
    warn = None
    if hud.get("cam_state") not in (None, "ok", "ended"):
        warn = f"Camera signal {hud['cam_state']} - check connection"
    elif pred is not None and pred.label == UNABLE and pred.source in ("gate", "model") and pred.reasons:
        warn = "Insufficient visibility: " + pred.reasons[0]
    if warn:
        cv2.rectangle(img, (0, h - int(78 * s)), (w, h - int(50 * s)), (0, 120, 220), -1)
        _text(img, warn, (12, h - int(57 * s)), 0.6 * s, WHITE, 2)
    if pred is not None and label in FEEDBACK:
        _text(img, FEEDBACK[label], (12, h - int(30 * s)), 0.6 * s, LABEL_COLOURS.get(label, WHITE), 1)
    _text(img, "Research prototype - visible positioning only; cannot assess airflow or dose delivery. Not a medical device.",
          (12, h - int(8 * s)), 0.4 * s, GREY)
