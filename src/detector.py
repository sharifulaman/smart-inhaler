"""YOLO11 inhaler / spacer detection with tracking and short-term hold.

Works with either a standard detect model (axis-aligned boxes) or an OBB model
(oriented boxes). The inhaler angle is computed from the object's own geometry:
the long axis of the oriented box. With a plain detect model only a quantised
horizontal/vertical axis is available, flagged as angle_source="box".
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import resolve
from .geometry import Point, axis_angle_deg, box_long_axis, obb_long_axis


@dataclass
class Detection:
    cls_name: str
    conf: float
    box: Tuple[float, float, float, float]
    track_id: Optional[int] = None
    polygon: Optional[np.ndarray] = None          # (4, 2) for OBB models
    axis: Optional[Tuple[Point, Point]] = None    # long-axis end points
    length: float = 0.0                           # long-axis length (px)
    angle_deg: Optional[float] = None             # undirected axis angle in [-90, 90)
    angle_source: str = "box"                     # "obb" (reliable) or "box" (quantised)
    held: bool = False                            # True when carried over from earlier frames

    @property
    def center(self) -> Point:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class InhalerDetector:
    def __init__(self, cfg: dict):
        from ultralytics import YOLO  # heavy import kept local

        d = cfg["detector"]
        weights = resolve(d["weights"])
        zs = d.get("zero_shot", {})
        self.zero_shot = False
        conf = d["conf"]
        if weights.exists():
            self.model = YOLO(str(weights))
        elif zs.get("enabled", False):
            self.model = self._load_zero_shot(zs)
            self.zero_shot = True
            conf = zs.get("conf", 0.05)
        else:
            raise FileNotFoundError(
                f"Detector weights not found: {weights}\n"
                "Place your trained model at models/best.pt (see README, step 2) or train one "
                "with training/train_yolo.py.")
        self.task = getattr(self.model, "task", "detect")
        self.kw = dict(imgsz=d["imgsz"], conf=conf, iou=d["iou"], device=d.get("device"),
                       half=d.get("half", False), verbose=False)
        self.tracker = d.get("tracker", "bytetrack.yaml")
        self.hold = int(d.get("hold_frames", 0))
        self._last: Dict[str, Detection] = {}
        self._age: Dict[str, int] = {}

    @staticmethod
    def _load_zero_shot(zs: dict):
        """Open-vocabulary YOLO-World: finds 'inhaler' from text prompts, no training data."""
        import shutil
        from pathlib import Path

        from ultralytics import YOLOWorld

        name = zs.get("model", "yolov8s-worldv2.pt")
        local = resolve("models/" + name)
        model = YOLOWorld(str(local if local.exists() else name))   # bare name -> auto-download
        if not local.exists() and Path(name).exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(name, local)                                # keep weights under models/
        model.set_classes(list(zs.get("prompts", ["inhaler"])))
        return model

    def reset(self) -> None:
        """Forget tracker state (call between unrelated videos)."""
        self._last.clear()
        self._age.clear()
        if hasattr(self.model, "predictor"):
            self.model.predictor = None  # forces a fresh tracker on the next call

    def detect(self, frame: np.ndarray) -> List[Detection]:
        r = self.model.track(frame, persist=True, tracker=self.tracker, **self.kw)[0]
        return self._apply_hold(self._parse(r))

    # ------------------------------------------------------------------ internals
    def _parse(self, r) -> List[Detection]:
        is_obb = getattr(r, "obb", None) is not None
        src = r.obb if is_obb else r.boxes
        if src is None or len(src) == 0:
            return []
        n = len(src)
        conf = src.conf.cpu().numpy()
        cls = src.cls.cpu().numpy().astype(int)
        xyxy = src.xyxy.cpu().numpy()
        ids = src.id.cpu().numpy().astype(int) if getattr(src, "id", None) is not None else [None] * n
        polys = src.xyxyxyxy.cpu().numpy() if is_obb else [None] * n
        names = r.names
        out: List[Detection] = []
        for i in range(n):
            box = tuple(float(v) for v in xyxy[i])
            if is_obb:
                a, b, length = obb_long_axis(polys[i])
                source = "obb"
            else:
                a, b, length = box_long_axis(box)
                source = "box"
            out.append(Detection(
                cls_name="inhaler" if self.zero_shot else str(names[int(cls[i])]).lower(),
                conf=float(conf[i]), box=box,
                track_id=None if ids[i] is None else int(ids[i]),
                polygon=polys[i], axis=(a, b), length=length,
                angle_deg=axis_angle_deg(a, b), angle_source=source))
        return out

    def _apply_hold(self, dets: List[Detection]) -> List[Detection]:
        out = list(dets)
        seen = set()
        for d in dets:
            seen.add(d.cls_name)
            self._last[d.cls_name] = d
            self._age[d.cls_name] = 0
        for name in list(self._last):
            if name in seen:
                continue
            self._age[name] += 1
            if self._age[name] <= self.hold:
                d = self._last[name]
                out.append(replace(d, held=True, conf=d.conf * (0.9 ** self._age[name])))
            else:
                del self._last[name], self._age[name]
        return out


def select_device(dets: List[Detection], cfg: dict) -> Optional[Detection]:
    """Pick the object whose mouthpiece is evaluated: the spacer when one is visible
    (and prefer_spacer is set), otherwise the inhaler."""
    d = cfg["detector"]
    inh = [x for x in dets if x.cls_name in d["inhaler_classes"]]
    spc = [x for x in dets if x.cls_name in d["spacer_classes"]]
    pool = spc if (d.get("prefer_spacer", True) and spc) else (inh or spc)
    return max(pool, key=lambda x: x.conf) if pool else None
