"""End-to-end per-frame pipeline: detect -> face/mouth -> hands/pose -> features -> classify."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from .classifier import LabelStabiliser, Prediction, TechniqueClassifier
from .detector import Detection, InhalerDetector, select_device
from .face_mouth import FaceMouthDetector, FaceResult
from .features import FeatureExtractor, SlidingWindow
from .hands_pose import HandPoseDetector, HandsPoseResult


@dataclass
class PipelineOutput:
    frame_idx: int
    detections: List[Detection]
    device: Optional[Detection]
    face: FaceResult
    hands: HandsPoseResult
    features: Dict[str, float]
    window_features: Optional[Dict[str, float]]
    prediction: Optional[Prediction]
    timings: Dict[str, float] = field(default_factory=dict)   # milliseconds per stage


class InhalerPipeline:
    def __init__(self, cfg: dict, load_classifier: bool = True, weights: Optional[str] = None,
                 load_detector: bool = True):
        if weights:
            cfg["detector"]["weights"] = weights
        self.cfg = cfg
        self.detector = InhalerDetector(cfg) if load_detector else None
        self.face = FaceMouthDetector(cfg)
        self.hp = HandPoseDetector(cfg)
        self.extractor = FeatureExtractor(cfg)
        self.clf = TechniqueClassifier(cfg) if load_classifier else None
        win = (self.clf.window_size if self.clf and self.clf.window_size
               else cfg["features"]["window_size"])
        self.window = SlidingWindow(win)
        self.stabiliser = LabelStabiliser(cfg["classifier"].get("stabiliser_frames", 5))
        self._idx = 0
        p = cfg.get("pipeline", {})
        self._enhance = bool(p.get("enhance_dark", False))
        self._dark_thr = float(p.get("dark_mean_threshold", 70))

    def reset(self) -> None:
        """Call between unrelated videos / sessions."""
        if self.detector is not None:
            self.detector.reset()
        self.face.reset()
        self.hp.reset()
        self.extractor.reset()
        self.window.clear()
        self.stabiliser.reset()
        self._idx = 0

    def _to_rgb(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self._enhance:
            mean = float(rgb[::8, ::8].mean())
            if mean < self._dark_thr:
                gamma = float(np.clip(np.log(0.5) / np.log(max(mean, 1.0) / 255.0), 0.4, 1.0))
                lut = (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)
                rgb = cv2.LUT(rgb, lut)
        return rgb

    def process(self, frame: np.ndarray, t: Optional[float] = None) -> PipelineOutput:
        t = time.perf_counter() if t is None else t
        tm: Dict[str, float] = {}
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        dets = self.detector.detect(frame) if self.detector is not None else []
        device = select_device(dets, self.cfg)
        tm["yolo"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        rgb = self._to_rgb(frame)
        tm["preprocess"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        face = self.face.process(rgb)
        tm["face"] = (time.perf_counter() - t0) * 1000

        hands = self.hp.process(rgb, device.box if device else None, self._idx)
        tm["hands"], tm["pose"] = self.hp.last_timings["hands"], self.hp.last_timings["pose"]

        t0 = time.perf_counter()
        feats = self.extractor.update(frame.shape, device, face, hands, t)
        tm["features"] = (time.perf_counter() - t0) * 1000

        wf = pred = None
        t0 = time.perf_counter()
        if self.clf is not None:
            self.window.push(feats)
            ready = self.window.ready()
            wf = self.window.features() if ready else None
            pred = self.clf.predict(wf, ready)
            pred.display_label = self.stabiliser.update(pred.label)
        tm["classifier"] = (time.perf_counter() - t0) * 1000

        tm["total"] = (time.perf_counter() - t_start) * 1000
        out = PipelineOutput(self._idx, dets, device, face, hands, feats, wf, pred, tm)
        self._idx += 1
        return out
