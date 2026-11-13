"""Per-frame feature CSV writer (dataset recording and offline extraction share this schema)."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict

from .features import META_COLS, frame_csv_columns


class FeatureRecorder:
    def __init__(self, path, meta: Dict[str, str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._cols = frame_csv_columns()
        self._meta = [meta.get(k, "") for k in META_COLS]
        self._w.writerow(META_COLS + self._cols)
        self.n_rows = 0

    def write(self, frame_idx: int, t: float, feats: Dict[str, float]) -> None:
        row = [frame_idx, f"{t:.4f}"]
        for c in self._cols[2:]:
            v = feats.get(c, float("nan"))
            row.append("" if (v is None or math.isnan(v)) else f"{v:.5f}")
        self._w.writerow(self._meta + row)
        self.n_rows += 1
        if self.n_rows % 100 == 0:
            self._f.flush()

    def close(self) -> None:
        self._f.close()


SCAN_COLS = ["frame", "time_s", "inhaler_found", "inhaler_conf", "inhaler_x1", "inhaler_y1",
             "inhaler_x2", "inhaler_y2", "face_found", "mouth_found", "hand_holding",
             "mouthpiece_mouth_dist", "inhaler_angle_deg", "technique", "technique_conf", "source"]


class ScanLogger:
    """One row per processed frame with the human-readable scan result (live app, `data/scans/`)."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow(SCAN_COLS)
        self.n_rows = 0

    @staticmethod
    def _num(v, nd=3):
        return "" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"

    def write(self, out, t: float) -> None:
        d, f, p = out.device, out.features, out.prediction
        box = [self._num(v, 1) for v in d.box] if d else [""] * 4
        self._w.writerow([
            out.frame_idx, f"{t:.3f}", int(d is not None), self._num(d.conf if d else None),
            *box, int(bool(out.face.detected)), int(bool(f.get("has_mouth"))),
            int(bool(f.get("has_hand"))), self._num(f.get("mp_mouth_dist")),
            self._num(f.get("inhaler_angle"), 1),
            (p.display_label or p.label) if p else "", self._num(p.confidence if p else None),
            p.source if p else ""])
        self.n_rows += 1
        if self.n_rows % 50 == 0:
            self._f.flush()

    def close(self) -> None:
        self._f.close()
