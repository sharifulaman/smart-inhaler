"""Synthetic per-frame feature CSVs for smoke-testing the training/evaluation code (not real data)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from src.features import ALL_COLS, FRAME_FEATURE_NAMES, META_COLS
from src.recorder import FeatureRecorder

CLASS_PARAMS = {   # mouthpiece distance (face-widths), angle vs head (deg), mouth-open ratio
    "Correct": (0.35, 5.0, 0.25),
    "Wrong Mouth Position": (0.95, 5.0, 0.03),
    "Wrong Inhaler Position": (0.45, 55.0, 0.22),
}


def make_dataset(out_dir, n_subjects: int = 12, frames: int = 120, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in range(n_subjects):
        off = rng.normal(0, 0.06, 3)
        for label, (d0, a0, o0) in CLASS_PARAMS.items():
            rec = FeatureRecorder(out_dir / f"P{s:02d}_{label.replace(' ', '')}.csv",
                                  dict(person_id=f"P{s:02d}", label=label, angle="front",
                                       lighting="normal", distance="medium", source="synthetic"))
            for i in range(frames):
                f = {n: float(rng.normal(0.5, 0.05)) for n in FRAME_FEATURE_NAMES}
                d = max(0.05, d0 + off[0] + rng.normal(0, 0.05))
                th = rng.uniform(0, 2 * math.pi)
                f.update(mp_mouth_dist=d, mp_rel_x=d * math.cos(th), mp_rel_y=d * math.sin(th),
                         inhaler_angle_rel_head=a0 + 8 * off[1] + rng.normal(0, 4),
                         mouth_open_ratio=max(0.0, o0 + off[2] + rng.normal(0, 0.02)),
                         face_vis=0.95, mouth_vis=0.9, inhaler_conf=0.8, hand_vis=0.9, pose_vis=0.9)
                f.update(has_inhaler=1.0, has_face=1.0, has_mouth=1.0, has_hand=1.0)
                if rng.random() < 0.05:   # occasional dropout
                    f["hand_inhaler_dist"] = float("nan")
                rec.write(i, i / 30.0, f)
            rec.close()
    return out_dir
