#!/usr/bin/env python
"""Run the perception pipeline over labelled videos and write per-frame feature CSVs.

  python training/extract_features.py --annotations data/annotations.csv --out data/features

annotations.csv columns (see data/annotations_template.csv):
  person_id, video_path, label, error_type, angle, lighting, distance, start_frame, end_frame
`label` is correct | wrong_mouth | wrong_inhaler. start_frame/end_frame may be blank (= whole video).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import pandas as pd

from src.classifier import UNABLE, normalise_label
from src.config import load_config, resolve
from src.pipeline import InhalerPipeline
from src.recorder import FeatureRecorder

REQUIRED = ["person_id", "video_path", "label"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", default="data/annotations.csv")
    ap.add_argument("--out", default="data/features")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--frame-step", type=int, default=1, help="process every Nth frame (keeps timing via t)")
    args = ap.parse_args(argv)

    ann = pd.read_csv(resolve(args.annotations))
    miss = [c for c in REQUIRED if c not in ann.columns]
    if miss:
        print(f"annotations missing columns: {miss}", file=sys.stderr)
        return 2
    cfg = load_config(args.config)
    pipe = InhalerPipeline(cfg, load_classifier=False, weights=args.weights)
    out_dir = resolve(args.out)

    for i, row in ann.iterrows():
        label = normalise_label(row["label"])
        if label is None or label == UNABLE:
            print(f"row {i}: skipping unknown label {row['label']!r}")
            continue
        video = resolve(row["video_path"])
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"row {i}: cannot open {video}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        start = int(row["start_frame"]) if "start_frame" in row and pd.notna(row["start_frame"]) else 0
        end = int(row["end_frame"]) if "end_frame" in row and pd.notna(row["end_frame"]) else None
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        meta = {k: ("" if pd.isna(row.get(k, "")) else str(row.get(k, "")))
                for k in ("person_id", "error_type", "angle", "lighting", "distance")}
        meta.update(label=label, source=video.name)
        dest = out_dir / f"{meta['person_id']}_{label.replace(' ', '')}_{video.stem}_{start}.csv"
        rec = FeatureRecorder(dest, meta)
        pipe.reset()
        idx = start
        while True:
            if end is not None and idx > end:
                break
            ok, frame = cap.read()
            if not ok:
                break
            if (idx - start) % args.frame_step == 0:
                out = pipe.process(frame, t=idx / fps)
                rec.write(idx, idx / fps, out.features)
            idx += 1
        rec.close()
        cap.release()
        print(f"[{i + 1}/{len(ann)}] {video.name}: {rec.n_rows} rows -> {dest.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
