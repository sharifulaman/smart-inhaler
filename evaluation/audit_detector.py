#!/usr/bin/env python
"""Step 1 of the project: audit the EXISTING best.pt before rebuilding anything.

  python evaluation/audit_detector.py --weights models/best.pt --source clip.mp4 [--data data/inhaler.yaml]

Prints model task/classes, optional validation mAP, and on a video: per-class detection coverage,
longest missed streak (hand occlusion shows up here), mean confidence and track-ID switches.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from src.config import resolve


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="models/best.pt")
    ap.add_argument("--source", default=None, help="video file or camera index to probe")
    ap.add_argument("--data", default=None, help="dataset yaml to compute validation mAP")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-frames", type=int, default=600)
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    w = resolve(args.weights)
    if not w.exists():
        print(f"weights not found: {w}", file=sys.stderr)
        return 2
    model = YOLO(str(w))
    print(f"Model: {w.name} | task: {model.task} | classes: {model.names}")

    if args.data:
        m = model.val(data=str(resolve(args.data)), imgsz=args.imgsz, verbose=False)
        print(f"Validation: mAP50 {m.box.map50:.3f} | mAP50-95 {m.box.map:.3f} | "
              f"P {m.box.mp:.3f} | R {m.box.mr:.3f}")
        for i, name in model.names.items():
            if i < len(m.box.ap50):
                print(f"  {name}: AP50 {m.box.ap50[i]:.3f}")

    if args.source is None:
        return 0
    src = int(args.source) if str(args.source).isdigit() else str(resolve(args.source))
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"cannot open {src}", file=sys.stderr)
        return 3
    names = {i: str(n).lower() for i, n in model.names.items()}
    present = {n: [] for n in names.values()}
    confs = {n: [] for n in names.values()}
    ids = {n: [] for n in names.values()}
    frames = 0
    while frames < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        r = model.track(frame, persist=True, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        src_obj = r.obb if getattr(r, "obb", None) is not None else r.boxes
        best = {}
        if src_obj is not None and len(src_obj):
            cls = src_obj.cls.cpu().numpy().astype(int)
            cf = src_obj.conf.cpu().numpy()
            tid = src_obj.id.cpu().numpy().astype(int) if src_obj.id is not None else [None] * len(cls)
            for c, p, t in zip(cls, cf, tid):
                n = names[int(c)]
                if n not in best or p > best[n][0]:
                    best[n] = (float(p), t)
        for n in present:
            present[n].append(n in best)
            if n in best:
                confs[n].append(best[n][0])
                ids[n].append(best[n][1])
    cap.release()

    print(f"\nProbed {frames} frames at conf>={args.conf}")
    for n, p in present.items():
        p = np.array(p, dtype=bool)
        longest = cur = 0
        for v in p:
            cur = 0 if v else cur + 1
            longest = max(longest, cur)
        seq = [i for i in ids[n] if i is not None]
        switches = int(np.sum(np.diff(seq) != 0)) if len(seq) > 1 else 0
        mc = float(np.mean(confs[n])) if confs[n] else float("nan")
        print(f"  {n:<8} coverage {p.mean():6.1%} | missed frames {int((~p).sum()):4d} | "
              f"longest miss streak {longest:3d} | mean conf {mc:.2f} | track-ID switches {switches}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
