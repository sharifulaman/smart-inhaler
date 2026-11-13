#!/usr/bin/env python
"""Fine-tune YOLO11 for inhaler / spacer detection and export models/best.pt.

  python training/train_yolo.py --data data/inhaler.yaml --model yolo11s.pt --epochs 100
  python training/train_yolo.py --data data/inhaler_obb.yaml --model yolo11s-obb.pt   # oriented boxes

Use an OBB model (or keypoints) if you want the inhaler angle measured from the object's own
geometry; a plain detect model only gives a horizontal/vertical axis estimate.

Tip: combine the Roboflow data with your own webcam frames (hand-occluded, blurred, dim/bright)
and run training/augment_dataset.py first to add synthetic hand occlusion and motion blur.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ROOT, resolve


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/inhaler.yaml")
    ap.add_argument("--model", default="yolo11s.pt", help="yolo11n.pt | yolo11s.pt | yolo11s-obb.pt ...")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--name", default="inhaler_yolo11")
    ap.add_argument("--out", default="models/best.pt")
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(resolve(args.data)), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, patience=args.patience, project=str(ROOT / "runs"), name=args.name,
        exist_ok=True,
        # photometric: lighting variation
        hsv_h=0.02, hsv_s=0.7, hsv_v=0.5,
        # geometric: camera angle / distance / small rotations
        degrees=10, translate=0.12, scale=0.5, shear=2.0, perspective=0.0005, fliplr=0.5,
        # composition
        mosaic=1.0, mixup=0.1, close_mosaic=10,
    )
    best = ROOT / "runs" / args.name / "weights" / "best.pt"
    if not best.exists():
        print(f"Training finished but {best} not found", file=sys.stderr)
        return 2
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out)
    print(f"Copied {best} -> {out}")

    metrics = YOLO(str(out)).val(data=str(resolve(args.data)), imgsz=args.imgsz, device=args.device)
    box = metrics.box
    print(f"\nValidation: mAP50 {box.map50:.3f} | mAP50-95 {box.map:.3f} | "
          f"precision {box.mp:.3f} | recall {box.mr:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
