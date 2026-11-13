#!/usr/bin/env python
"""Offline augmentation for the inhaler detector: synthetic hand occlusion, motion blur, lighting.

Hand occlusion is the main YOLO failure mode for this project, and Ultralytics' built-in
augmentations do not simulate it for detection. This script reads a YOLO-format split
(images/ + labels/, detect or OBB labels) and writes augmented copies next to a new folder.
Labels are copied unchanged: the object is still there, just partly covered.

  python training/augment_dataset.py --images data/roboflow/train/images \
      --labels data/roboflow/train/labels --out data/roboflow_aug/train --copies 2
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


def read_boxes(label_file: Path, w: int, h: int):
    """Return pixel boxes (x1, y1, x2, y2) from detect (5 tokens) or OBB (9 tokens) label lines."""
    boxes = []
    if not label_file.exists():
        return boxes
    for line in label_file.read_text().splitlines():
        t = line.split()
        if len(t) == 5:
            cx, cy, bw, bh = (float(v) for v in t[1:])
            boxes.append((int((cx - bw / 2) * w), int((cy - bh / 2) * h), int((cx + bw / 2) * w), int((cy + bh / 2) * h)))
        elif len(t) == 9:
            xs = [float(v) * w for v in t[1::2]]
            ys = [float(v) * h for v in t[2::2]]
            boxes.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))
    return boxes


def skin_colour(rng):
    base = np.array([rng.randint(110, 210), rng.randint(130, 180), rng.randint(150, 235)])  # BGR-ish
    return tuple(int(v) for v in base)


def add_occlusion(img, boxes, rng, max_cover=0.4):
    """Paste 'finger-like' skin-coloured blobs that cover part of each box (<= max_cover of its area)."""
    out = img.copy()
    for (x1, y1, x2, y2) in boxes:
        bw, bh = max(x2 - x1, 2), max(y2 - y1, 2)
        for _ in range(rng.randint(1, 3)):
            cover = rng.uniform(0.08, max_cover)
            ew = int(bw * rng.uniform(0.15, 0.5))
            eh = int(min(bh * cover * bw / max(ew, 1), bh * 1.2) + 2)
            cx = rng.randint(x1, x2)
            cy = rng.randint(y1, y2)
            col = skin_colour(rng)
            layer = out.copy()
            cv2.ellipse(layer, (cx, cy), (max(ew // 2, 2), max(eh // 2, 2)), rng.uniform(0, 180), 0, 360, col, -1, cv2.LINE_AA)
            layer = cv2.GaussianBlur(layer, (5, 5), 0)
            out = cv2.addWeighted(layer, 0.92, out, 0.08, 0)
    return out


def motion_blur(img, rng):
    k = rng.choice([5, 9, 13, 17])
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel)


def lighting(img, rng):
    gamma = rng.uniform(0.5, 1.8)
    lut = (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)
    out = cv2.LUT(img, lut)
    return np.clip(out.astype(np.float32) * rng.uniform(0.7, 1.25), 0, 255).astype(np.uint8)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True, help="output root; writes <out>/images and <out>/labels")
    ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--occlusion-prob", type=float, default=0.7)
    ap.add_argument("--blur-prob", type=float, default=0.4)
    ap.add_argument("--light-prob", type=float, default=0.5)
    ap.add_argument("--keep-originals", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    img_out, lab_out = Path(args.out) / "images", Path(args.out) / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lab_out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    n = 0
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        lf = Path(args.labels) / f"{p.stem}.txt"
        h, w = img.shape[:2]
        boxes = read_boxes(lf, w, h)
        if args.keep_originals:
            shutil.copy2(p, img_out / p.name)
            if lf.exists():
                shutil.copy2(lf, lab_out / lf.name)
        for k in range(args.copies):
            a = img
            if boxes and rng.random() < args.occlusion_prob:
                a = add_occlusion(a, boxes, rng)
            if rng.random() < args.blur_prob:
                a = motion_blur(a, rng)
            if rng.random() < args.light_prob:
                a = lighting(a, rng)
            name = f"{p.stem}_aug{k}"
            cv2.imwrite(str(img_out / f"{name}{p.suffix}"), a)
            if lf.exists():
                shutil.copy2(lf, lab_out / f"{name}.txt")
            n += 1
    print(f"wrote {n} augmented images to {img_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
