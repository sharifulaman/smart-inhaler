#!/usr/bin/env python
"""Build a synthetic YOLO detection dataset from a few reference inhaler photos.

No labelled data needed: each reference photo (data/reference_inhalers/, with one box per photo in
boxes.csv) is cut out with GrabCut and pasted onto many random backgrounds with random scale,
rotation, flip, colour shift, blur, lighting and fake finger/hand occlusion. Labels are exact
because we know where we pasted.

  python training/make_synthetic.py --out data/synthetic_inhaler --n 900
  python training/train_yolo.py --data data/inhaler_synth.yaml --model yolo11n.pt --imgsz 416

Limits: only as diverse as the reference photos, so it is a bootstrap. Validation images come from
the same photos, so validation scores are optimistic - judge it on real webcam frames. Put real
background photos (rooms, faces, hands; no inhaler) in data/backgrounds/ to improve realism.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import resolve

SKIN = [(120, 150, 200), (140, 170, 225), (100, 130, 180), (80, 105, 150), (170, 190, 235)]  # BGR


def load_cutouts(ref_dir: Path):
    """Return a list of (bgr_crop, mask) tightly cropped around each reference inhaler."""
    cuts = []
    for row in csv.DictReader(open(ref_dir / "boxes.csv")):
        img = cv2.imread(str(ref_dir / row["file"]))
        h, w = img.shape[:2]
        x1, y1, x2, y2 = (max(0, int(row[k])) for k in ("x1", "y1", "x2", "y2"))
        x2, y2 = min(x2, w), min(y2, h)
        crop = img[y1:y2, x1:x2].copy()
        ch, cw = crop.shape[:2]
        mask = np.full((ch, cw), cv2.GC_PR_BGD, np.uint8)
        mask[int(ch * .08):int(ch * .92), int(cw * .12):int(cw * .88)] = cv2.GC_PR_FGD
        mask[int(ch * .3):int(ch * .7), int(cw * .3):int(cw * .7)] = cv2.GC_FGD
        mask[:2, :] = mask[-2:, :] = mask[:, :2] = mask[:, -2:] = cv2.GC_BGD
        bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
        try:
            cv2.grabCut(crop, mask, None, bgd, fgd, 6, cv2.GC_INIT_WITH_MASK)
            fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8) * 255
        except cv2.error:
            fg = np.full((ch, cw), 255, np.uint8)
        if fg.mean() < 255 * 0.25:          # GrabCut lost the object: fall back to the whole box
            fg = np.full((ch, cw), 255, np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        ys, xs = np.where(fg > 0)
        cuts.append((crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1],
                     fg[ys.min():ys.max() + 1, xs.min():xs.max() + 1]))
    return cuts


def random_background(rng: random.Random, size, bg_files):
    w, h = size
    if bg_files and rng.random() < 0.6:
        img = cv2.imread(str(rng.choice(bg_files)))
        if img is not None:
            ih, iw = img.shape[:2]
            s = max(w / iw, h / ih) * rng.uniform(1.0, 1.6)
            img = cv2.resize(img, (int(iw * s) + 1, int(ih * s) + 1))
            y, x = rng.randint(0, img.shape[0] - h), rng.randint(0, img.shape[1] - w)
            return img[y:y + h, x:x + w].copy()
    c1 = np.array([rng.randint(0, 255) for _ in range(3)], np.float32)
    c2 = np.array([rng.randint(0, 255) for _ in range(3)], np.float32)
    t = np.linspace(0, 1, h if rng.random() < .5 else w, dtype=np.float32)
    grad = (c1[None, :] * (1 - t[:, None]) + c2[None, :] * t[:, None])
    img = (np.repeat(grad[:, None, :], w, 1) if len(t) == h else np.repeat(grad[None, :, :], h, 0))
    img = img + np.random.RandomState(rng.randint(0, 10 ** 6)).normal(0, rng.uniform(2, 14), img.shape)
    for _ in range(rng.randint(2, 8)):          # distractor blobs: skin-like and random colours
        col = rng.choice(SKIN) if rng.random() < .5 else tuple(rng.randint(0, 255) for _ in range(3))
        cv2.ellipse(img, (rng.randint(0, w), rng.randint(0, h)), (rng.randint(20, 200), rng.randint(20, 200)),
                    rng.randint(0, 180), 0, 360, col, -1, cv2.LINE_AA)
    return cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (0, 0), rng.uniform(0, 5))


def transform_cutout(rng, crop, mask, target_h):
    s = target_h / crop.shape[0]
    crop, mask = cv2.resize(crop, None, fx=s, fy=s), cv2.resize(mask, None, fx=s, fy=s)
    if rng.random() < .5:
        crop, mask = crop[:, ::-1], mask[:, ::-1]
    ang = rng.choice([rng.uniform(-30, 30)] * 3 + [rng.uniform(-90, 90)])
    h, w = crop.shape[:2]
    diag = int(np.hypot(h, w)) + 4
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    M[0, 2] += diag / 2 - w / 2
    M[1, 2] += diag / 2 - h / 2
    rc = cv2.warpAffine(np.ascontiguousarray(crop), M, (diag, diag), flags=cv2.INTER_LINEAR)
    rm = cv2.warpAffine(np.ascontiguousarray(mask), M, (diag, diag), flags=cv2.INTER_LINEAR)
    ys, xs = np.where(rm > 40)
    rc, rm = rc[ys.min():ys.max() + 1, xs.min():xs.max() + 1], rm[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    # colour / lighting jitter
    hsv = cv2.cvtColor(rc, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-8, 8)) % 180
    hsv[..., 1] *= rng.uniform(0.7, 1.25)
    hsv[..., 2] *= rng.uniform(0.45, 1.2)
    rc = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return rc, rm


def add_fingers(rng, img, box):
    """Draw 1-3 skin-coloured rounded blobs over part of the object (hand occlusion)."""
    x1, y1, x2, y2 = box
    col = rng.choice(SKIN)
    for _ in range(rng.randint(1, 3)):
        cx = rng.randint(x1, x2)
        cy = rng.randint(y1, y2)
        ax = rng.randint(8, max(9, (x2 - x1) // 2))
        ay = rng.randint(6, max(7, (y2 - y1) // 5))
        c = tuple(int(np.clip(v + rng.randint(-15, 15), 0, 255)) for v in col)
        cv2.ellipse(img, (cx, cy), (ax, ay), rng.randint(0, 180), 0, 360, c, -1, cv2.LINE_AA)


def make_one(rng, cuts, size, bg_files, negative):
    w, h = size
    img = random_background(rng, size, bg_files)
    if negative:
        return img, None
    crop, mask = rng.choice(cuts)
    rc, rm = transform_cutout(rng, crop, mask, rng.randint(int(h * .18), int(h * .85)))
    ch, cw = rc.shape[:2]
    if cw >= w or ch >= h:
        r = min((w - 2) / cw, (h - 2) / ch)
        rc, rm = cv2.resize(rc, None, fx=r, fy=r), cv2.resize(rm, None, fx=r, fy=r)
        ch, cw = rc.shape[:2]
    x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
    a = cv2.GaussianBlur(rm, (0, 0), 1.2).astype(np.float32)[..., None] / 255.0
    roi = img[y:y + ch, x:x + cw].astype(np.float32)
    img[y:y + ch, x:x + cw] = (rc * a + roi * (1 - a)).astype(np.uint8)
    box = (x, y, x + cw, y + ch)
    if rng.random() < .55:
        add_fingers(rng, img, box)
    # global degradations: blur, motion blur, noise, brightness
    if rng.random() < .4:
        k = rng.choice([5, 9, 13])
        kern = np.zeros((k, k), np.float32)
        kern[k // 2, :] = 1.0 / k
        M = cv2.getRotationMatrix2D((k / 2 - .5, k / 2 - .5), rng.uniform(0, 180), 1.0)
        img = cv2.filter2D(img, -1, cv2.warpAffine(kern, M, (k, k)))
    if rng.random() < .3:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.8, 2.5))
    img = np.clip(img.astype(np.float32) * rng.uniform(0.55, 1.25) + rng.uniform(-20, 20)
                  + np.random.RandomState(rng.randint(0, 10 ** 6)).normal(0, rng.uniform(0, 8), img.shape), 0, 255)
    return img.astype(np.uint8), box


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refs", default="data/reference_inhalers")
    ap.add_argument("--backgrounds", default="data/backgrounds")
    ap.add_argument("--out", default="data/synthetic_inhaler")
    ap.add_argument("--n", type=int, default=900, help="total images (85%% train / 15%% val)")
    ap.add_argument("--size", type=int, nargs=2, default=[640, 480], metavar=("W", "H"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    cuts = load_cutouts(resolve(args.refs))
    bgdir = resolve(args.backgrounds)
    bg_files = [p for p in bgdir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")] if bgdir.exists() else []
    print(f"{len(cuts)} reference cut-outs, {len(bg_files)} background photos")
    out = resolve(args.out)
    W, H = args.size
    for split, count in (("train", int(args.n * .85)), ("val", args.n - int(args.n * .85))):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(count):
            img, box = make_one(rng, cuts, (W, H), bg_files, negative=rng.random() < .12)
            name = f"{split}_{i:05d}"
            cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            lab = ""
            if box:
                x1, y1, x2, y2 = box
                lab = f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n"
            (out / "labels" / split / f"{name}.txt").write_text(lab)
    print(f"wrote {args.n} images to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
