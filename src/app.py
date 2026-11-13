#!/usr/bin/env python
"""Live demo / dataset recorder.

  python -m src.app                                  # webcam 0, live feedback
  python -m src.app --source 1                       # another camera
  python -m src.app --source rtsp://...              # phone stream
  python -m src.app --source clip.mp4                # video file
  python -m src.app --record --person-id P01 --label correct --angle front --lighting bright --distance near

Keys: q/Esc quit | r toggle feature recording | s save screenshot
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .camera import Camera
from .classifier import UNABLE, normalise_label
from .config import ROOT, load_config
from .overlay import draw
from .pipeline import InhalerPipeline
from .recorder import FeatureRecorder, ScanLogger


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--source", default=None, help="camera index, stream URL or video file")
    p.add_argument("--weights", default=None, help="override detector weights (best.pt)")
    p.add_argument("--camera-only", action="store_true",
                   help="open the camera without inhaler detection (useful before best.pt is available)")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--save-video", default=None, help="write the annotated output to this mp4")
    p.add_argument("--screenshot-dir", default="runs/screenshots")
    p.add_argument("--scan-dir", default="data/scans", help="folder for the per-frame scan-result CSV")
    p.add_argument("--no-scan-log", action="store_true", help="do not write the scan-result CSV")
    rec = p.add_argument_group("dataset recording")
    rec.add_argument("--record", action="store_true", help="start recording per-frame features immediately")
    rec.add_argument("--person-id", default="P00")
    rec.add_argument("--label", default="correct", help="correct | wrong_mouth | wrong_inhaler")
    rec.add_argument("--error-type", default="")
    rec.add_argument("--angle", default="front")
    rec.add_argument("--lighting", default="normal")
    rec.add_argument("--distance", default="medium")
    rec.add_argument("--out-dir", default="data/recordings")
    return p.parse_args(argv)


def _source(v):
    if v is None:
        return None
    return int(v) if str(v).isdigit() else v


def _open_recorder(args) -> FeatureRecorder:
    label = normalise_label(args.label)
    if label is None or label == UNABLE:
        raise SystemExit(f"--label must be correct | wrong_mouth | wrong_inhaler (got {args.label!r})")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = ROOT / args.out_dir / f"{args.person_id}_{args.label}_{stamp}.csv"
    meta = dict(person_id=args.person_id, label=label, error_type=args.error_type, angle=args.angle,
                lighting=args.lighting, distance=args.distance, source="live")
    print(f"[record] writing {path}")
    return FeatureRecorder(path, meta)


def _blank(message: str, w=960, h=540) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    cv2.putText(img, message, (30, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 180, 255), 2, cv2.LINE_AA)
    return img


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    c = cfg["camera"]
    source = _source(args.source) if args.source is not None else c["source"]

    try:
        pipe = InhalerPipeline(cfg, weights=args.weights, load_detector=not args.camera_only)
    except FileNotFoundError as e:
        if args.weights is not None:
            print(e, file=sys.stderr)
            return 2
        print(f"{e}\nStarting camera-only mode. Add models\\best.pt for inhaler detection.",
              file=sys.stderr)
        pipe = InhalerPipeline(cfg, load_detector=False)
    print(f"[init] classifier mode: {pipe.clf.mode}"
          + ("  (placeholder rule thresholds - train a model for real use)" if pipe.clf.mode == "rule-based" else ""))

    cam = Camera(source, c["width"], c["height"], c["fps"], c["reconnect_attempts"],
                 c["reconnect_delay_s"], c["stale_frame_timeout_s"])
    if not cam.start():
        print(f"Could not open camera/source {source!r}", file=sys.stderr)
        return 3

    recorder = _open_recorder(args) if args.record else None
    scan_log = None
    if not args.no_scan_log:
        scan_log = ScanLogger(ROOT / args.scan_dir / f"scan_{datetime.now():%Y%m%d_%H%M%S}.csv")
        print(f"[scan] writing {scan_log.path}")
    writer = None
    fps_ema = lat_ema = None
    last_done = None
    n = n_uncertain = n_pred = 0
    stage = {}
    t_begin = time.perf_counter()

    try:
        while True:
            fr = cam.read(timeout=0.5)
            if fr is None:
                if cam.ended:
                    break
                if not args.no_display:
                    cv2.imshow("Smart Inhaler CV", _blank(f"Camera signal {cam.state} - reconnecting..."))
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
                continue

            out = pipe.process(fr.image, fr.timestamp)
            n += 1
            for k, v in out.timings.items():
                stage[k] = stage.get(k, 0.0) + v
            if out.prediction is not None and out.prediction.source != "warmup":
                n_pred += 1
                n_uncertain += int(out.prediction.uncertain)
            if recorder is not None:
                recorder.write(out.frame_idx, fr.timestamp - t_begin, out.features)
            if scan_log is not None:
                scan_log.write(out, fr.timestamp - t_begin)

            now = time.perf_counter()
            if last_done is not None:
                inst = 1.0 / max(now - last_done, 1e-6)
                fps_ema = inst if fps_ema is None else 0.9 * fps_ema + 0.1 * inst
            last_done = now

            vis = draw(fr.image, out, dict(fps=fps_ema, latency_ms=lat_ema, cam_state=cam.state,
                                           recording=recorder is not None))
            lat = (time.perf_counter() - fr.timestamp) * 1000.0
            lat_ema = lat if lat_ema is None else 0.9 * lat_ema + 0.1 * lat

            if args.save_video:
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), c["fps"], (w, h))
                writer.write(vis)

            if not args.no_display:
                cv2.imshow("Smart Inhaler CV", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    if recorder is None:
                        recorder = _open_recorder(args)
                    else:
                        recorder.close()
                        print(f"[record] saved {recorder.n_rows} rows -> {recorder.path}")
                        recorder = None
                if key == ord("s"):
                    d = ROOT / args.screenshot_dir
                    d.mkdir(parents=True, exist_ok=True)
                    p = d / f"shot_{datetime.now():%Y%m%d_%H%M%S}.png"
                    cv2.imwrite(str(p), vis)
                    print(f"[screenshot] {p}")
            if args.max_frames and n >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        if scan_log is not None:
            scan_log.close()
            print(f"[scan] saved {scan_log.n_rows} rows -> {scan_log.path}")
        if writer is not None:
            writer.release()
        if recorder is not None:
            recorder.close()
            print(f"[record] saved {recorder.n_rows} rows -> {recorder.path}")
        cv2.destroyAllWindows()

    if n:
        wall = time.perf_counter() - t_begin
        print(f"\nSession summary: {n} frames in {wall:.1f}s -> {n / wall:.1f} FPS (incl. display)")
        print("Mean per-stage ms: " + ", ".join(f"{k}={v / n:.1f}" for k, v in stage.items()))
        if n_pred:
            print(f"Uncertain / unable predictions: {n_uncertain}/{n_pred} ({n_uncertain / n_pred:.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
