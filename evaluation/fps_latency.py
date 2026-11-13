#!/usr/bin/env python
"""Measure real-time performance on this machine. Reports measured values only.

  python evaluation/fps_latency.py --source 0 --frames 300          # live camera
  python evaluation/fps_latency.py --source clip.mp4 --frames 500   # video file

Metrics: FPS (processing + overlay), per-stage inference time, average inference time per frame,
end-to-end feedback latency (capture timestamp -> annotated frame ready), uncertain-prediction rate.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.camera import Camera
from src.config import load_config, resolve
from src.overlay import draw
from src.pipeline import InhalerPipeline


def hardware() -> dict:
    info = dict(platform=platform.platform(), python=platform.python_version(),
                processor=platform.processor() or platform.machine())
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return info


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=None)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out", default="evaluation/results/fps_latency.json")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    c = cfg["camera"]
    src = args.source if args.source is not None else c["source"]
    src = int(src) if str(src).isdigit() else src
    pipe = InhalerPipeline(cfg, weights=args.weights)
    cam = Camera(src, c["width"], c["height"], c["fps"])
    if not cam.start():
        print(f"cannot open {src!r}", file=sys.stderr)
        return 3

    stages, lat, totals = {}, [], []
    uncertain = preds = 0
    n = 0
    t_run0 = None
    while n < args.warmup + args.frames:
        fr = cam.read(timeout=2.0)
        if fr is None:
            if cam.ended:
                break
            continue
        out = pipe.process(fr.image, fr.timestamp)
        vis = draw(fr.image, out, {})
        t_end = time.perf_counter()
        n += 1
        if n == args.warmup:
            t_run0 = t_end
        if n > args.warmup:
            lat.append((t_end - fr.timestamp) * 1000.0)
            totals.append(out.timings["total"])
            for k, v in out.timings.items():
                stages.setdefault(k, []).append(v)
            if out.prediction is not None and out.prediction.source != "warmup":
                preds += 1
                uncertain += int(out.prediction.uncertain)
    cam.stop()
    measured = n - args.warmup
    if measured <= 0 or t_run0 is None:
        print("not enough frames measured", file=sys.stderr)
        return 4
    wall = time.perf_counter() - t_run0

    result = dict(
        source=str(src), frames_measured=measured, warmup_frames=args.warmup,
        classifier_mode=pipe.clf.mode, hardware=hardware(),
        fps=measured / wall,
        inference_ms_per_frame=dict(mean=float(np.mean(totals)), p50=pct(totals, 50), p95=pct(totals, 95)),
        stage_ms={k: dict(mean=float(np.mean(v)), p95=pct(v, 95)) for k, v in stages.items()},
        end_to_end_latency_ms=dict(mean=float(np.mean(lat)), p50=pct(lat, 50), p95=pct(lat, 95),
                                   max=float(np.max(lat))),
        uncertain_prediction_rate=(uncertain / preds) if preds else None,
    )
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    print(f"\nMeasured over {measured} frames on {result['hardware']['processor']}"
          f"{' / ' + result['hardware']['cuda'] if result['hardware'].get('cuda') else ''}")
    print(f"FPS: {result['fps']:.1f}   mean inference: {result['inference_ms_per_frame']['mean']:.1f} ms/frame "
          f"(p95 {result['inference_ms_per_frame']['p95']:.1f})")
    print(f"End-to-end latency: mean {result['end_to_end_latency_ms']['mean']:.1f} ms, "
          f"p95 {result['end_to_end_latency_ms']['p95']:.1f} ms")
    for k, v in result["stage_ms"].items():
        print(f"  {k:<11} mean {v['mean']:6.1f} ms   p95 {v['p95']:6.1f} ms")
    if result["uncertain_prediction_rate"] is not None:
        print(f"Uncertain predictions: {result['uncertain_prediction_rate']:.1%}")
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
