#!/usr/bin/env python
"""Detection-coverage and classification metrics from recorded data. Reports actual values.

  python evaluation/metrics.py --features-dirs data/features data/recordings \
         --predictions evaluation/results/loso_predictions_xgb.csv

* Coverage (from per-frame feature CSVs): % frames with an inhaler, a reliable face, a reliable
  mouth and a holding hand - overall and split by angle / lighting / distance.
* Classification (from LOSO predictions): accuracy, macro-F1, confusion matrix, uncertain rate,
  per-subject accuracy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.classifier import TRAINABLE, UNABLE
from src.config import resolve
from src.evaluation_utils import classification_summary, plot_confusion

FLAGS = {"has_inhaler": "Inhaler detection coverage", "has_face": "Face detection",
         "has_mouth": "Mouth landmark detection", "has_hand": "Holding-hand detection"}


def coverage(dirs):
    frames = []
    for d in dirs:
        for f in sorted(resolve(d).rglob("*.csv")):
            df = pd.read_csv(f)
            if set(FLAGS).issubset(df.columns):
                frames.append(df)
    if not frames:
        return None, None
    df = pd.concat(frames, ignore_index=True)
    overall = {FLAGS[k]: float(df[k].mean()) for k in FLAGS}
    overall["frames"] = int(len(df))
    groups = {}
    for col in ("angle", "lighting", "distance"):
        if col in df.columns and df[col].notna().any():
            g = df.groupby(col)[list(FLAGS)].mean()
            groups[col] = {str(i): {FLAGS[k]: float(r[k]) for k in FLAGS} for i, r in g.iterrows()}
    return overall, groups


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-dirs", nargs="*", default=["data/features", "data/recordings"])
    ap.add_argument("--predictions", default=None, help="loso_predictions_<model>.csv")
    ap.add_argument("--out-dir", default="evaluation/results")
    ap.add_argument("--threshold", type=float, default=None,
                    help="re-apply an uncertainty threshold (default: use stored pred_final)")
    args = ap.parse_args(argv)

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result, md = {}, ["# Evaluation report\n"]

    overall, groups = coverage(args.features_dirs)
    if overall:
        result["coverage"], result["coverage_by_condition"] = overall, groups
        md.append(f"## Detection coverage ({overall['frames']} frames)\n")
        for k in FLAGS.values():
            md.append(f"- {k}: **{overall[k]:.1%}**")
        for col, table in groups.items():
            md.append(f"\n### By {col}\n")
            md.append("| " + col + " | " + " | ".join(FLAGS.values()) + " |")
            md.append("|---|" + "---|" * len(FLAGS))
            for name, vals in table.items():
                md.append(f"| {name} | " + " | ".join(f"{vals[v]:.1%}" for v in FLAGS.values()) + " |")
        md.append("")
    else:
        print("no per-frame feature CSVs found for coverage metrics")

    if args.predictions:
        df = pd.read_csv(resolve(args.predictions))
        classes = [c for c in TRAINABLE if c in set(df["true"])]
        pred_labels = classes + [UNABLE]
        if args.threshold is not None:
            final = [p if c >= args.threshold else UNABLE for p, c in zip(df["pred"], df["conf"])]
        else:
            final = df["pred_final"].tolist()
        s = classification_summary(df["true"], final, classes, pred_labels)
        raw = classification_summary(df["true"], df["pred"], classes, classes + [UNABLE])
        unc = float(np.mean([f == UNABLE for f in final]))
        per_subj = {str(k): float(v) for k, v in
                    (df.assign(ok=(df["true"] == pd.Series(final, index=df.index)))
                       .groupby("person_id")["ok"].mean()).items()}
        result["classification"] = dict(with_uncertain=s, raw=raw, uncertain_rate=unc, per_subject_accuracy=per_subj)
        plot_confusion(s["confusion"], s["confusion_rows"], s["confusion_cols"], out_dir / "confusion_matrix.png",
                       f"LOSO confusion matrix (acc {s['accuracy']:.2f}, macro-F1 {s['macro_f1']:.2f})")
        md.append(f"## Classification (LOSO, {df['person_id'].nunique()} subjects, {len(df)} windows)\n")
        md.append(f"- Accuracy (uncertain counted as errors): **{s['accuracy']:.3f}**")
        md.append(f"- Accuracy (argmax, no uncertainty): {raw['accuracy']:.3f}")
        md.append(f"- Macro-F1: **{s['macro_f1']:.3f}**   Balanced accuracy: {s['balanced_accuracy']:.3f}")
        md.append(f"- Frequency of uncertain predictions: **{unc:.1%}**\n")
        md.append("| Class | Precision | Recall | F1 | Support |\n|---|---|---|---|---|")
        for k, v in s["per_class"].items():
            md.append(f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |")
        md.append("\nConfusion matrix (rows = true, cols = " + ", ".join(pred_labels) + "):\n")
        for r, row in zip(s["confusion_rows"], s["confusion"]):
            md.append(f"- {r}: {row}")
        md.append("\nPer-subject accuracy: " + ", ".join(f"{k}={v:.2f}" for k, v in per_subj.items()))

    (out_dir / "report.md").write_text("\n".join(md) + "\n")
    (out_dir / "report.json").write_text(json.dumps(result, indent=2))
    print("\n".join(md))
    print(f"\nSaved -> {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
