#!/usr/bin/env python
"""Train and evaluate the technique classifier with Leave-One-Subject-Out (LOSO) validation.

Reads per-frame feature CSVs (from `python -m src.app --record` or training/extract_features.py),
cuts them into sliding windows, and evaluates every model with LeaveOneGroupOut on person_id,
so no individual appears in both training and test data. The best model (by LOSO macro-F1) is
refit on all subjects and saved to models/technique_clf.joblib.

  python training/train_classifier.py --data data/features data/recordings --model all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.classifier import (MODEL_KINDS, PARAM_GRIDS, TRAINABLE, UNABLE, build_model, fit_kwargs,
                            normalise_label)
from src.config import load_config, resolve
from src.evaluation_utils import classification_summary, plot_confusion
from src.features import ALL_COLS, gate_check, select_window_features, window_features_from_array


def load_windows(dirs, cfg, window: int, stride: int):
    files = []
    for d in dirs:
        files += sorted(resolve(d).rglob("*.csv"))
    feats, meta = [], []
    n_files = n_gate = 0
    for f in files:
        df = pd.read_csv(f)
        if not {"person_id", "label", "frame_idx"}.issubset(df.columns):
            continue
        for c in ALL_COLS:
            if c not in df.columns:
                df[c] = np.nan
        n_files += 1
        df = df.sort_values("frame_idx").reset_index(drop=True)
        labels = df["label"].map(normalise_label).to_numpy(dtype=object)
        arr = df[ALL_COLS].to_numpy(dtype=float)
        for s in range(0, len(df) - window + 1, stride):
            seg = labels[s:s + window]
            if seg[0] not in TRAINABLE or any(x != seg[0] for x in seg):
                continue
            wf = window_features_from_array(arr[s:s + window])
            ok, _ = gate_check(wf, cfg)
            if not ok:
                n_gate += 1
                continue
            r = df.iloc[s]
            feats.append(wf)
            meta.append(dict(person_id=str(r["person_id"]), label=seg[0],
                             error_type=r.get("error_type", ""), angle=r.get("angle", ""),
                             lighting=r.get("lighting", ""), distance=r.get("distance", ""),
                             source_file=f.name, start_frame=int(r["frame_idx"])))
    return pd.DataFrame(feats), pd.DataFrame(meta), n_files, n_gate


def loso(kind, X, y, groups, classes, thr, seed):
    from sklearn.model_selection import LeaveOneGroupOut

    proba_all = np.zeros((len(y), len(classes)))
    majority_hits = 0
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        m = build_model(kind, seed)
        m.fit(X[tr], y[tr], **fit_kwargs(kind, y[tr]))
        proba = m.predict_proba(X[te])
        full = np.zeros((len(te), len(classes)))
        full[:, m.classes_.astype(int)] = proba
        proba_all[te] = full
        majority = np.bincount(y[tr], minlength=len(classes)).argmax()
        majority_hits += int(np.sum(y[te] == majority))
    pred = proba_all.argmax(1)
    conf = proba_all.max(1)
    return proba_all, pred, conf, majority_hits / len(y)


def tuned_model(kind, X, y, groups, seed):
    from sklearn.model_selection import GridSearchCV, GroupKFold

    n_splits = int(min(5, len(set(groups))))
    gs = GridSearchCV(build_model(kind, seed), PARAM_GRIDS[kind], scoring="f1_macro",
                      cv=GroupKFold(n_splits=n_splits), n_jobs=1)
    gs.fit(X, y, groups=groups, **fit_kwargs(kind, y))
    print(f"    tuned {kind}: {gs.best_params_} (inner CV macro-F1 {gs.best_score_:.3f})")
    return gs.best_estimator_


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", default=["data/features", "data/recordings"])
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--model", default="all", choices=list(MODEL_KINDS) + ["all"])
    ap.add_argument("--features", default="core", choices=["core", "all"])
    ap.add_argument("--exclude-prefix", nargs="*", default=[], help="ablation, e.g. hand_ head_")
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--tune", action="store_true", help="nested group-CV hyper-parameter search (final fit)")
    ap.add_argument("--out", default="models/technique_clf.joblib")
    ap.add_argument("--report-dir", default="evaluation/results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-subjects", type=int, default=3)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    window = args.window or cfg["features"]["window_size"]
    stride = args.stride or cfg["features"]["window_stride_train"]
    thr = cfg["classifier"]["uncertain_threshold"]
    report = resolve(args.report_dir)
    report.mkdir(parents=True, exist_ok=True)

    W, M, n_files, n_gate = load_windows(args.data, cfg, window, stride)
    if W.empty:
        print("No usable windows found. Record or extract features first (see README).", file=sys.stderr)
        return 2
    subjects = sorted(M["person_id"].unique())
    print(f"Loaded {n_files} files -> {len(W)} windows from {len(subjects)} subjects "
          f"({n_gate} windows dropped by the visibility gate)")
    print(M["label"].value_counts().to_string())
    if len(subjects) < args.min_subjects:
        print(f"Need at least {args.min_subjects} subjects for LOSO (found {len(subjects)}).", file=sys.stderr)
        return 3
    if len(subjects) < 10:
        print(f"WARNING: only {len(subjects)} subjects; the project requires at least 10.")

    names = select_window_features(args.features, args.exclude_prefix)
    X = W[names].to_numpy(dtype=float)
    classes = [c for c in TRAINABLE if c in set(M["label"])]
    y = np.array([classes.index(l) for l in M["label"]])
    groups = M["person_id"].to_numpy()
    pred_labels = classes + [UNABLE]
    print(f"{len(names)} features ({args.features}), classes: {classes}")

    kinds = list(MODEL_KINDS) if args.model == "all" else [args.model]
    summary = {}
    for kind in kinds:
        t0 = time.time()
        try:
            proba, pred, conf, base_acc = loso(kind, X, y, groups, classes, thr, args.seed)
        except ImportError as e:
            print(f"  skipping {kind}: {e}")
            continue
        true_l = [classes[i] for i in y]
        raw_l = [classes[i] for i in pred]
        final_l = [classes[i] if c >= thr else UNABLE for i, c in zip(pred, conf)]
        raw = classification_summary(true_l, raw_l, classes, pred_labels)
        fin = classification_summary(true_l, final_l, classes, pred_labels)
        confident = conf >= thr
        acc_conf = float(np.mean(pred[confident] == y[confident])) if confident.any() else float("nan")
        per_subj = {s: float(np.mean(pred[groups == s] == y[groups == s])) for s in subjects}
        summary[kind] = dict(raw=raw, with_uncertain=fin, uncertain_rate=float(1 - confident.mean()),
                             accuracy_on_confident=acc_conf, majority_baseline_accuracy=float(base_acc),
                             per_subject_accuracy=per_subj, n_windows=len(y), n_subjects=len(subjects))
        pdf = M.copy()
        pdf["true"], pdf["pred"], pdf["conf"], pdf["pred_final"] = true_l, raw_l, conf, final_l
        for j, c in enumerate(classes):
            pdf[f"p_{c}"] = proba[:, j]
        pdf.to_csv(report / f"loso_predictions_{kind}.csv", index=False)
        plot_confusion(fin["confusion"], fin["confusion_rows"], fin["confusion_cols"],
                       report / f"confusion_{kind}.png",
                       f"{kind.upper()} LOSO (thr {thr}) acc {fin['accuracy']:.2f} / F1 {fin['macro_f1']:.2f}")
        print(f"  {kind:>4}: acc {raw['accuracy']:.3f}  macro-F1 {raw['macro_f1']:.3f}  "
              f"uncertain {summary[kind]['uncertain_rate']:.1%}  baseline {base_acc:.3f}  ({time.time() - t0:.0f}s)")

    if not summary:
        print("No model could be trained.", file=sys.stderr)
        return 4
    best = max(summary, key=lambda k: summary[k]["raw"]["macro_f1"])
    print(f"Best by LOSO macro-F1: {best}")
    (report / "loso_summary.json").write_text(json.dumps(
        dict(features=args.features, n_features=len(names), window=window, stride=stride,
             threshold=thr, best=best, models=summary), indent=2))

    final = tuned_model(best, X, y, groups, args.seed) if args.tune else build_model(best, args.seed)
    if not args.tune:
        final.fit(X, y, **fit_kwargs(best, y))
    import joblib
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(pipeline=final, feature_names=names, classes=classes, kind=best,
                     window_size=window, stride=stride, feature_mode=args.features,
                     n_subjects=len(subjects), n_windows=len(y),
                     loso_macro_f1=summary[best]["raw"]["macro_f1"],
                     loso_accuracy=summary[best]["raw"]["accuracy"]), out)
    print(f"Saved model -> {out}\nReports -> {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
