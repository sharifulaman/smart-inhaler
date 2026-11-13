"""Shared classification metrics and plotting (used by training and evaluation scripts)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def classification_summary(y_true: Sequence[str], y_pred: Sequence[str],
                           true_labels: List[str], pred_labels: List[str]) -> Dict:
    """Accuracy, macro-F1, per-class P/R/F1 and a confusion matrix.

    Rows = true labels, columns = predicted labels (pred_labels may include "Unable to Determine",
    which counts as an error for accuracy/recall but never as a true label).
    """
    from sklearn.metrics import confusion_matrix

    y_true, y_pred = np.asarray(y_true, dtype=object), np.asarray(y_pred, dtype=object)
    n = len(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=list(pred_labels))
    idx = [pred_labels.index(t) for t in true_labels]
    cm_t = cm[idx, :]                                  # rows restricted to true labels
    per_class = {}
    f1s = []
    for t in true_labels:
        j = pred_labels.index(t)
        tp = float(cm[j, j])
        fn = float(cm[j, :].sum() - tp)
        fp = float(sum(cm[pred_labels.index(o), j] for o in true_labels if o != t))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        per_class[t] = dict(precision=p, recall=r, f1=f1, support=int(cm[j, :].sum()))
        f1s.append(f1)
    correct = float(sum(cm[pred_labels.index(t), pred_labels.index(t)] for t in true_labels))
    return dict(n=int(n), accuracy=correct / n if n else float("nan"),
                macro_f1=float(np.mean(f1s)) if f1s else float("nan"),
                balanced_accuracy=float(np.mean([v["recall"] for v in per_class.values()])) if per_class else float("nan"),
                per_class=per_class, confusion=cm_t.tolist(),
                confusion_rows=list(true_labels), confusion_cols=list(pred_labels))


def plot_confusion(cm: Sequence[Sequence[float]], row_labels: List[str], col_labels: List[str],
                   path: Path, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.6 * len(col_labels) + 2, 1.3 * len(row_labels) + 2))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(col_labels)), [c.replace(" ", "\n") for c in col_labels])
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{int(cm[i, j])}\n{norm[i, j]:.0%}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
