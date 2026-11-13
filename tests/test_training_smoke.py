import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib

from synthetic_data import make_dataset
from training import train_classifier
from evaluation import metrics


def test_loso_training_and_report(tmp_path):
    data = make_dataset(tmp_path / "feat", n_subjects=10, frames=100, seed=1)
    out = tmp_path / "clf.joblib"
    rep = tmp_path / "rep"
    rc = train_classifier.main(["--data", str(data), "--model", "rf", "--out", str(out),
                                "--report-dir", str(rep), "--stride", "10"])
    assert rc == 0 and out.exists()
    summ = json.loads((rep / "loso_summary.json").read_text())
    rf = summ["models"]["rf"]
    assert rf["n_subjects"] == 10
    assert rf["raw"]["accuracy"] > 0.85           # synthetic classes are separable
    assert rf["raw"]["accuracy"] > rf["majority_baseline_accuracy"] + 0.3
    bundle = joblib.load(out)
    assert bundle["classes"] == ["Correct", "Wrong Mouth Position", "Wrong Inhaler Position"]

    rc = metrics.main(["--features-dirs", str(data), "--predictions", str(rep / "loso_predictions_rf.csv"),
                       "--out-dir", str(rep)])
    assert rc == 0 and (rep / "confusion_matrix.png").exists() and (rep / "report.md").exists()
