"""Technique classification: visibility gate -> trained model (or rule-based fallback) -> label stabiliser.

Final output is one of four classes. "Unable to Determine" is produced by the gate or by low model
confidence; it is not a trained class. Classifier inputs are windowed geometric features only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .config import resolve
from .features import gate_check
from .geometry import axis_diff_deg

CORRECT = "Correct"
WRONG_MOUTH = "Wrong Mouth Position"
WRONG_INHALER = "Wrong Inhaler Position"
UNABLE = "Unable to Determine"
CLASSES = [CORRECT, WRONG_MOUTH, WRONG_INHALER, UNABLE]
TRAINABLE = CLASSES[:3]
MODEL_KINDS = ("rf", "svm", "xgb", "mlp")

_ALIASES = {
    "correct": CORRECT,
    "wrong_mouth": WRONG_MOUTH, "wrong_mouth_position": WRONG_MOUTH, "mouth": WRONG_MOUTH,
    "wrong_inhaler": WRONG_INHALER, "wrong_inhaler_position": WRONG_INHALER, "inhaler": WRONG_INHALER,
    "unable": UNABLE, "unable_to_determine": UNABLE, "uncertain": UNABLE,
}


def normalise_label(s) -> Optional[str]:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return None
    return _ALIASES.get(str(s).strip().lower().replace("-", "_").replace(" ", "_"))


@dataclass
class Prediction:
    label: str
    confidence: float
    source: str                                   # warmup | gate | model | rule-based | none
    probs: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    uncertain: bool = False
    display_label: str = ""                       # label after the stabiliser


class RuleBasedClassifier:
    """Bring-up fallback using placeholder thresholds. NOT clinically validated."""

    def __init__(self, cfg: dict):
        self.t = cfg["rule_thresholds"]

    def predict(self, wf: Dict[str, float]) -> Prediction:
        d = wf.get("mp_mouth_dist_mean", float("nan"))
        ang = wf.get("inhaler_angle_rel_head_mean", float("nan"))
        opn = wf.get("mouth_open_ratio_mean", float("nan"))
        if any(math.isnan(v) for v in (d, ang, opn)):
            return Prediction(UNABLE, 0.0, "rule-based", reasons=["required geometry missing"], uncertain=True)
        t = self.t
        dev = abs(axis_diff_deg(ang, t["expected_angle_deg"]))
        if dev > t["angle_tolerance_deg"]:
            label, margin = WRONG_INHALER, (dev - t["angle_tolerance_deg"]) / t["angle_tolerance_deg"]
        elif d > t["max_mouthpiece_distance"] or opn < t["min_mouth_open_ratio"]:
            margin = max((d - t["max_mouthpiece_distance"]) / t["max_mouthpiece_distance"],
                         (t["min_mouth_open_ratio"] - opn) / t["min_mouth_open_ratio"])
            label = WRONG_MOUTH
        else:
            label = CORRECT
            margin = min((t["max_mouthpiece_distance"] - d) / t["max_mouthpiece_distance"],
                         (t["angle_tolerance_deg"] - dev) / t["angle_tolerance_deg"])
        conf = 0.5 + 0.5 * min(1.0, max(margin, 0.0))
        return Prediction(label, conf, "rule-based", reasons=["uncalibrated rule-based fallback"])


class TechniqueClassifier:
    def __init__(self, cfg: dict, model_path: Optional[str] = None):
        self.cfg = cfg
        c = cfg["classifier"]
        self.threshold = float(c["uncertain_threshold"])
        self.bundle = None
        path = resolve(model_path or c["model_path"])
        if path.exists():
            import joblib
            self.bundle = joblib.load(path)
        self.rule = RuleBasedClassifier(cfg) if c.get("rule_based_fallback", True) else None

    @property
    def mode(self) -> str:
        return "model" if self.bundle else ("rule-based" if self.rule else "none")

    @property
    def window_size(self) -> Optional[int]:
        return int(self.bundle["window_size"]) if self.bundle else None

    def predict(self, wf: Optional[Dict[str, float]], ready: bool) -> Prediction:
        if not ready or wf is None:
            return Prediction(UNABLE, 0.0, "warmup", reasons=["collecting frames"])
        ok, reasons = gate_check(wf, self.cfg)
        if not ok:
            return Prediction(UNABLE, 0.0, "gate", reasons=reasons, uncertain=True)
        if self.bundle:
            names = self.bundle["feature_names"]
            x = np.array([[wf.get(n, float("nan")) for n in names]], dtype=float)
            proba = self.bundle["pipeline"].predict_proba(x)[0]
            classes = self.bundle["classes"]
            i = int(np.argmax(proba))
            probs = {c: float(p) for c, p in zip(classes, proba)}
            if proba[i] < self.threshold:
                return Prediction(UNABLE, float(proba[i]), "model", probs,
                                  [f"low model confidence ({proba[i]:.2f} < {self.threshold:.2f})"], True)
            return Prediction(classes[i], float(proba[i]), "model", probs)
        if self.rule:
            return self.rule.predict(wf)
        return Prediction(UNABLE, 0.0, "none", reasons=["no trained model"], uncertain=True)


class LabelStabiliser:
    """Hysteresis: the displayed label changes only after `hold` consecutive agreeing windows."""

    def __init__(self, hold: int = 5):
        self.hold = max(1, hold)
        self.reset()

    def reset(self) -> None:
        self.current = UNABLE
        self._cand: Optional[str] = None
        self._n = 0

    def update(self, label: str) -> str:
        if label == self.current:
            self._cand, self._n = None, 0
        elif label == self._cand:
            self._n += 1
            if self._n >= self.hold:
                self.current, self._cand, self._n = label, None, 0
        else:
            self._cand, self._n = label, 1
            if self.hold == 1:
                self.current, self._cand, self._n = label, None, 0
        return self.current


# --------------------------------------------------------------------------- model zoo
PARAM_GRIDS = {
    "rf": {"clf__max_depth": [4, 8, None], "clf__min_samples_leaf": [2, 5]},
    "svm": {"clf__C": [0.5, 1.0, 4.0], "clf__gamma": ["scale", 0.01]},
    "xgb": {"clf__max_depth": [2, 3, 4], "clf__n_estimators": [150, 300]},
    "mlp": {"clf__alpha": [1e-3, 1e-2], "clf__hidden_layer_sizes": [(32,), (64, 32)]},
}


def build_model(kind: str, seed: int = 42):
    """Imputer (+ scaler) + classifier. Small, regularised models: the dataset is small."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    imp = ("imp", SimpleImputer(strategy="median", keep_empty_features=True))
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                     n_jobs=-1, random_state=seed)
        return Pipeline([imp, ("clf", clf)])
    if kind == "svm":
        import warnings
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        # newer scikit-learn deprecates probability=True; it still works and keeps older versions supported
        warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.svm")
        clf = SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced", random_state=seed)
        return Pipeline([imp, ("scale", StandardScaler()), ("clf", clf)])
    if kind == "xgb":
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.7, reg_lambda=2.0, min_child_weight=2,
                            objective="multi:softprob", eval_metric="mlogloss", tree_method="hist",
                            n_jobs=-1, random_state=seed)
        return Pipeline([imp, ("clf", clf)])
    if kind == "mlp":
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
                            max_iter=500, random_state=seed)
        return Pipeline([imp, ("scale", StandardScaler()), ("clf", clf)])
    raise ValueError(f"unknown model kind {kind!r}; choose from {MODEL_KINDS}")


def fit_kwargs(kind: str, y: np.ndarray) -> dict:
    """XGBoost has no class_weight option, so pass balanced sample weights."""
    if kind == "xgb":
        from sklearn.utils.class_weight import compute_sample_weight
        return {"clf__sample_weight": compute_sample_weight("balanced", y)}
    return {}
