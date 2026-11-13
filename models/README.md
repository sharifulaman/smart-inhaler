# models/

- `best.pt` - your trained YOLO11 inhaler/spacer detector. **Not included**: copy your existing
  working model here (audit it first with `python evaluation/audit_detector.py`) or produce one with
  `python training/train_yolo.py`.
  Without `best.pt`, the app falls back to zero-shot YOLO-World (`detector.zero_shot` in
  `configs/config.yaml`; `yolov8s-worldv2.pt` is auto-downloaded here on first run). It needs no
  training data but is less accurate and gives boxes only.
- `technique_clf.joblib` - technique classifier written by `python training/train_classifier.py`.
  Until it exists, the app uses the placeholder rule-based fallback (clearly labelled on screen).
