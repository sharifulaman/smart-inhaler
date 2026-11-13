# Technique dataset protocol

The Roboflow dataset supports object detection only. The technique classifier needs its own data.

## Requirements

* **>= 10 individuals** (aim for 15+), multiple recordings each.
* Per individual: correct use, wrong mouth position, wrong inhaler position - **3+ takes each**.
* Variation: **2+ camera angles** (front, ~30 deg left/right), **2+ lighting conditions**
  (bright, dim), **2+ distances** (near, far).
* Use a **placebo / empty inhaler** (no medication).
* Ethics: obtain informed consent and institutional approval before recording people; store
  videos securely; only use de-identified `person_id` values in CSVs.

## Recording

Option A - record features directly (fast, no video kept):
```bash
python -m src.app --record --person-id P03 --label correct --angle front --lighting bright --distance near
```
Option B - record video, label later, extract features offline (keeps raw data for re-processing):
1. Record clips with any camera; name them `P03_correct_front_bright_near.mp4`.
2. Fill `data/annotations.csv` (template: `data/annotations_template.csv`).
3. `python training/extract_features.py --annotations data/annotations.csv --out data/features`

Annotation columns: `person_id, video_path, label, error_type, angle, lighting, distance,
start_frame, end_frame` (frame range optional). Both options write the same per-frame CSV schema.

## Checklist per recording session

- [ ] Consent form signed; `person_id` assigned
- [ ] Clinician-approved checklist (docs/TECHNIQUE_CHECKLIST.md) shown to the participant
- [ ] 3 classes x 3 takes x (angle, lighting, distance) combinations covered
- [ ] Face, mouth and inhaler visible at the start of every take
- [ ] Hands occasionally cover the inhaler (this is realistic and needed for robustness)
- [ ] Metadata filled for every file

## Splits and evaluation rules

* **Never split by frame or by video.** One person = one group. `train_classifier.py` uses
  Leave-One-Subject-Out on `person_id`.
* Tune hyper-parameters only inside training folds (`--tune` uses nested group CV).
* Report per-subject accuracy as well as the mean; with ~10 subjects the variance is large.
* Keep a final **held-out subject set** (never used during development) if you have 15+ people.
