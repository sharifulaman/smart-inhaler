# স্মার্ট ইনহেলার CV — চালানো, ট্রেনিং ও রেজাল্ট (বাংলা গাইড)

এই প্রজেক্ট ওয়েবক্যামে ইনহেলার, মুখ ও হাত দেখে বলে দেয় ইনহেলার ব্যবহারের ভঙ্গি ঠিক আছে কি না।

> **সতর্কতা:** এটি গবেষণা/ডেমো টুল, চিকিৎসা-সরঞ্জাম নয়। রুল-ভিত্তিক থ্রেশহোল্ডগুলো ক্লিনিক্যালি যাচাই করা নয়।

---

## ১. নতুন ডিভাইসে চালানোর নিয়ম (Windows)

### প্রয়োজন
- Python **3.10 – 3.12** (README-র সুপারিশ; 3.13-এও এই প্রজেক্টে চলেছে)
- একটি ওয়েবক্যাম, এবং ইন্টারনেট (প্রথমবার মডেল নামাতে)
- `git` (CLIP প্যাকেজ নামাতে লাগতে পারে)

### ধাপ
```powershell
# ১) প্রজেক্ট ফোল্ডারে যান
cd smart-inhaler-cv

# ২) ভার্চুয়াল এনভায়রনমেন্ট বানান ও চালু করুন
python -m venv .venv
.venv\Scripts\Activate.ps1

# ৩) লাইব্রেরি ইনস্টল
pip install -r requirements.txt

# ৪) MediaPipe মডেল (মুখ/হাত/পোজ) নামান — models/mediapipe/ এ থাকলে এড়িয়ে যাবে
python scripts/download_models.py

# ৫) অ্যাপ চালান
python -m src.app
```

### প্রথমবার চালালে কী হবে
- `models/best.pt` না থাকলে অ্যাপ **zero-shot YOLO-World** দিয়ে ইনহেলার খোঁজে। এর জন্য প্রথমবার
  `yolov8s-worldv2.pt` (~৩৪০ MB) অটো ডাউনলোড হয় এবং `models/` এ রাখা হয়। `clip` ও `lap`
  প্যাকেজও অটো ইনস্টল হতে পারে।
- ⚠️ **`--camera-only` দেবেন না।** ওই অপশন ইনহেলার ডিটেকশন বন্ধ করে দেয়।

### দরকারি অপশন
| কমান্ড | কাজ |
|---|---|
| `python -m src.app --source 1` | অন্য ক্যামেরা (১, ২ …) |
| `python -m src.app --source clip.mp4` | ভিডিও ফাইল থেকে |
| `python -m src.app --source rtsp://...` | ফোন/IP ক্যামেরা |
| `python -m src.app --weights models/best.pt` | নির্দিষ্ট ইনহেলার মডেল |
| `python -m src.app --no-scan-log` | স্ক্যান CSV সেভ বন্ধ |
| `python -m src.app --save-video out.mp4` | রেজাল্টসহ ভিডিও সেভ |

**কিবোর্ড:** `q` বা `Esc` = বন্ধ · `r` = ফিচার রেকর্ড চালু/বন্ধ · `s` = স্ক্রিনশট (`runs/screenshots/`)

---

## ২. ইনহেলার কীভাবে ডিটেক্ট হয়

প্রতিটি ফ্রেম এই ধাপগুলো পার হয়:

```
ক্যামেরা ফ্রেম
   │
   ├─► ১. ইনহেলার ডিটেক্টর (YOLO)          → ইনহেলারের বক্স, কনফিডেন্স, অক্ষের কোণ
   ├─► ২. মুখ (MediaPipe Face)             → মুখ ও ঠোঁটের পয়েন্ট
   ├─► ৩. হাত + শরীর (MediaPipe Hands/Pose)→ কোন হাত ইনহেলার ধরেছে
   │
   ├─► ৪. ফিচার: মাউথপিস ↔ মুখের দূরত্ব, ইনহেলারের কোণ, মুখ খোলা কি না…
   └─► ৫. ক্লাসিফায়ার (প্রায় ২০ ফ্রেমের উইন্ডো) → টেকনিক রেজাল্ট
```

**ইনহেলার খোঁজার দুই পদ্ধতি** (`configs/config.yaml` → `detector`):
1. **ট্রেনড মডেল** `models/best.pt` থাকলে সেটি ব্যবহার হয় (সবচেয়ে নির্ভুল)।
2. না থাকলে **zero-shot (YOLO-World)**: কোনো ট্রেনিং ডেটা ছাড়াই লেখা প্রম্পট
   (`inhaler`, `asthma inhaler`, `blue plastic inhaler`) দিয়ে খোঁজে। কনফিডেন্স কম (≈০.০৫–০.১) এবং
   হাতে ঢাকা পড়লে হারাতে পারে। `hold_frames: 8` — ডিটেকশন হারালে ৮ ফ্রেম পর্যন্ত আগের বক্স ধরে রাখে।

**ট্র্যাকিং:** ByteTrack দিয়ে ফ্রেম থেকে ফ্রেমে একই ইনহেলারকে একই `track_id` দেওয়া হয়।

**ভিজিবিলিটি গেট:** ইনহেলার/মুখ/মুখগহ্বর যথেষ্ট দেখা না গেলে (`min_valid_fraction`) অ্যাপ অনুমান না করে
**"Unable to Determine"** দেখায়।

---

## ৩. স্ক্রিনে রেজাল্ট কীভাবে দেখায়

বাঁ-দিকের প্যানেলে (HUD) প্রতিটি সারি:

| সারি | অর্থ | রং |
|---|---|---|
| **Inhaler** | `inhaler 0.62` = পাওয়া গেছে + কনফিডেন্স; `(held)` = সাময়িক হারিয়েছে, আগের বক্স ধরে রাখা; `not found` = নেই | সবুজ / হলুদ / লাল |
| **Face** | `ok 1.00` = মুখ স্পষ্ট | সবুজ |
| **Mouth** | `ok 0.80` = ঠোঁট স্পষ্ট; `low reliability` = কম নির্ভরযোগ্য | সবুজ / লাল |
| **Hand** | `holding` = ইনহেলার ধরে আছে; `detected (1), not holding inhaler` = হাত দেখা গেছে কিন্তু ইনহেলারের কাছে নয়; `not found` | সবুজ / হলুদ |
| **Mouthpiece-mouth** | মাউথপিস থেকে মুখের দূরত্ব (মুখের প্রস্থের এককে); ছোট = কাছে | সাদা |
| **Inhaler angle** | ইনহেলারের কোণ (ডিগ্রি) | সাদা |
| **Visibility (min)** | মুখ/ঠোঁট/ইনহেলারের সবচেয়ে কম দৃশ্যমানতা (০–১) | সাদা |
| **Technique** | চূড়ান্ত রেজাল্ট (নিচে) | রঙিন |
| শেষ লাইন | FPS, লেটেন্সি (ms), মডেলের উৎস | — |

**ছবিতে আঁকা জিনিস:** ইনহেলারের চারপাশে বক্স + নাম/কনফিডেন্স · মুখ ও ঠোঁটের রেখা · হাতের কঙ্কাল ·
মাউথপিস ও মুখের মধ্যে লিংক।

**Technique রেজাল্ট:**
| রেজাল্ট | রং | মানে |
|---|---|---|
| **Correct** | সবুজ | ভঙ্গি দেখে ঠিক মনে হচ্ছে |
| **Wrong Mouth Position** | লাল | ইনহেলার মুখের দিকে আনুন, ঠোঁট মাউথপিসে চেপে ধরুন |
| **Wrong Inhaler Position** | লাল | ইনহেলার সোজা ধরুন, মাউথপিস মুখের ভেতরে |
| **Unable to Determine** | হলুদ | মুখ/ইনহেলার যথেষ্ট দেখা যায়নি বা অনিশ্চিত |

ট্রেনড ক্লাসিফায়ার না থাকলে অ্যাপ শুরুতে লেখে: `classifier mode: rule-based (placeholder rule thresholds)` —
অর্থাৎ ফলাফল শুধু আনুমানিক।

---

## ৪. স্ক্যান করলে CSV সেভ (অটো)

অ্যাপ চালালেই প্রতিবার একটি ফাইল তৈরি হয়: **`data/scans/scan_তারিখ_সময়.csv`**
(বন্ধ করতে `--no-scan-log`)। প্রতি প্রসেস-করা ফ্রেমে একটি সারি:

| কলাম | অর্থ |
|---|---|
| `frame`, `time_s` | ফ্রেম নম্বর, শুরু থেকে সেকেন্ড |
| `inhaler_found`, `inhaler_conf` | ১/০, কনফিডেন্স |
| `inhaler_x1,y1,x2,y2` | বক্সের পিক্সেল স্থানাঙ্ক |
| `face_found`, `mouth_found` | ১/০ |
| `hand_holding` | ১ = ইনহেলার ধরে আছে |
| `mouthpiece_mouth_dist` | মাউথপিস ↔ মুখ দূরত্ব |
| `inhaler_angle_deg` | ইনহেলারের কোণ |
| `technique`, `technique_conf` | রেজাল্ট ও কনফিডেন্স |
| `source` | `warmup` (প্রথম ~২০ ফ্রেম), `gate`, `rule-based`, `model` |

> প্রথম ~২০ ফ্রেম `warmup`, তখন technique সবসময় "Unable to Determine" — স্বাভাবিক।

---

## ৫. ট্রেনিং ডেটা CSV-তে কোথায় থাকে

| ফোল্ডার/ফাইল | কী আছে | কীভাবে তৈরি হয় |
|---|---|---|
| `data/annotations.csv` | ভিডিওর তালিকা + লেবেল | হাতে লেখা (টেমপ্লেট: `data/annotations_template.csv`) |
| `data/recordings/*.csv` | লাইভ রেকর্ড করা ফ্রেম-ফিচার | `python -m src.app --record ...` |
| `data/features/*.csv` | ভিডিও থেকে বের করা ফ্রেম-ফিচার | `training/extract_features.py` |
| `data/reference_inhalers/` | ৬টি রেফারেন্স ইনহেলার ছবি + `boxes.csv` (বক্স) | হাতে |
| `data/synthetic_inhaler/` | কৃত্রিম ইনহেলার ডেটা (ছবি+লেবেল) | `training/make_synthetic.py` |
| `data/scans/*.csv` | লাইভ স্ক্যানের রেজাল্ট লগ | অ্যাপ নিজে |

**`annotations.csv` কলাম:**
`person_id, video_path, label, error_type, angle, lighting, distance, start_frame, end_frame`

```csv
person_id,video_path,label,error_type,angle,lighting,distance,start_frame,end_frame
P01,data/recordings/P01_correct_front_bright_near.mp4,correct,none,front,bright,near,0,
P01,data/recordings/P01_wrongmouth_front_bright_near.mp4,wrong_mouth,mouth_too_far,front,bright,near,0,
```
- `label`: `correct` | `wrong_mouth` | `wrong_inhaler`
- `person_id`: নাম নয়, শুধু বেনামী কোড (P01, P02…)
- `end_frame` ফাঁকা = পুরো ভিডিও

**ডেটা সংগ্রহের নিয়ম** (`docs/DATASET_PROTOCOL.md`): কমপক্ষে ১০ জন, প্রত্যেকের ৩ ধরন × ৩+ টেক,
২+ অ্যাঙ্গেল, ২+ আলো, ২+ দূরত্ব; প্লাসিবো/খালি ইনহেলার; **সম্মতি** নিয়ে রেকর্ডিং।

---

## ৬. ট্রেনিং কমান্ড

> সব কমান্ড প্রজেক্টের মূল ফোল্ডার থেকে, `.venv` চালু রেখে চালান।

### ক) ইনহেলার ডিটেক্টর (YOLO) — `models/best.pt`

**ছবি-ডেটা ছাড়া (রেফারেন্স ছবি থেকে কৃত্রিম ডেটা):**
```powershell
python training/make_synthetic.py --out data/synthetic_inhaler --n 600
python training/train_yolo.py --data data/inhaler_synth.yaml --model yolo11n.pt --imgsz 320 --epochs 15 --out models/best_synth.pt
```
- CPU-তে ধীর (≈২ মিনিট/ইপক)। GPU থাকলে `--imgsz 640 --epochs 50` দিন।
- কৃত্রিম ডেটার ভ্যালিডেশন স্কোর **বাড়িয়ে দেখায়** (একই ৬ ছবি থেকে তৈরি)। আসল ওয়েবক্যামে যাচাই করুন:
  `python -m src.app --weights models/best_synth.pt`
- ভালো হলে `models/best_synth.pt` কে `models/best.pt` নামে কপি করুন।
- আসল ব্যাকগ্রাউন্ড ছবি (ইনহেলার ছাড়া ঘর/মুখ/হাত) `data/backgrounds/` এ দিলে মান বাড়ে।

**আসল লেবেল করা ছবি থাকলে (সেরা ফল):** Roboflow/CVAT এ লেবেল করে `data/roboflow` এ রাখুন, তারপর:
```powershell
python training/augment_dataset.py --images data/roboflow/train/images --labels data/roboflow/train/labels --out data/roboflow_aug/train --copies 2
python training/train_yolo.py --data data/inhaler.yaml --model yolo11s.pt --epochs 100
```

### খ) টেকনিক ক্লাসিফায়ার — `models/technique_clf.joblib`

**১) ডেটা সংগ্রহ** (দুইটির যেকোনো একটি):
```powershell
# অপশন A: সরাসরি লাইভ রেকর্ড (ভিডিও থাকে না, শুধু ফিচার CSV)
python -m src.app --record --person-id P03 --label correct --angle front --lighting bright --distance near
#   --label: correct | wrong_mouth | wrong_inhaler

# অপশন B: ভিডিও রেকর্ড করে annotations.csv লিখে ফিচার বের করুন
python training/extract_features.py --annotations data/annotations.csv --out data/features
```

**২) ট্রেনিং** (Leave-One-Subject-Out — কমপক্ষে ৩ জন `person_id` লাগবে):
```powershell
python training/train_classifier.py --data data/features data/recordings --model all
# হাইপারপ্যারামিটার টিউন সহ:   --tune
# শুধু একটি মডেল:               --model rf   (rf | svm | xgb | mlp)
```
ফলাফল: `models/technique_clf.joblib` + রিপোর্ট `evaluation/results/`। এরপর অ্যাপ নিজে থেকেই
`rule-based` ছেড়ে ট্রেনড মডেল ব্যবহার করবে।

**৩) মূল্যায়ন (ঐচ্ছিক):**
```powershell
python evaluation/fps_latency.py        # গতি
python evaluation/audit_detector.py     # ডিটেক্টর যাচাই
python -m pytest -q                     # টেস্ট
```

---

## ৭. সমস্যা ও সমাধান

| সমস্যা | কারণ / সমাধান |
|---|---|
| Inhaler সবসময় `not found` | `--camera-only` দিয়ে চালিয়েছেন? ওটা বাদ দিন। অথবা `.venv` সক্রিয় নয় |
| ইনহেলার ঝিকমিক করে/মাঝে মাঝে হারায় | zero-shot কম নির্ভুল; `configs/config.yaml` এ `prompts` বদলান বা `hold_frames` বাড়ান; ভালো আলো দিন; পুরো ইনহেলার দেখান; ট্রেনড মডেল বানান |
| Hand `not found` | হাত ক্যামেরায় স্পষ্ট নয়, বা ইনহেলার বক্স নেই |
| Technique সবসময় `Unable to Determine` | ইনহেলার না পেলে/মুখ ঢাকা থাকলে এটাই স্বাভাবিক; প্রথম ২০ ফ্রেম warmup |
| খুব ধীর (কম FPS) | অন্য ভারী প্রোগ্রাম বন্ধ করুন; `configs/config.yaml` এ `imgsz` কমান; GPU থাকলে `device: "0"` |
| `FileNotFoundError ... best.pt` | `--weights` দিয়েছেন কিন্তু ফাইল নেই; ফাইল রাখুন বা অপশন বাদ দিন |
| ক্যামেরা খোলে না | `--source 1` বা `2` চেষ্টা করুন; অন্য অ্যাপ (Zoom/Teams) ক্যামেরা ধরে আছে কি দেখুন |
