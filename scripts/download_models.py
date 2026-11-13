#!/usr/bin/env python
"""Download the MediaPipe Tasks model bundles (face, hand, pose landmarkers) into models/mediapipe/."""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve
from src.mp_models import MODEL_URLS


def main() -> int:
    cfg = load_config()
    rc = 0
    for key, url in MODEL_URLS.items():
        dest = resolve(cfg["mediapipe"][key])
        if dest.exists() and dest.stat().st_size > 100_000:
            print(f"ok   {dest} (already present)")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"get  {url}")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"ok   {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        except Exception as e:  # network / proxy problems
            print(f"FAIL {key}: {e}\n     download it manually and save as {dest}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
