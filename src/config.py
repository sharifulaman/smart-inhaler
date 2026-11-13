"""Config loading and path resolution."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve(path: Union[str, Path]) -> Path:
    """Absolute paths pass through; relative paths use cwd if they exist, else the project root."""
    p = Path(path)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return ROOT / p


def load_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    cfg_path = resolve(path or "configs/config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
