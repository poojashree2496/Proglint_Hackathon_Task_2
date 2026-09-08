"""Shared helpers used by every script in the training pipeline."""

from pathlib import Path
import random

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPO_ROOT / "training"


def load_config(config_path: Path = None) -> dict:
    config_path = config_path or (TRAINING_ROOT / "config.yaml")
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(preference: str = "auto") -> str:
    try:
        import torch
    except ImportError:
        return "cpu"

    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def yolo_box_valid(cx, cy, w, h, min_w=1e-4, min_h=1e-4) -> bool:
    """A normalized YOLO box is valid if it is inside [0,1] and non-degenerate."""
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
        return False
    if w <= min_w or h <= min_h:
        return False
    if cx - w / 2 < -0.02 or cx + w / 2 > 1.02:
        return False
    if cy - h / 2 < -0.02 or cy + h / 2 > 1.02:
        return False
    return True


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


PERSON_CLASS_ID = 0  # single-class dataset: 0 == person, always
