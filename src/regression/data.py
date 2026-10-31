"""Dataset loaders. All raw data lives in the top-level ``data/`` folder."""
import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _load_csv(name: str) -> np.ndarray:
    return np.loadtxt(DATA_DIR / name, delimiter=",", skiprows=1)


def load_xy(name: str):
    """Load a CSV whose last column is the target. Returns (X, y)."""
    arr = _load_csv(name)
    X, y = arr[:, :-1], arr[:, -1]
    return (X[:, 0] if X.shape[1] == 1 else X), y


def load_logits(name: str = "q5_logits.csv") -> np.ndarray:
    return _load_csv(name)


def load_json(name: str) -> dict:
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)
