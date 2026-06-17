from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, data) -> None:
    ensure_parent(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def compare_arrays(a, b) -> dict:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return {"shape_match": False, "a_shape": list(a.shape), "b_shape": list(b.shape), "max_abs_error": None, "mean_abs_error": None, "relative_error": None}
    diff = np.abs(a.astype("float64") - b.astype("float64"))
    denom = max(float(np.mean(np.abs(b.astype("float64")))), 1e-12)
    return {"shape_match": True, "shape": list(a.shape), "max_abs_error": float(diff.max()) if diff.size else 0.0, "mean_abs_error": float(diff.mean()) if diff.size else 0.0, "relative_error": float(diff.mean() / denom)}
