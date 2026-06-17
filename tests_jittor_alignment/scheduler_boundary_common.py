from __future__ import annotations

from pathlib import Path

import numpy as np

from jittor_tadsr_full.utils import compare_arrays


ORACLE_DIR = Path("experiments/full_repro/scheduler_alignment/oracle_tensors_scheduler_boundary")


def load_array(name: str) -> np.ndarray:
    return np.load(ORACLE_DIR / name)


def compare_named(name: str, actual, expected, tolerance: float) -> dict:
    result = compare_arrays(actual, expected)
    result["name"] = name
    result["tolerance"] = tolerance
    result["pass"] = bool(result.get("shape_match") and (result.get("max_abs_error") or 0.0) <= tolerance)
    return result
