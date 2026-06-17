from __future__ import annotations

from pathlib import Path
import numpy as np


def find_lora_keys(npz_path: str | Path, max_rows: int = 500):
    data = np.load(npz_path)
    rows = []
    for key in data.files:
        low = key.lower()
        if "lora" in low or "adapter" in low:
            rows.append({"key": key, "shape": list(data[key].shape), "dtype": str(data[key].dtype)})
            if len(rows) >= max_rows:
                break
    return rows


def merge_lora_linear(base: np.ndarray, lora_a: np.ndarray, lora_b: np.ndarray, scale: float = 1.0) -> np.ndarray:
    if lora_a.ndim != 2 or lora_b.ndim != 2 or base.ndim != 2:
        raise NotImplementedError("Only 2D linear LoRA merge is implemented in the skeleton helper.")
    delta = lora_b @ lora_a
    if delta.shape != base.shape:
        raise ValueError(f"LoRA delta shape {delta.shape} does not match base {base.shape}")
    return base + scale * delta
