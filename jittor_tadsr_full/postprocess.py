from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image


def tensor_to_pil_minus1_1(tensor: np.ndarray) -> Image.Image:
    arr = np.asarray(tensor)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr * 0.5 + 0.5, 0.0, 1.0)
    return Image.fromarray((arr * 255.0).round().astype("uint8"))


def save_tensor_image(tensor: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil_minus1_1(tensor).save(path)
