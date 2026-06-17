from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image


def load_image_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def official_resize(image: Image.Image, process_size: int = 512, upscale: int = 4):
    ori_width, ori_height = image.size
    resize_flag = False
    if ori_width < process_size // upscale or ori_height < process_size // upscale:
        scale = (process_size // upscale) / min(ori_width, ori_height)
        image = image.resize((int(scale * ori_width), int(scale * ori_height)))
        resize_flag = True
    image = image.resize((image.size[0] * upscale, image.size[1] * upscale))
    new_width = image.width - image.width % 8
    new_height = image.height - image.height % 8
    image = image.resize((new_width, new_height), Image.LANCZOS)
    return image, {"original_size": (ori_width, ori_height), "resized_size": (new_width, new_height), "resize_flag": resize_flag}


def image_to_nchw_float(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image).astype("float32") / 255.0
    return np.transpose(arr, (2, 0, 1))[None, ...]


def preprocess_lq(path: str | Path, process_size: int = 512, upscale: int = 4):
    image = load_image_rgb(path)
    resized, meta = official_resize(image, process_size=process_size, upscale=upscale)
    lq_0_1 = image_to_nchw_float(resized)
    return {"image": resized, "lq_0_1": lq_0_1, "lq_minus1_1": lq_0_1 * 2.0 - 1.0, "metadata": meta}


def ram_normalize_from_lq(lq_0_1: np.ndarray) -> np.ndarray:
    im = np.transpose(lq_0_1[0], (1, 2, 0))
    pil = Image.fromarray(np.clip(im * 255.0, 0, 255).astype("uint8"))
    pil = pil.resize((384, 384), Image.BILINEAR)
    arr = np.asarray(pil).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")[None, :, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype="float32")[None, :, None, None]
    return (arr - mean) / std
