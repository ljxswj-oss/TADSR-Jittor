from __future__ import annotations

from pathlib import Path
from typing import Dict
import json
import numpy as np

DEFAULT_BASE = Path("/mnt/data/sj/checkpoints/TADSR/jittor_converted")
DEFAULT_DIFFUSERS = DEFAULT_BASE / "diffusers"


def load_npz_weights(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Converted Jittor weights not found: {path}")
    return np.load(path)


def summarize_npz(path: str | Path, max_keys: int = 20) -> Dict[str, object]:
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "status": "MISSING", "key_count": 0, "keys": []}
    data = np.load(path)
    keys = list(data.files)
    shapes = {k: list(data[k].shape) for k in keys[:max_keys]}
    dtypes = {k: str(data[k].dtype) for k in keys[:max_keys]}
    return {"path": str(path), "status": "PASS", "key_count": len(keys), "keys": keys[:max_keys], "shapes": shapes, "dtypes": dtypes}


def default_weight_paths() -> Dict[str, Path]:
    return {
        "tadsr": DEFAULT_BASE / "tadsr_weights.npz",
        "DAPE": DEFAULT_BASE / "DAPE_weights.npz",
        "RAM": DEFAULT_BASE / "ram_swin_large_14m_weights.npz",
        "unet": DEFAULT_DIFFUSERS / "unet_weights.npz",
        "vae": DEFAULT_DIFFUSERS / "vae_weights.npz",
        "time_vae": DEFAULT_DIFFUSERS / "time_vae_weights.npz",
        "text_encoder": DEFAULT_DIFFUSERS / "text_encoder_weights.npz",
        "bert_base_uncased": DEFAULT_DIFFUSERS / "bert_base_uncased_weights.npz",
    }


def summarize_all(paths: Dict[str, Path] | None = None) -> Dict[str, object]:
    paths = paths or default_weight_paths()
    components = {name: summarize_npz(path) for name, path in paths.items()}
    status = "PASS" if all(x["status"] == "PASS" for x in components.values()) else "PARTIAL"
    return {"status": status, "components": components}


def load_key_mapping(path: str | Path = "experiments/full_repro/weights/key_mapping.json"):
    p = Path(path)
    if not p.exists():
        return {"status": "MISSING", "rows": []}
    return json.loads(p.read_text())
