#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

TARGETS = [
    {"name": "unet", "source_dir": "unet", "candidates": ["diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin"], "target": "unet_weights.npz"},
    {"name": "vae", "source_dir": "vae", "candidates": ["diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin"], "target": "vae_weights.npz"},
    {"name": "time_vae", "source_dir": "time_vae", "candidates": ["diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin"], "target": "time_vae_weights.npz"},
    {"name": "text_encoder", "source_dir": "text_encoder", "candidates": ["model.safetensors", "pytorch_model.bin"], "target": "text_encoder_weights.npz"},
    {"name": "bert_base_uncased", "source_dir": "bert-base-uncased", "candidates": ["pytorch_model.bin", "model.safetensors"], "target": "bert_base_uncased_weights.npz"},
]


def sanitize_key(key: str) -> str:
    return key.replace("/", "__").replace(".", "__").replace(":", "__")


def array_stats(array: np.ndarray) -> Dict[str, object]:
    x = array.astype(np.float64, copy=False)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "mean": float(x.mean()) if x.size else 0.0,
        "std": float(x.std()) if x.size else 0.0,
        "min": float(x.min()) if x.size else 0.0,
        "max": float(x.max()) if x.size else 0.0,
        "numel": int(array.size),
        "nbytes": int(array.nbytes),
    }


def write_npz_stream(target: Path, arrays: Iterable[Tuple[str, np.ndarray]]) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(target, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for key, array in arrays:
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(array), allow_pickle=False)
            zf.writestr(key + ".npy", buffer.getvalue())
            count += 1
    return count


def flatten_torch_object(obj, prefix: str = ""):
    if hasattr(obj, "state_dict"):
        obj = obj.state_dict()
    if isinstance(obj, dict):
        for k, v in obj.items():
            name = f"{prefix}.{k}" if prefix else str(k)
            if hasattr(v, "detach"):
                yield name, v.detach().cpu().numpy()
            elif isinstance(v, dict):
                yield from flatten_torch_object(v, name)


def iter_safetensors(path: Path):
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            yield key, f.get_tensor(key).detach().cpu().numpy()


def iter_torch(path: Path):
    import torch
    obj = torch.load(path, map_location="cpu")
    yield from flatten_torch_object(obj)


def find_source(weights_dir: Path, spec: Dict[str, object]) -> Path | None:
    base = weights_dir / str(spec["source_dir"])
    for candidate in spec["candidates"]:
        p = base / str(candidate)
        if p.exists():
            return p
    return None


def convert_one(weights_dir: Path, output_dir: Path, spec: Dict[str, object]) -> Dict[str, object]:
    source = find_source(weights_dir, spec)
    item = {"name": spec["name"], "source": str(source) if source else None, "target": str(output_dir / str(spec["target"])), "status": "PENDING", "tensors": []}
    if source is None:
        item["status"] = "MISSING_SOURCE"
        return item
    if source.name.endswith(".safetensors"):
        iterator = iter_safetensors(source)
        item["source_format"] = "safetensors"
    elif source.suffix.lower() in {".bin", ".pth", ".pkl"}:
        iterator = iter_torch(source)
        item["source_format"] = "torch"
    else:
        item["status"] = "UNSUPPORTED_SOURCE_FORMAT"
        return item
    arrays = []
    try:
        for original_key, array in iterator:
            npz_key = sanitize_key(original_key)
            arr = np.asarray(array)
            item["tensors"].append({"original_key": original_key, "npz_key": npz_key, "source_file": str(source), **array_stats(arr)})
            arrays.append((npz_key, arr))
        item["tensor_count"] = write_npz_stream(Path(item["target"]), arrays)
        item["target_size_bytes"] = Path(item["target"]).stat().st_size
        item["status"] = "converted"
    except Exception as exc:
        item["status"] = "FAILED"
        item["error"] = repr(exc)
    return item


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_dir", default="/mnt/data/sj/checkpoints/TADSR/preset/weights")
    parser.add_argument("--output_dir", default="/mnt/data/sj/checkpoints/TADSR/jittor_converted/diffusers")
    parser.add_argument("--report_dir", default="experiments/full_repro/weights")
    args = parser.parse_args()
    weights_dir = Path(args.weights_dir)
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "PASS", "weights_dir": str(weights_dir), "output_dir": str(output_dir), "note": "NPZ files are stored outside the git repository under /mnt/data/sj.", "converted_files": []}
    key_rows = []
    for spec in TARGETS:
        item = convert_one(weights_dir, output_dir, spec)
        manifest["converted_files"].append(item)
        for tensor in item.get("tensors", []):
            key_rows.append({"component": item["name"], "target": item["target"], **tensor})
    failed = [x for x in manifest["converted_files"] if x.get("status") != "converted"]
    if failed:
        manifest["status"] = "PARTIAL" if len(failed) < len(TARGETS) else "FAIL"
    (report_dir / "diffusers_conversion_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (report_dir / "diffusers_key_mapping.json").write_text(json.dumps({"rows": key_rows}, indent=2), encoding="utf-8")
    md = ["# Diffusers Weight Conversion Manifest", "", f"Status: **{manifest['status']}**", "", "| Component | Status | Tensor count | Target |", "|---|---|---:|---|"]
    for item in manifest["converted_files"]:
        md.append(f"| `{item['name']}` | {item.get('status')} | {len(item.get('tensors', []))} | `{item.get('target')}` |")
    md += ["", "NPZ files are intentionally stored under `/mnt/data/sj/checkpoints/TADSR/jittor_converted/diffusers/`, not in git."]
    (report_dir / "diffusers_conversion_manifest.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    key_md = ["# Diffusers Key Mapping", "", f"Total tensor rows: {len(key_rows)}", "", "| Component | Original key | NPZ key | Shape | Dtype |", "|---|---|---|---|---|"]
    for row in key_rows[:300]:
        key_md.append(f"| `{row['component']}` | `{row['original_key']}` | `{row['npz_key']}` | `{row['shape']}` | `{row['dtype']}` |")
    (report_dir / "diffusers_key_mapping.md").write_text("\n".join(key_md) + "\n", encoding="utf-8")
    print(f"Diffusers conversion status: {manifest['status']} components={len(manifest['converted_files'])} failed={len(failed)}")
    return 0 if manifest["status"] in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
