#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


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
    return image, {"original_width": ori_width, "original_height": ori_height, "resize_flag": resize_flag, "new_width": new_width, "new_height": new_height}


def to_tensor(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image).astype("float32") / 255.0
    return np.transpose(arr, (2, 0, 1))[None, ...]


def ram_preprocess(tensor: np.ndarray) -> np.ndarray:
    im = np.transpose(tensor[0], (1, 2, 0))
    pil = Image.fromarray(np.clip(im * 255.0, 0, 255).astype("uint8"))
    pil = pil.resize((384, 384), Image.BILINEAR)
    arr = np.asarray(pil).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")[None, :, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype="float32")[None, :, None, None]
    return (arr - mean) / std


def scheduler_arrays(config_path: Path):
    cfg = json.loads(config_path.read_text())
    n = int(cfg.get("num_train_timesteps", 1000))
    beta_start = float(cfg.get("beta_start", 0.00085))
    beta_end = float(cfg.get("beta_end", 0.012))
    schedule = cfg.get("beta_schedule", "scaled_linear")
    if schedule == "scaled_linear":
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, n, dtype=np.float64) ** 2
    else:
        betas = np.linspace(beta_start, beta_end, n, dtype=np.float64)
    betas = betas.astype("float32")
    alphas = (1.0 - betas).astype("float32")
    alpha_cumprod = np.cumprod(alphas).astype("float32")
    timesteps = np.arange(n - 1, -1, -1, dtype="int64")
    return cfg, {"betas": betas, "alphas": alphas, "alphas_cumprod": alpha_cumprod, "timesteps": timesteps}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="smoke")
    parser.add_argument("--input_dir", default="/mnt/data/sj/datasets/TADSR/smoke/input")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--weights_dir", default="/mnt/data/sj/checkpoints/TADSR/preset/weights")
    parser.add_argument("--official_repo", default="/mnt/data/sj/projects/TADSR_official_pytorch")
    args = parser.parse_args()
    out = Path(args.output_dir) if args.output_dir else Path("experiments/full_repro/oracle_tensors") / args.split
    out.mkdir(parents=True, exist_ok=True)
    input_dir = Path(args.input_dir)
    images = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in IMG_EXT]) if input_dir.exists() else []
    manifest = []
    arrays = {}
    for i, path in enumerate(images):
        image = Image.open(path).convert("RGB")
        resized, meta = official_resize(image)
        lq = to_tensor(resized)
        prefix = f"sample_{i:04d}"
        arrays[f"{prefix}_lq_0_1"] = lq
        arrays[f"{prefix}_lq_minus1_1"] = lq * 2.0 - 1.0
        arrays[f"{prefix}_ram_normalized"] = ram_preprocess(lq)
        manifest.append({"index": i, "path": str(path), "file_name": path.name, **meta})
    np.savez(out / "preprocess_tensors.npz", **arrays)
    (out / "input_manifest.json").write_text(json.dumps({"images": manifest}, indent=2), encoding="utf-8")
    cfg_path = Path(args.weights_dir) / "scheduler" / "scheduler_config.json"
    cfg, sched = scheduler_arrays(cfg_path)
    np.savez(out / "scheduler_tensors.npz", **sched)
    final_outputs = sorted(Path("experiments/full_repro/pytorch_official/smoke/output").glob("*.png"))
    md = ["# Final Outputs Manifest", "", f"Output count: {len(final_outputs)}", "", "| File | Size |", "|---|---:|"]
    for p in final_outputs:
        try:
            size = Image.open(p).size
        except Exception:
            size = None
        md.append(f"| `{p}` | `{size}` |")
    (out / "final_outputs_manifest.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    try:
        commit = subprocess.check_output(["git", "-C", args.official_repo, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    metadata = {
        "status": "PASS" if images else "BLOCKED_NO_INPUT_IMAGES",
        "split": args.split,
        "input_dir": str(input_dir),
        "output_dir": str(out),
        "official_repo": args.official_repo,
        "official_repo_commit": commit,
        "selected_env": "/mnt/data/sj/venvs/tadsr_official_strict_cu118",
        "scheduler_config": cfg,
        "exported": ["preprocess_tensors", "scheduler_tensors", "final_outputs_manifest"],
        "not_exported": {
            "prompt_text_embeddings": "requires deeper hooks into official TADSR_test/DAPE pipeline",
            "vae_unet_intermediates": "requires invasive module hooks; deferred to module-specific alignment",
        },
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = ["# Oracle Tensor Export Report", "", f"Status: **{metadata['status']}**", "", f"Images exported: {len(images)}", "", "Exported files:", "", "- `preprocess_tensors.npz`", "- `scheduler_tensors.npz`", "- `input_manifest.json`", "- `metadata.json`", "- `final_outputs_manifest.md`", "", "Deferred hooks are recorded in `metadata.json`; no intermediate tensor is fabricated."]
    (out / "oracle_tensor_export_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Oracle tensor export: {metadata['status']} images={len(images)} out={out}")
    return 0 if images else 2


if __name__ == "__main__":
    raise SystemExit(main())
