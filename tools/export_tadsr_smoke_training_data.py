#!/usr/bin/env python3
"""Export deterministic output-tail conv_out smoke-training pairs.

This is an offline data-preparation utility for a small-data training
alignment experiment. It uses real TADSR output-tail tensors and the official
PyTorch output-tail conv_out weights, but it does not run full TADSR training,
the denoising loop, VAE decode, or any image/video generation path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STRICT_PYTHON = Path("/mnt/data/sj/venvs/tadsr_official_strict_cu118/bin/python")


def maybe_reexec_strict() -> None:
    if os.environ.get("TADSR_SKIP_STRICT_REEXEC") == "1":
        return
    if STRICT_PYTHON.exists() and Path(sys.executable).resolve() != STRICT_PYTHON.resolve():
        os.execv(str(STRICT_PYTHON), [str(STRICT_PYTHON), *sys.argv])


def load_base_features() -> list[np.ndarray]:
    candidates = [
        ROOT / "experiments/full_repro/unet_alignment/oracle_tensors_unet_output_tail/synthetic_output_tail_act_output.npy",
        ROOT / "experiments/full_repro/unet_alignment/oracle_tensors_unet_output_tail/entry_output_tail_act_output.npy",
    ]
    arrays: list[np.ndarray] = []
    for path in candidates:
        if path.exists():
            arr = np.load(path).astype("float32")
            if arr.ndim != 4 or arr.shape[1] != 320:
                raise ValueError(f"Unexpected output-tail feature shape in {path}: {arr.shape}")
            arrays.append(arr)
    if not arrays:
        raise FileNotFoundError("No output-tail activation oracle tensors found.")
    return arrays


def make_features(num_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bases = load_base_features()
    samples = []
    for idx in range(num_samples):
        base = bases[idx % len(bases)][0]
        rolled = np.roll(base, shift=(idx % 5) - 2, axis=1)
        rolled = np.roll(rolled, shift=((idx // 5) % 5) - 2, axis=2)
        scale = 1.0 + 0.015 * ((idx % 7) - 3)
        noise = rng.normal(0.0, 0.01, size=base.shape).astype("float32")
        samples.append((rolled * scale + noise).astype("float32"))
    return np.stack(samples, axis=0)


def main() -> int:
    maybe_reexec_strict()
    import torch
    import torch.nn.functional as F

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--train-samples", type=int, default=24)
    parser.add_argument("--val-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", default="experiments/smoke_training/output_tail")
    parser.add_argument("--perturb-std", type=float, default=0.02)
    args = parser.parse_args()

    if args.train_samples + args.val_samples > args.num_samples:
        raise ValueError("train_samples + val_samples must be <= num_samples")
    if args.train_samples <= 0 or args.val_samples <= 0:
        raise ValueError("train_samples and val_samples must both be positive")

    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    weights_path = ROOT / "experiments/full_repro/unet_alignment/converted_unet_output_tail_effective_weights.npz"
    weights = np.load(weights_path)
    official_weight = weights["output_tail_conv_out_weight"].astype("float32")
    official_bias = weights["output_tail_conv_out_bias"].astype("float32")

    features = make_features(args.num_samples, args.seed)
    with torch.no_grad():
        targets = F.conv2d(
            torch.from_numpy(features),
            torch.from_numpy(official_weight),
            torch.from_numpy(official_bias),
            padding=1,
        ).cpu().numpy().astype("float32")

    rng = np.random.default_rng(args.seed + 17)
    initial_weight = (official_weight + rng.normal(0.0, args.perturb_std, size=official_weight.shape)).astype("float32")
    initial_bias = (official_bias + rng.normal(0.0, args.perturb_std, size=official_bias.shape)).astype("float32")

    split_rng = np.random.default_rng(args.seed + 101)
    perm = split_rng.permutation(args.num_samples)
    train_indices = np.sort(perm[: args.train_samples]).astype("int64")
    val_indices = np.sort(perm[args.train_samples: args.train_samples + args.val_samples]).astype("int64")

    np.save(out / "smoke_features.npy", features)
    np.save(out / "smoke_targets.npy", targets)
    np.save(out / "train_indices.npy", train_indices)
    np.save(out / "val_indices.npy", val_indices)
    np.save(out / "initial_conv_weight.npy", initial_weight)
    np.save(out / "initial_conv_bias.npy", initial_bias)

    metadata = {
        "task_name": "tadsr_output_tail_conv_out_smoke_training",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_samples": int(args.num_samples),
        "train_samples": int(args.train_samples),
        "val_samples": int(args.val_samples),
        "train_indices": train_indices.astype(int).tolist(),
        "val_indices": val_indices.astype(int).tolist(),
        "feature_shape": list(features.shape),
        "target_shape": list(targets.shape),
        "train_feature_shape": list(features[train_indices].shape),
        "val_feature_shape": list(features[val_indices].shape),
        "train_target_shape": list(targets[train_indices].shape),
        "val_target_shape": list(targets[val_indices].shape),
        "seed": int(args.seed),
        "perturb_std": float(args.perturb_std),
        "trainable_module": "output_tail_conv_out",
        "trainable_parameters": int(np.prod(official_weight.shape) + np.prod(official_bias.shape)),
        "conv_weight_shape": list(official_weight.shape),
        "conv_bias_shape": list(official_bias.shape),
        "full_tadsr_training": False,
        "full_inference_executed": False,
        "full_denoising_loop_executed": False,
        "image_saved": False,
        "video_saved": False,
        "data_source": "deterministic_feature_target_pairs_from_official_output_tail",
        "source_weight_file": str(weights_path.relative_to(ROOT)),
        "source_feature_files": [
            "experiments/full_repro/unet_alignment/oracle_tensors_unet_output_tail/synthetic_output_tail_act_output.npy",
            "experiments/full_repro/unet_alignment/oracle_tensors_unet_output_tail/entry_output_tail_act_output.npy",
        ],
        "limitation": "Smoke training validates the small training loop, not full TADSR training or production inference.",
    }
    (out / "smoke_training_data_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    summary = [
        "# Output-Tail Conv-Out Smoke Training Data",
        "",
        "Status: **PASS**",
        "",
        f"- Task: `{metadata['task_name']}`",
        f"- Total samples: {args.num_samples}",
        f"- Train/validation split: {args.train_samples}/{args.val_samples}",
        f"- Feature shape: `{features.shape}`",
        f"- Target shape: `{targets.shape}`",
        f"- Train feature shape: `{features[train_indices].shape}`",
        f"- Validation feature shape: `{features[val_indices].shape}`",
        "- Trainable module: output-tail `conv_out` only.",
        "- Full TADSR training: false.",
        "- Full inference executed: false.",
        "- Full denoising loop executed: false.",
        "- Image/video saved: false.",
        "",
        "This dataset is a deterministic feature-target smoke-test set for validating",
        "PyTorch and Jittor training loops on a real TADSR output-tail component.",
        "Raw `.npy` tensors are intentionally ignored by git; committed artifacts keep",
        "only metadata, logs, summaries, and diagnostic plots.",
    ]
    (out / "smoke_training_data_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("TADSR_SMOKE_TRAINING_DATA_PREP: PASS")
    print("TADSR_SMOKE_TRAINING_TRAIN_VAL_SPLIT: PASS")
    print(f"feature_shape: {features.shape}")
    print(f"target_shape: {targets.shape}")
    print(f"train_samples: {len(train_indices)}")
    print(f"val_samples: {len(val_indices)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
