#!/usr/bin/env python3
"""Generate smoke-training diagnostic visualizations.

These figures are tensor diagnostics for train/validation loss and output-tail
prediction alignment. They are not final restored images or videos.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metric(rows: list[dict[str, str]], key: str, fallback: str | None = None) -> np.ndarray:
    values = []
    for row in rows:
        raw = row.get(key, "")
        if raw == "" and fallback is not None:
            raw = row.get(fallback, "")
        values.append(float(raw) if raw != "" else np.nan)
    return np.asarray(values, dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="experiments/smoke_training/output_tail")
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir
    out = data_dir / "visualizations"
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p_rows = read_csv(data_dir / "pytorch/loss.csv")
    j_rows = read_csv(data_dir / "jittor/loss.csv")
    p_val_rows = read_csv(data_dir / "pytorch/validation_loss.csv")
    j_val_rows = read_csv(data_dir / "jittor/validation_loss.csv")
    p_perf = read_csv(data_dir / "pytorch/performance_log.csv")
    j_perf = read_csv(data_dir / "jittor/performance_log.csv")

    p_steps = metric(p_rows, "step")
    j_steps = metric(j_rows, "step")
    p_train = metric(p_rows, "train_loss", "loss")
    j_train = metric(j_rows, "train_loss", "loss")
    p_val_steps = metric(p_val_rows, "eval_step")
    j_val_steps = metric(j_val_rows, "eval_step")
    p_val = metric(p_val_rows, "val_loss")
    j_val = metric(j_val_rows, "val_loss")

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    ax.plot(p_steps, p_train, label="PyTorch train", linewidth=2)
    ax.plot(j_steps, j_train, label="Jittor train", linewidth=2)
    ax.plot(p_val_steps, p_val, "o-", label="PyTorch val", markersize=3)
    ax.plot(j_val_steps, j_val, "o-", label="Jittor val", markersize=3)
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE loss")
    ax.set_title("Train/Validation Loss Alignment")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "train_val_loss_curve.png")
    plt.close(fig)

    n = min(len(p_train), len(j_train))
    train_gap = np.abs(p_train[:n] - j_train[:n])
    p_val_interp = np.interp(np.arange(n), p_val_steps, p_val)
    j_val_interp = np.interp(np.arange(n), j_val_steps, j_val)
    val_gap = np.abs(p_val_interp - j_val_interp)
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    ax.plot(np.arange(n), train_gap, label="train absolute gap")
    ax.plot(np.arange(n), val_gap, label="validation absolute gap")
    ax.set_xlabel("Step")
    ax.set_ylabel("Absolute loss gap")
    ax.set_title("PyTorch/Jittor Loss Gap")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "loss_gap_curve.png")
    plt.close(fig)

    rel_train_gap = train_gap / np.maximum(np.maximum(np.abs(p_train[:n]), np.abs(j_train[:n])), 1e-12)
    rel_val_gap = val_gap / np.maximum(np.maximum(np.abs(p_val_interp), np.abs(j_val_interp)), 1e-12)
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    ax.plot(np.arange(n), rel_train_gap, label="train relative gap")
    ax.plot(np.arange(n), rel_val_gap, label="validation relative gap")
    ax.set_xlabel("Step")
    ax.set_ylabel("Relative loss gap")
    ax.set_title("Relative PyTorch/Jittor Loss Gap")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "relative_loss_gap_curve.png")
    plt.close(fig)

    targets = np.load(data_dir / "smoke_targets.npy").astype("float32")
    val_indices = np.load(data_dir / "val_indices.npy").astype("int64")
    val_target = targets[val_indices]
    p_pred = np.load(data_dir / "pytorch/val_predictions.npy").astype("float32")
    j_pred = np.load(data_dir / "jittor/val_predictions.npy").astype("float32")
    target_map = val_target[0, 0]
    p_map = p_pred[0, 0]
    j_map = j_pred[0, 0]

    def heatmap_grid(path: Path, title: str, items: list[tuple[str, np.ndarray]]) -> None:
        fig, axes = plt.subplots(1, len(items), figsize=(4.2 * len(items), 3.8), dpi=150)
        if len(items) == 1:
            axes = [axes]
        for ax, (name, arr) in zip(axes, items):
            im = ax.imshow(arr, cmap="viridis")
            ax.set_title(name)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    heatmap_grid(out / "prediction_target_heatmap.png", "Validation Target vs Predictions", [
        ("target ch0", target_map),
        ("PyTorch pred ch0", p_map),
        ("Jittor pred ch0", j_map),
    ])
    heatmap_grid(out / "pytorch_jittor_prediction_heatmap.png", "PyTorch/Jittor Prediction Alignment", [
        ("PyTorch pred ch0", p_map),
        ("Jittor pred ch0", j_map),
        ("abs diff", np.abs(p_map - j_map)),
    ])
    heatmap_grid(out / "prediction_error_heatmap.png", "Prediction Error Diagnostics", [
        ("|PyTorch-target|", np.abs(p_map - target_map)),
        ("|Jittor-target|", np.abs(j_map - target_map)),
        ("|PyTorch-Jittor|", np.abs(p_map - j_map)),
    ])

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    ax.plot(metric(p_perf, "step"), metric(p_perf, "elapsed_sec"), label="PyTorch")
    ax.plot(metric(j_perf, "step"), metric(j_perf, "elapsed_sec"), label="Jittor")
    ax.set_xlabel("Step")
    ax.set_ylabel("Elapsed sec")
    ax.set_title("Smoke Training Step Time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "performance_step_time.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    ax.plot(metric(p_perf, "step"), metric(p_perf, "samples_per_sec"), label="PyTorch")
    ax.plot(metric(j_perf, "step"), metric(j_perf, "samples_per_sec"), label="Jittor")
    ax.set_xlabel("Step")
    ax.set_ylabel("Samples/sec")
    ax.set_title("Smoke Training Throughput")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "performance_samples_per_sec.png")
    plt.close(fig)

    print("TADSR_SMOKE_TRAINING_TRAIN_VAL_LOSS_CURVE: PASS")
    print("TADSR_SMOKE_TRAINING_LOSS_GAP_CURVE: PASS")
    print("TADSR_SMOKE_TRAINING_TENSOR_VISUALIZATION: PASS")
    print("TADSR_SMOKE_TRAINING_PERFORMANCE_VISUALIZATION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
