#!/usr/bin/env python3
"""Jittor smoke training for the TADSR output-tail conv_out head."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault("JITTOR_HOME", os.path.abspath(".jittor_cache"))
import jittor as jt
from jittor import nn


ROOT = Path(__file__).resolve().parents[1]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def gpu_memory_mb() -> float:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible:
        return 0.0
    first = visible.split(",")[0].strip()
    if first == "":
        return 0.0
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={first}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return float(out.strip().splitlines()[0])
    except Exception:
        return 0.0


def prediction_summary(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    diff = pred.astype("float64") - target.astype("float64")
    return {
        "val_mse": float(np.mean(diff * diff)),
        "val_mae": float(np.mean(np.abs(diff))),
        "val_max_abs_error": float(np.max(np.abs(diff))),
        "val_l2_error": float(np.linalg.norm(diff.reshape(-1))),
        "prediction_mean": float(np.mean(pred)),
        "prediction_std": float(np.std(pred)),
        "target_mean": float(np.mean(target)),
        "target_std": float(np.std(target)),
    }


class OutputTailConvOut(nn.Module):
    def __init__(self, weight: np.ndarray, bias: np.ndarray) -> None:
        super().__init__()
        self.conv = nn.Conv(weight.shape[1], weight.shape[0], 3, padding=1, bias=True)
        self.conv.weight.assign(jt.array(weight))
        self.conv.bias.assign(jt.array(bias))

    def execute(self, x):
        return self.conv(x)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="experiments/smoke_training/output_tail")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--use-cuda", type=int, default=int(os.environ.get("USE_CUDA", "0")))
    parser.add_argument("--eval-interval", type=int, default=10)
    args = parser.parse_args()

    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)
    jt.flags.use_cuda = 1 if args.use_cuda else 0

    data_dir = ROOT / args.data_dir
    out = data_dir / "jittor"
    out.mkdir(parents=True, exist_ok=True)
    features = np.load(data_dir / "smoke_features.npy").astype("float32")
    targets = np.load(data_dir / "smoke_targets.npy").astype("float32")
    init_weight = np.load(data_dir / "initial_conv_weight.npy").astype("float32")
    init_bias = np.load(data_dir / "initial_conv_bias.npy").astype("float32")
    metadata = json.loads((data_dir / "smoke_training_data_metadata.json").read_text(encoding="utf-8"))
    train_indices = np.load(data_dir / "train_indices.npy").astype("int64") if (data_dir / "train_indices.npy").exists() else np.arange(features.shape[0])
    val_indices = np.load(data_dir / "val_indices.npy").astype("int64") if (data_dir / "val_indices.npy").exists() else train_indices

    model = OutputTailConvOut(init_weight, init_bias)
    opt = nn.Adam(model.parameters(), lr=args.lr)
    train_order = np.array(train_indices, copy=True)
    rng = np.random.default_rng(args.seed)

    def eval_loss() -> tuple[float, np.ndarray]:
        vx = jt.array(features[val_indices])
        vy = jt.array(targets[val_indices])
        pred = model(vx)
        loss = ((pred - vy) * (pred - vy)).mean()
        return float(loss.data), np.asarray(pred.data).astype("float32")

    initial_train_loss = None
    initial_val_loss, _ = eval_loss()
    log_lines = [
        f"created_at_utc={datetime.now(timezone.utc).isoformat()}",
        "task=tadsr_output_tail_conv_out_smoke_training",
        "framework=jittor",
        f"use_cuda={args.use_cuda}",
        f"steps={args.steps}",
        f"batch_size={args.batch_size}",
        f"lr={args.lr}",
        f"eval_interval={args.eval_interval}",
        f"train_samples={len(train_indices)}",
        f"val_samples={len(val_indices)}",
        "full_tadsr_training=false",
        "full_inference_executed=false",
        "image_saved=false",
        "video_saved=false",
    ]
    loss_rows: list[dict[str, object]] = []
    val_rows = [{"eval_step": 0, "val_loss": initial_val_loss}]
    perf_rows: list[dict[str, object]] = []
    nan_inf = False

    for step in range(1, args.steps + 1):
        start_idx = ((step - 1) * args.batch_size) % len(train_order)
        if start_idx == 0:
            rng.shuffle(train_order)
        batch_indices = train_order[start_idx:start_idx + args.batch_size]
        if len(batch_indices) < args.batch_size:
            batch_indices = np.concatenate([batch_indices, train_order[: args.batch_size - len(batch_indices)]])
        bx = jt.array(features[batch_indices])
        by = jt.array(targets[batch_indices])
        t0 = time.time()
        pred = model(bx)
        train_loss = ((pred - by) * (pred - by)).mean()
        train_loss_value = float(train_loss.data)
        if not np.isfinite(train_loss_value):
            nan_inf = True
        opt.step(train_loss)
        elapsed = time.time() - t0
        if initial_train_loss is None:
            initial_train_loss = train_loss_value
        val_loss_value: float | str = ""
        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            val_loss_float, _ = eval_loss()
            val_loss_value = val_loss_float
            val_rows.append({"eval_step": step, "val_loss": val_loss_float})
        row = {
            "step": step,
            "train_loss": train_loss_value,
            "loss": train_loss_value,
            "val_loss": val_loss_value,
            "lr": args.lr,
            "elapsed_sec": elapsed,
            "samples_per_sec": args.batch_size / max(elapsed, 1e-12),
            "gpu_memory_used_mb": gpu_memory_mb(),
            "nan_inf_status": "FAIL" if nan_inf else "PASS",
        }
        loss_rows.append(row)
        perf_rows.append(row.copy())
        if step == 1 or step % 10 == 0 or step == args.steps:
            log_lines.append(f"step={step} train_loss={train_loss_value:.8f} val_loss={val_loss_value} elapsed_sec={elapsed:.6f}")

    final_val_loss, val_pred = eval_loss()
    val_targets = targets[val_indices]
    np.save(out / "val_predictions.npy", val_pred)
    pred_summary = prediction_summary(val_pred, val_targets)
    (out / "final_prediction_summary.json").write_text(json.dumps(pred_summary, indent=2) + "\n", encoding="utf-8")

    train_initial_loss = float(initial_train_loss if initial_train_loss is not None else loss_rows[0]["train_loss"])
    train_final_loss = float(loss_rows[-1]["train_loss"])
    val_initial_loss = float(initial_val_loss)
    val_final_loss = float(final_val_loss)
    train_drop = (train_initial_loss - train_final_loss) / max(abs(train_initial_loss), 1e-12)
    val_drop = (val_initial_loss - val_final_loss) / max(abs(val_initial_loss), 1e-12)
    status = "PASS" if not nan_inf and train_final_loss < train_initial_loss and val_final_loss <= val_initial_loss * 1.05 else "FAIL"
    loss_status = "PASS" if status == "PASS" and train_drop >= 0.2 and val_drop >= 0.1 else ("PARTIAL" if status == "PASS" else "FAIL")

    write_csv(out / "loss.csv", loss_rows)
    write_csv(out / "validation_loss.csv", val_rows)
    write_csv(out / "performance_log.csv", perf_rows)
    (out / "training.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    final_weight = np.asarray(model.conv.weight.data)
    final_bias = np.asarray(model.conv.bias.data)
    metrics = {
        "status": status,
        "loss_decrease_status": loss_status,
        "framework": "jittor",
        "use_cuda": bool(args.use_cuda),
        "selected_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "none") if args.use_cuda else "none",
        "cpu_fallback": not bool(args.use_cuda),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "eval_interval": args.eval_interval,
        "trainable_parameters": int(np.prod(init_weight.shape) + np.prod(init_bias.shape)),
        "initial_loss": train_initial_loss,
        "final_loss": train_final_loss,
        "loss_drop_ratio": train_drop,
        "train_initial_loss": train_initial_loss,
        "train_final_loss": train_final_loss,
        "train_loss_drop_ratio": train_drop,
        "val_initial_loss": val_initial_loss,
        "val_final_loss": val_final_loss,
        "val_loss_drop_ratio": val_drop,
        "no_nan_inf": not nan_inf,
        "step_time_mean": float(np.mean([float(r["elapsed_sec"]) for r in perf_rows])),
        "samples_per_sec_mean": float(np.mean([float(r["samples_per_sec"]) for r in perf_rows])),
        "training_scope": "output_tail_conv_out_only",
        "full_tadsr_training": False,
        "full_inference_executed": False,
        "image_saved": False,
        "video_saved": False,
        "data_metadata": metadata,
        "prediction_summary": pred_summary,
    }
    (out / "final_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    delta = {
        "weight_l2_delta": float(np.linalg.norm(final_weight - init_weight)),
        "bias_l2_delta": float(np.linalg.norm(final_bias - init_bias)),
    }
    (out / "final_weight_delta_summary.json").write_text(json.dumps(delta, indent=2) + "\n", encoding="utf-8")

    print(f"TADSR_JITTOR_SMOKE_TRAINING: {status}")
    print(f"TADSR_JITTOR_SMOKE_TRAINING_LOSS_DECREASE: {loss_status}")
    print(f"train_initial_loss: {train_initial_loss:.8f}")
    print(f"train_final_loss: {train_final_loss:.8f}")
    print(f"val_initial_loss: {val_initial_loss:.8f}")
    print(f"val_final_loss: {val_final_loss:.8f}")
    print(f"train_loss_drop_ratio: {train_drop:.6f}")
    print(f"val_loss_drop_ratio: {val_drop:.6f}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
