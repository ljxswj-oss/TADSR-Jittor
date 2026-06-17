#!/usr/bin/env python3
"""Plot and compare PyTorch/Jittor smoke-training loss curves."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
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
    return np.array(values, dtype=np.float64)


def finite_pairs(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    aa = a[:n]
    bb = b[:n]
    mask = np.isfinite(aa) & np.isfinite(bb)
    return aa[mask], bb[mask]


def corr(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = finite_pairs(a, b)
    if len(aa) < 2 or float(np.std(aa)) == 0.0 or float(np.std(bb)) == 0.0:
        return 1.0 if len(aa) else 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def has_no_nan_inf(rows: list[dict[str, str]]) -> bool:
    losses = metric(rows, "train_loss", "loss")
    statuses = [row.get("nan_inf_status", "PASS") for row in rows]
    return bool(np.isfinite(losses).all() and all(status == "PASS" for status in statuses))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch-log", required=True)
    parser.add_argument("--jittor-log", required=True)
    parser.add_argument("--output-dir", default="experiments/smoke_training/output_tail")
    args = parser.parse_args()

    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pytorch_rows = read_csv(ROOT / args.pytorch_log)
    jittor_rows = read_csv(ROOT / args.jittor_log)
    p_val_rows = read_csv((ROOT / args.pytorch_log).parent / "validation_loss.csv")
    j_val_rows = read_csv((ROOT / args.jittor_log).parent / "validation_loss.csv")

    p_steps = metric(pytorch_rows, "step")
    p_train = metric(pytorch_rows, "train_loss", "loss")
    p_val_inline = metric(pytorch_rows, "val_loss")
    j_steps = metric(jittor_rows, "step")
    j_train = metric(jittor_rows, "train_loss", "loss")
    j_val_inline = metric(jittor_rows, "val_loss")
    p_val_steps = metric(p_val_rows, "eval_step")
    p_val = metric(p_val_rows, "val_loss")
    j_val_steps = metric(j_val_rows, "eval_step")
    j_val = metric(j_val_rows, "val_loss")

    p_drop = (p_train[0] - p_train[-1]) / max(abs(p_train[0]), 1e-12)
    j_drop = (j_train[0] - j_train[-1]) / max(abs(j_train[0]), 1e-12)
    p_val_drop = (p_val[0] - p_val[-1]) / max(abs(p_val[0]), 1e-12)
    j_val_drop = (j_val[0] - j_val[-1]) / max(abs(j_val[0]), 1e-12)
    no_nan_inf = has_no_nan_inf(pytorch_rows) and has_no_nan_inf(jittor_rows)
    final_train_gap = abs(p_train[-1] - j_train[-1]) / max(abs(p_train[-1]), abs(j_train[-1]), 1e-12)
    final_val_gap = abs(p_val[-1] - j_val[-1]) / max(abs(p_val[-1]), abs(j_val[-1]), 1e-12)
    train_corr = corr(p_train, j_train)
    val_corr = corr(p_val, j_val)
    both_decrease = p_train[-1] < p_train[0] and j_train[-1] < j_train[0]
    val_decrease = p_val[-1] < p_val[0] and j_val[-1] < j_val[0]
    trend_status = "PASS" if (
        no_nan_inf and both_decrease and val_decrease and p_drop >= 0.2 and j_drop >= 0.2 and final_train_gap < 0.02 and final_val_gap < 0.02 and train_corr > 0.95 and val_corr > 0.95
    ) else (
        "PARTIAL" if no_nan_inf and both_decrease and p_drop >= 0.05 and j_drop >= 0.05 else "FAIL"
    )

    comparison_rows = []
    for idx in range(max(len(pytorch_rows), len(jittor_rows))):
        comparison_rows.append(
            {
                "index": idx,
                "pytorch_step": int(p_steps[idx]) if idx < len(p_steps) else "",
                "pytorch_train_loss": float(p_train[idx]) if idx < len(p_train) else "",
                "pytorch_val_loss": float(p_val_inline[idx]) if idx < len(p_val_inline) and np.isfinite(p_val_inline[idx]) else "",
                "jittor_step": int(j_steps[idx]) if idx < len(j_steps) else "",
                "jittor_train_loss": float(j_train[idx]) if idx < len(j_train) else "",
                "jittor_val_loss": float(j_val_inline[idx]) if idx < len(j_val_inline) and np.isfinite(j_val_inline[idx]) else "",
            }
        )
    write_csv(out / "loss_curve.csv", comparison_rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=150)
    ax.plot(p_steps, p_train, label="PyTorch train", linewidth=2)
    ax.plot(j_steps, j_train, label="Jittor train", linewidth=2)
    ax.plot(p_val_steps, p_val, "o-", label="PyTorch val", linewidth=2, markersize=3)
    ax.plot(j_val_steps, j_val, "o-", label="Jittor val", linewidth=2, markersize=3)
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE loss")
    ax.set_title("TADSR Output-Tail Conv-Out Smoke Training")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "loss_curve.png")
    fig.savefig(out / "train_val_loss_curve.png")
    plt.close(fig)

    p_perf = read_csv((ROOT / args.pytorch_log).parent / "performance_log.csv")
    j_perf = read_csv((ROOT / args.jittor_log).parent / "performance_log.csv")
    comparison = {
        "status": trend_status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pytorch_train_initial_loss": float(p_train[0]),
        "pytorch_train_final_loss": float(p_train[-1]),
        "pytorch_train_loss_drop_ratio": float(p_drop),
        "pytorch_val_initial_loss": float(p_val[0]),
        "pytorch_val_final_loss": float(p_val[-1]),
        "pytorch_val_loss_drop_ratio": float(p_val_drop),
        "jittor_train_initial_loss": float(j_train[0]),
        "jittor_train_final_loss": float(j_train[-1]),
        "jittor_train_loss_drop_ratio": float(j_drop),
        "jittor_val_initial_loss": float(j_val[0]),
        "jittor_val_final_loss": float(j_val[-1]),
        "jittor_val_loss_drop_ratio": float(j_val_drop),
        "final_loss_relative_gap": float(final_train_gap),
        "final_train_loss_relative_gap": float(final_train_gap),
        "final_val_loss_relative_gap": float(final_val_gap),
        "train_loss_correlation": float(train_corr),
        "val_loss_correlation": float(val_corr),
        "trend_alignment_status": trend_status,
        "no_nan_inf": bool(no_nan_inf),
        "pytorch_step_time_mean": float(np.mean(metric(p_perf, "elapsed_sec"))),
        "jittor_step_time_mean": float(np.mean(metric(j_perf, "elapsed_sec"))),
        "pytorch_samples_per_sec_mean": float(np.mean(metric(p_perf, "samples_per_sec"))),
        "jittor_samples_per_sec_mean": float(np.mean(metric(j_perf, "samples_per_sec"))),
        "selected_gpu": "none",
        "cpu_fallback": True,
        "full_tadsr_training": False,
        "full_inference_executed": False,
        "image_saved": False,
        "video_saved": False,
    }
    (out / "smoke_training_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")

    report = [
        "# TADSR Output-Tail Conv-Out Smoke Training Report",
        "",
        f"Status: **{trend_status}**",
        "",
        "This is small-data training-pipeline validation for a real TADSR",
        "output-tail `conv_out` head. It is not full TADSR training and does",
        "not run full inference or generate images/videos.",
        "",
        "## Metrics",
        "",
        "| Metric | PyTorch | Jittor |",
        "|---|---:|---:|",
        f"| Train initial loss | {p_train[0]:.8f} | {j_train[0]:.8f} |",
        f"| Train final loss | {p_train[-1]:.8f} | {j_train[-1]:.8f} |",
        f"| Train loss drop ratio | {p_drop:.6f} | {j_drop:.6f} |",
        f"| Validation initial loss | {p_val[0]:.8f} | {j_val[0]:.8f} |",
        f"| Validation final loss | {p_val[-1]:.8f} | {j_val[-1]:.8f} |",
        f"| Validation loss drop ratio | {p_val_drop:.6f} | {j_val_drop:.6f} |",
        f"| Mean step time sec | {comparison['pytorch_step_time_mean']:.6f} | {comparison['jittor_step_time_mean']:.6f} |",
        "",
        f"- Final train loss relative gap: {final_train_gap:.6f}",
        f"- Final validation loss relative gap: {final_val_gap:.6f}",
        f"- Train loss correlation: {train_corr:.6f}",
        f"- Validation loss correlation: {val_corr:.6f}",
        f"- No NaN/Inf: {no_nan_inf}",
        "- Loss curve: `experiments/smoke_training/output_tail/train_val_loss_curve.png`",
    ]
    (out / "smoke_training_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print("TADSR_SMOKE_TRAINING_LOSS_LOG: PASS")
    print("TADSR_SMOKE_TRAINING_VALIDATION_LOSS_LOG: PASS")
    print("TADSR_SMOKE_TRAINING_LOSS_CURVE: PASS")
    print("TADSR_SMOKE_TRAINING_TRAIN_VAL_LOSS_CURVE: PASS")
    print("TADSR_SMOKE_TRAINING_PERFORMANCE_LOG: PASS")
    print(f"TADSR_SMOKE_TRAINING_PYTORCH_JITTOR_LOSS_ALIGNMENT: {trend_status}")
    print(f"final_train_loss_relative_gap: {final_train_gap:.6f}")
    print(f"final_val_loss_relative_gap: {final_val_gap:.6f}")
    return 0 if trend_status in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
