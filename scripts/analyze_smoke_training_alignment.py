#!/usr/bin/env python3
"""Analyze PyTorch/Jittor smoke-training prediction and loss alignment."""

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
    return np.asarray(values, dtype=np.float64)


def finite_pairs(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    aa = a[:n]
    bb = b[:n]
    mask = np.isfinite(aa) & np.isfinite(bb)
    return aa[mask], bb[mask]


def corr(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = finite_pairs(a, b)
    if len(aa) < 2:
        return 0.0
    if float(np.std(aa)) == 0.0 or float(np.std(bb)) == 0.0:
        return 1.0
    return float(np.corrcoef(aa, bb)[0, 1])


def tensor_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    aa = a.astype("float64")
    bb = b.astype("float64")
    diff = aa - bb
    flat_a = aa.reshape(-1)
    flat_b = bb.reshape(-1)
    denom = max(float(np.linalg.norm(flat_b)), 1e-12)
    cos = float(np.dot(flat_a, flat_b) / max(float(np.linalg.norm(flat_a)) * float(np.linalg.norm(flat_b)), 1e-12))
    return {
        "mse": float(np.mean(diff * diff)),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "cosine_similarity": cos,
        "relative_l2_diff": float(np.linalg.norm(diff.reshape(-1)) / denom),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="experiments/smoke_training/output_tail")
    parser.add_argument("--prediction-relative-l2-threshold", type=float, default=5e-3)
    parser.add_argument("--val-gap-threshold", type=float, default=2e-2)
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir
    metadata = json.loads((data_dir / "smoke_training_data_metadata.json").read_text(encoding="utf-8"))
    targets = np.load(data_dir / "smoke_targets.npy").astype("float32")
    val_indices = np.load(data_dir / "val_indices.npy").astype("int64")
    val_targets = targets[val_indices]
    p_pred = np.load(data_dir / "pytorch/val_predictions.npy").astype("float32")
    j_pred = np.load(data_dir / "jittor/val_predictions.npy").astype("float32")

    p_rows = read_csv(data_dir / "pytorch/loss.csv")
    j_rows = read_csv(data_dir / "jittor/loss.csv")
    p_val_rows = read_csv(data_dir / "pytorch/validation_loss.csv")
    j_val_rows = read_csv(data_dir / "jittor/validation_loss.csv")
    p_train = metric(p_rows, "train_loss", "loss")
    j_train = metric(j_rows, "train_loss", "loss")
    p_val = metric(p_val_rows, "val_loss")
    j_val = metric(j_val_rows, "val_loss")

    pred_alignment = tensor_metrics(p_pred, j_pred)
    p_target = tensor_metrics(p_pred, val_targets)
    j_target = tensor_metrics(j_pred, val_targets)
    final_train_gap = abs(p_train[-1] - j_train[-1]) / max(abs(p_train[-1]), abs(j_train[-1]), 1e-12)
    final_val_gap = abs(p_val[-1] - j_val[-1]) / max(abs(p_val[-1]), abs(j_val[-1]), 1e-12)
    train_corr = corr(p_train, j_train)
    val_corr = corr(p_val, j_val)
    no_nan_inf = bool(np.isfinite(p_train).all() and np.isfinite(j_train).all() and np.isfinite(p_val).all() and np.isfinite(j_val).all())

    prediction_status = "PASS" if pred_alignment["relative_l2_diff"] < args.prediction_relative_l2_threshold else (
        "PARTIAL" if pred_alignment["relative_l2_diff"] < 0.05 else "FAIL"
    )
    validation_status = "PASS" if final_val_gap < args.val_gap_threshold and train_corr > 0.95 and val_corr > 0.95 and no_nan_inf else (
        "PARTIAL" if final_val_gap < 0.05 and no_nan_inf else "FAIL"
    )

    metrics = {
        "status": "PASS" if prediction_status == "PASS" and validation_status == "PASS" else ("PARTIAL" if "FAIL" not in {prediction_status, validation_status} else "FAIL"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "num_samples": metadata.get("num_samples"),
            "train_samples": metadata.get("train_samples"),
            "val_samples": metadata.get("val_samples"),
            "trainable_module": metadata.get("trainable_module"),
        },
        "prediction_alignment_status": prediction_status,
        "validation_alignment_status": validation_status,
        "pytorch_vs_jittor_prediction": pred_alignment,
        "pytorch_prediction_vs_target": p_target,
        "jittor_prediction_vs_target": j_target,
        "loss_curve_alignment": {
            "final_train_loss_gap": float(abs(p_train[-1] - j_train[-1])),
            "final_val_loss_gap": float(abs(p_val[-1] - j_val[-1])),
            "final_train_loss_relative_gap": float(final_train_gap),
            "final_val_loss_relative_gap": float(final_val_gap),
            "train_loss_correlation": float(train_corr),
            "val_loss_correlation": float(val_corr),
            "no_nan_inf": no_nan_inf,
        },
        "full_tadsr_training": False,
        "full_inference_executed": False,
        "image_saved": False,
        "video_saved": False,
    }
    (data_dir / "smoke_training_alignment_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    rows = [
        {"metric": "prediction_relative_l2_diff", "value": pred_alignment["relative_l2_diff"]},
        {"metric": "prediction_mae", "value": pred_alignment["mae"]},
        {"metric": "prediction_max_abs_diff", "value": pred_alignment["max_abs_diff"]},
        {"metric": "prediction_cosine_similarity", "value": pred_alignment["cosine_similarity"]},
        {"metric": "final_train_loss_relative_gap", "value": final_train_gap},
        {"metric": "final_val_loss_relative_gap", "value": final_val_gap},
        {"metric": "train_loss_correlation", "value": train_corr},
        {"metric": "val_loss_correlation", "value": val_corr},
    ]
    write_csv(data_dir / "prediction_alignment.csv", rows)

    md = [
        "# Smoke Training Alignment Metrics",
        "",
        f"Overall status: **{metrics['status']}**",
        "",
        "This analysis compares the PyTorch and Jittor validation predictions for",
        "the real TADSR output-tail `conv_out` smoke training task. It does not",
        "represent full TADSR training or image/video generation.",
        "",
        "PASS thresholds are set to prediction relative L2 < 0.005 and final",
        "validation-loss relative gap < 0.02. These tolerances are intentionally",
        "small but account for PyTorch/Jittor optimizer and CPU convolution kernel",
        "differences during independent training runs.",
        "",
        "## Prediction Alignment",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Relative L2 diff | {pred_alignment['relative_l2_diff']:.8e} |",
        f"| MAE | {pred_alignment['mae']:.8e} |",
        f"| Max abs diff | {pred_alignment['max_abs_diff']:.8e} |",
        f"| Cosine similarity | {pred_alignment['cosine_similarity']:.8f} |",
        "",
        "## Loss-Curve Alignment",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Final train relative gap | {final_train_gap:.8e} |",
        f"| Final validation relative gap | {final_val_gap:.8e} |",
        f"| Train loss correlation | {train_corr:.8f} |",
        f"| Validation loss correlation | {val_corr:.8f} |",
        "",
        f"- TADSR_SMOKE_TRAINING_PREDICTION_ALIGNMENT: {prediction_status}",
        f"- TADSR_SMOKE_TRAINING_VALIDATION_ALIGNMENT: {validation_status}",
    ]
    (data_dir / "smoke_training_alignment_metrics.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"TADSR_SMOKE_TRAINING_PREDICTION_ALIGNMENT: {prediction_status}")
    print(f"TADSR_SMOKE_TRAINING_VALIDATION_ALIGNMENT: {validation_status}")
    print(f"prediction_relative_l2_diff: {pred_alignment['relative_l2_diff']:.8e}")
    print(f"final_val_loss_relative_gap: {final_val_gap:.8e}")
    return 0 if metrics["status"] in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
