#!/usr/bin/env python3
"""Validate committed smoke-training evidence artifacts.

This test intentionally does not rerun training. It checks the logs, reports,
visualizations, multi-seed summaries, and safety flags produced by the
small-data output-tail conv_out smoke task.
"""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SMOKE_ROOT = ROOT / "experiments/smoke_training/output_tail"
OUT_JSON = SMOKE_ROOT / "smoke_training_artifacts_test.json"
OUT_MD = SMOKE_ROOT / "smoke_training_artifacts_test.md"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "FAIL", "error": repr(exc)}


def has_rows(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            return len(list(csv.DictReader(f))) > 0
    except Exception:
        return False


def nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def final_audit_marker(name: str) -> str:
    audit = load_json(ROOT / "experiments/final_audit_report.json")
    for row in audit.get("rows", []):
        if row.get("check") == name:
            return str(row.get("status", "BLOCKED"))
    return "BLOCKED"


def staged_tensor_artifacts() -> list[str]:
    try:
        out = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except Exception:
        return []
    return [line for line in out.splitlines() if line.endswith((".npy", ".npz"))]


def main() -> int:
    required_files = [
        SMOKE_ROOT / "smoke_training_data_metadata.json",
        SMOKE_ROOT / "smoke_training_data_summary.md",
        SMOKE_ROOT / "pytorch/loss.csv",
        SMOKE_ROOT / "pytorch/validation_loss.csv",
        SMOKE_ROOT / "pytorch/training.log",
        SMOKE_ROOT / "pytorch/performance_log.csv",
        SMOKE_ROOT / "pytorch/final_metrics.json",
        SMOKE_ROOT / "pytorch/final_prediction_summary.json",
        SMOKE_ROOT / "jittor/loss.csv",
        SMOKE_ROOT / "jittor/validation_loss.csv",
        SMOKE_ROOT / "jittor/training.log",
        SMOKE_ROOT / "jittor/performance_log.csv",
        SMOKE_ROOT / "jittor/final_metrics.json",
        SMOKE_ROOT / "jittor/final_prediction_summary.json",
        SMOKE_ROOT / "loss_curve.png",
        SMOKE_ROOT / "train_val_loss_curve.png",
        SMOKE_ROOT / "smoke_training_comparison.json",
        SMOKE_ROOT / "smoke_training_report.md",
        SMOKE_ROOT / "smoke_training_alignment_metrics.json",
        SMOKE_ROOT / "smoke_training_alignment_metrics.md",
        SMOKE_ROOT / "prediction_alignment.csv",
        SMOKE_ROOT / "visualizations/train_val_loss_curve.png",
        SMOKE_ROOT / "visualizations/loss_gap_curve.png",
        SMOKE_ROOT / "visualizations/relative_loss_gap_curve.png",
        SMOKE_ROOT / "visualizations/prediction_target_heatmap.png",
        SMOKE_ROOT / "visualizations/pytorch_jittor_prediction_heatmap.png",
        SMOKE_ROOT / "visualizations/prediction_error_heatmap.png",
        SMOKE_ROOT / "visualizations/performance_step_time.png",
        SMOKE_ROOT / "visualizations/performance_samples_per_sec.png",
        SMOKE_ROOT / "multiseed/multiseed_summary.csv",
        SMOKE_ROOT / "multiseed/multiseed_summary.json",
        SMOKE_ROOT / "multiseed/multiseed_summary.md",
        SMOKE_ROOT / "multiseed/multiseed_loss_curves.png",
        SMOKE_ROOT / "multiseed/multiseed_final_metrics_bar.png",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required_files if not nonempty(path)]

    metadata = load_json(SMOKE_ROOT / "smoke_training_data_metadata.json")
    comparison = load_json(SMOKE_ROOT / "smoke_training_comparison.json")
    alignment = load_json(SMOKE_ROOT / "smoke_training_alignment_metrics.json")
    multiseed = load_json(SMOKE_ROOT / "multiseed/multiseed_summary.json")
    pytorch_metrics = load_json(SMOKE_ROOT / "pytorch/final_metrics.json")
    jittor_metrics = load_json(SMOKE_ROOT / "jittor/final_metrics.json")

    flags_ok = all(
        metadata.get(key) is expected
        for key, expected in [
            ("full_tadsr_training", False),
            ("full_inference_executed", False),
            ("full_denoising_loop_executed", False),
            ("image_saved", False),
            ("video_saved", False),
        ]
    )
    metrics_flags_ok = all(
        metrics.get(key) is expected
        for metrics in [pytorch_metrics, jittor_metrics, comparison, alignment, multiseed]
        for key, expected in [
            ("full_tadsr_training", False),
            ("full_inference_executed", False),
            ("image_saved", False),
            ("video_saved", False),
        ]
    )
    split_ok = metadata.get("num_samples") == 32 and metadata.get("train_samples") == 24 and metadata.get("val_samples") == 8
    loss_logs_ok = has_rows(SMOKE_ROOT / "pytorch/loss.csv") and has_rows(SMOKE_ROOT / "jittor/loss.csv")
    val_logs_ok = has_rows(SMOKE_ROOT / "pytorch/validation_loss.csv") and has_rows(SMOKE_ROOT / "jittor/validation_loss.csv")
    perf_logs_ok = has_rows(SMOKE_ROOT / "pytorch/performance_log.csv") and has_rows(SMOKE_ROOT / "jittor/performance_log.csv")
    inference_guard_ok = final_audit_marker("JITTOR_FULL_INFERENCE") in {"NOT_COMPLETE", "BLOCKED"}
    staged_tensors = staged_tensor_artifacts()

    status = "PASS"
    reasons: list[str] = []
    if missing:
        status = "FAIL"
        reasons.append("missing=" + ",".join(missing))
    if not split_ok:
        status = "FAIL"
        reasons.append("train/validation split metadata is missing or unexpected")
    if not flags_ok or not metrics_flags_ok:
        status = "FAIL"
        reasons.append("smoke metadata incorrectly claims full training/inference/image/video")
    if not loss_logs_ok:
        status = "FAIL"
        reasons.append("loss logs missing or empty")
    if not val_logs_ok:
        status = "FAIL"
        reasons.append("validation logs missing or empty")
    if not perf_logs_ok:
        status = "FAIL"
        reasons.append("performance logs missing or empty")
    if alignment.get("status") not in {"PASS", "PARTIAL"}:
        status = "FAIL"
        reasons.append("alignment metrics not PASS/PARTIAL")
    if multiseed.get("status") not in {"PASS", "PARTIAL"}:
        status = "FAIL"
        reasons.append("multiseed summary not PASS/PARTIAL")
    if not inference_guard_ok:
        status = "FAIL"
        reasons.append("JITTOR_FULL_INFERENCE is not NOT_COMPLETE in final audit")
    if staged_tensors:
        status = "FAIL"
        reasons.append("staged tensor artifacts=" + ",".join(staged_tensors))

    result = {
        "status": status,
        "missing": missing,
        "split_ok": split_ok,
        "flags_ok": flags_ok,
        "metrics_flags_ok": metrics_flags_ok,
        "loss_logs_ok": loss_logs_ok,
        "validation_logs_ok": val_logs_ok,
        "performance_logs_ok": perf_logs_ok,
        "jittor_full_inference_marker": final_audit_marker("JITTOR_FULL_INFERENCE"),
        "staged_tensor_artifacts": staged_tensors,
        "trend_alignment_status": comparison.get("trend_alignment_status", "BLOCKED"),
        "prediction_alignment_status": alignment.get("prediction_alignment_status", "BLOCKED"),
        "validation_alignment_status": alignment.get("validation_alignment_status", "BLOCKED"),
        "multiseed_status": multiseed.get("status", "BLOCKED"),
        "reasons": reasons,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Smoke Training Artifacts Test",
        "",
        f"Status: **{status}**",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Required files | {'PASS' if not missing else 'FAIL'} |",
        f"| Train/validation split | {'PASS' if split_ok else 'FAIL'} |",
        f"| Safety flags | {'PASS' if flags_ok and metrics_flags_ok else 'FAIL'} |",
        f"| Loss logs | {'PASS' if loss_logs_ok else 'FAIL'} |",
        f"| Validation logs | {'PASS' if val_logs_ok else 'FAIL'} |",
        f"| Performance logs | {'PASS' if perf_logs_ok else 'FAIL'} |",
        f"| Prediction alignment | {alignment.get('prediction_alignment_status', 'BLOCKED')} |",
        f"| Validation alignment | {alignment.get('validation_alignment_status', 'BLOCKED')} |",
        f"| Multi-seed stability | {multiseed.get('status', 'BLOCKED')} |",
        f"| Full inference guard | {'PASS' if inference_guard_ok else 'FAIL'} |",
        f"| No staged .npy/.npz | {'PASS' if not staged_tensors else 'FAIL'} |",
    ]
    if reasons:
        lines += ["", "## Reasons", "", *[f"- {reason}" for reason in reasons]]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"TADSR_SMOKE_TRAINING_ARTIFACTS_TEST: {status}")
    if reasons:
        print("reasons: " + "; ".join(reasons))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
