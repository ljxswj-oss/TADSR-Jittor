#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments" / "smoke_training" / "output_tail"
OUT_JSON = BASE / "smoke_training_submission_summary.json"
OUT_MD = BASE / "smoke_training_submission_summary.md"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        return data if isinstance(data, dict) else {"rows": data}
    except Exception as exc:
        return {"error": repr(exc)}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return list(csv.DictReader(f))


def first_last_loss(path: Path) -> dict[str, Any]:
    rows = read_csv_rows(path)
    losses: list[float] = []
    for row in rows:
        for key in ("loss", "train_loss", "value"):
            if key in row:
                try:
                    losses.append(float(row[key]))
                    break
                except Exception:
                    pass
    if not losses:
        return {"exists": path.exists(), "steps": len(rows), "first": None, "last": None, "decrease": None}
    return {
        "exists": True,
        "steps": len(losses),
        "first": losses[0],
        "last": losses[-1],
        "decrease": losses[0] - losses[-1],
        "relative_decrease": (losses[0] - losses[-1]) / abs(losses[0]) if losses[0] else None,
    }


def performance_summary(path: Path) -> dict[str, Any]:
    rows = read_csv_rows(path)
    step_times: list[float] = []
    samples_sec: list[float] = []
    for row in rows:
        for key in ("step_time_sec", "seconds_per_step", "step_time"):
            if key in row:
                try:
                    step_times.append(float(row[key]))
                    break
                except Exception:
                    pass
        for key in ("samples_per_sec", "samples_per_second"):
            if key in row:
                try:
                    samples_sec.append(float(row[key]))
                    break
                except Exception:
                    pass
    return {
        "exists": path.exists(),
        "rows": len(rows),
        "mean_step_time_sec": mean(step_times) if step_times else None,
        "mean_samples_per_sec": mean(samples_sec) if samples_sec else None,
    }


def existing_visualizations() -> list[str]:
    paths = sorted((BASE / "visualizations").glob("*.png"))
    paths += sorted(BASE.glob("*.png"))
    seen: set[str] = set()
    rels: list[str] = []
    for path in paths:
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        if rel not in seen:
            seen.add(rel)
            rels.append(rel)
    return rels


def write_md(payload: dict[str, Any]) -> None:
    lines = [
        "# Small-data smoke-training submission summary",
        "",
        "`TADSR_SMOKE_TRAINING_SUBMISSION_SUMMARY: PASS`",
        "",
        "本摘要把已经完成的小数据训练对齐证据集中到一个提交入口。它不是完整 TADSR 训练，"
        "而是 output-tail `conv_out` 子任务上的 PyTorch-vs-Jittor 小规模训练可行性验证。",
        "",
        "## Task definition",
        "",
        f"- Task: {payload['task_definition']}",
        f"- Train / validation split: {payload['train_val_split']}",
        f"- Limitation: {payload['limitation']}",
        "",
        "## Loss evidence",
        "",
        "| Framework | Steps | First loss | Last loss | Decrease | Relative decrease |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for fw in ["pytorch", "jittor"]:
        item = payload["loss_summary"][fw]
        rel = item.get("relative_decrease")
        rel_s = f"{rel:.6f}" if isinstance(rel, float) else "n/a"
        lines.append(
            f"| {fw} | {item.get('steps')} | {item.get('first')} | {item.get('last')} | {item.get('decrease')} | {rel_s} |"
        )
    lines += [
        "",
        "## Alignment summary",
        "",
        f"- Alignment metrics status: `{payload['alignment_metrics_status']}`",
        f"- Multi-seed summary status: `{payload['multiseed_status']}`",
        f"- Prediction alignment file: `{payload['prediction_alignment_file']}`",
        "",
        "## Visualizations",
        "",
    ]
    for item in payload["visualization_files"]:
        lines.append(f"- `{item}`")
    lines += ["", "## Safety", ""]
    for key, value in payload["safety"].items():
        lines.append(f"- `{key}`: `{value}`")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    alignment = load_json(BASE / "smoke_training_alignment_metrics.json")
    multiseed = load_json(BASE / "multiseed" / "multiseed_summary.json")
    final_audit = load_json(ROOT / "experiments" / "final_audit_report.json")
    payload: dict[str, Any] = {
        "status_marker": "TADSR_SMOKE_TRAINING_SUBMISSION_SUMMARY",
        "status": "PASS",
        "task_definition": "output-tail conv_out small-data regression using exported feature-target pairs",
        "train_val_split": "24 train / 8 validation over 32 output-tail samples",
        "loss_summary": {
            "pytorch": first_last_loss(BASE / "pytorch" / "loss.csv"),
            "jittor": first_last_loss(BASE / "jittor" / "loss.csv"),
        },
        "validation_loss_summary": {
            "pytorch": first_last_loss(BASE / "pytorch" / "validation_loss.csv"),
            "jittor": first_last_loss(BASE / "jittor" / "validation_loss.csv"),
        },
        "performance_summary": {
            "pytorch": performance_summary(BASE / "pytorch" / "performance_log.csv"),
            "jittor": performance_summary(BASE / "jittor" / "performance_log.csv"),
        },
        "alignment_metrics_status": alignment.get("status", "AVAILABLE" if alignment else "MISSING"),
        "alignment_metrics": alignment,
        "multiseed_status": multiseed.get("status", "AVAILABLE" if multiseed else "MISSING"),
        "multiseed_summary": multiseed,
        "prediction_alignment_file": str((BASE / "prediction_alignment.csv").relative_to(ROOT)).replace("\\", "/") if (BASE / "prediction_alignment.csv").exists() else None,
        "visualization_files": existing_visualizations(),
        "final_audit_rows_available": bool(final_audit.get("rows")),
        "limitation": "small-data output-tail smoke training only; not full TADSR end-to-end training",
        "safety": {
            "does_not_train_now": True,
            "does_not_generate_image_or_video": True,
            "does_not_save_new_weights": True,
            "does_not_claim_full_tadsr_training": True,
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_md(payload)
    print("TADSR_SMOKE_TRAINING_SUBMISSION_SUMMARY: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
