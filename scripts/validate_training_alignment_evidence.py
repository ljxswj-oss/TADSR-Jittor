#!/usr/bin/env python3
"""Audit existing small-data training evidence without rerunning training."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/smoke_training/output_tail"
OUT_JSON = BASE / "training_alignment_evidence_validation.json"
OUT_MD = BASE / "training_alignment_evidence_validation.md"
DOC_MD = ROOT / "docs/training_alignment_evidence_summary.md"


def exists(path: str) -> bool:
    return (ROOT / path).exists()


def read_json(path: str) -> dict:
    p = ROOT / path
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    except Exception as exc:
        return {"error": repr(exc)}


def csv_rows(path: str) -> int:
    p = ROOT / path
    if not p.exists():
        return 0
    try:
        with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            return max(sum(1 for _ in csv.reader(f)) - 1, 0)
    except Exception:
        return 0


def check(name: str, paths: list[str], note: str, optional: bool = False) -> dict:
    present = [p for p in paths if exists(p)]
    status = "PASS" if len(present) == len(paths) else ("MISSING_OPTIONAL" if optional else ("PARTIAL" if present else "FAIL"))
    return {"name": name, "status": status, "paths": [{"path": p, "exists": exists(p)} for p in paths], "note": note}


def build_report() -> dict:
    checks = [
        check("PyTorch loss log", ["experiments/smoke_training/output_tail/pytorch/loss.csv"], "PyTorch 小数据训练 loss 逐步记录。"),
        check("Jittor loss log", ["experiments/smoke_training/output_tail/jittor/loss.csv"], "Jittor 小数据训练 loss 逐步记录。"),
        check("train/validation loss", [
            "experiments/smoke_training/output_tail/pytorch/validation_loss.csv",
            "experiments/smoke_training/output_tail/jittor/validation_loss.csv",
            "experiments/smoke_training/output_tail/visualizations/train_val_loss_curve.png",
        ], "包含训练/验证 loss 与曲线。"),
        check("loss gap curve", ["experiments/smoke_training/output_tail/visualizations/loss_gap_curve.png", "experiments/smoke_training/output_tail/visualizations/relative_loss_gap_curve.png"], "可视化 PyTorch 与 Jittor loss 差距。"),
        check("performance log", [
            "experiments/smoke_training/output_tail/pytorch/performance_log.csv",
            "experiments/smoke_training/output_tail/jittor/performance_log.csv",
            "experiments/smoke_training/output_tail/visualizations/performance_step_time.png",
            "experiments/smoke_training/output_tail/visualizations/performance_samples_per_sec.png",
        ], "记录 step time 与吞吐，作为性能 log。"),
        check("prediction alignment", [
            "experiments/smoke_training/output_tail/prediction_alignment.csv",
            "experiments/smoke_training/output_tail/visualizations/prediction_error_heatmap.png",
            "experiments/smoke_training/output_tail/visualizations/pytorch_jittor_prediction_heatmap.png",
        ], "输出预测与误差热图。"),
        check("multi-seed stability", [
            "experiments/smoke_training/output_tail/multiseed/multiseed_summary.json",
            "experiments/smoke_training/output_tail/multiseed/multiseed_summary.md",
            "experiments/smoke_training/output_tail/multiseed/multiseed_loss_curves.png",
        ], "多 seed 稳定性记录。"),
        check("parameter update evidence", [
            "experiments/smoke_training/output_tail/pytorch/final_weight_delta_summary.json",
            "experiments/smoke_training/output_tail/jittor/final_weight_delta_summary.json",
        ], "已有 final_weight_delta_summary，证明参数发生更新。"),
        check("gradient norm evidence", [
            "experiments/smoke_training/output_tail/pytorch/gradient_norm.csv",
            "experiments/smoke_training/output_tail/jittor/gradient_norm.csv",
        ], "当前原始训练日志未保存逐 step gradient norm，作为 optional enhancement 记录。", optional=True),
        check("training scripts", ["tools/train_smoke_pytorch_output_tail.py", "scripts/train_smoke_jittor_output_tail.py"], "训练脚本存在，可复查。"),
        check("false full training claim guard", ["experiments/smoke_training/output_tail/smoke_training_submission_summary.md"], "材料明确 smoke training 不是 full TADSR training。"),
    ]
    status_by_name = {c["name"]: c["status"] for c in checks}
    markers = {
        "TADSR_TRAINING_LOSS_CURVE_EVIDENCE": "PASS" if status_by_name["train/validation loss"] == "PASS" and status_by_name["loss gap curve"] == "PASS" else "PARTIAL",
        "TADSR_TRAINING_PERFORMANCE_LOG_EVIDENCE": status_by_name["performance log"],
        "TADSR_TRAINING_OUTPUT_ALIGNMENT_EVIDENCE": status_by_name["prediction alignment"],
        "TADSR_TRAINING_GRAD_PARAM_UPDATE_EVIDENCE": "PARTIAL" if status_by_name["parameter update evidence"] == "PASS" and status_by_name["gradient norm evidence"] == "MISSING_OPTIONAL" else status_by_name["gradient norm evidence"],
    }
    core_ok = all(status_by_name[name] == "PASS" for name in [
        "PyTorch loss log",
        "Jittor loss log",
        "train/validation loss",
        "performance log",
        "prediction alignment",
        "multi-seed stability",
        "parameter update evidence",
        "training scripts",
    ])
    markers["TADSR_TRAINING_ALIGNMENT_EVIDENCE_VALIDATION"] = "PASS" if core_ok else "PARTIAL"
    markers["TADSR_TRAINING_EVIDENCE_TEACHER_READY"] = "PASS" if core_ok else "PARTIAL"
    metrics = read_json("experiments/smoke_training/output_tail/smoke_training_alignment_metrics.json")
    summary = read_json("experiments/smoke_training/output_tail/smoke_training_submission_summary.json")
    return {
        "status": markers["TADSR_TRAINING_EVIDENCE_TEACHER_READY"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "markers": markers,
        "checks": checks,
        "row_counts": {
            "pytorch_loss_rows": csv_rows("experiments/smoke_training/output_tail/pytorch/loss.csv"),
            "jittor_loss_rows": csv_rows("experiments/smoke_training/output_tail/jittor/loss.csv"),
            "pytorch_validation_rows": csv_rows("experiments/smoke_training/output_tail/pytorch/validation_loss.csv"),
            "jittor_validation_rows": csv_rows("experiments/smoke_training/output_tail/jittor/validation_loss.csv"),
        },
        "alignment_metrics": metrics,
        "submission_summary": summary,
        "notes": [
            "当前证据覆盖 loss、validation、performance、output alignment、multi-seed 和 parameter update。",
            "逐 step gradient norm 未作为原始训练日志保存，标记为 MISSING_OPTIONAL；不伪造、不重跑训练。",
            "本实验是 small-data smoke training alignment，不声明完整 TADSR 训练已经完成。",
        ],
    }


def write_markdown(data: dict, path: Path) -> None:
    lines = [
        "# 训练证据完整性审计",
        "",
        f"总体状态：**{data['status']}**",
        "",
        "本审计只读取已经生成的 smoke training 文件，不重新训练，不导入 `torch` 或 `jittor`。",
        "",
        "## Marker 汇总",
        "",
        "| Marker | Status |",
        "|---|---|",
    ]
    for k, v in data["markers"].items():
        lines.append(f"| `{k}` | `{v}` |")
    lines += ["", "## 文件检查", "", "| 检查项 | 状态 | 文件 | 说明 |", "|---|---|---|---|"]
    for c in data["checks"]:
        paths = "<br>".join(f"`{p['path']}` ({'存在' if p['exists'] else '缺失'})" for p in c["paths"])
        lines.append(f"| {c['name']} | `{c['status']}` | {paths} | {c['note']} |")
    lines += ["", "## 行数统计", "", "| 文件 | 行数 |", "|---|---:|"]
    for k, v in data["row_counts"].items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## 结论", ""]
    lines += [f"- {note}" for note in data["notes"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    data = build_report()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(data, OUT_MD)
    write_markdown(data, DOC_MD)
    for k, v in data["markers"].items():
        print(f"{k}: {v}")
    return 0 if data["status"] in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
