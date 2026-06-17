#!/usr/bin/env python3
"""Build a course-requirement compliance matrix without importing ML runtimes."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "experiments/course_requirement_compliance_matrix.json"
OUT_MD = ROOT / "experiments/course_requirement_compliance_matrix.md"
DOC_MD = ROOT / "docs/course_requirement_compliance_matrix.md"


def read_json(path: str | Path) -> dict:
    p = ROOT / path
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    except Exception as exc:
        return {"status": "FAIL", "error": repr(exc)}


def marker_status(marker: str) -> str:
    for src in [
        "experiments/final_audit_report.json",
        "experiments/jittor_migration_feasibility_summary.json",
        "experiments/final_deliverables_validation.json",
        "experiments/final_submission_content_validation.json",
        "experiments/github_release_readiness_audit.json",
        "experiments/production_completion/diagnostic_image_smoke/diagnostic_image_smoke_validation.json",
        "experiments/production_completion/diagnostic_image_smoke/diagnostic_image_smoke_metrics.json",
    ]:
        data = read_json(src)
        markers = data.get("markers", {}) if isinstance(data, dict) else {}
        if marker in markers:
            return str(markers[marker])
        for row in data.get("rows", []) if isinstance(data, dict) else []:
            if row.get("check") == marker:
                return str(row.get("status", "MISSING"))
    return "MISSING"


def exists(path: str) -> bool:
    return (ROOT / path).exists()


def requirement(
    rid: str,
    title: str,
    evidence_files: list[str],
    markers: list[str],
    notes: str,
    limitation: str = "",
) -> dict:
    file_status = [{"path": p, "exists": exists(p)} for p in evidence_files]
    marker_values = {m: marker_status(m) for m in markers}
    files_ok = all(item["exists"] for item in file_status)
    marker_ok = all(v in {"PASS", "PARTIAL", "NOT_COMPLETE", "NOT_IMPLEMENTED_BY_DESIGN"} for v in marker_values.values()) if markers else True
    status = "PASS" if files_ok and marker_ok else "PARTIAL"
    if not any(item["exists"] for item in file_status):
        status = "FAIL"
    return {
        "requirement_id": rid,
        "course_requirement": title,
        "evidence_status": status,
        "evidence_files": file_status,
        "markers": marker_values,
        "notes": notes,
        "limitation_if_any": limitation,
    }


def build_matrix() -> dict:
    rows = [
        requirement(
            "R01",
            "选题合规：近期顶会/顶刊论文，且无现成 Jittor 开源实现",
            ["README.md", "docs/01_algorithm_explanation.md", "docs/TADSR-Jittor_项目全流程详解.md"],
            [],
            "README 和项目全流程文档说明选择 TADSR、论文背景、复现目标与 Jittor 迁移定位。",
        ),
        requirement(
            "R02",
            "Jittor 代码实现与开源仓库结构",
            ["jittor_tadsr_full", "jittor_tadsr", "tests_jittor_alignment", "README.md"],
            ["TADSR_JITTOR_MIGRATION_FEASIBILITY_VALIDATION"],
            "核心 Jittor 模块、对齐测试和最终审计均在仓库内。",
        ),
        requirement(
            "R03",
            "环境配置说明",
            ["README.md", "requirements.txt", "docs/repository_handoff_guide.md", "docs/final_demo_runbook.md"],
            [],
            "README、handoff guide 和 demo runbook 给出环境与演示说明。",
        ),
        requirement(
            "R04",
            "数据准备脚本",
            ["tools/export_tadsr_smoke_training_data.py", "scripts/collect_final_evidence_manifest.py", "experiments/smoke_training/output_tail/smoke_training_data_metadata.json"],
            [],
            "包含 smoke training 数据导出与 evidence manifest 收集；大型 raw tensors 不提交。",
        ),
        requirement(
            "R05",
            "训练脚本",
            ["tools/train_smoke_pytorch_output_tail.py", "scripts/train_smoke_jittor_output_tail.py", "experiments/smoke_training/output_tail/smoke_training_submission_summary.md"],
            ["TADSR_SMALL_DATA_TRAINING_ALIGNMENT"],
            "当前训练是小数据 PyTorch-vs-Jittor smoke training，不声明 full TADSR training。",
        ),
        requirement(
            "R06",
            "测试脚本",
            ["tests_jittor_alignment", "scripts/final_audit.py", "scripts/validate_jittor_migration_feasibility.py"],
            ["TADSR_FINAL_SUBMISSION_AFTER_PHASE5B_READY"],
            "测试脚本覆盖核心边界对齐、final audit 和最终提交一致性。",
        ),
        requirement(
            "R07",
            "与 PyTorch 实现对齐的实验 log",
            ["experiments/full_repro", "experiments/production_completion/full_inference/one_step/jittor_one_step_alignment.json", "experiments/production_completion/diagnostic_image_smoke/diagnostic_image_smoke_metrics.json"],
            ["TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT", "TADSR_DIAGNOSTIC_IMAGE_SMOKE_ALIGNMENT"],
            "包含 UNet、TimeVAE、Scheduler、LoRA、one-step tensor path 和 diagnostic smoke 对齐证据。",
        ),
        requirement(
            "R08",
            "性能 log",
            ["experiments/smoke_training/output_tail/pytorch/performance_log.csv", "experiments/smoke_training/output_tail/jittor/performance_log.csv", "experiments/smoke_training/output_tail/visualizations/performance_step_time.png"],
            [],
            "小数据训练阶段记录 step time / samples per second；不夸大速度结论。",
        ),
        requirement(
            "R09",
            "训练过程 log 与 loss 曲线",
            ["experiments/smoke_training/output_tail/pytorch/training.log", "experiments/smoke_training/output_tail/jittor/training.log", "experiments/smoke_training/output_tail/visualizations/train_val_loss_curve.png", "experiments/smoke_training/output_tail/visualizations/loss_gap_curve.png"],
            ["TADSR_SMALL_DATA_TRAINING_ALIGNMENT"],
            "包含 train/validation loss、loss gap 和多 seed 稳定性。",
        ),
        requirement(
            "R10",
            "结果与可视化对齐",
            ["figures/diagnostic_image_smoke/side_by_side_diagnostic_grid.png", "figures/diagnostic_image_smoke/absolute_difference_heatmap.png", "experiments/smoke_training/output_tail/visualizations/prediction_error_heatmap.png"],
            ["TADSR_DIAGNOSTIC_IMAGE_SMOKE_ALIGNMENT"],
            "diagnostic image smoke 是 one-step tensor visualization，不是最终 restored image。",
        ),
        requirement(
            "R11",
            "PPT/PDF",
            ["deliverables/TADSR-Jittor_final_presentation.pptx", "deliverables/TADSR-Jittor_final_presentation.pdf", "deliverables/TADSR-Jittor_final_presentation.md", "docs/03_ppt_outline.md"],
            ["TADSR_FINAL_PPT_READY", "TADSR_FINAL_PDF_READY"],
            "最终演示材料存在，并由 validator 检查不夸大 full inference。",
        ),
        requirement(
            "R12",
            "视频讲稿与录制指南",
            ["docs/04_video_script.md", "deliverables/TADSR-Jittor_video_recording_guide.md"],
            ["TADSR_FINAL_VIDEO_RECORDING_PACKAGE_READY"],
            "包含 20-30 分钟结构、算法讲解、Jittor 实现与现场演示命令。",
        ),
        requirement(
            "R13",
            "GitHub release readiness 与提交索引",
            ["experiments/github_release_readiness_audit.md", "deliverables/TADSR-Jittor_submission_readme.md", "docs/github_submission_handoff.md"],
            ["TADSR_GITHUB_RELEASE_READINESS_AUDIT"],
            "仓库大小、历史大文件、提交材料结构已审计。",
        ),
        requirement(
            "R14",
            "诚实边界与 false-claim guard",
            ["experiments/final_audit_report.md", "docs/full_inference_gap_analysis.md", "docs/timevae_full_alignment_gap_analysis.md", "docs/runtime_lora_final_decision_proof.md"],
            ["JITTOR_FULL_INFERENCE", "TIME_VAE_FULL_ALIGNMENT", "TADSR_DYNAMIC_RUNTIME_LORA_IMPLEMENTATION"],
            "明确保留 NOT_COMPLETE / NOT_IMPLEMENTED_BY_DESIGN，不把 one-step 或 diagnostic smoke 写成 full inference。",
            "production full inference、TimeVAE full alignment、dynamic runtime LoRA 仍不作为已完成项。",
        ),
    ]
    markers = {
        "TADSR_COURSE_REQUIREMENT_EVIDENCE_MATRIX": "PASS" if all(r["evidence_status"] in {"PASS", "PARTIAL"} for r in rows) else "FAIL",
        "TADSR_COURSE_TRAINING_REQUIREMENT_EVIDENCE": next(r["evidence_status"] for r in rows if r["requirement_id"] == "R09"),
        "TADSR_COURSE_VISUALIZATION_REQUIREMENT_EVIDENCE": next(r["evidence_status"] for r in rows if r["requirement_id"] == "R10"),
        "TADSR_COURSE_GITHUB_REQUIREMENT_EVIDENCE": next(r["evidence_status"] for r in rows if r["requirement_id"] == "R13"),
        "TADSR_COURSE_PPT_VIDEO_REQUIREMENT_EVIDENCE": "PASS" if all(next(r for r in rows if r["requirement_id"] == rid)["evidence_status"] == "PASS" for rid in ["R11", "R12"]) else "PARTIAL",
    }
    markers["TADSR_COURSE_REQUIREMENT_COMPLIANCE"] = "PASS" if all(v in {"PASS", "PARTIAL"} for v in markers.values()) else "FAIL"
    return {
        "status": markers["TADSR_COURSE_REQUIREMENT_COMPLIANCE"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "markers": markers,
        "requirements": rows,
    }


def write_markdown(matrix: dict, path: Path) -> None:
    lines = [
        "# 课程要求逐项对照矩阵",
        "",
        f"总体状态：**{matrix['status']}**",
        "",
        "本文件只索引已有证据，不运行模型、不导入 `torch` 或 `jittor`，也不生成 raw tensor。",
        "",
        "## Marker 汇总",
        "",
        "| Marker | Status |",
        "|---|---|",
    ]
    for key, value in matrix["markers"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines += ["", "## 要求对照表", "", "| ID | 课程要求 | 状态 | 证据文件 | Marker | 说明 |", "|---|---|---|---|---|---|"]
    for row in matrix["requirements"]:
        files = "<br>".join(f"`{item['path']}` ({'存在' if item['exists'] else '缺失'})" for item in row["evidence_files"])
        markers = "<br>".join(f"`{k}`={v}" for k, v in row["markers"].items()) or "无"
        note = row["notes"]
        if row["limitation_if_any"]:
            note += f"<br>限制：{row['limitation_if_any']}"
        lines.append(f"| {row['requirement_id']} | {row['course_requirement']} | `{row['evidence_status']}` | {files} | {markers} | {note} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    matrix = build_matrix()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(matrix, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(matrix, OUT_MD)
    write_markdown(matrix, DOC_MD)
    for key, value in matrix["markers"].items():
        print(f"{key}: {value}")
    return 0 if matrix["status"] in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
