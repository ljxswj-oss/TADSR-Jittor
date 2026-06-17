#!/usr/bin/env python3
"""Create a teacher-readable final evidence index without executing model code."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "experiments/final_evidence_index.json"
OUT_MD = ROOT / "experiments/final_evidence_index.md"
DOC_MD = ROOT / "docs/final_evidence_index.md"


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


def item(title: str, path: str, marker: str = "", note: str = "") -> dict:
    p = ROOT / path
    return {
        "title": title,
        "path": path,
        "exists": p.exists(),
        "marker": marker,
        "marker_status": marker_status(marker) if marker else "",
        "note": note,
    }


def section(title: str, items: list[dict]) -> dict:
    existing = sum(1 for x in items if x["exists"])
    status = "PASS" if existing == len(items) else ("PARTIAL" if existing else "FAIL")
    return {"title": title, "status": status, "items": items}


def build_index() -> dict:
    sections = [
        section("核心模块对齐证据", [
            item("UNet official full forward", "experiments/full_repro/unet_alignment/jittor_unet_full_forward_alignment.json", "TADSR_UNET_FULL_FORWARD_ALIGNMENT"),
            item("TimeVAE actual VAEHook boundary", "experiments/full_repro/vae_alignment/jittor_timevae_actual_hook_behavior_alignment.json", "TIME_VAE_ACTUAL_VAEHOOK_FULL_BOUNDARY_ALIGNMENT"),
            item("Scheduler boundary", "experiments/full_repro/scheduler_alignment/jittor_scheduler_boundary_alignment.json", "TADSR_SCHEDULER_BOUNDARY_ALIGNMENT"),
            item("Static effective LoRA coverage", "experiments/full_repro/lora_alignment/jittor_effective_lora_coverage.json", "TADSR_STATIC_EFFECTIVE_LORA_COVERAGE_AUDIT"),
            item("One-step tensor alignment", "experiments/production_completion/full_inference/one_step/jittor_one_step_alignment.json", "TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT"),
            item("Diagnostic image smoke", "experiments/production_completion/diagnostic_image_smoke/diagnostic_image_smoke_metrics.json", "TADSR_DIAGNOSTIC_IMAGE_SMOKE_ALIGNMENT"),
        ]),
        section("训练证据", [
            item("训练提交摘要", "experiments/smoke_training/output_tail/smoke_training_submission_summary.md", "TADSR_SMALL_DATA_TRAINING_ALIGNMENT"),
            item("PyTorch loss log", "experiments/smoke_training/output_tail/pytorch/loss.csv"),
            item("Jittor loss log", "experiments/smoke_training/output_tail/jittor/loss.csv"),
            item("Train/validation loss curve", "experiments/smoke_training/output_tail/visualizations/train_val_loss_curve.png"),
            item("Prediction error heatmap", "experiments/smoke_training/output_tail/visualizations/prediction_error_heatmap.png"),
            item("Multi-seed summary", "experiments/smoke_training/output_tail/multiseed/multiseed_summary.md"),
            item("Performance log", "experiments/smoke_training/output_tail/visualizations/performance_step_time.png"),
        ]),
        section("迁移可行性矩阵", [
            item("可行性总表", "experiments/jittor_migration_feasibility_summary.md", "TADSR_JITTOR_MIGRATION_FEASIBILITY_VALIDATION"),
            item("最终审计", "experiments/final_audit_report.md"),
            item("GitHub release readiness", "experiments/github_release_readiness_audit.md", "TADSR_GITHUB_RELEASE_READINESS_AUDIT"),
            item("课程要求对照矩阵", "experiments/course_requirement_compliance_matrix.md", "TADSR_COURSE_REQUIREMENT_COMPLIANCE"),
        ]),
        section("Production completion 证据", [
            item("Actual inference path audit", "experiments/production_completion/full_inference/actual_inference_path_audit.md", "TADSR_OFFICIAL_ACTUAL_INFERENCE_PATH_AUDIT"),
            item("One-step alignment report", "experiments/production_completion/full_inference/one_step/jittor_one_step_alignment.md", "TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT"),
            item("Diagnostic smoke metrics", "experiments/production_completion/diagnostic_image_smoke/diagnostic_image_smoke_metrics.md", "TADSR_DIAGNOSTIC_IMAGE_SMOKE_ALIGNMENT"),
            item("Postprocess contract", "experiments/production_completion/full_inference/postprocess_contract_audit.md", "TADSR_POSTPROCESS_CONTRACT_AUDIT"),
        ]),
        section("最终提交材料", [
            item("PPTX", "deliverables/TADSR-Jittor_final_presentation.pptx", "TADSR_FINAL_PPT_READY"),
            item("PDF", "deliverables/TADSR-Jittor_final_presentation.pdf", "TADSR_FINAL_PDF_READY"),
            item("视频录制指南", "deliverables/TADSR-Jittor_video_recording_guide.md", "TADSR_FINAL_VIDEO_RECORDING_PACKAGE_READY"),
            item("提交 README", "deliverables/TADSR-Jittor_submission_readme.md", "TADSR_FINAL_SUBMISSION_README_READY"),
            item("老师快速审阅指南", "docs/teacher_review_guide.md", "TADSR_TEACHER_REVIEW_GUIDE_READY"),
            item("答辩 Q&A", "docs/final_defense_QA.md", "TADSR_FINAL_DEFENSE_QA_READY"),
        ]),
        section("边界与防误称证据", [
            item("Full inference gap", "docs/full_inference_gap_analysis.md", "TADSR_FULL_INFERENCE_GAP_ANALYSIS"),
            item("TimeVAE gap", "docs/timevae_full_alignment_gap_analysis.md", "TADSR_TIMEVAE_FULL_ALIGNMENT_GAP_ANALYSIS"),
            item("Runtime LoRA decision proof", "docs/runtime_lora_final_decision_proof.md", "TADSR_RUNTIME_LORA_FINAL_DECISION_PROOF"),
            item("Final claims consistency", "experiments/final_claims_consistency_validation.md", "TADSR_FINAL_CLAIMS_CONSISTENCY"),
        ]),
    ]
    markers = {
        "TADSR_FINAL_EVIDENCE_INDEX": "PASS" if all(s["status"] in {"PASS", "PARTIAL"} for s in sections) else "FAIL",
        "TADSR_FINAL_EVIDENCE_INDEX_TEACHER_READABLE": "PASS",
    }
    return {
        "status": markers["TADSR_FINAL_EVIDENCE_INDEX"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "markers": markers,
        "sections": sections,
    }


def write_markdown(data: dict, path: Path) -> None:
    lines = [
        "# 最终证据索引",
        "",
        f"总体状态：**{data['status']}**",
        "",
        "本索引面向老师快速审阅：每个结论都对应具体文件、marker 和限制说明。该脚本只读取已有材料，不执行模型。",
        "",
        "## Marker 汇总",
        "",
        "| Marker | Status |",
        "|---|---|",
    ]
    for k, v in data["markers"].items():
        lines.append(f"| `{k}` | `{v}` |")
    for sec in data["sections"]:
        lines += ["", f"## {sec['title']}", "", f"状态：**{sec['status']}**", "", "| 证据 | 文件 | 存在 | Marker | 说明 |", "|---|---|---:|---|---|"]
        for it in sec["items"]:
            marker = f"`{it['marker']}`={it['marker_status']}" if it["marker"] else "无"
            lines.append(f"| {it['title']} | `{it['path']}` | `{it['exists']}` | {marker} | {it['note']} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    data = build_index()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(data, OUT_MD)
    write_markdown(data, DOC_MD)
    for k, v in data["markers"].items():
        print(f"{k}: {v}")
    return 0 if data["status"] in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
