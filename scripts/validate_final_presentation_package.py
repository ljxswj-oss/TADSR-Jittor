#!/usr/bin/env python3
"""Validate final presentation/video package guardrails."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "experiments/final_presentation_package_validation.json"
OUT_MD = ROOT / "experiments/final_presentation_package_validation.md"

FALSE_CLAIMS = (
    "full inference complete",
    "production pipeline complete",
    "image generation complete",
    "video generation complete",
    "image/video generation complete",
    "full TADSR training complete",
    "dynamic runtime LoRA complete",
    "runtime LoRA complete",
    "TimeVAE full alignment complete",
)

NEGATION_HINTS = (
    "not",
    "no ",
    "do not",
    "does not",
    "without",
    "never",
    "NOT_COMPLETE",
    "NOT_IMPLEMENTED",
    "NotImplementedError",
    "未完成",
    "未实现",
    "不声称",
    "没有",
    "不会",
    "不包含",
)


def read_text(path: str | Path) -> str:
    p = ROOT / path
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="ignore")


def read_json(path: str | Path) -> dict:
    p = ROOT / path
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}


def has_any(text: str, variants: list[str]) -> bool:
    return any(v in text for v in variants)


def has_groups(text: str, groups: list[list[str]]) -> bool:
    return all(has_any(text, group) for group in groups)


def line_is_negated(line: str) -> bool:
    low = line.lower()
    return any(token.lower() in low for token in NEGATION_HINTS)


def find_false_claims(paths: list[str]) -> list[dict]:
    hits: list[dict] = []
    for path in paths:
        for lineno, line in enumerate(read_text(path).splitlines(), 1):
            low = line.lower()
            for phrase in FALSE_CLAIMS:
                if phrase.lower() in low and not line_is_negated(line):
                    hits.append({"path": path, "line": lineno, "phrase": phrase, "text": line.strip()})
            if "JITTOR_FULL_INFERENCE: PASS" in line or "TIME_VAE_FULL_ALIGNMENT: PASS" in line:
                hits.append({"path": path, "line": lineno, "phrase": "forbidden PASS marker", "text": line.strip()})
    return hits


def marker_status(marker: str) -> str:
    candidates = [
        read_json("experiments/jittor_migration_feasibility_summary.json").get("markers", {}),
        {row.get("check"): row.get("status") for row in read_json("experiments/final_audit_report.json").get("rows", [])},
    ]
    for mapping in candidates:
        if marker in mapping:
            return str(mapping[marker])
    return "MISSING"


def validate() -> dict:
    paths = {
        "ppt_outline": "docs/03_ppt_outline.md",
        "video_script": "docs/04_video_script.md",
        "runbook": "docs/final_demo_runbook.md",
        "handoff": "docs/repository_handoff_guide.md",
        "checklist": "docs/final_submission_checklist.md",
        "presentation_md": "deliverables/TADSR-Jittor_final_presentation.md",
        "presentation_pptx": "deliverables/TADSR-Jittor_final_presentation.pptx",
        "presentation_pdf": "deliverables/TADSR-Jittor_final_presentation.pdf",
        "submission_readme": "deliverables/TADSR-Jittor_submission_readme.md",
        "feasibility_md": "experiments/jittor_migration_feasibility_summary.md",
        "feasibility_json": "experiments/jittor_migration_feasibility_summary.json",
        "final_audit_json": "experiments/final_audit_report.json",
        "full_inference_gap": "docs/full_inference_gap_analysis.md",
        "timevae_gap": "docs/timevae_full_alignment_gap_analysis.md",
        "lora_gap": "docs/lora_runtime_gap_analysis.md",
        "teacher_summary": "docs/final_teacher_status_summary.md",
        "timevae_closure": "docs/timevae_full_alignment_closure_plan.md",
        "runtime_lora_decision": "docs/runtime_lora_final_decision_proof.md",
        "diagnostic_image_smoke_metrics": "experiments/production_completion/diagnostic_image_smoke/diagnostic_image_smoke_metrics.md",
        "diagnostic_image_smoke_validation": "experiments/production_completion/diagnostic_image_smoke/diagnostic_image_smoke_validation.md",
        "diagnostic_cli": "experiments/production_completion/diagnostic_cli/one_step_diagnostic_cli_test.md",
        "defense_risk_response_pack": "docs/final_defense_risk_response_pack.md",
        "defense_short_answers": "docs/defense_short_answers_zh.md",
        "defense_long_answers": "docs/defense_long_answers_zh.md",
        "defense_do_not_say": "docs/defense_do_not_say.md",
        "defense_evidence_lookup": "docs/defense_evidence_lookup_table.md",
        "release_candidate_signoff": "docs/final_release_candidate_signoff.md",
        "command_index": "docs/final_command_index.md",
        "github_upload_checklist": "docs/final_github_upload_checklist.md",
        "links_paths_validation": "experiments/final_links_and_paths_validation.md",
        "chinese_materials_validation": "experiments/final_chinese_materials_validation.md",
        "human_submission_instructions": "docs/final_human_submission_instructions.md",
        "video_rehearsal_checklist": "docs/final_video_rehearsal_checklist.md",
        "submission_freeze_tag": "docs/final_submission_freeze_tag.md",
        "human_submission_lock_report": "docs/final_human_submission_lock_report.md",
        "video_submission_check": "docs/final_video_submission_check.md",
        "human_submission_instructions_marker": "experiments/final_human_submission_instructions.md",
        "video_rehearsal_checklist_marker": "experiments/final_video_rehearsal_checklist.md",
        "submission_freeze_tag_marker": "experiments/final_submission_freeze_tag.md",
        "human_submission_lock_report_marker": "experiments/final_human_submission_lock_report.md",
        "video_submission_check_marker": "experiments/final_video_submission_check.md",
        "clean_public_release_manifest": "experiments/clean_public_release_package_manifest.md",
        "clean_public_release_validation": "experiments/clean_public_release_package_validation.md",
        "github_url_status": "experiments/final_github_url_status.md",
    }
    file_status = {
        name: {"path": path, "exists": (ROOT / path).exists(), "size_bytes": (ROOT / path).stat().st_size if (ROOT / path).exists() else 0}
        for name, path in paths.items()
    }

    combined = "\n".join(read_text(path) for path in paths.values() if str(path).endswith(".md"))
    false_claim_hits = find_false_claims([path for path in paths.values() if str(path).endswith(".md")])

    required_groups = [
        ["TADSR_UNET_FULL_FORWARD_ALIGNMENT: PASS", "TADSR_UNET_FULL_FORWARD_ALIGNMENT"],
        ["TIME_VAE_ACTUAL_VAEHOOK_FULL_BOUNDARY_ALIGNMENT: PASS", "TIME_VAE_ACTUAL_VAEHOOK_FULL_BOUNDARY_ALIGNMENT"],
        ["TADSR_STATIC_EFFECTIVE_LORA_COVERAGE_AUDIT: PASS", "TADSR_STATIC_EFFECTIVE_LORA_COVERAGE_AUDIT"],
        ["TADSR_SCHEDULER_BOUNDARY_ALIGNMENT: PASS", "TADSR_SCHEDULER_BOUNDARY_ALIGNMENT"],
        ["TADSR_MINIMAL_ONE_STEP_DECODE_DRY_RUN: PASS", "TADSR_MINIMAL_ONE_STEP_DECODE_DRY_RUN"],
        ["JITTOR_FULL_INFERENCE: NOT_COMPLETE"],
        ["NotImplementedError"],
        ["TADSR_JITTOR_MIGRATION_FEASIBILITY_VALIDATION"],
        ["TADSR_BOUNDARY_LEVEL_REPRODUCTION"],
        ["TADSR_SMALL_DATA_TRAINING_ALIGNMENT"],
        ["TADSR_FULL_INFERENCE_GAP_ANALYSIS"],
        ["TADSR_TIMEVAE_FULL_ALIGNMENT_GAP_ANALYSIS"],
        ["TADSR_LORA_RUNTIME_GAP_ANALYSIS"],
        ["TADSR_GAP_ANALYSIS_READINESS"],
        ["TIME_VAE_FULL_ALIGNMENT: NOT_COMPLETE"],
        ["TADSR_DYNAMIC_RUNTIME_LORA_IMPLEMENTATION: NOT_IMPLEMENTED_BY_DESIGN"],
        ["full_inference_gap_analysis.md"],
        ["timevae_full_alignment_gap_analysis.md"],
        ["lora_runtime_gap_analysis.md"],
        ["final_teacher_status_summary.md"],
        ["rigorous boundary-level Jittor migration evidence", "边界级 Jittor 迁移证据", "边界级迁移证据"],
        ["Small-Data Smoke Training", "小数据训练"],
        ["24 train", "24 个训练"],
        ["8 validation", "8 个验证"],
        ["multi-seed", "多 seed"],
        ["TADSR_DIAGNOSTIC_IMAGE_SMOKE_READY"],
        ["TADSR_DIAGNOSTIC_IMAGE_SMOKE_EXECUTED"],
        ["TADSR_DIAGNOSTIC_IMAGE_SMOKE_NOT_FULL_INFERENCE"],
        ["TADSR_DIAGNOSTIC_IMAGE_SMOKE_VALIDATION"],
        ["TADSR_ONE_STEP_DIAGNOSTIC_CLI_READY"],
        ["TADSR_ONE_STEP_DIAGNOSTIC_CLI_NOT_FULL_INFERENCE"],
        ["TADSR_TIMEVAE_FULL_ALIGNMENT_CLOSURE_SUMMARY"],
        ["TADSR_TIMEVAE_FULL_ALIGNMENT_REMAINS_SCOPED"],
        ["TADSR_RUNTIME_LORA_FINAL_DECISION_PROOF"],
        ["TADSR_DYNAMIC_RUNTIME_LORA_NOT_REQUIRED_FOR_AUDITED_PATH"],
        ["TADSR_FINAL_SUBMISSION_WITH_DIAGNOSTIC_SMOKE_READY"],
        ["TADSR_DEFENSE_RISK_RESPONSE_PACK"],
        ["TADSR_DEFENSE_SHORT_ANSWERS_READY"],
        ["TADSR_DEFENSE_LONG_ANSWERS_READY"],
        ["TADSR_DEFENSE_FALSE_CLAIM_GUARD"],
        ["TADSR_DEFENSE_EVIDENCE_LOOKUP_READY"],
        ["TADSR_FINAL_RELEASE_CANDIDATE_SIGNOFF"],
        ["TADSR_FINAL_LINKS_AND_PATHS_VALIDATION"],
        ["TADSR_FINAL_CHINESE_MATERIALS_VALIDATION"],
        ["TADSR_FINAL_COMMAND_INDEX"],
        ["TADSR_FINAL_GITHUB_UPLOAD_CHECKLIST"],
        ["TADSR_CLEAN_PUBLIC_RELEASE_PACKAGE_BUILD"],
        ["TADSR_CLEAN_PUBLIC_RELEASE_PACKAGE_VALIDATION"],
        ["TADSR_FINAL_GITHUB_URL_UPDATE_SCRIPT_READY"],
        ["TADSR_FINAL_HUMAN_SUBMISSION_INSTRUCTIONS"],
        ["TADSR_FINAL_VIDEO_REHEARSAL_CHECKLIST"],
        ["TADSR_FINAL_SUBMISSION_FREEZE_TAG_DOC"],
        ["TADSR_FINAL_HUMAN_SUBMISSION_LOCK_REPORT"],
        ["TADSR_FINAL_VIDEO_SUBMISSION_CHECK"],
    ]
    missing_groups = [group for group in required_groups if not has_any(combined, group)]

    feasibility_markers = [
        "TADSR_JITTOR_MIGRATION_FEASIBILITY_VALIDATION",
        "TADSR_MODULE_COVERAGE_MATRIX",
        "TADSR_WEIGHT_LOADING_COVERAGE_MATRIX",
        "TADSR_LORA_EFFECTIVE_WEIGHT_COVERAGE_MATRIX",
        "TADSR_NUMERICAL_ALIGNMENT_COVERAGE_MATRIX",
        "TADSR_INTEGRATION_PATH_COVERAGE_MATRIX",
        "TADSR_TRAINING_PATH_FEASIBILITY_MATRIX",
        "TADSR_BOUNDARY_LEVEL_REPRODUCTION",
        "TADSR_SMALL_DATA_TRAINING_ALIGNMENT",
        "TADSR_FULL_INFERENCE_GUARD_VALIDATION",
        "TADSR_FULL_INFERENCE_GAP_ANALYSIS",
        "TADSR_TIMEVAE_FULL_ALIGNMENT_GAP_ANALYSIS",
        "TADSR_LORA_RUNTIME_GAP_ANALYSIS",
        "TADSR_RUNTIME_LORA_LAYER_FORMULA_ALIGNMENT",
        "TADSR_DIAGNOSTIC_IMAGE_SMOKE_NOT_FULL_INFERENCE",
        "TADSR_ONE_STEP_DIAGNOSTIC_CLI_READY",
        "TADSR_TIMEVAE_FULL_ALIGNMENT_CLOSURE_SUMMARY",
        "TADSR_RUNTIME_LORA_FINAL_DECISION_PROOF",
        "TADSR_FINAL_SUBMISSION_WITH_DIAGNOSTIC_SMOKE_READY",
        "TADSR_SUBMISSION_FACING_STATUS_SUMMARY",
        "TADSR_ENVIRONMENT_BLOCKERS_EXPLAINED",
        "TADSR_GAP_ANALYSIS_READINESS",
    ]
    marker_map = {m: marker_status(m) for m in feasibility_markers}
    markers_ok = all(v == "PASS" for v in marker_map.values())

    presentation_ok = all(file_status[name]["exists"] and file_status[name]["size_bytes"] > 0 for name in ["ppt_outline", "presentation_md", "presentation_pptx", "presentation_pdf"]) and not missing_groups and not false_claim_hits
    video_ok = file_status["video_script"]["exists"] and not false_claim_hits
    demo_runbook_ok = file_status["runbook"]["exists"] and file_status["runbook"]["size_bytes"] > 0 and not false_claim_hits
    repository_handoff_ok = file_status["handoff"]["exists"] and file_status["handoff"]["size_bytes"] > 0 and not false_claim_hits
    submission_ok = file_status["submission_readme"]["exists"] and not false_claim_hits
    feasibility_ok = file_status["feasibility_json"]["exists"] and markers_ok
    defense_ok = all(file_status[name]["exists"] and file_status[name]["size_bytes"] > 0 for name in [
        "defense_risk_response_pack",
        "defense_short_answers",
        "defense_long_answers",
        "defense_do_not_say",
        "defense_evidence_lookup",
    ])
    final_qa_ok = all(file_status[name]["exists"] and file_status[name]["size_bytes"] > 0 for name in [
        "release_candidate_signoff",
        "command_index",
        "github_upload_checklist",
        "links_paths_validation",
        "chinese_materials_validation",
        "human_submission_instructions",
        "video_rehearsal_checklist",
        "submission_freeze_tag",
        "human_submission_lock_report",
        "video_submission_check",
        "human_submission_lock_report_marker",
        "video_submission_check_marker",
        "clean_public_release_manifest",
        "clean_public_release_validation",
        "github_url_status",
    ])
    overall_ok = presentation_ok and video_ok and demo_runbook_ok and repository_handoff_ok and submission_ok and feasibility_ok and defense_ok and final_qa_ok

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if overall_ok else "FAIL",
        "files": file_status,
        "missing_required_groups": missing_groups,
        "false_claim_hits": false_claim_hits,
        "feasibility_marker_status": marker_map,
        "markers": {
            "TADSR_FINAL_PRESENTATION_PACKAGE": "PASS" if presentation_ok else "FAIL",
            "TADSR_VIDEO_SCRIPT_READY": "PASS" if video_ok else "FAIL",
            "TADSR_DEMO_RUNBOOK_READY": "PASS" if demo_runbook_ok else "FAIL",
            "TADSR_REPOSITORY_HANDOFF_READY": "PASS" if repository_handoff_ok else "FAIL",
            "TADSR_FINAL_SUBMISSION_README_READY": "PASS" if submission_ok else "FAIL",
            "TADSR_DEFENSE_RISK_RESPONSE_PACK": "PASS" if defense_ok else "FAIL",
            "TADSR_FINAL_RELEASE_QA_READY": "PASS" if final_qa_ok else "FAIL",
            "TADSR_JITTOR_MIGRATION_FEASIBILITY_VALIDATION": "PASS" if feasibility_ok else "FAIL",
            "TADSR_FINAL_PRESENTATION_PACKAGE_VALIDATION": "PASS" if overall_ok else "FAIL",
        },
    }
    return result


def write_outputs(result: dict) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# 最终 PPT / 视频 / 提交材料验证",
        "",
        f"状态：**{result['status']}**",
        "",
        "## Markers",
        "",
        "| Marker | Status |",
        "|---|---|",
    ]
    for key, value in result["markers"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines += ["", "## Feasibility Markers", "", "| Marker | Status |", "|---|---|"]
    for key, value in result["feasibility_marker_status"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines += ["", "## 缺失内容组", ""]
    if result["missing_required_groups"]:
        for group in result["missing_required_groups"]:
            lines.append(f"- {group}")
    else:
        lines.append("无。")
    lines += ["", "## 误导性完成表述", ""]
    if result["false_claim_hits"]:
        for hit in result["false_claim_hits"]:
            lines.append(f"- `{hit['path']}:{hit['line']}` phrase `{hit['phrase']}`: {hit['text']}")
    else:
        lines.append("无。")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    result = validate()
    write_outputs(result)
    for key in [
        "TADSR_FINAL_PRESENTATION_PACKAGE",
        "TADSR_VIDEO_SCRIPT_READY",
        "TADSR_DEMO_RUNBOOK_READY",
        "TADSR_REPOSITORY_HANDOFF_READY",
        "TADSR_FINAL_SUBMISSION_README_READY",
        "TADSR_DEFENSE_RISK_RESPONSE_PACK",
        "TADSR_JITTOR_MIGRATION_FEASIBILITY_VALIDATION",
        "TADSR_FINAL_PRESENTATION_PACKAGE_VALIDATION",
    ]:
        print(f"{key}: {result['markers'][key]}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
