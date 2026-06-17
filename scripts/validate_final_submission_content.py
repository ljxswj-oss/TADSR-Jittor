#!/usr/bin/env python3
"""Validate final submission wording and guardrails without importing torch."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "experiments/final_submission_content_validation.json"
OUT_MD = ROOT / "experiments/final_submission_content_validation.md"

FILES = {
    "readme": "README.md",
    "ppt_markdown": "deliverables/TADSR-Jittor_final_presentation.md",
    "ppt_outline": "docs/03_ppt_outline.md",
    "video_script": "docs/04_video_script.md",
    "video_guide": "deliverables/TADSR-Jittor_video_recording_guide.md",
    "submission_readme": "deliverables/TADSR-Jittor_submission_readme.md",
    "final_checklist": "docs/final_submission_checklist.md",
    "feasibility_report": "experiments/jittor_migration_feasibility_summary.md",
    "final_audit_report": "experiments/final_audit_report.md",
    "full_inference_gap": "docs/full_inference_gap_analysis.md",
    "timevae_gap": "docs/timevae_full_alignment_gap_analysis.md",
    "lora_gap": "docs/lora_runtime_gap_analysis.md",
    "teacher_summary": "docs/final_teacher_status_summary.md",
    "phase5b_freeze_summary": "docs/final_phase5b_submission_freeze_summary.md",
    "diagnostic_image_smoke_plan": "docs/production_completion/diagnostic_image_smoke_plan.md",
    "timevae_closure": "docs/timevae_full_alignment_closure_plan.md",
    "runtime_lora_decision": "docs/runtime_lora_final_decision_proof.md",
    "diagnostic_image_smoke_metrics": "experiments/production_completion/diagnostic_image_smoke/diagnostic_image_smoke_metrics.md",
    "diagnostic_image_smoke_validation": "experiments/production_completion/diagnostic_image_smoke/diagnostic_image_smoke_validation.md",
    "diagnostic_cli": "experiments/production_completion/diagnostic_cli/one_step_diagnostic_cli_test.md",
    "course_requirement_matrix": "docs/course_requirement_compliance_matrix.md",
    "final_evidence_index": "docs/final_evidence_index.md",
    "training_evidence_summary": "docs/training_alignment_evidence_summary.md",
    "teacher_review_guide": "docs/teacher_review_guide.md",
    "final_defense_qa": "docs/final_defense_QA.md",
    "engineering_roadmap": "docs/full_engineering_completion_roadmap.md",
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

REQUIRED_GROUPS = [
    ["TADSR-Jittor"],
    ["TADSR_UNET_FULL_FORWARD_ALIGNMENT: PASS", "TADSR_UNET_FULL_FORWARD_ALIGNMENT"],
    ["TIME_VAE_ACTUAL_VAEHOOK_FULL_BOUNDARY_ALIGNMENT: PASS", "TIME_VAE_ACTUAL_VAEHOOK_FULL_BOUNDARY_ALIGNMENT"],
    ["TADSR_STATIC_EFFECTIVE_LORA_COVERAGE_AUDIT: PASS", "TADSR_STATIC_EFFECTIVE_LORA_COVERAGE_AUDIT"],
    ["TADSR_SCHEDULER_BOUNDARY_ALIGNMENT: PASS", "TADSR_SCHEDULER_BOUNDARY_ALIGNMENT"],
    ["TADSR_MINIMAL_ONE_STEP_DECODE_DRY_RUN: PASS", "TADSR_MINIMAL_ONE_STEP_DECODE_DRY_RUN"],
    ["JITTOR_FULL_INFERENCE: NOT_COMPLETE"],
    ["NotImplementedError"],
    ["static effective LoRA", "静态 effective LoRA"],
    ["TADSR_SMALL_DATA_TRAINING_READINESS: PASS", "TADSR_SMALL_DATA_TRAINING_READINESS"],
    ["TADSR_SMALL_DATA_TRAINING_ALIGNMENT"],
    ["TADSR_JITTOR_MIGRATION_FEASIBILITY_VALIDATION"],
    ["TADSR_BOUNDARY_LEVEL_REPRODUCTION"],
    ["TADSR_FULL_INFERENCE_GAP_ANALYSIS"],
    ["TADSR_TIMEVAE_FULL_ALIGNMENT_GAP_ANALYSIS"],
    ["TADSR_LORA_RUNTIME_GAP_ANALYSIS"],
    ["TADSR_GAP_ANALYSIS_READINESS"],
    ["TADSR_TIMEVAE_METADATA_PARTIAL_DIAGNOSIS"],
    ["TADSR_OFFICIAL_DIFFUSERS_OVERLAY_READY"],
    ["TADSR_OFFICIAL_RUNTIME_DEPENDENCY_REPAIR_READINESS"],
    ["TADSR_OFFICIAL_RUNTIME_DEPENDENCY_OVERLAY_ACTIVE"],
    ["TADSR_OFFICIAL_RUNTIME_DEPENDENCY_DIAGNOSIS"],
    ["TADSR_TIMEVAE_METADATA_REPAIR_ATTEMPT"],
    ["TADSR_TIMEVAE_METADATA_FIELDS_COMPLETE"],
    ["TADSR_TIMEVAE_METADATA_READY_FOR_ONE_STEP"],
    ["TADSR_TIMEVAE_LIVE_METADATA_COMPLETION"],
    ["TADSR_TIMEVAE_LIVE_ENCODE_METADATA"],
    ["TADSR_TIMEVAE_LIVE_DECODE_METADATA"],
    ["TADSR_TIMEVAE_LIVE_SAFETY_FLAGS"],
    ["TADSR_OFFICIAL_ACTUAL_INFERENCE_PATH_AUDIT"],
    ["TADSR_OFFICIAL_INFERENCE_MULTISTEP_REQUIREMENT_AUDIT"],
    ["TADSR_POSTPROCESS_CONTRACT_AUDIT"],
    ["TADSR_POSTPROCESS_NOT_EXECUTED_GUARD"],
    ["TADSR_TINY_MULTISTEP_ALIGNMENT_APPLICABILITY"],
    ["TADSR_TINY_MULTISTEP_ALIGNMENT_DECISION"],
    ["TADSR_EXPERIMENTAL_CLI_METADATA_ONLY_PLAN"],
    ["TADSR_PHASE5B_SUBMISSION_FREEZE_SUMMARY"],
    ["TADSR_DIAGNOSTIC_IMAGE_SMOKE_PLAN"],
    ["TADSR_DIAGNOSTIC_IMAGE_SMOKE_READY"],
    ["TADSR_DIAGNOSTIC_IMAGE_SMOKE_EXECUTED"],
    ["TADSR_DIAGNOSTIC_IMAGE_SMOKE_NOT_FULL_INFERENCE"],
    ["TADSR_DIAGNOSTIC_IMAGE_SMOKE_NO_RAW_TENSORS"],
    ["TADSR_DIAGNOSTIC_IMAGE_SMOKE_VALIDATION"],
    ["TADSR_ONE_STEP_DIAGNOSTIC_CLI_READY"],
    ["TADSR_ONE_STEP_DIAGNOSTIC_CLI_NOT_FULL_INFERENCE"],
    ["TADSR_TIMEVAE_FULL_ALIGNMENT_CLOSURE_SUMMARY"],
    ["TADSR_TIMEVAE_FULL_ALIGNMENT_REMAINS_SCOPED"],
    ["TADSR_RUNTIME_LORA_FINAL_DECISION_PROOF"],
    ["TADSR_DYNAMIC_RUNTIME_LORA_NOT_REQUIRED_FOR_AUDITED_PATH"],
    ["TADSR_OFFICIAL_ACTUAL_SINGLE_STEP_PATH_DOCUMENTED"],
    ["TADSR_MULTISTEP_NOT_REQUIRED_FOR_OFFICIAL_ACTUAL_PATH"],
    ["TADSR_CONTROLLED_ONE_STEP_EVIDENCE_DOCUMENTED"],
    ["TADSR_FINAL_SUBMISSION_AFTER_PHASE5B_READY"],
    ["TADSR_FINAL_SUBMISSION_WITH_DIAGNOSTIC_SMOKE_READY"],
    ["single-step / get_x0_from_res", "single_step_get_x0_from_res"],
    ["tiny multi-step alignment 不是 official actual inference 的必需项", "tiny multi-step is not an official requirement"],
    ["diagnostic image-smoke"],
    ["NOT_EXECUTED"],
    ["TADSR_RUNTIME_LORA_DECISION_FINALIZED_FOR_FIXED_INFERENCE"],
    ["TADSR_SMOKE_TRAINING_SUBMISSION_SUMMARY"],
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
]

REQUIRED_GROUPS.extend([
    ["TADSR_DEFENSE_RISK_RESPONSE_PACK"],
    ["TADSR_DEFENSE_SHORT_ANSWERS_READY"],
    ["TADSR_DEFENSE_LONG_ANSWERS_READY"],
    ["TADSR_DEFENSE_FALSE_CLAIM_GUARD"],
    ["TADSR_DEFENSE_EVIDENCE_LOOKUP_READY"],
    ["TADSR_FINAL_RELEASE_CANDIDATE_SIGNOFF"],
    ["TADSR_FINAL_RELEASE_CANDIDATE_SCOPE_GUARD"],
    ["TADSR_FINAL_LINKS_AND_PATHS_VALIDATION"],
    ["TADSR_FINAL_CHINESE_MATERIALS_VALIDATION"],
    ["TADSR_FINAL_COMMAND_INDEX"],
    ["TADSR_FINAL_GITHUB_UPLOAD_CHECKLIST"],
    ["TADSR_FINAL_READY_FOR_HUMAN_SUBMISSION"],
    ["TADSR_CLEAN_PUBLIC_RELEASE_PACKAGE_BUILD"],
    ["TADSR_CLEAN_PUBLIC_RELEASE_PACKAGE_VALIDATION"],
    ["TADSR_FINAL_GITHUB_URL_UPDATE_SCRIPT_READY"],
    ["TADSR_FINAL_HUMAN_SUBMISSION_INSTRUCTIONS"],
    ["TADSR_FINAL_VIDEO_REHEARSAL_CHECKLIST"],
    ["TADSR_FINAL_SUBMISSION_FREEZE_TAG_DOC"],
    ["TADSR_FINAL_HUMAN_SUBMISSION_LOCK_REPORT"],
    ["TADSR_FINAL_VIDEO_SUBMISSION_CHECK"],
])

FALSE_CLAIMS = [
    "full inference complete",
    "production pipeline complete",
    "image generation complete",
    "video generation complete",
    "image/video generation complete",
    "full TADSR training complete",
    "dynamic runtime LoRA complete",
    "runtime LoRA complete",
    "TimeVAE full alignment complete",
]

FORBIDDEN_MARKERS = [
    "JITTOR_FULL_INFERENCE: PASS",
    "TIME_VAE_FULL_ALIGNMENT: PASS",
]

NEGATION_ALLOWLIST = [
    "not complete",
    "not implemented",
    "not claimed",
    "do not claim",
    "does not claim",
    "no ",
    "without",
    "never",
    "intentionally guarded",
    "NOT_COMPLETE",
    "NotImplementedError",
    "未完成",
    "未实现",
    "不声称",
    "没有",
    "不会",
    "不包含",
]


def rel(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def read_text(path: str | Path) -> str:
    p = ROOT / path
    return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""


def read_json(path: str | Path) -> dict:
    p = ROOT / path
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}


def has_any(text: str, variants: list[str]) -> bool:
    return any(variant in text for variant in variants)


def is_negated(line: str) -> bool:
    low = line.lower()
    return any(token.lower() in low for token in NEGATION_ALLOWLIST)


def scan_false_claims(text_by_file: dict[str, str]) -> list[dict[str, object]]:
    hits: list[dict[str, object]] = []
    for name, text in text_by_file.items():
        for lineno, line in enumerate(text.splitlines(), 1):
            low = line.lower()
            for claim in FALSE_CLAIMS:
                if claim.lower() in low and not is_negated(line):
                    hits.append({"file": name, "line": lineno, "phrase": claim, "text": line.strip()})
            for marker in FORBIDDEN_MARKERS:
                if marker in line:
                    hits.append({"file": name, "line": lineno, "phrase": marker, "text": line.strip()})
    return hits


def marker_status(marker: str) -> str:
    feasibility = read_json("experiments/jittor_migration_feasibility_summary.json").get("markers", {})
    if marker in feasibility:
        return str(feasibility[marker])
    for row in read_json("experiments/final_audit_report.json").get("rows", []):
        if row.get("check") == marker:
            return str(row.get("status", "MISSING"))
    return "MISSING"


def main() -> int:
    file_status = {}
    text_by_file = {}
    for name, rel_path in FILES.items():
        path = ROOT / rel_path
        file_status[name] = {
            "path": rel(rel_path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
        }
        text_by_file[name] = read_text(rel_path)

    combined_text = "\n".join(text_by_file.values())
    missing_groups = [group for group in REQUIRED_GROUPS if not has_any(combined_text, group)]
    false_claim_hits = scan_false_claims(text_by_file)

    feasibility_markers = {
        key: marker_status(key)
        for key in [
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
            "TADSR_SUBMISSION_FACING_STATUS_SUMMARY",
            "TADSR_ENVIRONMENT_BLOCKERS_EXPLAINED",
            "TADSR_GAP_ANALYSIS_READINESS",
        ]
    }

    files_ok = all(item["exists"] and item["size_bytes"] > 0 for item in file_status.values())
    terms_ok = not missing_groups
    false_claims_ok = not false_claim_hits
    feasibility_ok = all(v == "PASS" for v in feasibility_markers.values())
    defense_validation = read_json("experiments/defense_risk_response_pack_validation.json")
    defense_markers = defense_validation.get("markers", {}) if isinstance(defense_validation, dict) else {}
    required_defense_markers = [
        "TADSR_DEFENSE_RISK_RESPONSE_PACK",
        "TADSR_DEFENSE_SHORT_ANSWERS_READY",
        "TADSR_DEFENSE_LONG_ANSWERS_READY",
        "TADSR_DEFENSE_FALSE_CLAIM_GUARD",
        "TADSR_DEFENSE_EVIDENCE_LOOKUP_READY",
    ]
    defense_ok = all(defense_markers.get(marker) == "PASS" for marker in required_defense_markers)
    final_qa_sources = [
        ("experiments/final_release_candidate_signoff.json", "TADSR_FINAL_RELEASE_CANDIDATE_SIGNOFF"),
        ("experiments/final_links_and_paths_validation.json", "TADSR_FINAL_LINKS_AND_PATHS_VALIDATION"),
        ("experiments/final_chinese_materials_validation.json", "TADSR_FINAL_CHINESE_MATERIALS_VALIDATION"),
        ("experiments/final_command_index.json", "TADSR_FINAL_COMMAND_INDEX"),
        ("experiments/final_github_upload_checklist.json", "TADSR_FINAL_GITHUB_UPLOAD_CHECKLIST"),
        ("experiments/clean_public_release_package_manifest.json", "TADSR_CLEAN_PUBLIC_RELEASE_PACKAGE_BUILD"),
        ("experiments/clean_public_release_package_validation.json", "TADSR_CLEAN_PUBLIC_RELEASE_PACKAGE_VALIDATION"),
        ("experiments/final_github_url_status.json", "TADSR_FINAL_GITHUB_URL_UPDATE_SCRIPT_READY"),
        ("experiments/final_human_submission_instructions.json", "TADSR_FINAL_HUMAN_SUBMISSION_INSTRUCTIONS"),
        ("experiments/final_video_rehearsal_checklist.json", "TADSR_FINAL_VIDEO_REHEARSAL_CHECKLIST"),
        ("experiments/final_submission_freeze_tag.json", "TADSR_FINAL_SUBMISSION_FREEZE_TAG_DOC"),
        ("experiments/final_human_submission_lock_report.json", "TADSR_FINAL_HUMAN_SUBMISSION_LOCK_REPORT"),
        ("experiments/final_video_submission_check.json", "TADSR_FINAL_VIDEO_SUBMISSION_CHECK"),
    ]
    final_qa_status = {}
    for path, marker in final_qa_sources:
        data = read_json(path)
        final_qa_status[marker] = str(data.get("markers", {}).get(marker, "MISSING")) if isinstance(data, dict) else "MISSING"
    final_qa_ok = all(value == "PASS" for value in final_qa_status.values())

    markers = {
        "TADSR_FINAL_PPT_CONTENT_GUARDRAIL": "PASS" if files_ok and terms_ok and false_claims_ok else "PARTIAL",
        "TADSR_FINAL_VIDEO_SCRIPT_GUARDRAIL": "PASS" if file_status["video_script"]["exists"] and false_claims_ok else "PARTIAL",
        "TADSR_FINAL_SUBMISSION_README_GUARDRAIL": "PASS" if file_status["submission_readme"]["exists"] and false_claims_ok else "PARTIAL",
        "TADSR_JITTOR_MIGRATION_FEASIBILITY_VALIDATION": "PASS" if feasibility_ok else "PARTIAL",
        "TADSR_FINAL_PERFECTION_CONTENT_READY": "PASS" if all(file_status[name]["exists"] for name in [
            "course_requirement_matrix",
            "final_evidence_index",
            "training_evidence_summary",
            "teacher_review_guide",
            "final_defense_qa",
            "engineering_roadmap",
        ]) and false_claims_ok else "PARTIAL",
        "TADSR_DEFENSE_RISK_RESPONSE_CONTENT_READY": "PASS" if defense_ok and false_claims_ok else "PARTIAL",
        "TADSR_FINAL_RELEASE_CANDIDATE_CONTENT_READY": "PASS" if final_qa_ok and false_claims_ok else "PARTIAL",
    }
    markers["TADSR_FINAL_SUBMISSION_CONTENT_VALIDATION"] = (
        "PASS" if files_ok and terms_ok and false_claims_ok and feasibility_ok and defense_ok and final_qa_ok else "PARTIAL"
    )

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": markers["TADSR_FINAL_SUBMISSION_CONTENT_VALIDATION"],
        "files": file_status,
        "missing_required_groups": missing_groups,
        "false_claim_hits": false_claim_hits,
        "feasibility_marker_status": feasibility_markers,
        "final_qa_marker_status": final_qa_status,
        "markers": markers,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Final Submission Content Validation",
        "",
        f"Status: **{result['status']}**",
        "",
        "## Markers",
        "",
        "| Marker | Status |",
        "|---|---|",
    ]
    for marker, status in markers.items():
        lines.append(f"| `{marker}` | `{status}` |")
    lines.extend(["", "## Feasibility Markers", "", "| Marker | Status |", "|---|---|"])
    for marker, status in feasibility_markers.items():
        lines.append(f"| `{marker}` | `{status}` |")
    lines.extend(["", "## Final QA Markers", "", "| Marker | Status |", "|---|---|"])
    for marker, status in final_qa_status.items():
        lines.append(f"| `{marker}` | `{status}` |")
    lines.extend(["", "## Missing Required Groups", ""])
    if missing_groups:
        for group in missing_groups:
            lines.append(f"- {group}")
    else:
        lines.append("None.")
    lines.extend(["", "## False Claim Hits", ""])
    if false_claim_hits:
        for hit in false_claim_hits:
            lines.append(f"- `{hit['file']}:{hit['line']}` phrase `{hit['phrase']}`: {hit['text']}")
    else:
        lines.append("None.")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for marker in [
        "TADSR_FINAL_SUBMISSION_CONTENT_VALIDATION",
        "TADSR_FINAL_PPT_CONTENT_GUARDRAIL",
        "TADSR_FINAL_VIDEO_SCRIPT_GUARDRAIL",
        "TADSR_FINAL_SUBMISSION_README_GUARDRAIL",
        "TADSR_JITTOR_MIGRATION_FEASIBILITY_VALIDATION",
        "TADSR_DEFENSE_RISK_RESPONSE_CONTENT_READY",
        "TADSR_FINAL_RELEASE_CANDIDATE_CONTENT_READY",
    ]:
        print(f"{marker}: {markers[marker]}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
