#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


OUT_JSON = Path("experiments/final_deliverables_validation.json")
OUT_MD = Path("experiments/final_deliverables_validation.md")


DELIVERABLES = [
    Path("deliverables/TADSR-Jittor_final_presentation.md"),
    Path("deliverables/TADSR-Jittor_final_presentation.pptx"),
    Path("deliverables/TADSR-Jittor_final_presentation.pdf"),
    Path("deliverables/TADSR-Jittor_video_recording_guide.md"),
    Path("deliverables/TADSR-Jittor_submission_readme.md"),
    Path("deliverables/TADSR-Jittor_final_checklist.md"),
]

MISLEADING_PHRASES = [
    "full inference complete",
    "production pipeline complete",
    "image generation complete",
    "video generation complete",
    "image/video generation complete",
    "dynamic runtime LoRA complete",
]

NEGATIONS = [
    "not",
    "no ",
    "do not",
    "does not",
    "without",
    "never",
    "NOT_COMPLETE",
    "NOT_IMPLEMENTED",
    "不",
    "没有",
    "未",
    "不要",
]


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def line_is_negated(line: str) -> bool:
    low = line.lower()
    return any(token.lower() in low for token in NEGATIONS)


def find_false_claims(paths: list[Path]) -> list[dict]:
    hits = []
    for path in paths:
        if not path.exists() or path.suffix.lower() not in {".md", ".txt", ".py"}:
            continue
        for lineno, line in enumerate(read_text(path).splitlines(), 1):
            low = line.lower()
            for phrase in MISLEADING_PHRASES:
                if phrase.lower() in low and not line_is_negated(line):
                    hits.append({"path": str(path), "line": lineno, "phrase": phrase, "text": line.strip()})
    return hits


def audit_status(check: str) -> str:
    report = Path("experiments/final_audit_report.json")
    if not report.exists():
        return "MISSING"
    data = json.loads(report.read_text(encoding="utf-8"))
    for row in data.get("rows", []):
        if row.get("check") == check:
            return str(row.get("status", "MISSING"))
    return "MISSING"


def contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def contains_all_flexible(text: str, groups: list[list[str]]) -> bool:
    return all(contains_any(text, group) for group in groups)


def validate() -> dict:
    files = {str(path): {"exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0} for path in DELIVERABLES}
    total_size = sum(item["size_bytes"] for item in files.values())
    false_claim_hits = find_false_claims(DELIVERABLES + [
        Path("README.md"),
        Path("docs/03_ppt_outline.md"),
        Path("docs/04_video_script.md"),
        Path("docs/final_presentation_handoff.md"),
        Path("docs/final_demo_runbook.md"),
        Path("docs/repository_handoff_guide.md"),
        Path("docs/final_phase5b_submission_freeze_summary.md"),
        Path("docs/production_completion/diagnostic_image_smoke_plan.md"),
        Path("docs/timevae_full_alignment_closure_plan.md"),
        Path("docs/runtime_lora_final_decision_proof.md"),
        Path("docs/final_defense_risk_response_pack.md"),
        Path("docs/defense_short_answers_zh.md"),
        Path("docs/defense_long_answers_zh.md"),
        Path("docs/defense_do_not_say.md"),
        Path("docs/defense_evidence_lookup_table.md"),
        Path("docs/final_release_candidate_signoff.md"),
        Path("docs/final_command_index.md"),
        Path("docs/final_github_upload_checklist.md"),
        Path("docs/final_human_submission_instructions.md"),
        Path("docs/final_video_rehearsal_checklist.md"),
        Path("docs/final_submission_freeze_tag.md"),
        Path("docs/final_human_submission_lock_report.md"),
        Path("docs/final_video_submission_check.md"),
        Path("experiments/final_links_and_paths_validation.md"),
        Path("experiments/final_chinese_materials_validation.md"),
        Path("experiments/clean_public_release_package_manifest.md"),
        Path("experiments/clean_public_release_package_validation.md"),
        Path("experiments/final_github_url_status.md"),
        Path("experiments/production_completion/diagnostic_image_smoke/diagnostic_image_smoke_metrics.md"),
        Path("experiments/production_completion/diagnostic_image_smoke/diagnostic_image_smoke_validation.md"),
        Path("experiments/production_completion/diagnostic_cli/one_step_diagnostic_cli_test.md"),
    ])
    ppt_markdown = DELIVERABLES[0]
    pptx = DELIVERABLES[1]
    pdf = DELIVERABLES[2]
    video_guide = DELIVERABLES[3]
    submission_readme = DELIVERABLES[4]
    ppt_ready = files[str(pptx)]["exists"] and files[str(pptx)]["size_bytes"] > 0
    pdf_ready = files[str(pdf)]["exists"] and files[str(pdf)]["size_bytes"] > 0
    video_ready = files[str(video_guide)]["exists"] and contains_any(
        read_text(video_guide),
        ["20-30 minutes", "20-30 分钟", "20–30 分钟"],
    )
    submission_text = read_text(submission_readme)
    submission_ready = (
        files[str(submission_readme)]["exists"]
        and "JITTOR_FULL_INFERENCE" in submission_text
        and "NOT_COMPLETE" in submission_text
    )
    size_ok = total_size < 100 * 1024 * 1024
    guard_ok = audit_status("JITTOR_FULL_INFERENCE") == "NOT_COMPLETE"
    no_false_claims = len(false_claim_hits) == 0
    ppt_smoke = files[str(ppt_markdown)]["exists"] and contains_all_flexible(
        read_text(ppt_markdown),
        [
            ["TADSR_SMALL_DATA_TRAINING_READINESS: PASS", "小数据 PyTorch-vs-Jittor 训练对齐"],
            ["24 train", "24 个训练", "24 train / 8 validation"],
            ["8 validation", "8 个验证", "24 train / 8 validation"],
            ["prediction relative L2"],
            ["multi-seed", "多 seed"],
        ],
    )
    video_smoke = contains_all_flexible(
        read_text(video_guide) + "\n" + read_text(Path("docs/04_video_script.md")),
        [
            ["Small-Data Smoke Training", "小数据", "PyTorch-vs-Jittor"],
            ["output-tail"],
            ["train/validation loss", "train/validation loss log", "loss 曲线", "loss log", "loss"],
            ["multi-seed", "多 seed", "seed"],
        ],
    )
    readme_smoke = contains_all_flexible(
        read_text(submission_readme),
        [
            ["Small-Data Smoke Training Evidence", "小数据训练对齐"],
            ["24 training samples", "24 train", "24 个训练"],
            ["8 validation samples", "8 validation", "8 个验证"],
            ["multi-seed stability", "多 seed", "multi-seed"],
        ],
    )
    smoke_included = ppt_smoke and video_smoke and readme_smoke
    defense_docs = [
        Path("docs/final_defense_risk_response_pack.md"),
        Path("docs/defense_short_answers_zh.md"),
        Path("docs/defense_long_answers_zh.md"),
        Path("docs/defense_do_not_say.md"),
        Path("docs/defense_evidence_lookup_table.md"),
    ]
    defense_pack_ready = all(path.exists() and path.stat().st_size > 0 for path in defense_docs)
    final_qa_docs = [
        Path("docs/final_release_candidate_signoff.md"),
        Path("docs/final_command_index.md"),
        Path("docs/final_github_upload_checklist.md"),
        Path("docs/final_human_submission_instructions.md"),
        Path("docs/final_video_rehearsal_checklist.md"),
        Path("docs/final_submission_freeze_tag.md"),
        Path("docs/final_human_submission_lock_report.md"),
        Path("docs/final_video_submission_check.md"),
        Path("experiments/final_human_submission_lock_report.md"),
        Path("experiments/final_video_submission_check.md"),
        Path("experiments/final_links_and_paths_validation.md"),
        Path("experiments/final_chinese_materials_validation.md"),
        Path("experiments/clean_public_release_package_manifest.md"),
        Path("experiments/clean_public_release_package_validation.md"),
        Path("experiments/final_github_url_status.md"),
    ]
    final_qa_ready = all(path.exists() and path.stat().st_size > 0 for path in final_qa_docs)
    phase4_text = "\n".join(
        read_text(path)
        for path in [
            Path("README.md"),
            Path("docs/03_ppt_outline.md"),
            Path("docs/04_video_script.md"),
            Path("docs/final_teacher_status_summary.md"),
            Path("deliverables/TADSR-Jittor_final_presentation.md"),
            Path("deliverables/TADSR-Jittor_submission_readme.md"),
            Path("docs/final_phase5b_submission_freeze_summary.md"),
            Path("docs/production_completion/diagnostic_image_smoke_plan.md"),
        ]
    )
    phase4_evidence = contains_all_flexible(
        phase4_text,
        [
            ["TADSR_TIMEVAE_METADATA_PARTIAL_DIAGNOSIS"],
            ["TADSR_OFFICIAL_DIFFUSERS_OVERLAY_READY"],
            ["TADSR_OFFICIAL_RUNTIME_DEPENDENCY_REPAIR_READINESS"],
            ["TADSR_OFFICIAL_RUNTIME_DEPENDENCY_OVERLAY_ACTIVE"],
            ["TADSR_OFFICIAL_RUNTIME_DEPENDENCY_DIAGNOSIS"],
            ["TADSR_TIMEVAE_METADATA_REPAIR_ATTEMPT"],
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
            ["TADSR_SMOKE_TRAINING_SUBMISSION_SUMMARY"],
        ],
    )
    overall = (
        ppt_ready
        and pdf_ready
        and video_ready
        and submission_ready
        and smoke_included
        and defense_pack_ready
        and final_qa_ready
        and phase4_evidence
        and size_ok
        and guard_ok
        and no_false_claims
    )
    marker = lambda ok: "PASS" if ok else "PARTIAL"
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if overall else "PARTIAL",
        "files": files,
        "total_size_bytes": total_size,
        "total_size_mb": total_size / (1024 * 1024),
        "false_claim_hits": false_claim_hits,
        "markers": {
            "TADSR_FINAL_PPT_READY": marker(ppt_ready),
            "TADSR_FINAL_PDF_READY": marker(pdf_ready),
            "TADSR_FINAL_VIDEO_RECORDING_PACKAGE_READY": marker(video_ready),
            "TADSR_FINAL_SUBMISSION_README_READY": marker(submission_ready),
            "TADSR_FINAL_DELIVERABLE_SIZE_AUDIT": marker(size_ok),
            "TADSR_FINAL_DELIVERABLES_INCLUDE_SMOKE_TRAINING": marker(smoke_included),
            "TADSR_FINAL_DELIVERABLES_INCLUDE_DEFENSE_RISK_PACK": marker(defense_pack_ready),
            "TADSR_FINAL_DELIVERABLES_INCLUDE_RELEASE_QA": marker(final_qa_ready),
            "TADSR_FINAL_PPT_INCLUDES_SMOKE_TRAINING": marker(ppt_smoke),
            "TADSR_VIDEO_SCRIPT_INCLUDES_SMOKE_TRAINING": marker(video_smoke),
            "TADSR_SUBMISSION_README_INCLUDES_SMOKE_TRAINING": marker(readme_smoke),
            "TADSR_FINAL_DELIVERABLES_INCLUDE_PHASE4A_EVIDENCE": marker(phase4_evidence),
            "TADSR_FINAL_DELIVERABLES_READY": marker(overall),
        },
        "guard": {
            "JITTOR_FULL_INFERENCE": audit_status("JITTOR_FULL_INFERENCE"),
            "JITTOR_FULL_PORT": audit_status("JITTOR_FULL_PORT"),
        },
    }


def write_outputs(result: dict) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# 最终提交材料验证",
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
    lines += ["", "## 文件", "", "| 路径 | 是否存在 | 字节数 |", "|---|---:|---:|"]
    for path, info in result["files"].items():
        lines.append(f"| `{path}` | `{info['exists']}` | {info['size_bytes']} |")
    lines += ["", f"总大小：**{result['total_size_mb']:.3f} MB**", ""]
    lines += ["## 误导性表述命中", ""]
    if result["false_claim_hits"]:
        for hit in result["false_claim_hits"]:
            lines.append(f"- `{hit['path']}:{hit['line']}` phrase `{hit['phrase']}`: {hit['text']}")
    else:
        lines.append("无。")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    result = validate()
    write_outputs(result)
    for key, value in result["markers"].items():
        print(f"{key}: {value}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
