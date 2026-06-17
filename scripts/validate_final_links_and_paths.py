#!/usr/bin/env python3
"""Validate final Markdown links, critical paths and placeholders.

This is a text-only release QA script. It intentionally does not import
PyTorch/Jittor and does not execute model code.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "experiments/final_links_and_paths_validation.json"
OUT_MD = ROOT / "experiments/final_links_and_paths_validation.md"

SCAN_ROOTS = [
    ROOT / "README.md",
    ROOT / "docs",
    ROOT / "deliverables",
    ROOT / "experiments",
]

CRITICAL_PATHS = [
    "README.md",
    "docs/final_release_candidate_signoff.md",
    "docs/final_command_index.md",
    "docs/final_github_upload_checklist.md",
    "docs/final_human_submission_instructions.md",
    "docs/final_video_rehearsal_checklist.md",
    "docs/final_submission_freeze_tag.md",
    "docs/final_human_submission_lock_report.md",
    "docs/final_video_submission_check.md",
    "docs/final_defense_risk_response_pack.md",
    "docs/defense_short_answers_zh.md",
    "docs/defense_long_answers_zh.md",
    "docs/defense_do_not_say.md",
    "docs/defense_evidence_lookup_table.md",
    "docs/course_requirement_compliance_matrix.md",
    "docs/final_evidence_index.md",
    "docs/training_alignment_evidence_summary.md",
    "docs/teacher_review_guide.md",
    "docs/final_teacher_status_summary.md",
    "docs/final_submission_checklist.md",
    "deliverables/TADSR-Jittor_final_presentation.md",
    "deliverables/TADSR-Jittor_final_presentation.pptx",
    "deliverables/TADSR-Jittor_final_presentation.pdf",
    "deliverables/TADSR-Jittor_video_recording_guide.md",
    "deliverables/TADSR-Jittor_submission_readme.md",
    "experiments/final_audit_report.md",
    "experiments/final_release_candidate_signoff.md",
    "experiments/final_command_index.md",
    "experiments/final_github_upload_checklist.md",
    "experiments/final_human_submission_instructions.md",
    "experiments/final_video_rehearsal_checklist.md",
    "experiments/final_submission_freeze_tag.md",
    "experiments/final_human_submission_lock_report.md",
    "experiments/final_video_submission_check.md",
    "experiments/clean_public_release_package_manifest.md",
    "experiments/clean_public_release_package_validation.md",
    "experiments/final_github_url_status.md",
    "scripts/build_clean_public_release_package.py",
    "scripts/validate_clean_public_release_package.py",
    "scripts/set_final_github_url.py",
]

PLACEHOLDER_PATTERNS = [
    "your_github_repo_here",
    "<GitHub URL>",
    "YOUR_GITHUB",
    "YOUR_USERNAME",
    "TODO",
    "FIXME",
    "待补充",
    "填写 GitHub 链接",
]

LINK_RE = re.compile(r"(!?\[[^\]]*\]\(([^)]+)\))")
CODE_FENCE_RE = re.compile(r"^\s*```")


def iter_markdown_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        if root.is_file() and root.suffix.lower() == ".md":
            files.append(root)
        elif root.is_dir():
            files.extend(path for path in root.rglob("*.md") if path.is_file())
    return sorted(set(files))


def normalize_link(raw: str) -> str:
    target = raw.strip().strip("<>").strip()
    if " " in target and not target.startswith(("http://", "https://")):
        target = target.split()[0]
    return unquote(target.split("#", 1)[0])


def is_external_or_special(target: str) -> bool:
    lowered = target.lower()
    return (
        not target
        or lowered.startswith(("http://", "https://", "mailto:", "tel:"))
        or target.startswith("#")
        or lowered.startswith("javascript:")
    )


def resolve_relative(source: Path, target: str) -> Path:
    target_path = Path(target)
    if target_path.is_absolute():
        return target_path
    return (source.parent / target_path).resolve()


def is_generated_context(line: str) -> bool:
    lowered = line.lower()
    return any(token in lowered for token in ["future work", "下一步", "示例", "example", "占位说明"])


def scan_links(files: list[Path]) -> list[dict[str, object]]:
    broken: list[dict[str, object]] = []
    for path in files:
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        in_code = False
        for lineno, line in enumerate(lines, 1):
            if CODE_FENCE_RE.match(line):
                in_code = not in_code
                continue
            if in_code:
                continue
            for _, raw_target in LINK_RE.findall(line):
                target = normalize_link(raw_target)
                if is_external_or_special(target):
                    continue
                if target.startswith(("C:", "D:", "/mnt/", "\\\\")):
                    # Absolute local paths are allowed as execution notes, not as GitHub critical links.
                    continue
                if any(ch in target for ch in ["{", "}"]):
                    continue
                resolved = resolve_relative(path, target)
                if not resolved.exists():
                    broken.append(
                        {
                            "file": str(path.relative_to(ROOT)).replace("\\", "/"),
                            "line": lineno,
                            "target": raw_target,
                            "resolved": str(resolved),
                            "critical": str(path.relative_to(ROOT)).replace("\\", "/") in {"README.md"},
                        }
                    )
    return broken


def scan_placeholders(files: list[Path]) -> list[dict[str, object]]:
    hits: list[dict[str, object]] = []
    for path in files:
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for lineno, line in enumerate(lines, 1):
            for pattern in PLACEHOLDER_PATTERNS:
                if pattern in line and not is_generated_context(line):
                    hits.append(
                        {
                            "file": str(path.relative_to(ROOT)).replace("\\", "/"),
                            "line": lineno,
                            "pattern": pattern,
                            "text": line.strip(),
                            "critical": path.name in {"README.md", "TADSR-Jittor_submission_readme.md"},
                        }
                    )
    return hits


def validate() -> dict:
    files = iter_markdown_files()
    broken_links = scan_links(files)
    placeholder_hits = scan_placeholders(files)
    missing_critical = [rel for rel in CRITICAL_PATHS if not (ROOT / rel).exists()]
    broken_critical = [hit for hit in broken_links if hit.get("critical")]
    placeholder_critical = [hit for hit in placeholder_hits if hit.get("critical")]

    critical_ok = not missing_critical and not broken_critical and not placeholder_critical
    markdown_links_status = "PASS" if not broken_links else "PARTIAL"
    placeholder_status = "PASS" if not placeholder_hits else "PARTIAL"
    overall_status = "PASS" if critical_ok else "FAIL"

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": overall_status,
        "markdown_file_count": len(files),
        "missing_critical_paths": missing_critical,
        "broken_links": broken_links,
        "placeholder_hits": placeholder_hits,
        "markers": {
            "TADSR_FINAL_LINKS_AND_PATHS_VALIDATION": overall_status,
            "TADSR_FINAL_MARKDOWN_LINKS_VALIDATION": markdown_links_status,
            "TADSR_FINAL_PLACEHOLDER_SCAN": placeholder_status,
        },
    }


def write_outputs(result: dict) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# 最终链接、路径与占位符检查",
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
    lines += ["", "## 缺失关键路径", ""]
    if result["missing_critical_paths"]:
        for item in result["missing_critical_paths"]:
            lines.append(f"- `{item}`")
    else:
        lines.append("无。")
    lines += ["", "## Markdown 链接警告", ""]
    if result["broken_links"]:
        for hit in result["broken_links"][:80]:
            lines.append(f"- `{hit['file']}:{hit['line']}` -> `{hit['target']}`")
        if len(result["broken_links"]) > 80:
            lines.append(f"- 其余 {len(result['broken_links']) - 80} 条见 JSON。")
    else:
        lines.append("无。")
    lines += ["", "## 占位符警告", ""]
    if result["placeholder_hits"]:
        for hit in result["placeholder_hits"][:80]:
            lines.append(f"- `{hit['file']}:{hit['line']}` `{hit['pattern']}`: {hit['text']}")
        if len(result["placeholder_hits"]) > 80:
            lines.append(f"- 其余 {len(result['placeholder_hits']) - 80} 条见 JSON。")
    else:
        lines.append("无。")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    result = validate()
    write_outputs(result)
    for marker, status in result["markers"].items():
        print(f"{marker}: {status}")
    print(f"broken_link_count: {len(result['broken_links'])}")
    print(f"placeholder_count: {len(result['placeholder_hits'])}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
