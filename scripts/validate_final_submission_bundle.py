#!/usr/bin/env python3
"""Validate a compact final submission materials folder without creating archives."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


FORBIDDEN_SUFFIXES = {
    ".npy",
    ".npz",
    ".zip",
    ".pth",
    ".pt",
    ".ckpt",
    ".safetensors",
    ".pkl",
}
FORBIDDEN_DIR_NAMES = {"local_tensors", "__pycache__", "weights", "checkpoints"}


def scan_files(materials_root: Path) -> tuple[list[dict[str, object]], int, int]:
    forbidden: list[dict[str, object]] = []
    total_size = 0
    file_count = 0
    for root, dirs, files in os.walk(materials_root):
        root_path = Path(root)
        dirs[:] = [d for d in dirs if d != ".git"]
        for name in files:
            path = root_path / name
            file_count += 1
            try:
                total_size += path.stat().st_size
            except OSError:
                pass
            rel = path.relative_to(materials_root)
            if path.suffix.lower() in FORBIDDEN_SUFFIXES or any(part in FORBIDDEN_DIR_NAMES for part in rel.parts):
                forbidden.append({"path": str(rel).replace("\\", "/"), "size_bytes": path.stat().st_size})
    return forbidden, total_size, file_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--materials-root", required=True)
    parser.add_argument("--create-zip", default="0")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    materials_root = Path(args.materials_root).resolve()
    out_json = repo_root / "experiments/final_submission_bundle_validation.json"
    out_md = repo_root / "experiments/final_submission_bundle_validation.md"

    forbidden, total_size, file_count = scan_files(materials_root)
    required = [
        "README.md",
        "docs/final_phase5b_submission_freeze_summary.md",
        "docs/production_completion/diagnostic_image_smoke_plan.md",
        "deliverables/TADSR-Jittor_submission_readme.md",
        "experiments/final_phase5b_submission_freeze_summary.json",
        "experiments/production_completion/full_inference/diagnostic_image_smoke_plan.json",
        "experiments/production_completion/diagnostic_image_smoke/diagnostic_image_smoke_metrics.json",
        "experiments/production_completion/diagnostic_image_smoke/diagnostic_image_smoke_validation.json",
        "experiments/production_completion/diagnostic_cli/one_step_diagnostic_cli_test.json",
        "docs/timevae_full_alignment_closure_plan.md",
        "experiments/timevae_full_alignment_closure_summary.json",
        "docs/runtime_lora_final_decision_proof.md",
        "experiments/runtime_lora_final_decision_proof.json",
        "docs/final_release_candidate_signoff.md",
        "docs/final_command_index.md",
        "docs/final_github_upload_checklist.md",
        "docs/final_defense_risk_response_pack.md",
        "docs/defense_short_answers_zh.md",
        "docs/defense_long_answers_zh.md",
        "docs/defense_do_not_say.md",
        "docs/defense_evidence_lookup_table.md",
        "experiments/final_release_candidate_signoff.json",
        "experiments/final_links_and_paths_validation.json",
        "experiments/final_chinese_materials_validation.json",
        "experiments/final_command_index.json",
        "experiments/final_github_upload_checklist.json",
        "scripts/validate_final_links_and_paths.py",
        "scripts/validate_final_chinese_materials.py",
        "docs/final_human_submission_instructions.md",
        "docs/final_video_rehearsal_checklist.md",
        "docs/final_submission_freeze_tag.md",
        "docs/final_human_submission_lock_report.md",
        "docs/final_video_submission_check.md",
        "experiments/final_human_submission_instructions.json",
        "experiments/final_video_rehearsal_checklist.json",
        "experiments/final_submission_freeze_tag.json",
        "experiments/final_human_submission_lock_report.json",
        "experiments/final_video_submission_check.json",
        "experiments/clean_public_release_package_manifest.json",
        "experiments/clean_public_release_package_validation.json",
        "experiments/final_github_url_status.json",
        "scripts/build_clean_public_release_package.py",
        "scripts/validate_clean_public_release_package.py",
        "scripts/set_final_github_url.py",
    ]
    missing = [rel for rel in required if not (materials_root / rel).exists()]
    size_ok = total_size < 100 * 1024 * 1024
    status = "PASS" if not forbidden and not missing and size_ok else "PARTIAL"
    markers = {
        "TADSR_FINAL_SUBMISSION_BUNDLE_VALIDATION": status,
        "TADSR_FINAL_SUBMISSION_BUNDLE_NO_RAW_TENSORS": "PASS" if not forbidden else "PARTIAL",
        "TADSR_FINAL_SUBMISSION_BUNDLE_SIZE_AUDIT": "PASS" if size_ok else "PARTIAL",
    }
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "materials_root": str(materials_root),
        "create_zip_requested": str(args.create_zip) == "1",
        "zip_created": False,
        "file_count": file_count,
        "total_size_bytes": total_size,
        "total_size_mb": total_size / (1024 * 1024),
        "forbidden_files": forbidden,
        "missing_required_files": missing,
        "markers": markers,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Final Submission Bundle Validation",
        "",
        f"Status: **{status}**",
        "",
        "## Markers",
        "",
        "| Marker | Status |",
        "|---|---|",
    ]
    for marker, value in markers.items():
        lines.append(f"| `{marker}` | `{value}` |")
    lines += [
        "",
        f"- Materials root: `{materials_root}`",
        f"- File count: `{file_count}`",
        f"- Total size: `{total_size / (1024 * 1024):.3f} MB`",
        "",
        "## Forbidden Files",
        "",
    ]
    if forbidden:
        for item in forbidden:
            lines.append(f"- `{item['path']}` ({item['size_bytes']} bytes)")
    else:
        lines.append("None.")
    lines += ["", "## Missing Required Files", ""]
    if missing:
        for item in missing:
            lines.append(f"- `{item}`")
    else:
        lines.append("None.")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for marker, value in markers.items():
        print(f"{marker}: {value}")
    print(f"file_count: {file_count}")
    print(f"total_size_mb: {total_size / (1024 * 1024):.3f}")
    print(f"forbidden_count: {len(forbidden)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
