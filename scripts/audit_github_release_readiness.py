#!/usr/bin/env python3
"""Audit GitHub publication readiness without deleting files or importing torch."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MB = 1024 * 1024
GITHUB_HARD_LIMIT_BYTES = 100 * MB
LARGE_BLOB_WARN_BYTES = 50 * MB
DELIVERABLE_LIMIT_BYTES = 100 * MB


def run(cmd: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, check=check)


def human_size(num_bytes: int | float) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024 or unit == "GB":
            return f"{value:.3f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.3f} GB"


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def dir_size(path: Path, *, exclude_git: bool = False) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, dirs, files in os.walk(path):
        root_path = Path(root)
        if exclude_git and root_path == ROOT:
            dirs[:] = [d for d in dirs if d != ".git"]
        for name in files:
            full = root_path / name
            if exclude_git and ".git" in full.relative_to(ROOT).parts:
                continue
            try:
                total += full.stat().st_size
            except OSError:
                pass
    return total


def git_lines(args: list[str]) -> list[str]:
    proc = run(["git", *args])
    if proc.returncode != 0:
        return []
    return [line for line in proc.stdout.splitlines() if line.strip()]


def load_manifest_text() -> str:
    chunks = []
    for rel in [
        "experiments/final_evidence_manifest.json",
        "experiments/final_evidence_manifest.md",
        "experiments/final_audit_report.json",
        "experiments/final_audit_report.md",
    ]:
        path = ROOT / rel
        if path.exists():
            chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(chunks)


def tracked_tensor_artifacts(manifest_text: str) -> tuple[list[dict[str, object]], int]:
    paths = [p for p in git_lines(["ls-files"]) if p.endswith((".npy", ".npz"))]
    rows = []
    total = 0
    for rel in paths:
        size = file_size(ROOT / rel)
        total += size
        under_full_repro = rel.startswith("experiments/full_repro/")
        referenced = rel in manifest_text
        rows.append(
            {
                "path": rel,
                "size_bytes": size,
                "size_mb": size / MB,
                "under_experiments_full_repro": under_full_repro,
                "referenced_by_final_evidence_or_audit": referenced,
                "likely_safe_to_untrack_future_cleanup": under_full_repro and not referenced,
            }
        )
    rows.sort(key=lambda item: int(item["size_bytes"]), reverse=True)
    return rows, total


def ignored_tensor_artifacts() -> list[dict[str, object]]:
    paths = [p for p in git_lines(["ls-files", "-io", "--exclude-standard"]) if p.endswith((".npy", ".npz"))]
    rows = []
    for rel in paths:
        rows.append({"path": rel, "size_bytes": file_size(ROOT / rel), "size_mb": file_size(ROOT / rel) / MB})
    rows.sort(key=lambda item: int(item["size_bytes"]), reverse=True)
    return rows


def worktree_tensor_status() -> dict[str, object]:
    status_lines = git_lines(["status", "--short"])
    tensor_lines = [line for line in status_lines if line.strip().endswith((".npy", ".npz"))]
    staged_paths = [p for p in git_lines(["diff", "--cached", "--name-only"]) if p.endswith((".npy", ".npz"))]
    return {
        "status_tensor_lines": tensor_lines,
        "staged_tensor_paths": staged_paths,
        "staged_tensor_count": len(staged_paths),
        "worktree_tensor_status_count": len(tensor_lines),
    }


def deliverables_summary() -> dict[str, object]:
    deliverables = ROOT / "deliverables"
    pptx = deliverables / "TADSR-Jittor_final_presentation.pptx"
    pdf = deliverables / "TADSR-Jittor_final_presentation.pdf"
    total = dir_size(deliverables)
    return {
        "total_size_bytes": total,
        "total_size_mb": total / MB,
        "pptx_size_bytes": file_size(pptx),
        "pptx_size_mb": file_size(pptx) / MB,
        "pdf_size_bytes": file_size(pdf),
        "pdf_size_mb": file_size(pdf) / MB,
        "under_100mb": total < DELIVERABLE_LIMIT_BYTES,
    }


def phase5b_after_release_ready() -> dict[str, object]:
    required_paths = [
        ROOT / "docs/final_phase5b_submission_freeze_summary.md",
        ROOT / "docs/production_completion/diagnostic_image_smoke_plan.md",
        ROOT / "experiments/final_phase5b_submission_freeze_summary.json",
        ROOT / "experiments/production_completion/full_inference/diagnostic_image_smoke_plan.json",
        ROOT / "experiments/production_completion/full_inference/actual_inference_path_audit.json",
        ROOT / "experiments/production_completion/full_inference/postprocess_contract_audit.json",
        ROOT / "experiments/production_completion/full_inference/multistep_applicability_decision.json",
        ROOT / "experiments/production_completion/diagnostic_image_smoke/diagnostic_image_smoke_metrics.json",
        ROOT / "experiments/production_completion/diagnostic_image_smoke/diagnostic_image_smoke_validation.json",
        ROOT / "experiments/production_completion/diagnostic_cli/one_step_diagnostic_cli_test.json",
        ROOT / "docs/timevae_full_alignment_closure_plan.md",
        ROOT / "experiments/timevae_full_alignment_closure_summary.json",
        ROOT / "docs/runtime_lora_final_decision_proof.md",
        ROOT / "experiments/runtime_lora_final_decision_proof.json",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required_paths if not path.exists()]
    return {
        "required_paths": [str(path.relative_to(ROOT)) for path in required_paths],
        "missing_paths": missing,
        "status": "PASS" if not missing else "PARTIAL",
    }


def history_blob_audit() -> dict[str, object]:
    revs = run(["git", "rev-list", "--objects", "--all"])
    if revs.returncode != 0:
        return {
            "status": "FAIL",
            "error": revs.stderr.strip(),
            "history_blob_count": 0,
            "largest_100_blobs": [],
            "blobs_over_100mb": [],
            "blobs_over_50mb": [],
        }

    object_lines = [line for line in revs.stdout.splitlines() if line.strip()]
    object_path = {}
    hashes = []
    for line in object_lines:
        parts = line.split(" ", 1)
        obj = parts[0]
        path = parts[1] if len(parts) > 1 else ""
        object_path[obj] = path
        hashes.append(obj)

    batch = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objecttype) %(objectname) %(objectsize)"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        input="\n".join(hashes) + "\n",
        capture_output=True,
        check=False,
    )
    blobs = []
    head_paths = set(git_lines(["ls-files"]))
    for line in batch.stdout.splitlines():
        fields = line.split()
        if len(fields) != 3 or fields[0] != "blob":
            continue
        _, obj, size_text = fields
        size = int(size_text)
        path = object_path.get(obj, "")
        suffix = Path(path).suffix.lower()
        kind = (
            "tensor"
            if suffix in {".npy", ".npz"}
            else "checkpoint"
            if suffix in {".pt", ".pth", ".ckpt", ".safetensors", ".bin"}
            else "presentation"
            if suffix in {".ppt", ".pptx", ".pdf"}
            else "other"
        )
        blobs.append(
            {
                "object": obj,
                "path": path,
                "size_bytes": size,
                "size_mb": size / MB,
                "kind": kind,
                "exists_in_head": path in head_paths,
                "historical_only": bool(path) and path not in head_paths,
            }
        )
    blobs.sort(key=lambda item: int(item["size_bytes"]), reverse=True)
    over_100 = [b for b in blobs if int(b["size_bytes"]) > GITHUB_HARD_LIMIT_BYTES]
    over_50 = [b for b in blobs if int(b["size_bytes"]) > LARGE_BLOB_WARN_BYTES]
    git_size = dir_size(ROOT / ".git")
    return {
        "status": "PASS" if not over_50 else ("FAIL_GITHUB_HARD_LIMIT_RISK" if over_100 else "PARTIAL_GITHUB_SIZE_RISK"),
        "history_blob_count": len(blobs),
        "max_blob_size_bytes": int(blobs[0]["size_bytes"]) if blobs else 0,
        "max_blob_size_mb": float(blobs[0]["size_mb"]) if blobs else 0.0,
        "largest_100_blobs": blobs[:100],
        "blobs_over_100mb": over_100,
        "blobs_over_50mb": over_50,
        "github_hard_limit_risk": bool(over_100),
        "github_large_repo_risk": bool(over_50) or git_size > 500 * MB,
        "git_size_bytes": git_size,
        "git_size_mb": git_size / MB,
    }


def main() -> int:
    manifest_text = load_manifest_text()
    tracked_rows, tracked_total = tracked_tensor_artifacts(manifest_text)
    ignored_rows = ignored_tensor_artifacts()
    worktree = worktree_tensor_status()
    deliverables = deliverables_summary()
    history = history_blob_audit()
    phase5b_release = phase5b_after_release_ready()
    git_size = dir_size(ROOT / ".git")
    worktree_size = dir_size(ROOT, exclude_git=True)
    remotes = "\n".join(git_lines(["remote", "-v"]))

    cleanup_candidates = [row for row in tracked_rows if row["likely_safe_to_untrack_future_cleanup"]]
    keep_metadata = [
        "README.md",
        "deliverables/TADSR-Jittor_submission_readme.md",
        "docs/github_submission_handoff.md",
        "docs/github_release_slimming_decision.md",
        "experiments/final_audit_report.md",
        "experiments/final_evidence_manifest.md",
        "experiments/github_release_readiness_audit.md",
    ]
    evidence_manifest_requires_tensor_files = any(row["referenced_by_final_evidence_or_audit"] for row in tracked_rows)

    marker_head = "PASS" if tracked_rows and all(row["under_experiments_full_repro"] for row in tracked_rows) else ("PASS" if not tracked_rows else "PARTIAL")
    marker_deliverables = "PASS" if deliverables["under_100mb"] else "FAIL"
    marker_worktree = "PASS" if worktree["staged_tensor_count"] == 0 and worktree["worktree_tensor_status_count"] == 0 else "FAIL"
    marker_history = "FAIL" if history["github_hard_limit_risk"] else ("PARTIAL" if history["github_large_repo_risk"] else "PASS")
    marker_evidence = "PASS" if not evidence_manifest_requires_tensor_files else "PARTIAL"
    marker_slimming_decision = "PASS" if (ROOT / "docs/github_release_slimming_decision.md").exists() else "PARTIAL"
    marker_release = "FAIL" if history["github_hard_limit_risk"] else ("PARTIAL" if history["github_large_repo_risk"] else "PASS")
    marker_phase5b_release = (
        "PASS"
        if phase5b_release["status"] == "PASS"
        and marker_worktree == "PASS"
        and marker_deliverables == "PASS"
        and marker_release in {"PASS", "PARTIAL"}
        else "PARTIAL"
    )

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": marker_release,
        "git": {
            "branch": "\n".join(git_lines(["branch", "--show-current"])),
            "latest_commit": "\n".join(git_lines(["log", "-1", "--oneline"])),
            "remotes": remotes,
            "remote_configured": bool(remotes.strip()),
        },
        "head_tracked_tensors": {
            "count": len(tracked_rows),
            "total_size_bytes": tracked_total,
            "total_size_mb": tracked_total / MB,
            "largest_50": tracked_rows[:50],
        },
        "ignored_local_tensors": {
            "count": len(ignored_rows),
            "largest_50": ignored_rows[:50],
        },
        "worktree_large_artifacts": worktree,
        "deliverables": deliverables,
        "repository_size": {
            "git_size_bytes": git_size,
            "git_size_mb": git_size / MB,
            "working_tree_excluding_git_size_bytes": worktree_size,
            "working_tree_excluding_git_size_mb": worktree_size / MB,
            "total_estimate_bytes": git_size + worktree_size,
            "total_estimate_mb": (git_size + worktree_size) / MB,
        },
        "history": history,
        "phase5b_after_release_ready": phase5b_release,
        "evidence_dependency": {
            "cleanup_candidate_tracked_tensors_count": len(cleanup_candidates),
            "cleanup_candidate_tracked_tensors_largest_50": cleanup_candidates[:50],
            "keep_required_metadata_files": keep_metadata,
            "evidence_manifest_requires_tensor_files": evidence_manifest_requires_tensor_files,
            "proposed_cleanup_strategy": [
                "Keep metadata JSON/Markdown/report files.",
                "If the user chooses a cleanup pass, git rm --cached tracked .npy/.npz files instead of deleting local evidence.",
                "Regenerate final evidence manifest and final audit after cleanup.",
                "Use history rewrite only in a separate explicitly authorized stage if >100MB historical blobs block GitHub.",
            ],
        },
        "markers": {
            "TADSR_GITHUB_HEAD_ARTIFACT_AUDIT": marker_head,
            "TADSR_GITHUB_DELIVERABLE_SIZE_AUDIT": marker_deliverables,
            "TADSR_GITHUB_WORKTREE_LARGE_ARTIFACT_AUDIT": marker_worktree,
            "TADSR_GIT_HISTORY_LARGE_BLOB_AUDIT": marker_history,
            "TADSR_EVIDENCE_DEPENDENCY_AUDIT": marker_evidence,
            "TADSR_GITHUB_RELEASE_SLIMMING_DECISION": marker_slimming_decision,
            "TADSR_GITHUB_RELEASE_READINESS_AUDIT": marker_release,
            "TADSR_GITHUB_RELEASE_AFTER_PHASE5B_READY": marker_phase5b_release,
        },
        "recommendation": (
            "Do not push until a separate history cleanup is authorized; >100MB blobs are present."
            if history["github_hard_limit_risk"]
            else "Manual push is technically possible, but repository slimming is recommended before a clean public release."
            if history["github_large_repo_risk"]
            else "Manual push is safe from GitHub hard-limit and large-blob perspectives; a separate cleanup pass is optional."
        ),
    }

    out_json = ROOT / "experiments/github_release_readiness_audit.json"
    out_md = ROOT / "experiments/github_release_readiness_audit.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# GitHub Release Readiness Audit",
        "",
        f"Status: **{result['status']}**",
        "",
        "## Markers",
        "",
        "| Marker | Status |",
        "|---|---|",
    ]
    for marker, status in result["markers"].items():
        lines.append(f"| `{marker}` | `{status}` |")
    lines.extend([
        "",
        "## Size Summary",
        "",
        f"- Deliverables: {human_size(deliverables['total_size_bytes'])}",
        f"- `.git`: {human_size(git_size)}",
        f"- Working tree excluding `.git`: {human_size(worktree_size)}",
        f"- Tracked `.npy/.npz`: {len(tracked_rows)} files, {human_size(tracked_total)}",
        f"- Ignored local `.npy/.npz`: {len(ignored_rows)} files",
        f"- Largest history blob: {human_size(history['max_blob_size_bytes'])}",
        f"- Blobs >100MB: {len(history['blobs_over_100mb'])}",
        f"- Blobs >50MB: {len(history['blobs_over_50mb'])}",
        "",
        "## Largest Tracked Tensor Artifacts",
        "",
        "| Size | Path | Manifest reference | Future cleanup candidate |",
        "|---:|---|---|---|",
    ])
    for row in tracked_rows[:20]:
        lines.append(
            f"| {human_size(row['size_bytes'])} | `{row['path']}` | `{row['referenced_by_final_evidence_or_audit']}` | `{row['likely_safe_to_untrack_future_cleanup']}` |"
        )
    lines.extend([
        "",
        "## Largest History Blobs",
        "",
        "| Size | Kind | HEAD | Path |",
        "|---:|---|---|---|",
    ])
    for row in history["largest_100_blobs"][:25]:
        lines.append(f"| {human_size(row['size_bytes'])} | `{row['kind']}` | `{row['exists_in_head']}` | `{row['path']}` |")
    lines.extend([
        "",
        "## Recommendation",
        "",
        result["recommendation"],
        "",
        "This audit did not delete files, untrack files, rewrite history, add remotes, or push to GitHub.",
    ])
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for marker in [
        "TADSR_GITHUB_HEAD_ARTIFACT_AUDIT",
        "TADSR_GITHUB_DELIVERABLE_SIZE_AUDIT",
        "TADSR_GITHUB_WORKTREE_LARGE_ARTIFACT_AUDIT",
        "TADSR_GIT_HISTORY_LARGE_BLOB_AUDIT",
        "TADSR_EVIDENCE_DEPENDENCY_AUDIT",
        "TADSR_GITHUB_RELEASE_SLIMMING_DECISION",
        "TADSR_GITHUB_RELEASE_READINESS_AUDIT",
        "TADSR_GITHUB_RELEASE_AFTER_PHASE5B_READY",
    ]:
        print(f"{marker}: {result['markers'][marker]}")
    print(f"tracked_npy_npz_count: {len(tracked_rows)}")
    print(f"tracked_npy_npz_total_size_mb: {tracked_total / MB:.3f}")
    print(f"largest_blob_size_mb: {history['max_blob_size_mb']:.3f}")
    print(f"blobs_over_100mb: {len(history['blobs_over_100mb'])}")
    print(f"blobs_over_50mb: {len(history['blobs_over_50mb'])}")
    print(f"recommendation: {result['recommendation']}")
    return 0 if marker_release != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
