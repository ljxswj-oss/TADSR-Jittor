#!/usr/bin/env python3
"""Validate repository handoff readiness without importing torch."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    Path("README.md"),
    Path("deliverables/TADSR-Jittor_final_presentation.pptx"),
    Path("deliverables/TADSR-Jittor_final_presentation.pdf"),
    Path("deliverables/TADSR-Jittor_submission_readme.md"),
    Path("docs/final_demo_runbook.md"),
    Path("docs/repository_handoff_guide.md"),
    Path("docs/github_submission_handoff.md"),
    Path("docs/video_recording_preflight_checklist.md"),
    Path("experiments/final_audit_report.md"),
    Path("experiments/final_audit_report.json"),
    Path("experiments/final_evidence_manifest.md"),
    Path("experiments/final_evidence_manifest.json"),
]

KNOWN_TRACKED_LARGE_PREFIXES = [
    "experiments/full_repro/",
]


def run_git(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    text = (proc.stdout + proc.stderr).strip()
    return proc.returncode, text


def rel(path: Path) -> str:
    return str(path).replace("\\", "/")


def main() -> int:
    file_status = {}
    for path in REQUIRED_FILES:
        full = ROOT / path
        file_status[rel(path)] = {"exists": full.exists(), "size_bytes": full.stat().st_size if full.exists() else 0}

    branch_code, branch = run_git(["branch", "--show-current"])
    commit_code, commit = run_git(["log", "-1", "--oneline"])
    remote_code, remotes = run_git(["remote", "-v"])
    status_code, status = run_git(["status", "--short"])
    tracked_code, tracked = run_git(["ls-files"])
    staged_code, staged = run_git(["diff", "--cached", "--name-only"])

    tracked_large = [p for p in tracked.splitlines() if p.endswith((".npy", ".npz"))]
    historical_allowed_tracked_large = [
        p for p in tracked_large if any(p.startswith(prefix) for prefix in KNOWN_TRACKED_LARGE_PREFIXES)
    ]
    disallowed_tracked_large = [p for p in tracked_large if p not in historical_allowed_tracked_large]
    staged_large = [p for p in staged.splitlines() if p.endswith((".npy", ".npz"))]
    status_large = [
        line for line in status.splitlines()
        if line.strip().endswith((".npy", ".npz"))
    ]
    missing = [path for path, info in file_status.items() if not info["exists"] or info["size_bytes"] <= 0]
    git_readable = branch_code == 0 and commit_code == 0 and status_code == 0 and tracked_code == 0 and staged_code == 0

    guide_text = (ROOT / "docs/github_submission_handoff.md").read_text(encoding="utf-8", errors="ignore") if (ROOT / "docs/github_submission_handoff.md").exists() else ""
    github_guide_ready = all(
        phrase in guide_text
        for phrase in [
            "git status --short",
            "git log -1 --oneline",
            "python3 scripts/final_audit.py",
            "git remote",
            ".npy/.npz",
            "Do not push automatically",
        ]
    )

    markers = {
        "TADSR_GITHUB_SUBMISSION_GUIDE_READY": "PASS" if github_guide_ready else "PARTIAL",
        "TADSR_TRACKED_LARGE_ARTIFACT_AUDIT": "PASS" if not disallowed_tracked_large and not staged_large and not status_large else "PARTIAL",
    }
    markers["TADSR_REPOSITORY_HANDOFF_VALIDATION"] = (
        "PASS"
        if git_readable and not missing and markers["TADSR_GITHUB_SUBMISSION_GUIDE_READY"] == "PASS" and markers["TADSR_TRACKED_LARGE_ARTIFACT_AUDIT"] == "PASS"
        else "PARTIAL"
    )

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": markers["TADSR_REPOSITORY_HANDOFF_VALIDATION"],
        "git": {
            "branch": branch,
            "latest_commit": commit,
            "remotes": remotes,
            "status_short": status,
            "remote_configured": bool(remotes.strip()),
        },
        "files": file_status,
        "missing_or_empty_files": missing,
        "tracked_large_artifacts": tracked_large,
        "historical_allowed_tracked_large_artifacts": historical_allowed_tracked_large,
        "disallowed_tracked_large_artifacts": disallowed_tracked_large,
        "staged_large_artifacts": staged_large,
        "worktree_large_artifact_status_lines": status_large,
        "markers": markers,
    }

    out_json = ROOT / "experiments/repository_handoff_validation.json"
    out_md = ROOT / "experiments/repository_handoff_validation.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Repository Handoff Validation",
        "",
        f"Status: **{result['status']}**",
        "",
        "## Git",
        "",
        f"- Branch: `{branch or '(none)'}`",
        f"- Latest commit: `{commit or '(unavailable)'}`",
        f"- Remote configured: `{bool(remotes.strip())}`",
        "",
        "## Markers",
        "",
        "| Marker | Status |",
        "|---|---|",
    ]
    for marker, status_value in markers.items():
        lines.append(f"| `{marker}` | `{status_value}` |")
    lines.extend(["", "## Missing Or Empty Files", ""])
    lines.extend([f"- `{path}`" for path in missing] or ["None."])
    lines.extend(["", "## Large Artifact Audit", ""])
    lines.append(f"- Historical allowed tracked `.npy/.npz`: {len(historical_allowed_tracked_large)}")
    lines.append(f"- Disallowed tracked `.npy/.npz`: {len(disallowed_tracked_large)}")
    lines.append(f"- Staged `.npy/.npz`: {len(staged_large)}")
    lines.append(f"- Worktree changed/untracked `.npy/.npz` status lines: {len(status_large)}")
    if disallowed_tracked_large:
        lines.extend(f"  - `{p}`" for p in disallowed_tracked_large)
    if staged_large:
        lines.extend(f"  - `{p}`" for p in staged_large)
    if status_large:
        lines.extend(f"  - `{p}`" for p in status_large)
    lines.extend([
        "",
        "Historical note: existing tracked tensor artifacts under",
        "`experiments/full_repro/` are treated as a known legacy evidence set.",
        "This validator blocks newly staged or out-of-policy `.npy/.npz` files;",
        "a separate controlled repository-slimming pass is recommended if a",
        "public release must remove all historical tensor artifacts.",
    ])
    lines.extend(["", "## Remote", "", "```text", remotes or "(no remote configured)", "```"])
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for marker in [
        "TADSR_REPOSITORY_HANDOFF_VALIDATION",
        "TADSR_GITHUB_SUBMISSION_GUIDE_READY",
        "TADSR_TRACKED_LARGE_ARTIFACT_AUDIT",
    ]:
        print(f"{marker}: {markers[marker]}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
