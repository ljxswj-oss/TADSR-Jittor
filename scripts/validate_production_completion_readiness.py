#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "production_completion"

BASELINE_MARKERS = [
    "TADSR_JITTOR_MIGRATION_FEASIBILITY_VALIDATION",
    "TADSR_BOUNDARY_LEVEL_REPRODUCTION",
    "TADSR_SMALL_DATA_TRAINING_ALIGNMENT",
    "TADSR_FINAL_SUBMISSION_READY",
    "TADSR_GAP_ANALYSIS_READINESS",
]

MUST_REMAIN = {
    "JITTOR_FULL_INFERENCE": "NOT_COMPLETE",
    "JITTOR_FULL_PORT": "PARTIAL",
    "TIME_VAE_FULL_ALIGNMENT": "NOT_COMPLETE",
    "TADSR_DYNAMIC_RUNTIME_LORA_IMPLEMENTATION": "NOT_IMPLEMENTED_BY_DESIGN",
}

FALSE_CLAIMS = [
    "full inference complete",
    "production pipeline complete",
    "image generation complete",
    "dynamic runtime lora complete",
]


def run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.returncode, proc.stdout


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as exc:
        return {"status": "FAIL", "error": repr(exc)}


def final_audit_statuses() -> dict[str, str]:
    report = load_json(ROOT / "experiments" / "final_audit_report.json")
    return {str(row.get("check")): str(row.get("status")) for row in report.get("rows", [])}


def git_branch() -> str:
    code, out = run(["git", "branch", "--show-current"])
    return out.strip() if code == 0 else ""


def branch_exists(name: str) -> bool:
    code, out = run(["git", "branch", "--list", name])
    return code == 0 and bool(out.strip())


def staged_raw_tensors() -> list[str]:
    code, out = run(["git", "diff", "--cached", "--name-only"])
    if code != 0:
        return ["<git diff --cached failed>"]
    return [line for line in out.splitlines() if line.endswith((".npy", ".npz"))]


def torch_import_hits() -> list[str]:
    hits: list[str] = []
    for subdir in ["jittor_tadsr_full", "jittor_tadsr"]:
        root = ROOT / subdir
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for lineno, line in enumerate(text.splitlines(), start=1):
                stripped = line.strip()
                if stripped.startswith("import torch") or stripped.startswith("from torch"):
                    hits.append(f"{path.relative_to(ROOT)}:{lineno}:{stripped}")
    return hits


def false_claim_hits() -> list[str]:
    hits: list[str] = []
    scan_paths = [
        ROOT / "docs" / "production_completion",
        ROOT / "README.md",
        ROOT / "docs" / "final_teacher_status_summary.md",
        ROOT / "docs" / "full_inference_gap_analysis.md",
        ROOT / "docs" / "timevae_full_alignment_gap_analysis.md",
        ROOT / "docs" / "lora_runtime_gap_analysis.md",
    ]
    for target in scan_paths:
        paths = [target] if target.is_file() else list(target.rglob("*.md")) if target.exists() else []
        for path in paths:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            for phrase in FALSE_CLAIMS:
                if phrase in text:
                    hits.append(f"{path.relative_to(ROOT)} contains '{phrase}'")
    return hits


def guard_status() -> tuple[str, str]:
    code, out = run([sys.executable, "-m", "jittor_tadsr_full.tadsr_full", "--full-inference"])
    ok = code != 0 and "NotImplementedError" in out and "Full Jittor TADSR inference is not complete" in out
    return ("PASS" if ok else "FAIL", out.strip()[-2000:])


def write_outputs(payload: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "readiness.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Production completion readiness",
        "",
        f"`TADSR_PRODUCTION_COMPLETION_READINESS: {payload['markers']['TADSR_PRODUCTION_COMPLETION_READINESS']}`",
        f"`TADSR_PRODUCTION_COMPLETION_BRANCH_READY: {payload['markers']['TADSR_PRODUCTION_COMPLETION_BRANCH_READY']}`",
        f"`TADSR_PRODUCTION_COMPLETION_BASELINE_PRESERVED: {payload['markers']['TADSR_PRODUCTION_COMPLETION_BASELINE_PRESERVED']}`",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for item in payload["checks"]:
        lines.append(f"| {item['name']} | {item['status']} | {item['detail']} |")
    (OUT_DIR / "readiness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    statuses = final_audit_statuses()
    branch = git_branch()
    guard, guard_output = guard_status()
    dirs = [
        "experiments/production_completion",
        "experiments/production_completion/timevae_full",
        "experiments/production_completion/runtime_lora",
        "experiments/production_completion/full_inference",
        "experiments/production_completion/blockers",
        "docs/production_completion/production_completion_plan.md",
    ]
    missing_dirs = [p for p in dirs if not (ROOT / p).exists()]
    baseline_bad = {
        marker: statuses.get(marker, "MISSING")
        for marker in BASELINE_MARKERS
        if statuses.get(marker) != "PASS"
    }
    must_bad = {
        marker: statuses.get(marker, "MISSING")
        for marker, expected in MUST_REMAIN.items()
        if statuses.get(marker) != expected
    }
    raw = staged_raw_tensors()
    torch_hits = torch_import_hits()
    claim_hits = false_claim_hits()

    branch_ready = branch == "codex/tadsr-production-completion" and branch_exists("submission-freeze-fb3b820")
    baseline_ok = not baseline_bad and not must_bad
    ready = branch_ready and baseline_ok and not missing_dirs and guard == "PASS" and not raw and not torch_hits and not claim_hits

    checks = [
        {"name": "current_branch", "status": "PASS" if branch == "codex/tadsr-production-completion" else "PARTIAL", "detail": branch or "<unknown>"},
        {"name": "submission_freeze_ref", "status": "PASS" if branch_exists("submission-freeze-fb3b820") else "PARTIAL", "detail": "submission-freeze-fb3b820"},
        {"name": "baseline_markers", "status": "PASS" if not baseline_bad else "FAIL", "detail": json.dumps(baseline_bad, ensure_ascii=False)},
        {"name": "must_remain_statuses", "status": "PASS" if not must_bad else "FAIL", "detail": json.dumps(must_bad, ensure_ascii=False)},
        {"name": "directory_readiness", "status": "PASS" if not missing_dirs else "FAIL", "detail": ", ".join(missing_dirs) if missing_dirs else "all required directories/files exist"},
        {"name": "full_inference_guard", "status": guard, "detail": "NotImplementedError guard preserved"},
        {"name": "staged_raw_tensors", "status": "PASS" if not raw else "FAIL", "detail": ", ".join(raw) if raw else "none"},
        {"name": "runtime_torch_import_scan", "status": "PASS" if not torch_hits else "FAIL", "detail": f"{len(torch_hits)} hit(s)"},
        {"name": "false_claim_scan", "status": "PASS" if not claim_hits else "FAIL", "detail": "; ".join(claim_hits) if claim_hits else "none"},
    ]
    payload = {
        "status": "PASS" if ready else "PARTIAL" if branch_ready and baseline_ok else "FAIL",
        "branch": branch,
        "guard_output_tail": guard_output,
        "checks": checks,
        "markers": {
            "TADSR_PRODUCTION_COMPLETION_READINESS": "PASS" if ready else "PARTIAL" if branch_ready and baseline_ok else "FAIL",
            "TADSR_PRODUCTION_COMPLETION_BRANCH_READY": "PASS" if branch_ready else "PARTIAL",
            "TADSR_PRODUCTION_COMPLETION_BASELINE_PRESERVED": "PASS" if baseline_ok and guard == "PASS" else "FAIL",
        },
    }
    write_outputs(payload)
    for key, value in payload["markers"].items():
        print(f"{key}: {value}")
    return 0 if payload["status"] in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
