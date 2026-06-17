#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


FORBIDDEN_SUFFIXES = {".npy", ".npz"}
FORBIDDEN_PARTS = {".git", "__pycache__", "local_tensors", "checkpoints", "weights"}
ALLOWED_PREFIXES = (
    "experiments/production_completion/",
    "experiments/final_audit_report.json",
    "experiments/final_audit_report.md",
)
CODE_SUFFIXES = {".py", ".sh"}


def is_forbidden(name: str) -> bool:
    path = Path(name)
    parts = set(path.parts)
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        return True
    return bool(parts & FORBIDDEN_PARTS)


def is_allowed_destination(name: str) -> bool:
    normalized = name.replace("\\", "/")
    return any(normalized == p or normalized.startswith(p) for p in ALLOWED_PREFIXES)


def write_report(repo: Path, payload: dict) -> None:
    out_json = repo / "experiments" / "production_completion" / "phase3_import_validation.json"
    out_md = repo / "experiments" / "production_completion" / "phase3_import_validation.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Phase 3-B live results import validation",
        "",
        f"`TADSR_PHASE3B_LIVE_RESULTS_IMPORT_VALIDATION: {payload['status']}`",
        f"`TADSR_PHASE3B_LIVE_RESULTS_NO_RAW_TENSORS: {payload['no_raw_tensors_status']}`",
        "",
        f"- dry_run: `{payload['dry_run']}`",
        f"- package: `{payload['package']}`",
        f"- file_count: `{payload['file_count']}`",
        f"- imported_count: `{payload['imported_count']}`",
        "",
        "## Notes",
        "",
        payload.get("note", ""),
    ]
    if payload.get("blocked_files"):
        lines += ["", "## Blocked files", ""]
        for item in payload["blocked_files"]:
            lines.append(f"- `{item}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_cmd(repo: Path, cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.returncode, proc.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description="Import Phase 3-B Linux live audit result package into Windows/local repo.")
    parser.add_argument("--package", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--dry-run", type=int, default=0)
    parser.add_argument("--allow-code-overwrite", type=int, default=0)
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    package = Path(args.package).resolve()
    dry_run = bool(args.dry_run)
    allow_code = bool(args.allow_code_overwrite)

    if not package.exists():
        payload = {
            "status_marker": "TADSR_PHASE3B_LIVE_RESULTS_IMPORT_VALIDATION",
            "status": "FAIL",
            "no_raw_tensors_status": "FAIL",
            "dry_run": dry_run,
            "package": str(package),
            "file_count": 0,
            "imported_count": 0,
            "blocked_files": [str(package)],
            "note": "Package does not exist.",
        }
        write_report(repo, payload)
        print("TADSR_PHASE3B_LIVE_RESULTS_IMPORT_VALIDATION: FAIL")
        return 1

    blocked: list[str] = []
    importable: list[zipfile.ZipInfo] = []
    with zipfile.ZipFile(package) as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            if info.is_dir():
                continue
            if name.startswith("/") or ".." in Path(name).parts:
                blocked.append(name)
                continue
            if is_forbidden(name):
                blocked.append(name)
                continue
            if not is_allowed_destination(name):
                blocked.append(name)
                continue
            if Path(name).suffix.lower() in CODE_SUFFIXES and not allow_code:
                blocked.append(name)
                continue
            importable.append(info)

        if blocked:
            payload = {
                "status_marker": "TADSR_PHASE3B_LIVE_RESULTS_IMPORT_VALIDATION",
                "status": "FAIL",
                "no_raw_tensors_status": "FAIL",
                "dry_run": dry_run,
                "package": str(package),
                "file_count": len(zf.infolist()),
                "imported_count": 0,
                "blocked_files": blocked,
                "note": "Forbidden files detected. Nothing was extracted.",
            }
            write_report(repo, payload)
            print("TADSR_PHASE3B_LIVE_RESULTS_IMPORT_VALIDATION: FAIL")
            print("TADSR_PHASE3B_LIVE_RESULTS_NO_RAW_TENSORS: FAIL")
            return 1

        if not dry_run:
            for info in importable:
                target = repo / info.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

    validation_output = ""
    final_audit_output = ""
    validation_code = 0
    final_code = 0
    if not dry_run:
        validation_code, validation_output = run_cmd(repo, [sys.executable, "scripts/validate_production_completion_phase3.py"])
        final_code, final_audit_output = run_cmd(repo, [sys.executable, "scripts/final_audit.py"])

    status = "PASS" if validation_code == 0 and final_code == 0 else "PARTIAL"
    payload = {
        "status_marker": "TADSR_PHASE3B_LIVE_RESULTS_IMPORT_VALIDATION",
        "status": status,
        "no_raw_tensors_status": "PASS",
        "dry_run": dry_run,
        "package": str(package),
        "file_count": len(importable),
        "imported_count": 0 if dry_run else len(importable),
        "blocked_files": [],
        "validation_returncode": validation_code,
        "final_audit_returncode": final_code,
        "validation_output_tail": validation_output[-4000:],
        "final_audit_output_tail": final_audit_output[-4000:],
        "note": "Package passed safety checks. Results were imported." if not dry_run else "Dry run passed safety checks. Nothing was extracted.",
    }
    write_report(repo, payload)
    print(f"TADSR_PHASE3B_LIVE_RESULTS_IMPORT_VALIDATION: {status}")
    print("TADSR_PHASE3B_LIVE_RESULTS_NO_RAW_TENSORS: PASS")
    print(f"imported_count: {payload['imported_count']}")
    return 0 if status in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
