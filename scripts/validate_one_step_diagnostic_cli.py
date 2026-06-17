#!/usr/bin/env python3
"""Validate the metadata-only one-step diagnostic CLI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run_cmd(repo_root: Path, args: list[str]) -> dict[str, object]:
    proc = subprocess.run(
        [sys.executable, "-m", *args],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return {"returncode": proc.returncode, "output": proc.stdout}


def write_reports(out_json: Path, out_md: Path, payload: dict[str, object]) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# One-Step Diagnostic CLI Validation",
        "",
        f"Status: **{payload['status']}**",
        "",
        "| Marker | Status |",
        "|---|---|",
    ]
    for key, value in payload["markers"].items():  # type: ignore[index]
        lines.append(f"| `{key}` | `{value}` |")
    lines += [
        "",
        "The CLI is metadata-only and does not provide production full inference.",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    out_dir = repo_root / "experiments/production_completion/diagnostic_cli"
    summary = run_cmd(repo_root, ["jittor_tadsr_full.tadsr_diagnostic", "--one-step-summary"])
    smoke = run_cmd(repo_root, ["jittor_tadsr_full.tadsr_diagnostic", "--check-diagnostic-image-smoke"])
    full = run_cmd(repo_root, ["jittor_tadsr_full.tadsr_diagnostic", "--full-inference"])
    summary_ok = summary["returncode"] == 0 and "full_inference_executed" in str(summary["output"])
    smoke_ok = smoke["returncode"] == 0 and "diagnostic_smoke" in str(smoke["output"])
    full_guard_ok = full["returncode"] != 0 and "NotImplementedError" in str(full["output"])
    status = "PASS" if summary_ok and smoke_ok and full_guard_ok else "FAIL"
    markers = {
        "TADSR_ONE_STEP_DIAGNOSTIC_CLI_READY": status,
        "TADSR_ONE_STEP_DIAGNOSTIC_CLI_NOT_FULL_INFERENCE": "PASS" if full_guard_ok else "FAIL",
    }
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "markers": markers,
        "summary_command": summary,
        "diagnostic_smoke_command": smoke,
        "full_inference_guard_command": full,
    }
    write_reports(out_dir / "one_step_diagnostic_cli_test.json", out_dir / "one_step_diagnostic_cli_test.md", payload)
    for key, value in markers.items():
        print(f"{key}: {value}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
