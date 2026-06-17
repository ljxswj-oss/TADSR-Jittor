from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "jittor_tadsr_full.tadsr_diagnostic", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def test_one_step_summary_metadata_only() -> None:
    proc = run_cli("--one-step-summary")
    assert proc.returncode == 0, proc.stdout
    assert "full_inference_executed" in proc.stdout
    assert "False" in proc.stdout or "false" in proc.stdout


def test_diagnostic_smoke_metadata_only() -> None:
    proc = run_cli("--check-diagnostic-image-smoke")
    assert proc.returncode == 0, proc.stdout
    assert "diagnostic_smoke" in proc.stdout


def test_full_inference_guard_preserved() -> None:
    proc = run_cli("--full-inference")
    assert proc.returncode != 0
    assert "NotImplementedError" in proc.stdout
