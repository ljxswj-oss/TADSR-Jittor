#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


REQUIRED_FILES = [
    Path("experiments/final_evidence_manifest.json"),
    Path("experiments/final_evidence_manifest.md"),
    Path("docs/production_cli_design_audit.md"),
    Path("docs/final_submission_checklist.md"),
    Path("experiments/final_audit_report.json"),
]


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    missing = [str(p) for p in REQUIRED_FILES if not p.exists()]
    audit = load_json(Path("experiments/final_audit_report.json"))
    rows = {row.get("check"): row.get("status") for row in audit.get("rows", [])}
    required_markers = {
        "TADSR_UNET_FULL_FORWARD_ALIGNMENT": "PASS",
        "TIME_VAE_ACTUAL_VAEHOOK_FULL_BOUNDARY_ALIGNMENT": "PASS",
        "TADSR_STATIC_EFFECTIVE_LORA_COVERAGE_AUDIT": "PASS",
        "TADSR_SCHEDULER_BOUNDARY_ALIGNMENT": "PASS",
        "TADSR_MINIMAL_ONE_STEP_DECODE_DRY_RUN": "PASS",
        "JITTOR_FULL_INFERENCE": "NOT_COMPLETE",
        "JITTOR_FULL_PORT": "PARTIAL",
    }
    marker_failures = {
        key: {"expected": expected, "actual": rows.get(key)}
        for key, expected in required_markers.items()
        if rows.get(key) != expected
    }
    manifest = load_json(Path("experiments/final_evidence_manifest.json"))
    manifest_status = manifest.get("status", "MISSING")
    status = "PASS" if not missing and not marker_failures and manifest_status in {"PASS", "PARTIAL"} else "FAIL"
    result = {
        "status": status,
        "missing": missing,
        "marker_failures": marker_failures,
        "manifest_status": manifest_status,
    }
    out = Path("experiments/final_evidence_manifest_test.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"TADSR_FINAL_EVIDENCE_MANIFEST_TEST: {status}")
    if missing:
        print("missing:", missing)
    if marker_failures:
        print("marker_failures:", marker_failures)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
