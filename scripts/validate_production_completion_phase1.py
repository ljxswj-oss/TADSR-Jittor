#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "production_completion"

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


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as exc:
        return {"status": "FAIL", "error": repr(exc)}


def final_audit_statuses() -> dict[str, str]:
    data = load_json(ROOT / "experiments" / "final_audit_report.json")
    return {str(row.get("check")): str(row.get("status")) for row in data.get("rows", [])}


def false_claim_hits() -> list[str]:
    hits: list[str] = []
    paths = [
        ROOT / "docs" / "production_completion",
        ROOT / "experiments" / "production_completion",
        ROOT / "README.md",
        ROOT / "docs" / "final_teacher_status_summary.md",
        ROOT / "docs" / "full_inference_gap_analysis.md",
        ROOT / "docs" / "timevae_full_alignment_gap_analysis.md",
        ROOT / "docs" / "lora_runtime_gap_analysis.md",
    ]
    for target in paths:
        files = [target] if target.is_file() else list(target.rglob("*.md")) if target.exists() else []
        for path in files:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            for phrase in FALSE_CLAIMS:
                if phrase in text:
                    hits.append(f"{path.relative_to(ROOT)} contains '{phrase}'")
    return hits


def marker(data: dict, name: str, default: str = "BLOCKED") -> str:
    markers = data.get("markers", {})
    if isinstance(markers, dict) and name in markers:
        return str(markers[name])
    return str(data.get(name, data.get("status", default)))


def write(payload: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "phase1_validation.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Production completion Phase 1 validation",
        "",
        f"`TADSR_PRODUCTION_COMPLETION_PHASE1_VALIDATION: {payload['markers']['TADSR_PRODUCTION_COMPLETION_PHASE1_VALIDATION']}`",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for item in payload["checks"]:
        lines.append(f"| {item['name']} | {item['status']} | {item['detail']} |")
    lines += [
        "",
        "## Must-remain statuses",
        "",
    ]
    for key, value in payload["must_remain_statuses"].items():
        lines.append(f"- `{key}: {value}`")
    (OUT_DIR / "phase1_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    readiness = load_json(OUT_DIR / "readiness.json")
    timevae = load_json(OUT_DIR / "timevae_full" / "official_timevae_full_path_audit.json")
    lora = load_json(OUT_DIR / "runtime_lora" / "official_runtime_lora_behavior_audit.json")
    plan = load_json(OUT_DIR / "full_inference" / "controlled_validation_plan.json")
    blockers = load_json(OUT_DIR / "blockers" / "production_completion_blockers.json")
    statuses = final_audit_statuses()
    must_bad = {k: statuses.get(k, "MISSING") for k, v in MUST_REMAIN.items() if statuses.get(k) != v}
    claims = false_claim_hits()

    readiness_status = marker(readiness, "TADSR_PRODUCTION_COMPLETION_READINESS")
    timevae_status = str(timevae.get("audit_status", timevae.get("status", "BLOCKED")))
    lora_status = str(lora.get("audit_status", lora.get("status", "BLOCKED")))
    lora_not_required = str(lora.get("dynamic_runtime_lora_not_required_marker", "BLOCKED"))
    plan_status = str(plan.get("status", "BLOCKED"))
    blocker_status = str(blockers.get("status", "BLOCKED"))

    timevae_ok = timevae_status in {"PASS", "PARTIAL", "PASS_WITH_EXISTING_EVIDENCE"}
    lora_ok = lora_status in {"PASS", "PARTIAL", "PASS_WITH_EXISTING_EVIDENCE"}
    base_ok = readiness_status in {"PASS", "PARTIAL"} and plan_status == "PASS" and blocker_status == "PASS"
    safe_ok = not must_bad and not claims
    phase_status = "PASS" if base_ok and timevae_status == "PASS" and lora_status == "PASS" and safe_ok else (
        "PASS_WITH_BLOCKERS" if base_ok and timevae_ok and lora_ok and safe_ok else "FAIL"
    )

    checks = [
        {"name": "readiness", "status": readiness_status, "detail": "production completion readiness report"},
        {"name": "timevae_full_production_path_audit", "status": timevae_status, "detail": timevae.get("evidence_source", "")},
        {"name": "official_runtime_lora_behavior_audit", "status": lora_status, "detail": lora.get("evidence_source", "")},
        {"name": "dynamic_runtime_lora_not_required", "status": lora_not_required, "detail": str(lora.get("conclusion", ""))},
        {"name": "controlled_full_inference_plan", "status": plan_status, "detail": "plan exists; stages were not executed"},
        {"name": "blocker_report", "status": blocker_status, "detail": f"{len(blockers.get('blockers', []))} blocker(s) recorded"},
        {"name": "must_remain_statuses", "status": "PASS" if not must_bad else "FAIL", "detail": json.dumps(must_bad, ensure_ascii=False)},
        {"name": "false_claim_scan", "status": "PASS" if not claims else "FAIL", "detail": "; ".join(claims) if claims else "none"},
    ]
    payload = {
        "status": phase_status,
        "checks": checks,
        "must_remain_statuses": {k: statuses.get(k, "MISSING") for k in MUST_REMAIN},
        "markers": {
            "TADSR_PRODUCTION_COMPLETION_PHASE1_VALIDATION": phase_status,
            "TADSR_TIMEVAE_FULL_PRODUCTION_PATH_AUDIT": timevae_status,
            "TADSR_OFFICIAL_RUNTIME_LORA_BEHAVIOR_AUDIT": lora_status,
            "TADSR_DYNAMIC_RUNTIME_LORA_NOT_REQUIRED_FOR_OFFICIAL_INFERENCE": lora_not_required,
            "TADSR_FULL_INFERENCE_CONTROLLED_VALIDATION_PLAN": plan_status,
            "TADSR_PRODUCTION_COMPLETION_BLOCKER_REPORT": blocker_status,
        },
    }
    write(payload)
    for key, value in payload["markers"].items():
        print(f"{key}: {value}")
    return 0 if phase_status in {"PASS", "PASS_WITH_BLOCKERS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
