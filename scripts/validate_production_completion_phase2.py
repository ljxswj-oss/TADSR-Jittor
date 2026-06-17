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


def final_statuses() -> dict[str, str]:
    data = load_json(ROOT / "experiments" / "final_audit_report.json")
    return {str(row.get("check")): str(row.get("status")) for row in data.get("rows", [])}


def false_claim_hits() -> list[str]:
    roots = [
        ROOT / "README.md",
        ROOT / "docs",
        ROOT / "experiments" / "production_completion",
    ]
    hits: list[str] = []
    for root in roots:
        files = [root] if root.is_file() else list(root.rglob("*.md")) if root.exists() else []
        for path in files:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            for phrase in FALSE_CLAIMS:
                if phrase in text:
                    hits.append(f"{path.relative_to(ROOT)} contains '{phrase}'")
    return hits


def status_of(data: dict, key: str = "status") -> str:
    return str(data.get(key, "BLOCKED")) if data else "BLOCKED"


def write(payload: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "phase2_validation.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Production completion Phase 2 validation",
        "",
        f"`TADSR_PRODUCTION_COMPLETION_PHASE2_VALIDATION: {payload['markers']['TADSR_PRODUCTION_COMPLETION_PHASE2_VALIDATION']}`",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for item in payload["checks"]:
        lines.append(f"| {item['name']} | {item['status']} | {item['detail']} |")
    lines += ["", "## Must-remain statuses", ""]
    for key, value in payload["must_remain_statuses"].items():
        lines.append(f"- `{key}: {value}`")
    (OUT_DIR / "phase2_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_blockers(payload: dict) -> None:
    path = OUT_DIR / "blockers" / "production_completion_blockers.json"
    blockers = load_json(path)
    if not blockers:
        return
    blockers["phase2_status"] = {
        "official_env_resolution_status": payload["markers"]["TADSR_PRODUCTION_OFFICIAL_ENV_RESOLUTION"],
        "timevae_oracle_metadata_status": payload["markers"]["TADSR_TIMEVAE_PRODUCTION_ORACLE_METADATA"],
        "timevae_alignment_preflight_status": payload["markers"]["TADSR_TIMEVAE_PRODUCTION_ALIGNMENT_PREFLIGHT"],
        "lora_live_audit_status": payload["markers"]["TADSR_OFFICIAL_RUNTIME_LORA_BEHAVIOR_AUDIT"],
        "full_inference_metadata_contract_status": payload["markers"]["TADSR_FULL_INFERENCE_METADATA_DRY_RUN_CONTRACT"],
    }
    blockers["phase2_next_action"] = (
        "Set TADSR_OFFICIAL_REPO, TADSR_OFFICIAL_WEIGHTS and TADSR_OFFICIAL_PYTHON on a machine with the official runtime, "
        "then rerun TimeVAE metadata oracle export and LoRA live audit."
    )
    path.write_text(json.dumps(blockers, indent=2, ensure_ascii=False), encoding="utf-8")
    md = OUT_DIR / "blockers" / "production_completion_blockers.md"
    existing = md.read_text(encoding="utf-8", errors="ignore") if md.exists() else "# Production completion blocker report\n"
    section = [
        "",
        "## Phase 2 blocker status",
        "",
        "| Item | Status |",
        "|---|---|",
    ]
    for key, value in blockers["phase2_status"].items():
        section.append(f"| `{key}` | `{value}` |")
    section += ["", blockers["phase2_next_action"], ""]
    marker = "## Phase 2 blocker status"
    if marker in existing:
        existing = existing.split(marker)[0].rstrip() + "\n" + "\n".join(section)
    else:
        existing = existing.rstrip() + "\n" + "\n".join(section)
    md.write_text(existing, encoding="utf-8")


def main() -> int:
    statuses = final_statuses()
    env = load_json(OUT_DIR / "env" / "official_env_resolution.json")
    timevae_audit = load_json(OUT_DIR / "timevae_full" / "official_timevae_full_path_audit.json")
    oracle = load_json(OUT_DIR / "timevae_full" / "timevae_production_oracle_metadata.json")
    preflight = load_json(OUT_DIR / "timevae_full" / "jittor_timevae_production_alignment_preflight.json")
    lora = load_json(OUT_DIR / "runtime_lora" / "official_runtime_lora_behavior_audit.json")
    contract = load_json(OUT_DIR / "full_inference" / "metadata_dry_run_contract.json")
    blockers = load_json(OUT_DIR / "blockers" / "production_completion_blockers.json")
    must_bad = {k: statuses.get(k, "MISSING") for k, v in MUST_REMAIN.items() if statuses.get(k) != v}
    claims = false_claim_hits()

    markers = {
        "TADSR_PRODUCTION_OFFICIAL_ENV_RESOLUTION": status_of(env),
        "TADSR_TIMEVAE_FULL_PRODUCTION_PATH_AUDIT": str(timevae_audit.get("audit_status", timevae_audit.get("status", "BLOCKED"))) if timevae_audit else "BLOCKED",
        "TADSR_TIMEVAE_PRODUCTION_ORACLE_METADATA": status_of(oracle),
        "TADSR_TIMEVAE_PRODUCTION_ALIGNMENT_PREFLIGHT": status_of(preflight),
        "TADSR_OFFICIAL_RUNTIME_LORA_BEHAVIOR_AUDIT": str(lora.get("audit_status", lora.get("status", "BLOCKED"))) if lora else "BLOCKED",
        "TADSR_DYNAMIC_RUNTIME_LORA_NOT_REQUIRED_FOR_OFFICIAL_INFERENCE": str(lora.get("dynamic_runtime_lora_not_required_marker", "BLOCKED")) if lora else "BLOCKED",
        "TADSR_FULL_INFERENCE_METADATA_DRY_RUN_CONTRACT": status_of(contract),
        "TADSR_PRODUCTION_COMPLETION_BLOCKER_REPORT": status_of(blockers),
    }
    acceptable = {"PASS", "PARTIAL", "BLOCKED"}
    base_ok = all(value in acceptable for value in markers.values()) and not must_bad and not claims
    if not base_ok:
        phase2 = "FAIL"
    elif markers["TADSR_PRODUCTION_OFFICIAL_ENV_RESOLUTION"] == "PASS" and markers["TADSR_TIMEVAE_PRODUCTION_ORACLE_METADATA"] == "PASS":
        phase2 = "PASS"
    else:
        phase2 = "PASS_WITH_BLOCKERS"
    markers["TADSR_PRODUCTION_COMPLETION_PHASE2_VALIDATION"] = phase2

    checks = [
        {"name": key, "status": value, "detail": "phase2 marker"} for key, value in markers.items() if key != "TADSR_PRODUCTION_COMPLETION_PHASE2_VALIDATION"
    ]
    checks += [
        {"name": "must_remain_statuses", "status": "PASS" if not must_bad else "FAIL", "detail": json.dumps(must_bad, ensure_ascii=False)},
        {"name": "false_claim_scan", "status": "PASS" if not claims else "FAIL", "detail": "; ".join(claims) if claims else "none"},
        {"name": "full_inference_executed", "status": "PASS", "detail": "no"},
        {"name": "denoising_loop_executed", "status": "PASS", "detail": "no"},
        {"name": "image_or_video_output_generated", "status": "PASS", "detail": "no"},
    ]
    payload = {
        "status": phase2,
        "checks": checks,
        "markers": markers,
        "must_remain_statuses": {k: statuses.get(k, "MISSING") for k in MUST_REMAIN},
    }
    write(payload)
    update_blockers(payload)
    for key, value in markers.items():
        print(f"{key}: {value}")
    return 0 if phase2 in {"PASS", "PASS_WITH_BLOCKERS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
