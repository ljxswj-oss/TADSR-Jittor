#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "production_completion" / "full_inference"


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


def guard_ok() -> bool:
    proc = subprocess.run(
        [sys.executable, "-m", "jittor_tadsr_full.tadsr_full", "--full-inference"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.returncode != 0 and "NotImplementedError" in proc.stdout and "Full Jittor TADSR inference is not complete" in proc.stdout


def write(payload: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metadata_dry_run_contract.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# Full inference metadata dry-run contract",
        "",
        f"`TADSR_FULL_INFERENCE_METADATA_DRY_RUN_CONTRACT: {payload['status']}`",
        "",
        "| Stage | Status | Detail |",
        "|---|---|---|",
    ]
    for item in payload["checks"]:
        lines.append(f"| {item['stage']} | {item['status']} | {item['detail']} |")
    lines += ["", "## Phase 4-A metadata gate markers", ""]
    for key, value in payload.get("markers", {}).items():
        lines.append(f"- `{key}: {value}`")
    (OUT_DIR / "metadata_dry_run_contract.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    statuses = final_statuses()
    oracle = load_json(ROOT / "experiments" / "production_completion" / "timevae_full" / "timevae_production_oracle_metadata.json")
    completion = load_json(ROOT / "experiments" / "production_completion" / "timevae_full" / "timevae_live_metadata_completion.json")
    lora = load_json(ROOT / "experiments" / "production_completion" / "runtime_lora" / "official_runtime_lora_behavior_audit.json")
    plan = load_json(ROOT / "experiments" / "production_completion" / "full_inference" / "controlled_validation_plan.json")
    feasibility = load_json(ROOT / "experiments" / "jittor_migration_feasibility_summary.json")

    checks = [
        {"stage": "preprocess contract", "status": "PASS" if statuses.get("JITTOR_PREPROCESS_ALIGNMENT") == "PASS" else "PARTIAL", "detail": statuses.get("JITTOR_PREPROCESS_ALIGNMENT", "MISSING")},
        {"stage": "condition/control contract", "status": "PASS" if feasibility else "PARTIAL", "detail": "feasibility summary loaded" if feasibility else "missing feasibility summary"},
        {"stage": "TimeVAE encode metadata", "status": oracle.get("status", "BLOCKED"), "detail": "production oracle metadata"},
        {"stage": "scheduler timesteps evidence", "status": "PASS" if statuses.get("TADSR_SCHEDULER_BOUNDARY_ALIGNMENT") == "PASS" else "PARTIAL", "detail": statuses.get("TADSR_SCHEDULER_BOUNDARY_ALIGNMENT", "MISSING")},
        {"stage": "UNet full forward evidence", "status": "PASS" if statuses.get("TADSR_UNET_FULL_FORWARD_ALIGNMENT") == "PASS" else "PARTIAL", "detail": statuses.get("TADSR_UNET_FULL_FORWARD_ALIGNMENT", "MISSING")},
        {"stage": "get_x0_from_res evidence", "status": "PASS" if statuses.get("TADSR_GET_X0_FROM_RES_ALIGNMENT") == "PASS" else "PARTIAL", "detail": statuses.get("TADSR_GET_X0_FROM_RES_ALIGNMENT", "MISSING")},
        {"stage": "TimeVAE decode metadata", "status": oracle.get("status", "BLOCKED"), "detail": "production oracle metadata"},
        {"stage": "postprocess contract", "status": "PASS" if plan else "PARTIAL", "detail": "documented in controlled plan"},
        {"stage": "dynamic LoRA status", "status": lora.get("dynamic_runtime_lora_not_required_marker", "BLOCKED"), "detail": str(lora.get("conclusion", ""))},
        {"stage": "full inference guard", "status": "PASS" if guard_ok() else "FAIL", "detail": "NotImplementedError preserved"},
        {"stage": "final image/video output", "status": "NOT_EXECUTED", "detail": "no image/video generated in this metadata contract"},
    ]
    required_metadata_fields = [
        "encode_input_shape",
        "latent_shape",
        "scaled_latent_shape",
        "decode_input_shape",
        "decode_output_shape",
        "clamp_policy",
    ]
    blocked_tokens = {None, "", "unknown", "blocked_or_not_executed", "requires live production metadata"}
    def missing_value(value):
        if isinstance(value, (list, tuple)):
            return len(value) == 0 or value == ["blocked_or_not_executed"]
        if isinstance(value, dict):
            return len(value) == 0
        return value in blocked_tokens
    missing_metadata_fields = [
        key
        for key in required_metadata_fields
        if key not in oracle or missing_value(oracle.get(key))
    ]
    completion_markers = completion.get("markers", {}) if isinstance(completion.get("markers"), dict) else {}
    completion_status = str(completion.get("status", "MISSING"))
    repair_attempt_status = "PASS" if oracle.get("status") == "PASS" else ("PARTIAL" if oracle else "BLOCKED")
    fields_complete_status = (
        "PASS"
        if completion_status == "PASS" or (not missing_metadata_fields and oracle.get("status") == "PASS")
        else ("PARTIAL" if oracle else "BLOCKED")
    )
    ready_for_one_step_status = (
        "PASS"
        if completion_status == "PASS" or (oracle.get("ready_for_one_step_contract") is True and not missing_metadata_fields)
        else ("PARTIAL" if oracle else "BLOCKED")
    )
    hard_fail = any(item["status"] == "FAIL" for item in checks)
    if hard_fail:
        status = "FAIL"
    elif oracle.get("status") == "PASS" and all(item["status"] in {"PASS", "NOT_EXECUTED"} for item in checks):
        status = "PASS"
    elif oracle and statuses.get("JITTOR_FULL_INFERENCE") == "NOT_COMPLETE":
        status = "PARTIAL"
    else:
        status = "BLOCKED"
    payload = {
        "status_marker": "TADSR_FULL_INFERENCE_METADATA_DRY_RUN_CONTRACT",
        "status": status,
        "checks": checks,
        "missing_timevae_metadata_fields": missing_metadata_fields,
        "markers": {
            "TADSR_TIMEVAE_METADATA_REPAIR_ATTEMPT": repair_attempt_status,
            "TADSR_TIMEVAE_METADATA_FIELDS_COMPLETE": fields_complete_status,
            "TADSR_TIMEVAE_METADATA_READY_FOR_ONE_STEP": ready_for_one_step_status,
            "TADSR_TIMEVAE_LIVE_METADATA_COMPLETION": str(completion_markers.get("TADSR_TIMEVAE_LIVE_METADATA_COMPLETION", completion_status if completion else "BLOCKED")),
            "TADSR_TIMEVAE_LIVE_ENCODE_METADATA": str(completion_markers.get("TADSR_TIMEVAE_LIVE_ENCODE_METADATA", "BLOCKED")),
            "TADSR_TIMEVAE_LIVE_DECODE_METADATA": str(completion_markers.get("TADSR_TIMEVAE_LIVE_DECODE_METADATA", "BLOCKED")),
            "TADSR_TIMEVAE_LIVE_SAFETY_FLAGS": str(completion_markers.get("TADSR_TIMEVAE_LIVE_SAFETY_FLAGS", "BLOCKED")),
        },
        "ready_for_one_step_tensor_alignment": bool(status == "PASS" and oracle.get("status") == "PASS" and fields_complete_status == "PASS" and ready_for_one_step_status == "PASS" and guard_ok()),
        "timevae_live_metadata_completion_status": completion_status,
        "full_inference_executed": False,
        "denoising_loop_executed": False,
        "image_or_video_output_generated": False,
        "jittor_full_inference_status": statuses.get("JITTOR_FULL_INFERENCE", "MISSING"),
    }
    write(payload)
    print(f"TADSR_FULL_INFERENCE_METADATA_DRY_RUN_CONTRACT: {status}")
    return 0 if status in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
