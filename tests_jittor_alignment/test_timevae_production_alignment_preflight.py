#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "production_completion" / "timevae_full"


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


def staged_raw_tensors() -> list[str]:
    try:
        proc = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except Exception:
        return ["unable_to_check_git_index"]
    return [line for line in proc.stdout.splitlines() if line.endswith((".npy", ".npz"))]


def write(payload: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "jittor_timevae_production_alignment_preflight.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# Jittor TimeVAE production alignment preflight",
        "",
        f"`TADSR_TIMEVAE_PRODUCTION_ALIGNMENT_PREFLIGHT: {payload['status']}`",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for item in payload["checks"]:
        lines.append(f"| {item['name']} | {item['status']} | {item['detail']} |")
    lines += ["", "## Phase 4-A metadata gate markers", ""]
    for key, value in payload.get("markers", {}).items():
        lines.append(f"- `{key}: {value}`")
    (OUT_DIR / "jittor_timevae_production_alignment_preflight.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    oracle = load_json(OUT_DIR / "timevae_production_oracle_metadata.json")
    completion = load_json(OUT_DIR / "timevae_live_metadata_completion.json")
    audit = load_json(OUT_DIR / "official_timevae_full_path_audit.json")
    gap = load_json(ROOT / "experiments" / "timevae_full_alignment_gap_analysis.json")
    statuses = final_statuses()

    oracle_status = str(oracle.get("status", "MISSING"))
    actual_boundary = str(statuses.get("TIME_VAE_ACTUAL_VAEHOOK_FULL_BOUNDARY_ALIGNMENT", gap.get("actual_vae_hook_boundary_status", "MISSING")))
    timevae_full = str(statuses.get("TIME_VAE_FULL_ALIGNMENT", "MISSING"))
    policy_documented = bool(audit) and bool(oracle)
    remaining_gaps = audit.get("not_complete_subpaths") or gap.get("not_complete_subpaths") or []
    raw_staged = staged_raw_tensors()
    required_metadata_fields = [
        "encode_input_shape",
        "encode_input_dtype",
        "encode_input_range",
        "encode_input_stats",
        "posterior_mean_shape",
        "posterior_logvar_shape",
        "latent_shape",
        "scaling_factor",
        "scaled_latent_shape",
        "decode_input_shape",
        "decode_output_shape",
        "decode_output_stats",
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
    repair_attempt_status = "PASS" if oracle_status == "PASS" else ("PARTIAL" if oracle else "BLOCKED")
    metadata_fields_status = (
        "PASS"
        if completion_status == "PASS" or (not missing_metadata_fields and oracle_status == "PASS")
        else ("PARTIAL" if oracle else "BLOCKED")
    )
    ready_for_one_step_marker = (
        "PASS"
        if completion_status == "PASS" or (bool(oracle.get("ready_for_one_step_contract")) and not missing_metadata_fields)
        else ("PARTIAL" if oracle else "BLOCKED")
    )

    if oracle_status == "PASS" and actual_boundary == "PASS" and policy_documented and timevae_full == "NOT_COMPLETE":
        status = "PASS"
    elif oracle_status in {"PARTIAL", "BLOCKED"} and actual_boundary == "PASS" and policy_documented and timevae_full == "NOT_COMPLETE":
        status = "PARTIAL"
    else:
        status = "BLOCKED"

    checks = [
        {"name": "oracle_metadata_exists", "status": "PASS" if oracle else "BLOCKED", "detail": oracle_status},
        {"name": "actual_vaehook_boundary_pass", "status": "PASS" if actual_boundary == "PASS" else "BLOCKED", "detail": actual_boundary},
        {"name": "shape_contract_compatible", "status": "PARTIAL" if oracle_status in {"BLOCKED", "PARTIAL"} else "PASS", "detail": "live tensor shapes not exported in this phase"},
        {"name": "scaling_clamp_policy_documented", "status": "PASS" if policy_documented else "BLOCKED", "detail": "metadata/audit files inspected"},
        {"name": "decoder_original_forward_policy_documented", "status": "PASS" if audit.get("original_forward_called") else "PARTIAL", "detail": str(audit.get("original_forward_called", "unknown"))},
        {"name": "tiled_policy_documented", "status": "PASS" if audit else "BLOCKED", "detail": f"remaining gaps: {len(remaining_gaps)}"},
        {"name": "timevae_full_alignment_not_upgraded", "status": "PASS" if timevae_full == "NOT_COMPLETE" else "FAIL", "detail": timevae_full},
        {"name": "raw_tensors_not_staged", "status": "PASS" if not raw_staged else "FAIL", "detail": ", ".join(raw_staged) if raw_staged else "no staged .npy/.npz"},
        {"name": "metadata_repair_attempt_recorded", "status": repair_attempt_status, "detail": f"oracle status: {oracle_status}"},
        {"name": "metadata_required_fields_complete", "status": metadata_fields_status, "detail": ", ".join(missing_metadata_fields) if missing_metadata_fields else "all required fields complete"},
        {"name": "metadata_ready_for_one_step", "status": ready_for_one_step_marker, "detail": str(oracle.get("ready_for_one_step_contract", False))},
    ]
    markers = {
        "TADSR_TIMEVAE_METADATA_REPAIR_ATTEMPT": repair_attempt_status,
        "TADSR_TIMEVAE_METADATA_FIELDS_COMPLETE": metadata_fields_status,
        "TADSR_TIMEVAE_METADATA_READY_FOR_ONE_STEP": ready_for_one_step_marker,
        "TADSR_TIMEVAE_LIVE_METADATA_COMPLETION": str(completion_markers.get("TADSR_TIMEVAE_LIVE_METADATA_COMPLETION", completion_status if completion else "BLOCKED")),
        "TADSR_TIMEVAE_LIVE_ENCODE_METADATA": str(completion_markers.get("TADSR_TIMEVAE_LIVE_ENCODE_METADATA", "BLOCKED")),
        "TADSR_TIMEVAE_LIVE_DECODE_METADATA": str(completion_markers.get("TADSR_TIMEVAE_LIVE_DECODE_METADATA", "BLOCKED")),
        "TADSR_TIMEVAE_LIVE_SAFETY_FLAGS": str(completion_markers.get("TADSR_TIMEVAE_LIVE_SAFETY_FLAGS", "BLOCKED")),
    }
    payload = {
        "status_marker": "TADSR_TIMEVAE_PRODUCTION_ALIGNMENT_PREFLIGHT",
        "status": status,
        "oracle_metadata_status": oracle_status,
        "actual_vaehook_boundary_status": actual_boundary,
        "timevae_full_alignment_status": timevae_full,
        "remaining_gaps": remaining_gaps,
        "ready_for_one_step_tensor_alignment": bool(status == "PASS" and oracle_status == "PASS" and metadata_fields_status == "PASS" and ready_for_one_step_marker == "PASS" and not raw_staged),
        "timevae_live_metadata_completion_status": completion_status,
        "missing_metadata_fields": missing_metadata_fields,
        "markers": markers,
        "checks": checks,
        "full_tadsr_inference_executed": False,
        "image_or_video_output_generated": False,
    }
    write(payload)
    print(f"TADSR_TIMEVAE_PRODUCTION_ALIGNMENT_PREFLIGHT: {status}")
    return 0 if status in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
