#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "production_completion" / "timevae_full"
OUT_JSON = OUT_DIR / "timevae_metadata_partial_diagnosis.json"
OUT_MD = OUT_DIR / "timevae_metadata_partial_diagnosis.md"

INPUTS = {
    "oracle": OUT_DIR / "timevae_production_oracle_metadata.json",
    "audit": OUT_DIR / "official_timevae_full_path_audit.json",
    "preflight": OUT_DIR / "jittor_timevae_production_alignment_preflight.json",
    "contract": ROOT / "experiments" / "production_completion" / "full_inference" / "metadata_dry_run_contract.json",
    "phase3": ROOT / "experiments" / "production_completion" / "phase3_validation.json",
    "blockers": ROOT / "experiments" / "production_completion" / "blockers" / "production_completion_blockers.json",
}

REQUIRED_FIELDS = [
    "official_python",
    "official_torch_version",
    "diffusers_available",
    "dependency_status",
    "dependency_blockers",
    "encode_input_shape",
    "encode_input_dtype",
    "encode_input_range",
    "encode_input_stats",
    "encoder_hook_path",
    "encoder_tiled_used",
    "quant_conv_used",
    "latent_distribution_type",
    "posterior_mean_shape",
    "posterior_logvar_shape",
    "posterior_sample_policy",
    "fixed_epsilon_used",
    "latent_shape",
    "latent_dtype",
    "latent_stats",
    "scaling_factor",
    "scaled_latent_shape",
    "scaled_latent_stats",
    "decode_input_shape",
    "decode_input_dtype",
    "post_quant_conv_used",
    "decoder_hook_path",
    "decoder_tiled_used",
    "decoder_original_forward_used",
    "decode_output_shape",
    "decode_output_dtype",
    "decode_output_range",
    "decode_output_stats",
    "clamp_policy",
    "full_inference_executed",
    "denoising_loop_executed",
    "image_or_video_generated",
    "raw_tensors_committed",
    "local_tensor_dir_ignored",
    "ready_for_timevae_preflight",
    "ready_for_one_step_contract",
]

OPTIONAL_FIELDS = [
    "summary_stats",
    "sha256",
    "local_tensor_paths",
    "partial_reasons",
    "blockers",
    "next_action",
]

BLOCKED_VALUES = {
    "",
    None,
    "unknown",
    "blocked",
    "blocked_or_not_executed",
    "requires live production metadata",
    "MISSING",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        return data if isinstance(data, dict) else {"rows": data}
    except Exception as exc:
        return {"status": "FAIL", "error": repr(exc)}


def status(data: dict[str, Any], default: str = "MISSING") -> str:
    if not data:
        return default
    return str(data.get("audit_status", data.get("status", default)))


def is_missing_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip() in BLOCKED_VALUES
    if isinstance(value, list):
        return len(value) == 0 or value == ["blocked_or_not_executed"]
    if isinstance(value, dict):
        return len(value) == 0
    return value in BLOCKED_VALUES


def missing_fields(data: dict[str, Any], fields: list[str]) -> list[str]:
    missing: list[str] = []
    for field in fields:
        if field not in data or is_missing_value(data.get(field)):
            missing.append(field)
    return missing


def bool_from_any(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "pass"}
    return False


def write_md(payload: dict[str, Any]) -> None:
    lines = [
        "# TimeVAE production metadata PARTIAL diagnosis",
        "",
        f"`TADSR_TIMEVAE_METADATA_PARTIAL_DIAGNOSIS: {payload['status']}`",
        "",
        "## Current status",
        "",
        f"- `current_status`: `{payload['current_status']}`",
        f"- `preflight_status`: `{payload['preflight_status']}`",
        f"- `metadata_contract_status`: `{payload['metadata_contract_status']}`",
        f"- `can_attempt_live_repair_on_linux`: `{payload['can_attempt_live_repair_on_linux']}`",
        "",
        "## Missing required fields",
        "",
    ]
    if payload["missing_required_fields"]:
        lines.extend(f"- `{item}`" for item in payload["missing_required_fields"])
    else:
        lines.append("- none")
    lines += ["", "## Blocker classification", ""]
    for key in [
        "blocked_by_dependency",
        "blocked_by_forward_execution",
        "blocked_by_missing_diffusers",
        "blocked_by_missing_weight",
        "blocked_by_missing_shape_contract",
    ]:
        lines.append(f"- `{key}`: `{payload[key]}`")
    lines += ["", "## Recommended repair actions", ""]
    for action in payload["recommended_repair_actions"]:
        lines.append(f"- {action}")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    data = {name: load_json(path) for name, path in INPUTS.items()}
    oracle = data["oracle"]
    audit = data["audit"]
    preflight = data["preflight"]
    contract = data["contract"]
    phase3 = data["phase3"]
    blocker_report = data["blockers"]

    current_status = status(oracle)
    missing_required = missing_fields(oracle, REQUIRED_FIELDS)
    missing_optional = missing_fields(oracle, OPTIONAL_FIELDS)
    dependency_blockers = oracle.get("dependency_blockers") or []
    blockers = list(oracle.get("blockers") or [])
    if blocker_report.get("phase3_blockers"):
        blockers.extend(str(item) for item in blocker_report.get("phase3_blockers", []))

    diffusers_available = oracle.get("diffusers_available")
    blocked_by_missing_diffusers = diffusers_available is False or "diffusers" in " ".join(map(str, dependency_blockers + blockers)).lower()
    blocked_by_dependency = bool(dependency_blockers) or blocked_by_missing_diffusers
    blocked_by_forward_execution = bool(
        oracle.get("forward_executed") is False
        or oracle.get("encode_input_shape") in {None, "blocked_or_not_executed"}
        or oracle.get("decode_output_shape") in {None, "blocked_or_not_executed"}
    )
    blocked_by_missing_weight = "weight" in " ".join(map(str, blockers)).lower() or status(data["audit"]) == "BLOCKED"
    shape_fields = ["encode_input_shape", "latent_shape", "decode_input_shape", "decode_output_shape"]
    blocked_by_missing_shape_contract = any(field in missing_required for field in shape_fields)
    env_status = str(phase3.get("official_env_resolution_status", oracle.get("official_env_status", "MISSING")))
    can_attempt_live_repair = env_status == "PASS" and not blocked_by_missing_weight

    actions: list[str] = []
    if blocked_by_missing_diffusers:
        actions.append("Official Python cannot import diffusers; record dependency gap or run in an environment that already has diffusers.")
    if blocked_by_forward_execution:
        actions.append("Run the metadata exporter in Linux official env in metadata-only mode; do not save or commit raw tensors.")
    if blocked_by_missing_shape_contract:
        actions.append("Populate encode/decode shape, dtype, range, sampling, scaling and clamp policy fields.")
    if not can_attempt_live_repair:
        actions.append("Resolve official repo / weights / strict Python availability before attempting live repair.")
    if not actions:
        actions.append("Metadata fields look complete; rerun TimeVAE preflight and full inference metadata contract.")

    payload: dict[str, Any] = {
        "status_marker": "TADSR_TIMEVAE_METADATA_PARTIAL_DIAGNOSIS",
        "status": "PASS",
        "current_status": current_status,
        "preflight_status": status(preflight),
        "metadata_contract_status": status(contract),
        "phase3_status": status(phase3),
        "audit_status": status(audit),
        "missing_required_fields": missing_required,
        "missing_optional_fields": missing_optional,
        "blocked_by_dependency": blocked_by_dependency,
        "blocked_by_forward_execution": blocked_by_forward_execution,
        "blocked_by_missing_diffusers": blocked_by_missing_diffusers,
        "blocked_by_missing_weight": blocked_by_missing_weight,
        "blocked_by_missing_shape_contract": blocked_by_missing_shape_contract,
        "dependency_blockers": dependency_blockers,
        "blockers": blockers,
        "recommended_repair_actions": actions,
        "can_attempt_live_repair_on_linux": can_attempt_live_repair,
        "safe_scope": {
            "imports_torch": False,
            "imports_jittor": False,
            "executes_model": False,
            "runs_full_inference": False,
            "generates_image_or_video": False,
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_md(payload)
    print("TADSR_TIMEVAE_METADATA_PARTIAL_DIAGNOSIS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
