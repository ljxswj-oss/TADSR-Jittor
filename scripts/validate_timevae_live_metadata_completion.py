#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
IN_JSON = ROOT / "experiments" / "production_completion" / "timevae_full" / "timevae_production_oracle_metadata.json"
OUT_JSON = ROOT / "experiments" / "production_completion" / "timevae_full" / "timevae_live_metadata_completion.json"
OUT_MD = ROOT / "experiments" / "production_completion" / "timevae_full" / "timevae_live_metadata_completion.md"


GROUPS: dict[str, list[str]] = {
    "dependency": [
        "diffusers_available",
        "overlay_used",
        "strict_env_modified",
    ],
    "encode_input": [
        "encode_input_shape",
        "encode_input_dtype",
        "encode_input_min",
        "encode_input_max",
        "encode_input_mean",
        "encode_input_std",
        "encode_input_finite",
        "encode_input_seed",
        "encode_input_generation_policy",
    ],
    "posterior": [
        "encoder_forward_called",
        "quant_conv_used",
        "posterior_object_type",
        "posterior_mean_shape",
        "posterior_mean_dtype",
        "posterior_mean_min",
        "posterior_mean_max",
        "posterior_mean_mean",
        "posterior_mean_std",
        "posterior_logvar_shape",
        "posterior_logvar_dtype",
        "posterior_logvar_min",
        "posterior_logvar_max",
        "posterior_logvar_mean",
        "posterior_logvar_std",
        "posterior_sample_policy",
        "posterior_mode_policy",
        "fixed_epsilon_used",
    ],
    "latent": [
        "latent_shape",
        "latent_dtype",
        "latent_min",
        "latent_max",
        "latent_mean",
        "latent_std",
        "latent_finite",
    ],
    "scaling": [
        "scaling_factor",
        "scaled_latent_shape",
        "scaled_latent_dtype",
        "scaled_latent_min",
        "scaled_latent_max",
        "scaled_latent_mean",
        "scaled_latent_std",
        "scaled_latent_finite",
    ],
    "decode": [
        "decode_input_shape",
        "decode_input_dtype",
        "post_quant_conv_used",
        "decoder_forward_called",
        "decode_output_shape",
        "decode_output_dtype",
        "decode_output_min",
        "decode_output_max",
        "decode_output_mean",
        "decode_output_std",
        "decode_output_finite",
    ],
    "clamp": [
        "clamp_policy",
        "clamped_output_min",
        "clamped_output_max",
        "clamped_output_mean",
        "clamped_output_std",
        "clamped_output_finite",
    ],
    "safety": [
        "full_inference_executed",
        "denoising_loop_executed",
        "unet_called",
        "scheduler_loop_called",
        "image_or_video_generated",
        "raw_tensors_committed",
        "save_local_tensors",
    ],
}

SAFETY_FALSE_FIELDS = {
    "full_inference_executed",
    "denoising_loop_executed",
    "unet_called",
    "scheduler_loop_called",
    "image_or_video_generated",
    "raw_tensors_committed",
    "save_local_tensors",
}

FINITE_TRUE_FIELDS = {
    "encode_input_finite",
    "latent_finite",
    "scaled_latent_finite",
    "decode_output_finite",
    "clamped_output_finite",
}

BLOCKED_VALUES = {
    None,
    "",
    "unknown",
    "blocked_or_not_executed",
    "requires live production metadata",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        return {"status": "FAIL", "error": repr(exc)}


def is_missing(value: Any) -> bool:
    if isinstance(value, list):
        return not value or value == ["blocked_or_not_executed"]
    if isinstance(value, dict):
        return not value
    return value in BLOCKED_VALUES


def group_status(data: dict[str, Any], keys: list[str]) -> tuple[str, list[str], list[str]]:
    missing = [key for key in keys if key not in data or is_missing(data.get(key))]
    bad = []
    for key in keys:
        if key in SAFETY_FALSE_FIELDS and data.get(key) is not False:
            bad.append(key)
        if key in FINITE_TRUE_FIELDS and data.get(key) is not True:
            bad.append(key)
    if not missing and not bad:
        return "PASS", missing, bad
    if data:
        return "PARTIAL", missing, bad
    return "BLOCKED", missing, bad


def marker_for_group(data: dict[str, Any], group: str) -> str:
    status, _, _ = group_status(data, GROUPS[group])
    return status


def write_outputs(result: dict[str, Any]) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# TimeVAE live metadata completion",
        "",
    ]
    for key, value in result["markers"].items():
        lines.append(f"`{key}: {value}`")
    lines += [
        "",
        "| Group | Status | Missing fields | Bad fields |",
        "|---|---|---|---|",
    ]
    for name, row in result["groups"].items():
        lines.append(
            f"| `{name}` | `{row['status']}` | `{', '.join(row['missing_fields']) or 'none'}` | `{', '.join(row['bad_fields']) or 'none'}` |"
        )
    lines += [
        "",
        "## Safety boundary",
        "",
        "- This validator does not import torch or jittor.",
        "- This validator reads only metadata JSON.",
        "- PASS here means the controlled TimeVAE metadata gate is complete; it does not mean full TADSR inference is complete.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    data = load_json(IN_JSON)
    groups = {}
    all_missing: list[str] = []
    all_bad: list[str] = []
    for name, keys in GROUPS.items():
        status, missing, bad = group_status(data, keys)
        groups[name] = {"status": status, "missing_fields": missing, "bad_fields": bad}
        all_missing.extend(missing)
        all_bad.extend(bad)

    encode_status = "PASS" if all(groups[name]["status"] == "PASS" for name in ["dependency", "encode_input", "posterior", "latent", "scaling"]) else ("PARTIAL" if data else "BLOCKED")
    decode_status = "PASS" if all(groups[name]["status"] == "PASS" for name in ["decode", "clamp"]) else ("PARTIAL" if data else "BLOCKED")
    safety_status = marker_for_group(data, "safety")
    completion_status = "PASS" if encode_status == "PASS" and decode_status == "PASS" and safety_status == "PASS" and data.get("status") == "PASS" else ("PARTIAL" if data else "BLOCKED")

    result = {
        "status_marker": "TADSR_TIMEVAE_LIVE_METADATA_COMPLETION",
        "status": completion_status,
        "source": str(IN_JSON.relative_to(ROOT)),
        "oracle_status": data.get("status", "MISSING"),
        "required_field_count": sum(len(keys) for keys in GROUPS.values()),
        "present_required_field_count": sum(
            1
            for keys in GROUPS.values()
            for key in keys
            if key in data and not is_missing(data.get(key))
        ),
        "missing_required_fields": sorted(set(all_missing)),
        "bad_required_fields": sorted(set(all_bad)),
        "groups": groups,
        "ready_for_timevae_preflight": completion_status == "PASS",
        "ready_for_one_step_contract": completion_status == "PASS",
        "markers": {
            "TADSR_TIMEVAE_LIVE_METADATA_COMPLETION": completion_status,
            "TADSR_TIMEVAE_LIVE_ENCODE_METADATA": encode_status,
            "TADSR_TIMEVAE_LIVE_DECODE_METADATA": decode_status,
            "TADSR_TIMEVAE_LIVE_SAFETY_FLAGS": safety_status,
        },
    }
    write_outputs(result)
    for key, value in result["markers"].items():
        print(f"{key}: {value}")
    return 0 if completion_status in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
