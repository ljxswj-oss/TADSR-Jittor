#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "experiments" / "production_completion" / "full_inference" / "one_step"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as exc:
        return {"status": "FAIL", "error": repr(exc)}


def marker_status(payload: dict, marker: str, default: str = "BLOCKED") -> str:
    markers = payload.get("markers", {}) if isinstance(payload, dict) else {}
    if isinstance(markers, dict) and marker in markers:
        return str(markers[marker])
    if payload.get("status_marker") == marker:
        return str(payload.get("audit_status", payload.get("status", default)))
    return default


def write_reports(out_dir: Path, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "one_step_official_path_audit.json"
    md_path = out_dir / "one_step_official_path_audit.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Official TADSR one-step tensor path audit",
        "",
        f"`TADSR_ONE_STEP_OFFICIAL_PATH_AUDIT: {payload['status']}`",
        "",
        "This audit only checks whether the controlled one-step tensor path is safe and well defined.",
        "It does not run a full denoising loop, does not invoke the production full-inference CLI, and does not save images or videos.",
        "",
        "## Stage contract",
        "",
        "| Stage | Status | Note |",
        "|---|---|---|",
    ]
    for row in payload["stages"]:
        lines.append(f"| `{row['name']}` | `{row['status']}` | {row['note']} |")
    lines += [
        "",
        "## Safety flags",
        "",
        "| Flag | Value |",
        "|---|---|",
    ]
    for key, value in payload["safety_flags"].items():
        lines.append(f"| `{key}` | `{value}` |")
    if payload["blockers"]:
        lines += ["", "## Blockers", ""]
        for item in payload["blockers"]:
            lines.append(f"- {item}")
    lines += [
        "",
        "## Must-remain statuses",
        "",
        "| Marker | Expected |",
        "|---|---|",
    ]
    for key, value in payload["must_remain_statuses"].items():
        lines.append(f"| `{key}` | `{value}` |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the official controlled one-step TADSR tensor path.")
    parser.add_argument("--official-repo", default="/mnt/data/sj/projects/TADSR_official_pytorch")
    parser.add_argument("--official-weights", default="/mnt/data/sj/checkpoints/TADSR/preset/weights")
    parser.add_argument("--official-python", default="/mnt/data/sj/venvs/tadsr_official_strict_cu118/bin/python")
    parser.add_argument("--pythonpath-overlay", default="/mnt/data/sj/tmp/tadsr_official_dependency_overlay_diffusers")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    completion = load_json(ROOT / "experiments" / "production_completion" / "timevae_full" / "timevae_live_metadata_completion.json")
    phase3 = load_json(ROOT / "experiments" / "production_completion" / "phase3_validation.json")
    unet = load_json(ROOT / "experiments" / "full_repro" / "unet_alignment" / "jittor_unet_full_forward_alignment.json")
    minimal = load_json(ROOT / "experiments" / "full_repro" / "integration_alignment" / "jittor_minimal_latent_integration_alignment.json")
    scheduler = load_json(ROOT / "experiments" / "full_repro" / "scheduler_alignment" / "jittor_scheduler_boundary_alignment.json")

    env_paths = {
        "official_repo": Path(args.official_repo),
        "official_weights": Path(args.official_weights),
        "official_python": Path(args.official_python),
        "pythonpath_overlay": Path(args.pythonpath_overlay),
    }
    env_status = {
        key: "PASS" if path.exists() else "NOT_PRESENT_ON_THIS_MACHINE"
        for key, path in env_paths.items()
    }
    timevae_ready = marker_status(completion, "TADSR_TIMEVAE_LIVE_METADATA_COMPLETION") == "PASS"
    one_step_contract_ready = marker_status(phase3, "TADSR_FULL_INFERENCE_CONTRACT_READY_FOR_ONE_STEP") == "PASS"
    protocol_ready = marker_status(phase3, "TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT_PROTOCOL_READY") == "PASS"
    existing_minimal_pass = str(minimal.get("status", "BLOCKED")) == "PASS"
    existing_unet_pass = str(unet.get("status", "BLOCKED")) == "PASS"
    scheduler_pass = str(scheduler.get("status", "BLOCKED")) == "PASS"

    stages = [
        {"name": "deterministic_input_tensor", "status": "PASS", "note": "fixed seed tensor contract only"},
        {"name": "timevae_encode", "status": "PASS" if timevae_ready else "BLOCKED", "note": "requires live TimeVAE metadata completion"},
        {"name": "scaling_factor", "status": "PASS" if timevae_ready else "BLOCKED", "note": "read from official TimeAwareAutoencoderKL config"},
        {"name": "scheduler_timestep", "status": "PASS" if scheduler_pass or existing_minimal_pass else "BLOCKED", "note": "single audited timestep only"},
        {"name": "unet_full_forward_once", "status": "PASS" if existing_unet_pass else "BLOCKED", "note": "single tensor forward boundary, not denoising loop"},
        {"name": "get_x0_from_res", "status": "PASS" if existing_minimal_pass else "BLOCKED", "note": "x0 = latent / sqrt(alpha) - model_pred"},
        {"name": "timevae_decode", "status": "PASS" if existing_minimal_pass else "BLOCKED", "note": "tensor-only decode boundary"},
        {"name": "clamp_tensor", "status": "PASS" if existing_minimal_pass else "BLOCKED", "note": "clamp to [-1, 1], no image postprocess"},
        {"name": "image_or_video_postprocess", "status": "SKIPPED_BY_DESIGN", "note": "explicitly outside this phase"},
    ]
    blockers = [
        f"{row['name']} is {row['status']}"
        for row in stages
        if row["status"] not in {"PASS", "SKIPPED_BY_DESIGN"}
    ]
    protocol_note = (
        "Existing Phase 3 protocol gate is PASS."
        if protocol_ready
        else "Existing Phase 3 protocol gate is not PASS, but this execution-phase audit can still proceed using the already audited minimal one-step contract."
    )

    safety_flags = {
        "full_inference_executed": False,
        "denoising_loop_executed": False,
        "production_cli_used": False,
        "image_or_video_generated": False,
        "raw_tensor_committed": False,
        "dynamic_runtime_lora_implemented": False,
    }
    status = "PASS" if not blockers else "BLOCKED"
    payload = {
        "status_marker": "TADSR_ONE_STEP_OFFICIAL_PATH_AUDIT",
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "official_environment_paths": {k: str(v) for k, v in env_paths.items()},
        "official_environment_path_status": env_status,
        "timevae_live_metadata_completion_status": marker_status(completion, "TADSR_TIMEVAE_LIVE_METADATA_COMPLETION"),
        "one_step_contract_ready_status": marker_status(phase3, "TADSR_FULL_INFERENCE_CONTRACT_READY_FOR_ONE_STEP"),
        "one_step_protocol_ready_status": marker_status(phase3, "TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT_PROTOCOL_READY"),
        "one_step_protocol_note": protocol_note,
        "existing_minimal_latent_one_step_status": str(minimal.get("status", "BLOCKED")),
        "existing_unet_full_forward_alignment_status": str(unet.get("status", "BLOCKED")),
        "existing_scheduler_boundary_alignment_status": str(scheduler.get("status", "BLOCKED")),
        "stages": stages,
        "safety_flags": safety_flags,
        "must_remain_statuses": {
            "JITTOR_FULL_INFERENCE": "NOT_COMPLETE",
            "JITTOR_FULL_PORT": "PARTIAL",
            "TIME_VAE_FULL_ALIGNMENT": "NOT_COMPLETE",
            "TADSR_DYNAMIC_RUNTIME_LORA_IMPLEMENTATION": "NOT_IMPLEMENTED_BY_DESIGN",
        },
        "blockers": blockers,
        "next_required_action": (
            "Run export_tadsr_one_step_tensor_oracle.py in the official Linux environment, then run the Jittor one-step alignment."
            if status == "PASS"
            else "Resolve the listed one-step path blockers before exporting or comparing tensors."
        ),
    }
    write_reports(out_dir, payload)
    print(f"TADSR_ONE_STEP_OFFICIAL_PATH_AUDIT: {status}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
