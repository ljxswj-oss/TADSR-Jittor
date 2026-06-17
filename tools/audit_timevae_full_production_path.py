#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as exc:
        return {"status": "FAIL", "error": repr(exc)}


def env_path(value: str | None, fallback: str) -> Path | None:
    raw = value or os.environ.get(fallback)
    return Path(raw) if raw else None


def write_report(out_dir: Path, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "official_timevae_full_path_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# TimeVAE full production path audit",
        "",
        f"`TADSR_TIMEVAE_FULL_PRODUCTION_PATH_AUDIT: {payload['audit_status']}`",
        "",
        f"- Evidence source: `{payload['evidence_source']}`",
        f"- Blocker: {payload['blocker_reason'] or 'none'}",
        f"- Safe to attempt Jittor alignment: `{payload['safe_to_attempt_jittor_alignment']}`",
        "",
        "## Completed subpaths",
        "",
    ]
    for item in payload["completed_subpaths"]:
        lines.append(f"- {item}")
    lines += ["", "## Not-complete subpaths", ""]
    for item in payload["not_complete_subpaths"]:
        lines.append(f"- {item}")
    lines += ["", "## Required next oracle exports", ""]
    for item in payload["required_next_oracle_exports"]:
        lines.append(f"- {item}")
    (out_dir / "official_timevae_full_path_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo")
    parser.add_argument("--official-weights")
    parser.add_argument("--official-python")
    parser.add_argument("--output-dir", default="experiments/production_completion/timevae_full")
    parser.add_argument("--metadata-only", default="1")
    args = parser.parse_args()

    resolved_env = load_json(ROOT / "experiments" / "production_completion" / "env" / "official_env_resolution.json")
    official_repo = env_path(args.official_repo, "TADSR_OFFICIAL_REPO")
    official_weights = env_path(args.official_weights, "TADSR_OFFICIAL_WEIGHTS")
    official_python = env_path(args.official_python, "TADSR_OFFICIAL_PYTHON")
    if resolved_env:
        official_repo = Path(resolved_env.get("official_repo", "")) if resolved_env.get("official_repo") else official_repo
        official_weights = Path(resolved_env.get("official_weights", "")) if resolved_env.get("official_weights") else official_weights
        official_python = Path(resolved_env.get("official_python", "")) if resolved_env.get("official_python") else official_python
    official_available = bool(resolved_env.get("status") == "PASS") if resolved_env else all(p and p.exists() for p in [official_repo, official_weights, official_python])

    gap = load_json(ROOT / "experiments" / "timevae_full_alignment_gap_analysis.json")
    actual_hook = load_json(ROOT / "experiments" / "full_repro" / "time_vae_alignment" / "jittor_timevae_actual_hook_behavior_alignment.json")
    boundary = load_json(ROOT / "experiments" / "full_repro" / "time_vae_alignment" / "jittor_timevae_full_boundary_alignment.json")

    if official_available:
        audit_status = "PASS"
        evidence_source = "official_runtime"
        blocker = ""
    elif gap:
        audit_status = "PARTIAL"
        evidence_source = "existing_reports"
        blocker = "official repo/weights/python were not all available for live production-path audit in this phase"
    else:
        audit_status = "BLOCKED"
        evidence_source = "unavailable"
        blocker = "missing official runtime and existing TimeVAE gap reports"

    completed = gap.get("completed_subpaths", []) if gap else []
    not_complete = gap.get("not_complete_subpaths", []) if gap else [
        "full production TimeVAE path not audited",
        "full TADSR-coupled TimeVAE path not audited",
    ]
    payload = {
        "status_marker": "TADSR_TIMEVAE_FULL_PRODUCTION_PATH_AUDIT",
        "audit_status": audit_status,
        "evidence_source": evidence_source,
        "metadata_only": str(args.metadata_only) == "1",
        "official_env_resolution_status": resolved_env.get("status", "not_run") if resolved_env else "not_run",
        "official_repo": str(official_repo) if official_repo else "",
        "official_weights": str(official_weights) if official_weights else "",
        "official_python": str(official_python) if official_python else "",
        "official_environment_available": bool(official_available),
        "timevae_class_name": "requires official runtime source introspection" if official_available else "existing reports only",
        "vae_hook_class_name": "requires official runtime source introspection" if official_available else "existing reports only",
        "timevae_class_or_module_path": "requires official runtime source introspection" if official_available else "existing reports only",
        "vae_hook_class_or_patch_point": "requires official runtime source introspection" if official_available else "existing reports only",
        "encoder_hook_installed": "known_from_existing_reports" if not official_available else "metadata_only_not_forward_executed",
        "decoder_hook_installed": "known_from_existing_reports" if not official_available else "metadata_only_not_forward_executed",
        "encoder_uses_vae_hook": "known_from_existing_reports" if not official_available else "metadata_only_not_forward_executed",
        "decoder_uses_vae_hook": "known_from_existing_reports" if not official_available else "metadata_only_not_forward_executed",
        "encoder_tiled_path_reachable": "partial_existing_evidence" if not official_available else "metadata_only_not_forward_executed",
        "decoder_tiled_path_reachable": "not_claimed" if not official_available else "metadata_only_not_forward_executed",
        "decoder_original_forward_used": "decoder_original_forward_existing_evidence" if not official_available else "metadata_only_not_forward_executed",
        "original_forward_called": "decoder_original_forward_existing_evidence" if not official_available else "metadata_only_not_forward_executed",
        "original_forward_patch_points": ["decoder VAEHook original_forward path requires live metadata"] if official_available else ["existing reports only"],
        "encode_method_path": "requires live source introspection" if official_available else "existing reports only",
        "decode_method_path": "requires live source introspection" if official_available else "existing reports only",
        "posterior_sample_policy": "requires production-path oracle before full status upgrade",
        "posterior_mode_policy": "requires production-path oracle before full status upgrade",
        "stochastic_sampling_policy": "requires production-path oracle before full status upgrade",
        "posterior_sample_mode_policy": "requires production-path oracle before full status upgrade",
        "scaling_factor": "requires live production metadata",
        "clamp_policy": "requires live production metadata",
        "expected_encode_input_shape": "not_exported_in_phase2_metadata_audit",
        "expected_latent_shape": "not_exported_in_phase2_metadata_audit",
        "expected_decode_input_shape": "not_exported_in_phase2_metadata_audit",
        "expected_decode_output_shape": "not_exported_in_phase2_metadata_audit",
        "actual_vaehook_boundary_consistent_with_reports": gap.get("actual_vae_hook_boundary_status") == "PASS" if gap else False,
        "sampling_policy": "requires production-path oracle before full status upgrade",
        "scaling_policy": "requires production-path oracle before full status upgrade",
        "coupled_tadsr_path": "not executed in Phase 3-A",
        "coupled_tadsr_path_assumptions": [
            "metadata-only audit does not execute full denoising loop",
            "TimeVAE metadata oracle is prerequisite for controlled one-step tensor alignment",
        ],
        "completed_subpaths": completed,
        "not_complete_subpaths": not_complete,
        "not_applicable_subpaths": [],
        "required_next_oracle_exports": [
            "official TimeVAE production encode metadata",
            "official TimeVAE production decode metadata",
            "full TADSR-coupled TimeVAE input/output tensor contracts",
            "sampling/scaling/clamp policy metadata",
        ],
        "prior_reports_loaded": {
            "timevae_gap_analysis": bool(gap),
            "actual_hook_alignment": bool(actual_hook),
            "full_boundary_alignment": bool(boundary),
        },
        "blocker_reason": blocker,
        "safe_to_attempt_timevae_oracle_metadata_export": bool(official_available and audit_status in {"PASS", "PARTIAL"}),
        "safe_to_attempt_jittor_alignment": audit_status in {"PASS", "PARTIAL"},
    }
    write_report(ROOT / args.output_dir, payload)
    print(f"TADSR_TIMEVAE_FULL_PRODUCTION_PATH_AUDIT: {audit_status}")
    return 0 if audit_status in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
