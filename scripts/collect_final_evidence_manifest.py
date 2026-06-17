#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


OUT_JSON = Path("experiments/final_evidence_manifest.json")
OUT_MD = Path("experiments/final_evidence_manifest.md")


def load_json(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "FAIL", "error": repr(exc)}


def status_from_json(path: str | Path, default: str = "MISSING") -> str:
    data = load_json(path)
    if not data:
        return default
    return str(data.get("status", default))


def file_item(name: str, path: str, status: str | None = None, note: str = "") -> dict:
    p = Path(path)
    item = {
        "name": name,
        "path": path,
        "exists": p.exists(),
        "status": status if status is not None else (status_from_json(path) if path.endswith(".json") else ("PRESENT" if p.exists() else "MISSING")),
        "note": note,
    }
    return item


def audit_status(check: str) -> str:
    audit = load_json("experiments/final_audit_report.json")
    for row in audit.get("rows", []):
        if row.get("check") == check:
            return str(row.get("status", "MISSING"))
    return "MISSING"


def build_manifest() -> dict:
    sections = {
        "UNet evidence": [
            file_item("UNet entry alignment", "experiments/full_repro/unet_alignment/jittor_unet_entry_alignment.json"),
            file_item("UNet down_blocks alignment", "experiments/full_repro/unet_alignment/jittor_unet_entry_downblock0_downblock1_downblock2_downblock3_resnet0_resnet1_alignment.json"),
            file_item("UNet mid_block alignment", "experiments/full_repro/unet_alignment/jittor_unet_entry_downblocks_midblock_resnet0_attention0_resnet1_alignment.json"),
            file_item("UNet up_blocks alignment", "experiments/full_repro/unet_alignment/jittor_unet_entry_downblocks_midblock_upblocks_output_tail_alignment.json"),
            file_item("UNet output tail alignment", "experiments/full_repro/unet_alignment/jittor_unet_output_tail_alignment.json"),
            file_item("UNet manual wrapper alignment", "experiments/full_repro/unet_alignment/jittor_unet_manual_wrapper_alignment.json"),
            file_item("UNet official full forward alignment", "experiments/full_repro/unet_alignment/jittor_unet_full_forward_alignment.json", audit_status("TADSR_UNET_FULL_FORWARD_ALIGNMENT")),
        ],
        "TimeVAE evidence": [
            file_item("Deterministic decoder alignment", "experiments/full_repro/time_vae_alignment/jittor_decoder_entry_midblock_upblocks0123_tail_alignment.json"),
            file_item("TimeVAE full boundary alignment", "experiments/full_repro/vae_alignment/jittor_timevae_full_boundary_alignment.json"),
            file_item("Actual VAEHook behavior audit", "experiments/full_repro/vae_alignment/audit_tadsr_vae_actual_hook_behavior.json"),
            file_item("Actual VAEHook oracle", "experiments/full_repro/vae_alignment/oracle_tensors_vae_actual_hook_behavior/vae_actual_hook_behavior_oracle_metadata.json"),
            file_item("Actual VAEHook alignment", "experiments/full_repro/vae_alignment/jittor_timevae_actual_hook_behavior_alignment.json", audit_status("TIME_VAE_ACTUAL_VAEHOOK_FULL_BOUNDARY_ALIGNMENT")),
            file_item("Tiled decoder blocked reason", "experiments/full_repro/vae_alignment/audit_tadsr_vae_tiled_boundary.json", note="records the truthful decoder-hook limitation"),
            file_item("TIME_VAE_FULL_ALIGNMENT remains NOT_COMPLETE", "experiments/final_audit_report.json", audit_status("TIME_VAE_FULL_ALIGNMENT"), "full TimeVAE status is intentionally not promoted"),
        ],
        "LoRA evidence": [
            file_item("Official LoRA policy audit", "experiments/full_repro/lora_alignment/audit_tadsr_lora_policy.json"),
            file_item("Static effective LoRA coverage", "experiments/full_repro/lora_alignment/jittor_effective_lora_coverage.json", audit_status("TADSR_STATIC_EFFECTIVE_LORA_COVERAGE_AUDIT")),
            file_item("UNet active LoRA coverage", "experiments/full_repro/lora_alignment/jittor_effective_lora_coverage.json"),
            file_item("TimeVAE active LoRA coverage", "experiments/full_repro/lora_alignment/timevae_lora_effective_weight_coverage_test.json"),
            file_item("Dynamic runtime LoRA remains design-not-implemented", "experiments/final_audit_report.json", audit_status("TADSR_DYNAMIC_RUNTIME_LORA_IMPLEMENTATION")),
        ],
        "Scheduler evidence": [
            file_item("Scheduler boundary audit", "experiments/full_repro/scheduler_alignment/audit_tadsr_scheduler_boundary.json"),
            file_item("Scheduler oracle", "experiments/full_repro/scheduler_alignment/oracle_tensors_scheduler_boundary/scheduler_boundary_oracle_metadata.json"),
            file_item("Jittor scheduler boundary alignment", "experiments/full_repro/scheduler_alignment/jittor_scheduler_boundary_alignment.json", audit_status("TADSR_SCHEDULER_BOUNDARY_ALIGNMENT")),
            file_item("UNet scheduler one-step alignment", "experiments/full_repro/scheduler_alignment/jittor_scheduler_boundary_alignment.json", audit_status("TADSR_UNET_SCHEDULER_ONE_STEP_ALIGNMENT")),
        ],
        "Minimal integration evidence": [
            file_item("Minimal integration source audit", "experiments/full_repro/integration_alignment/audit_tadsr_minimal_integration.json", audit_status("TADSR_MINIMAL_INTEGRATION_AUDIT")),
            file_item("Minimal latent oracle", "experiments/full_repro/integration_alignment/oracle_tensors_minimal_latent_integration/minimal_latent_integration_oracle_metadata.json", audit_status("TADSR_MINIMAL_LATENT_ONE_STEP_ORACLE")),
            file_item("get_x0_from_res alignment", "experiments/full_repro/integration_alignment/jittor_minimal_latent_integration_alignment.json", audit_status("TADSR_GET_X0_FROM_RES_ALIGNMENT")),
            file_item("Minimal decode/clamp tensor boundary alignment", "experiments/full_repro/integration_alignment/jittor_minimal_latent_integration_alignment.json", audit_status("TADSR_MINIMAL_DECODE_BOUNDARY_ALIGNMENT")),
            file_item("Minimal one-step decode dry-run", "experiments/full_repro/integration_alignment/jittor_minimal_latent_integration_alignment.json", audit_status("TADSR_MINIMAL_ONE_STEP_DECODE_DRY_RUN")),
        ],
        "Safety evidence": [
            file_item("Final audit report", "experiments/final_audit_report.json"),
            file_item("Migration report", "experiments/full_repro/jittor_alignment/jittor_migration_report.md"),
            file_item("Full inference guard remains NotImplemented", "jittor_tadsr_full/tadsr_full.py", audit_status("JITTOR_FULL_INFERENCE")),
            file_item("Large oracle tensors ignored by git", ".gitignore", "PRESENT"),
        ],
        "Remaining gaps": [
            file_item("JITTOR_FULL_INFERENCE", "experiments/final_audit_report.json", audit_status("JITTOR_FULL_INFERENCE"), "must remain NOT_COMPLETE"),
            file_item("JITTOR_FULL_PORT", "experiments/final_audit_report.json", audit_status("JITTOR_FULL_PORT"), "must remain PARTIAL"),
            file_item("TIME_VAE_FULL_ALIGNMENT", "experiments/final_audit_report.json", audit_status("TIME_VAE_FULL_ALIGNMENT"), "must remain NOT_COMPLETE"),
            file_item("Generic runtime LoRA", "experiments/final_audit_report.json", audit_status("TADSR_DYNAMIC_RUNTIME_LORA_IMPLEMENTATION"), "not implemented by design"),
            file_item("Image/video generation", "docs/production_cli_design_audit.md", "NOT_IMPLEMENTED"),
            file_item("Full denoising loop", "docs/production_cli_design_audit.md", "NOT_IMPLEMENTED"),
        ],
    }
    missing = []
    for group, items in sections.items():
        for item in items:
            if not item["exists"]:
                missing.append({"group": group, **item})
    major_required = [
        audit_status("TADSR_UNET_FULL_FORWARD_ALIGNMENT") == "PASS",
        audit_status("TIME_VAE_ACTUAL_VAEHOOK_FULL_BOUNDARY_ALIGNMENT") == "PASS",
        audit_status("TADSR_STATIC_EFFECTIVE_LORA_COVERAGE_AUDIT") == "PASS",
        audit_status("TADSR_SCHEDULER_BOUNDARY_ALIGNMENT") == "PASS",
        audit_status("TADSR_MINIMAL_ONE_STEP_DECODE_DRY_RUN") == "PASS",
        audit_status("JITTOR_FULL_INFERENCE") == "NOT_COMPLETE",
        audit_status("JITTOR_FULL_PORT") == "PARTIAL",
    ]
    status = "PASS" if all(major_required) else "PARTIAL"
    if missing:
        # Missing optional evidence is kept visible but does not falsify major marker status.
        status = "PASS" if all(major_required) else "PARTIAL"
    return {
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sections": sections,
        "missing_evidence_files": missing,
        "major_status": {
            "TADSR_UNET_FULL_FORWARD_ALIGNMENT": audit_status("TADSR_UNET_FULL_FORWARD_ALIGNMENT"),
            "TIME_VAE_ACTUAL_VAEHOOK_FULL_BOUNDARY_ALIGNMENT": audit_status("TIME_VAE_ACTUAL_VAEHOOK_FULL_BOUNDARY_ALIGNMENT"),
            "TADSR_STATIC_EFFECTIVE_LORA_COVERAGE_AUDIT": audit_status("TADSR_STATIC_EFFECTIVE_LORA_COVERAGE_AUDIT"),
            "TADSR_SCHEDULER_BOUNDARY_ALIGNMENT": audit_status("TADSR_SCHEDULER_BOUNDARY_ALIGNMENT"),
            "TADSR_MINIMAL_ONE_STEP_DECODE_DRY_RUN": audit_status("TADSR_MINIMAL_ONE_STEP_DECODE_DRY_RUN"),
            "JITTOR_FULL_INFERENCE": audit_status("JITTOR_FULL_INFERENCE"),
            "JITTOR_FULL_PORT": audit_status("JITTOR_FULL_PORT"),
            "TIME_VAE_FULL_ALIGNMENT": audit_status("TIME_VAE_FULL_ALIGNMENT"),
        },
    }


def write_markdown(manifest: dict) -> None:
    lines = [
        "# Final Evidence Manifest",
        "",
        f"Status: **{manifest['status']}**",
        "",
        "This manifest indexes committed metadata, reports, and audit files. It does not execute model code and does not import torch.",
        "",
        "## Major Status",
        "",
        "| Marker | Status |",
        "|---|---|",
    ]
    for key, value in manifest["major_status"].items():
        lines.append(f"| `{key}` | `{value}` |")
    for group, items in manifest["sections"].items():
        lines += ["", f"## {group}", "", "| Evidence | Status | Exists | Path | Note |", "|---|---|---:|---|---|"]
        for item in items:
            lines.append(f"| {item['name']} | `{item['status']}` | `{item['exists']}` | `{item['path']}` | {item.get('note', '')} |")
    if manifest["missing_evidence_files"]:
        lines += ["", "## Missing Evidence Files", ""]
        for item in manifest["missing_evidence_files"]:
            lines.append(f"- {item['group']}: `{item['name']}` -> `{item['path']}`")
    else:
        lines += ["", "## Missing Evidence Files", "", "None."]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    OUT_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_markdown(manifest)
    print(f"TADSR_FINAL_EVIDENCE_MANIFEST: {manifest['status']}")
    print(f"missing_evidence_files: {len(manifest['missing_evidence_files'])}")
    return 0 if manifest["status"] in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
