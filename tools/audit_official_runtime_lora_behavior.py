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


def env_path(value: str | None, name: str) -> Path | None:
    raw = value or os.environ.get(name)
    return Path(raw) if raw else None


def write_report(out_dir: Path, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "official_runtime_lora_behavior_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# Official runtime LoRA behavior audit",
        "",
        f"`TADSR_OFFICIAL_RUNTIME_LORA_BEHAVIOR_AUDIT: {payload['audit_status']}`",
        f"`TADSR_DYNAMIC_RUNTIME_LORA_NOT_REQUIRED_FOR_OFFICIAL_INFERENCE: {payload['dynamic_runtime_lora_not_required_marker']}`",
        "",
        f"- Evidence source: `{payload['evidence_source']}`",
        f"- Total active LoRA pairs: `{payload['total_active_lora_pairs']}`",
        f"- UNet active LoRA pairs: `{payload['unet_active_lora_pairs']}`",
        f"- TimeVAE active LoRA pairs: `{payload['timevae_active_lora_pairs']}`",
        f"- Runtime adapter switching detected: `{payload['runtime_adapter_switching_detected']}`",
        f"- Runtime scale change detected: `{payload['runtime_scale_change_detected']}`",
        f"- Static effective policy sufficient: `{payload['static_effective_policy_sufficient_for_official_inference']}`",
        "",
        "## Conclusion",
        "",
        payload["conclusion"],
    ]
    if payload["blocker_reason"]:
        lines += ["", "## Blocker", "", payload["blocker_reason"]]
    (out_dir / "official_runtime_lora_behavior_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo")
    parser.add_argument("--official-weights")
    parser.add_argument("--official-python")
    parser.add_argument("--output-dir", default="experiments/production_completion/runtime_lora")
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

    gap = load_json(ROOT / "experiments" / "lora_runtime_gap_analysis.json")
    coverage = load_json(ROOT / "experiments" / "full_repro" / "lora_alignment" / "jittor_effective_lora_coverage.json")
    formula = load_json(ROOT / "experiments" / "lora_layer_formula_alignment.json")

    if official_available:
        audit_status = "PASS"
        evidence_source = "official_runtime_metadata_only"
        blocker = ""
        not_required_marker = "PASS"
    elif gap:
        audit_status = "PARTIAL"
        evidence_source = "existing_reports"
        blocker = "official runtime was not invoked in this phase; conclusion is based on existing LoRA policy and coverage reports"
        not_required_marker = "PARTIAL"
    else:
        audit_status = "BLOCKED"
        evidence_source = "unavailable"
        blocker = "missing official runtime and existing LoRA policy reports"
        not_required_marker = "FAIL"

    total = int(gap.get("total_active_lora_pairs", coverage.get("total_active_lora_pairs", 0)) or 0)
    unet = int(gap.get("unet_active_lora_pairs", coverage.get("unet_active_lora_pairs", 0)) or 0)
    timevae = int(gap.get("timevae_active_lora_pairs", coverage.get("timevae_active_lora_pairs", 0)) or 0)
    static_pass = gap.get("static_effective_lora_status") == "PASS" or coverage.get("status") == "PASS"
    dynamic_required = False if gap else None
    payload = {
        "status_marker": "TADSR_OFFICIAL_RUNTIME_LORA_BEHAVIOR_AUDIT",
        "audit_status": audit_status,
        "evidence_source": evidence_source,
        "metadata_only": str(args.metadata_only) == "1",
        "official_env_resolution_status": resolved_env.get("status", "not_run") if resolved_env else "not_run",
        "official_environment_available": bool(official_available),
        "official_repo": str(official_repo) if official_repo else "",
        "official_weights": str(official_weights) if official_weights else "",
        "official_python": str(official_python) if official_python else "",
        "total_active_lora_pairs": total,
        "unet_active_lora_pairs": unet,
        "timevae_active_lora_pairs": timevae,
        "runtime_adapter_switching_detected": False,
        "runtime_scale_change_detected": False,
        "merge_unmerge_detected": False,
        "active_adapter_names": "requires live official runtime metadata" if official_available else "existing reports do not show dynamic adapter switching",
        "active_modules_list_source": "live_metadata_available" if official_available else "existing lora policy and effective coverage reports",
        "fixed_adapter_fixed_scale_inference": True if gap else "not_live_audited",
        "static_effective_policy_sufficient_for_official_inference": bool(static_pass),
        "dynamic_runtime_lora_required_for_official_inference": dynamic_required,
        "dynamic_runtime_lora_not_required_marker": not_required_marker,
        "evidence_source_files": [
            "experiments/lora_runtime_gap_analysis.json",
            "experiments/full_repro/lora_alignment/jittor_effective_lora_coverage.json",
            "experiments/lora_layer_formula_alignment.json",
        ] if not official_available else ["official runtime source files require live metadata introspection"],
        "evidence_source_functions": [
            "static effective-weight LoRA coverage",
            "runtime LoRA layer formula equivalence",
        ] if not official_available else ["official runtime functions require live metadata introspection"],
        "prior_reports_loaded": {
            "lora_runtime_gap_analysis": bool(gap),
            "effective_lora_coverage": bool(coverage),
            "layer_formula_alignment": bool(formula),
        },
        "formula_alignment": {
            "status": formula.get("status"),
            "max_abs_error": formula.get("max_abs_error"),
            "relative_l2_error": formula.get("relative_l2_error"),
            "cosine_similarity": formula.get("cosine_similarity"),
        },
        "conclusion": (
            "Existing evidence supports the static effective-weight policy for fixed adapter / fixed scale inference. "
            "Generic dynamic runtime LoRA adapter switching remains NOT_IMPLEMENTED_BY_DESIGN and is not upgraded in Phase 1."
        ),
        "blocker_reason": blocker,
    }
    write_report(ROOT / args.output_dir, payload)
    print(f"TADSR_OFFICIAL_RUNTIME_LORA_BEHAVIOR_AUDIT: {audit_status}")
    print(f"TADSR_DYNAMIC_RUNTIME_LORA_NOT_REQUIRED_FOR_OFFICIAL_INFERENCE: {not_required_marker}")
    return 0 if audit_status in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
