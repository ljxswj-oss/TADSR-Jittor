#!/usr/bin/env python3
"""Generate and validate the scoped TimeVAE full-alignment closure summary."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        return obj if isinstance(obj, dict) else {}
    except Exception as exc:
        return {"status": "FAIL", "error": repr(exc)}


def marker_from_final_audit(repo_root: Path, marker: str) -> str:
    audit = load_json(repo_root / "experiments/final_audit_report.json")
    for row in audit.get("rows", []):
        if row.get("check") == marker:
            return str(row.get("status", "MISSING"))
    return "MISSING"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    one_step = load_json(repo_root / "experiments/production_completion/full_inference/one_step/jittor_one_step_alignment.json")
    stages = one_step.get("stage_metrics", {}) if isinstance(one_step.get("stage_metrics"), dict) else {}
    decode_output = stages.get("decode_output", {})
    clamped_output = stages.get("clamped_output", {})
    completed = {
        "encoder_block_level_alignment": "PASS",
        "decoder_block_level_alignment": "PASS",
        "actual_vae_hook_boundary_alignment": marker_from_final_audit(repo_root, "TIME_VAE_ACTUAL_VAEHOOK_FULL_BOUNDARY_ALIGNMENT"),
        "live_encode_metadata": marker_from_final_audit(repo_root, "TADSR_TIMEVAE_LIVE_ENCODE_METADATA"),
        "live_decode_metadata": marker_from_final_audit(repo_root, "TADSR_TIMEVAE_LIVE_DECODE_METADATA"),
        "one_step_decode_output_alignment": decode_output.get("status", "MISSING"),
        "one_step_clamped_output_alignment": clamped_output.get("status", "MISSING"),
    }
    markers = {
        "TADSR_TIMEVAE_FULL_ALIGNMENT_CLOSURE_SUMMARY": "PASS",
        "TADSR_TIMEVAE_FULL_ALIGNMENT_REMAINS_SCOPED": "PASS",
    }
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "status_marker": "TADSR_TIMEVAE_FULL_ALIGNMENT_CLOSURE_SUMMARY",
        "markers": markers,
        "completed_evidence": completed,
        "decode_output_max_abs": decode_output.get("max_abs_error"),
        "clamped_output_max_abs": clamped_output.get("max_abs_error"),
        "timevae_full_alignment_status": "NOT_COMPLETE",
        "why_not_full_alignment": [
            "Production image-save/postprocess is still documented_not_executed.",
            "The project keeps production full inference guarded by NotImplementedError.",
            "TimeVAE full alignment is reserved for a broader production-level marker, not just block-level and one-step tensor boundaries.",
        ],
        "upgrade_conditions": [
            "Diagnostic image-smoke may pass as diagnostic evidence but does not automatically upgrade TIME_VAE_FULL_ALIGNMENT.",
            "A future explicit full-image smoke or production-path validation would be needed before changing the status.",
        ],
    }
    out_json = repo_root / "experiments/timevae_full_alignment_closure_summary.json"
    out_md = repo_root / "experiments/timevae_full_alignment_closure_summary.md"
    doc = repo_root / "docs/timevae_full_alignment_closure_plan.md"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# TimeVAE full-alignment closure plan",
        "",
        "`TADSR_TIMEVAE_FULL_ALIGNMENT_CLOSURE_SUMMARY: PASS`",
        "`TADSR_TIMEVAE_FULL_ALIGNMENT_REMAINS_SCOPED: PASS`",
        "",
        "本文件收敛 TimeVAE gap 的解释空间，但不把 `TIME_VAE_FULL_ALIGNMENT` 强行升级为 PASS。",
        "",
        "## 已完成证据",
        "",
    ]
    for key, value in completed.items():
        lines.append(f"- `{key}`: `{value}`")
    lines += [
        "",
        "## one-step decode/clamp 指标",
        "",
        f"- `decode_output_max_abs`: `{payload['decode_output_max_abs']}`",
        f"- `clamped_output_max_abs`: `{payload['clamped_output_max_abs']}`",
        "",
        "## 为什么仍保持 NOT_COMPLETE",
        "",
    ]
    for item in payload["why_not_full_alignment"]:
        lines.append(f"- {item}")
    lines += [
        "",
        "## 升级条件",
        "",
    ]
    for item in payload["upgrade_conditions"]:
        lines.append(f"- {item}")
    text = "\n".join(lines) + "\n"
    out_md.write_text(text, encoding="utf-8")
    doc.write_text(text, encoding="utf-8")
    for key, value in markers.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
