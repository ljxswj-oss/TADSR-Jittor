#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INFERENCE_DIR = ROOT / "experiments" / "production_completion" / "full_inference"
ACTUAL_AUDIT_JSON = INFERENCE_DIR / "actual_inference_path_audit.json"
OUT_JSON = INFERENCE_DIR / "multistep_applicability_decision.json"
OUT_MD = INFERENCE_DIR / "multistep_applicability_decision.md"


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as exc:
        return {"status": "FAIL", "error": repr(exc)}


def main() -> int:
    audit = read_json(ACTUAL_AUDIT_JSON)
    path_type = str(audit.get("official_actual_path_type", "unknown"))
    multi_required = audit.get("multi_step_required_for_official_actual_inference")

    if path_type == "single_step_get_x0_from_res" and multi_required is False:
        applicability = "NOT_REQUIRED_FOR_OFFICIAL_ACTUAL_INFERENCE"
        decision = "PASS"
        next_action = "Skip tiny multi-step as an official requirement. The next optional phase is a diagnostic image-smoke plan only after explicit approval, while production full inference remains guarded."
        reason = "The official actual inference path uses set_timesteps(1), UNet prediction, get_x0_from_res, and VAE decode/clamp. No denoising timestep loop or scheduler.step call is required by the audited actual path."
    elif multi_required is True or path_type == "multi_step_loop":
        applicability = "REQUIRED_NEXT"
        decision = "PASS"
        next_action = "Plan tiny two-step latent-only alignment. Do not generate image/video and do not open production full inference."
        reason = "The official actual inference path requires a multi-step denoising loop."
    else:
        applicability = "PARTIAL_UNKNOWN"
        decision = "PARTIAL"
        next_action = "Resolve official actual inference path audit before any multi-step experiment."
        reason = "The official actual inference path classification is unknown or incomplete."

    markers = {
        "TADSR_TINY_MULTISTEP_ALIGNMENT_APPLICABILITY": applicability,
        "TADSR_TINY_MULTISTEP_ALIGNMENT_DECISION": decision,
    }
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": decision,
        "official_actual_path_type": path_type,
        "multi_step_required_for_official_actual_inference": multi_required,
        "tiny_multi_step_alignment_applicability": applicability,
        "tiny_multi_step_alignment_executed": False,
        "tiny_multi_step_alignment_claimed_pass": False,
        "next_action": next_action,
        "reason": reason,
        "markers": markers,
        "must_remain_statuses": {
            "JITTOR_FULL_INFERENCE": "NOT_COMPLETE",
            "JITTOR_FULL_PORT": "PARTIAL",
            "TIME_VAE_FULL_ALIGNMENT": "NOT_COMPLETE",
            "TADSR_DYNAMIC_RUNTIME_LORA_IMPLEMENTATION": "NOT_IMPLEMENTED_BY_DESIGN",
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Multi-step alignment applicability decision",
        "",
        f"`TADSR_TINY_MULTISTEP_ALIGNMENT_APPLICABILITY: {applicability}`",
        f"`TADSR_TINY_MULTISTEP_ALIGNMENT_DECISION: {decision}`",
        "",
        f"- official_actual_path_type: `{path_type}`",
        f"- multi_step_required_for_official_actual_inference: `{multi_required}`",
        f"- tiny_multi_step_alignment_executed: `false`",
        "",
        "## Reason",
        "",
        reason,
        "",
        "## Next action",
        "",
        next_action,
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for marker, value in markers.items():
        print(f"{marker}: {value}")
    return 0 if decision in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
