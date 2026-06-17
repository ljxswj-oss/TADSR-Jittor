from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ONE_STEP_ALIGNMENT = ROOT / "experiments/production_completion/full_inference/one_step/jittor_one_step_alignment.json"
ONE_STEP_VALIDATION = ROOT / "experiments/production_completion/full_inference/one_step/one_step_tensor_alignment_validation.json"
DIAGNOSTIC_VALIDATION = ROOT / "experiments/production_completion/diagnostic_image_smoke/diagnostic_image_smoke_validation.json"
DIAGNOSTIC_METRICS = ROOT / "experiments/production_completion/diagnostic_image_smoke/diagnostic_image_smoke_metrics.json"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        return obj if isinstance(obj, dict) else {}
    except Exception as exc:
        return {"status": "FAIL", "error": repr(exc)}


def one_step_summary() -> dict[str, Any]:
    alignment = load_json(ONE_STEP_ALIGNMENT)
    validation = load_json(ONE_STEP_VALIDATION)
    stages = alignment.get("stage_metrics", {}) if isinstance(alignment.get("stage_metrics"), dict) else {}
    compact = {}
    for name in ["encoded_latent", "unet_model_pred", "x0_from_res", "decode_output", "clamped_output"]:
        row = stages.get(name, {})
        compact[name] = {
            "status": row.get("status"),
            "shape_match": row.get("shape_match"),
            "max_abs_error": row.get("max_abs_error"),
            "mean_abs_error": row.get("mean_abs_error"),
            "relative_l2": row.get("relative_l2"),
            "cosine_similarity": row.get("cosine_similarity"),
        }
    return {
        "status_marker": "TADSR_ONE_STEP_DIAGNOSTIC_CLI_READY",
        "status": "PASS" if validation else "PARTIAL",
        "mode": "metadata_only",
        "full_inference_executed": False,
        "denoising_loop_executed": False,
        "image_or_video_generated": False,
        "one_step_validation_markers": validation.get("markers", {}),
        "stage_metrics": compact,
        "note": "This is a diagnostic one-step summary only, not full production TADSR inference.",
    }


def diagnostic_smoke_summary() -> dict[str, Any]:
    metrics = load_json(DIAGNOSTIC_METRICS)
    validation = load_json(DIAGNOSTIC_VALIDATION)
    return {
        "status_marker": "TADSR_ONE_STEP_DIAGNOSTIC_CLI_READY",
        "status": "PASS" if validation else "PARTIAL",
        "mode": "metadata_only",
        "full_inference_executed": False,
        "denoising_loop_executed": False,
        "image_or_video_generated": False,
        "diagnostic_smoke_status": metrics.get("status", "MISSING"),
        "diagnostic_smoke_markers": metrics.get("markers", {}),
        "diagnostic_smoke_validation_markers": validation.get("markers", {}),
        "blocker": metrics.get("blocker"),
        "note": "Diagnostic image-smoke evidence is not a production restoration result.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Metadata-only TADSR one-step diagnostic CLI.")
    parser.add_argument("--one-step-summary", action="store_true")
    parser.add_argument("--check-diagnostic-image-smoke", action="store_true")
    parser.add_argument("--full-inference", action="store_true")
    args = parser.parse_args()

    if args.full_inference:
        raise NotImplementedError("Diagnostic CLI does not provide production full inference.")
    if args.one_step_summary:
        print(json.dumps(one_step_summary(), indent=2, ensure_ascii=False))
        return 0
    if args.check_diagnostic_image_smoke:
        print(json.dumps(diagnostic_smoke_summary(), indent=2, ensure_ascii=False))
        return 0
    print(json.dumps({"status": "NO_OP", "hint": "Use --one-step-summary or --check-diagnostic-image-smoke."}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
