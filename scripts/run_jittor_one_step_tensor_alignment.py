#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "experiments" / "production_completion" / "full_inference" / "one_step"
REQUIRED_LOCAL_TENSORS = [
    "input_tensor",
    "scheduler_timestep",
    "encoder_hidden_states",
    "sample_epsilon",
    "encoded_latent",
    "scaled_latent",
    "unet_model_pred",
    "alpha_prod_t",
    "sqrt_alpha_prod_t",
    "x0_from_res",
    "decode_input",
    "decode_output",
    "clamped_output",
]


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as exc:
        return {"status": "FAIL", "error": repr(exc)}


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = np.linalg.norm(aa) * np.linalg.norm(bb)
    if denom == 0:
        return 1.0 if np.linalg.norm(aa - bb) == 0 else None
    return float(np.dot(aa, bb) / denom)


def metric(got: np.ndarray, expected: np.ndarray, tolerance: float) -> dict:
    g = np.asarray(got)
    e = np.asarray(expected)
    shape_match = tuple(g.shape) == tuple(e.shape)
    dtype_match = str(g.dtype) == str(e.dtype)
    if not shape_match:
        return {
            "status": "FAIL",
            "shape_match": False,
            "dtype_match": dtype_match,
            "got_shape": list(g.shape),
            "expected_shape": list(e.shape),
            "got_dtype": str(g.dtype),
            "expected_dtype": str(e.dtype),
            "tolerance": tolerance,
        }
    diff = g.astype(np.float64) - e.astype(np.float64)
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    mean_abs = float(np.mean(np.abs(diff))) if diff.size else 0.0
    denom = float(np.linalg.norm(e.astype(np.float64).reshape(-1)))
    rel_l2 = float(np.linalg.norm(diff.reshape(-1)) / denom) if denom else (0.0 if max_abs == 0 else float("inf"))
    status = "PASS" if max_abs <= tolerance else ("PARTIAL" if max_abs <= max(1e-3, tolerance * 10.0) else "FAIL")
    return {
        "status": status,
        "shape_match": True,
        "dtype_match": dtype_match,
        "got_shape": list(g.shape),
        "expected_shape": list(e.shape),
        "got_dtype": str(g.dtype),
        "expected_dtype": str(e.dtype),
        "got_min": float(g.min()) if g.size else None,
        "got_max": float(g.max()) if g.size else None,
        "expected_min": float(e.min()) if e.size else None,
        "expected_max": float(e.max()) if e.size else None,
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "relative_l2": rel_l2,
        "cosine_similarity": cosine(g, e),
        "finite": bool(np.isfinite(g).all() and np.isfinite(e).all()),
        "tolerance": tolerance,
    }


def write_outputs(out_dir: Path, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "jittor_one_step_alignment.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = out_dir / "jittor_one_step_stage_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stage",
                "status",
                "shape_match",
                "dtype_match",
                "got_shape",
                "expected_shape",
                "got_dtype",
                "expected_dtype",
                "max_abs_error",
                "mean_abs_error",
                "relative_l2",
                "cosine_similarity",
                "tolerance",
            ],
        )
        writer.writeheader()
        for stage, m in payload.get("stage_metrics", {}).items():
            writer.writerow({
                "stage": stage,
                "status": m.get("status"),
                "shape_match": m.get("shape_match"),
                "dtype_match": m.get("dtype_match"),
                "got_shape": json.dumps(m.get("got_shape")),
                "expected_shape": json.dumps(m.get("expected_shape")),
                "got_dtype": m.get("got_dtype"),
                "expected_dtype": m.get("expected_dtype"),
                "max_abs_error": m.get("max_abs_error"),
                "mean_abs_error": m.get("mean_abs_error"),
                "relative_l2": m.get("relative_l2"),
                "cosine_similarity": m.get("cosine_similarity"),
                "tolerance": m.get("tolerance"),
            })
    lines = [
        "# Jittor one-step tensor alignment",
        "",
        f"`TADSR_ONE_STEP_JITTOR_TENSOR_RUN: {payload['markers']['TADSR_ONE_STEP_JITTOR_TENSOR_RUN']}`",
        f"`TADSR_ONE_STEP_TENSOR_STAGE_ALIGNMENT: {payload['markers']['TADSR_ONE_STEP_TENSOR_STAGE_ALIGNMENT']}`",
        f"`TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT: {payload['markers']['TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT']}`",
        "",
        "This is a controlled one-step tensor comparison only. It does not run a full denoising loop, production full inference, image generation, video generation, or dynamic runtime LoRA.",
        "",
    ]
    if payload.get("blockers"):
        lines += ["## Blockers", ""]
        for item in payload["blockers"]:
            lines.append(f"- {item}")
        lines.append("")
    lines += [
        "## Stage metrics",
        "",
        "| Stage | Status | Shape match | Max abs | Mean abs | Relative L2 | Cosine |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for stage, m in payload.get("stage_metrics", {}).items():
        lines.append(
            f"| `{stage}` | `{m.get('status')}` | `{m.get('shape_match')}` | `{m.get('max_abs_error')}` | "
            f"`{m.get('mean_abs_error')}` | `{m.get('relative_l2')}` | `{m.get('cosine_similarity')}` |"
        )
    (out_dir / "jittor_one_step_alignment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def blocked_payload(reason: str, oracle_meta: dict, out_dir: Path, status: str = "PARTIAL") -> dict:
    markers = {
        "TADSR_ONE_STEP_JITTOR_TENSOR_RUN": status,
        "TADSR_ONE_STEP_TENSOR_STAGE_ALIGNMENT": status,
        "TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT": status,
    }
    return {
        "status_marker": "TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT",
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "markers": markers,
        "oracle_status": oracle_meta.get("status", "MISSING") if oracle_meta else "MISSING",
        "stage_metrics": {},
        "blockers": [reason],
        "policy": {
            "full_inference_executed": False,
            "denoising_loop_executed": False,
            "production_cli_used": False,
            "image_or_video_generated": False,
            "runtime_lora_dynamic_loading": False,
            "raw_tensor_committed": False,
        },
        "output_dir": str(out_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Jittor controlled one-step tensor alignment against official oracle tensors.")
    parser.add_argument("--oracle-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--tolerance", type=float, default=2e-3)
    parser.add_argument(
        "--save-diagnostic-tensors",
        action="store_true",
        help=(
            "Save Jittor got tensors beside ignored official local_tensors for "
            "diagnostic image-smoke generation. These .npy files are raw evidence "
            "and must remain untracked."
        ),
    )
    args = parser.parse_args()

    oracle_dir = Path(args.oracle_dir)
    if not oracle_dir.is_absolute():
        oracle_dir = ROOT / oracle_dir
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    oracle_meta = load_json(oracle_dir / "official_one_step_oracle_metadata.json")
    if not oracle_meta:
        payload = blocked_payload("official_one_step_oracle_metadata.json is missing", oracle_meta, out_dir)
        write_outputs(out_dir, payload)
        print("TADSR_ONE_STEP_JITTOR_TENSOR_RUN: PARTIAL")
        print("TADSR_ONE_STEP_TENSOR_STAGE_ALIGNMENT: PARTIAL")
        print("TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT: PARTIAL")
        return 1
    if oracle_meta.get("status") != "PASS":
        payload = blocked_payload("official one-step oracle did not PASS", oracle_meta, out_dir)
        write_outputs(out_dir, payload)
        print("TADSR_ONE_STEP_JITTOR_TENSOR_RUN: PARTIAL")
        print("TADSR_ONE_STEP_TENSOR_STAGE_ALIGNMENT: PARTIAL")
        print("TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT: PARTIAL")
        return 1

    tensor_dir = oracle_dir / str(oracle_meta.get("local_tensor_dir") or "local_tensors")
    missing = [name for name in REQUIRED_LOCAL_TENSORS if not (tensor_dir / f"{name}.npy").exists()]
    if missing:
        payload = blocked_payload(f"official local tensor files are missing: {missing}", oracle_meta, out_dir)
        payload["available_metadata_stage_count"] = len(oracle_meta.get("tensor_stats", {}))
        payload["missing_local_tensor_names"] = missing
        write_outputs(out_dir, payload)
        print("TADSR_ONE_STEP_JITTOR_TENSOR_RUN: PARTIAL")
        print("TADSR_ONE_STEP_TENSOR_STAGE_ALIGNMENT: PARTIAL")
        print("TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT: PARTIAL")
        return 1

    arrays = {name: np.load(tensor_dir / f"{name}.npy") for name in REQUIRED_LOCAL_TENSORS}
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "tests_jittor_alignment"))
    try:
        from minimal_integration_common import minimal_tester
    except Exception as exc:
        payload = blocked_payload(f"failed to import Jittor minimal integration tester: {exc!r}", oracle_meta, out_dir, status="BLOCKED")
        write_outputs(out_dir, payload)
        print("TADSR_ONE_STEP_JITTOR_TENSOR_RUN: BLOCKED")
        print("TADSR_ONE_STEP_TENSOR_STAGE_ALIGNMENT: BLOCKED")
        print("TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT: BLOCKED")
        return 1

    try:
        tester = minimal_tester(float(oracle_meta.get("scaling_factor", 0.18215)))
        got = tester.run_minimal_latent_one_step_for_alignment(
            arrays["input_tensor"].astype(np.float32),
            arrays["scheduler_timestep"].astype(np.int64),
            arrays["encoder_hidden_states"].astype(np.float32),
            arrays["sample_epsilon"].astype(np.float32),
            return_intermediates=True,
            decode_boundary=True,
        )
        stage_metrics = {
            "encoded_latent": metric(got["posterior_sample"], arrays["encoded_latent"], args.tolerance),
            "scaled_latent": metric(got["scaled_latent"], arrays["scaled_latent"], args.tolerance),
            "unet_model_pred": metric(got["unet_model_pred"], arrays["unet_model_pred"], args.tolerance),
            "alpha_prod_t": metric(got["alpha_prod_t"], arrays["alpha_prod_t"], 1e-8),
            "sqrt_alpha_prod_t": metric(got["sqrt_alpha_prod_t"], arrays["sqrt_alpha_prod_t"], 1e-8),
            "x0_from_res": metric(got["x0_from_res"], arrays["x0_from_res"], args.tolerance),
            "decode_input": metric(got["decode_input"], arrays["decode_input"], args.tolerance),
            "decode_output": metric(got["decoded_output"], arrays["decode_output"], args.tolerance),
            "clamped_output": metric(got["final_clamped_output"], arrays["clamped_output"], args.tolerance),
        }
        saved_diagnostic_tensors = []
        if args.save_diagnostic_tensors:
            diagnostic_tensors = {
                "encoded_latent": got["posterior_sample"],
                "scaled_latent": got["scaled_latent"],
                "unet_model_pred": got["unet_model_pred"],
                "alpha_prod_t": got["alpha_prod_t"],
                "sqrt_alpha_prod_t": got["sqrt_alpha_prod_t"],
                "x0_from_res": got["x0_from_res"],
                "decode_input": got["decode_input"],
                "decode_output": got["decoded_output"],
                "clamped_output": got["final_clamped_output"],
            }
            for stage_name, value in diagnostic_tensors.items():
                path = tensor_dir / f"{stage_name}_got.npy"
                np.save(path, np.asarray(value, dtype=np.float32))
                saved_diagnostic_tensors.append(str(path))
        statuses = [m["status"] for m in stage_metrics.values()]
        stage_status = "PASS" if all(s == "PASS" for s in statuses) else ("FAIL" if any(s == "FAIL" for s in statuses) else "PARTIAL")
        tensor_run_status = "PASS" if stage_status in {"PASS", "PARTIAL"} else "FAIL"
        full_one_step_status = "PASS" if stage_status == "PASS" else stage_status
        markers = {
            "TADSR_ONE_STEP_JITTOR_TENSOR_RUN": tensor_run_status,
            "TADSR_ONE_STEP_TENSOR_STAGE_ALIGNMENT": stage_status,
            "TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT": full_one_step_status,
        }
        payload = {
            "status_marker": "TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT",
            "status": full_one_step_status,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "markers": markers,
            "oracle_dir": str(oracle_dir),
            "tensor_dir": str(tensor_dir),
            "diagnostic_tensors_saved": bool(saved_diagnostic_tensors),
            "diagnostic_tensor_paths": saved_diagnostic_tensors,
            "stage_metrics": stage_metrics,
            "blockers": [] if full_one_step_status == "PASS" else [f"one or more stages are {stage_status}"],
            "policy": {
                "full_inference_executed": False,
                "denoising_loop_executed": False,
                "production_cli_used": False,
                "image_or_video_generated": False,
                "runtime_lora_dynamic_loading": False,
                "raw_tensor_committed": False,
            },
            "must_remain_statuses": {
                "JITTOR_FULL_INFERENCE": "NOT_COMPLETE",
                "JITTOR_FULL_PORT": "PARTIAL",
                "TIME_VAE_FULL_ALIGNMENT": "NOT_COMPLETE",
                "TADSR_DYNAMIC_RUNTIME_LORA_IMPLEMENTATION": "NOT_IMPLEMENTED_BY_DESIGN",
            },
        }
        write_outputs(out_dir, payload)
        for marker, status in markers.items():
            print(f"{marker}: {status}")
        return 0 if full_one_step_status == "PASS" else 1
    except Exception as exc:
        payload = blocked_payload(f"Jittor one-step tensor execution failed: {exc!r}", oracle_meta, out_dir, status="BLOCKED")
        write_outputs(out_dir, payload)
        print("TADSR_ONE_STEP_JITTOR_TENSOR_RUN: BLOCKED")
        print("TADSR_ONE_STEP_TENSOR_STAGE_ALIGNMENT: BLOCKED")
        print("TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT: BLOCKED")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
