#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jittor_tadsr_full.scheduler import TADSRSchedulerBoundaryTester
from jittor_tadsr_full.utils import write_json
from tests_jittor_alignment.scheduler_boundary_common import ORACLE_DIR, compare_named, load_array


OUT = Path("experiments/full_repro/scheduler_alignment")
CONFIG = Path("/mnt/data/sj/checkpoints/TADSR/preset/weights/scheduler/scheduler_config.json")


def status_from_rows(rows: list[dict]) -> str:
    return "PASS" if all(r.get("pass") for r in rows) else "FAIL"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    meta_path = ORACLE_DIR / "scheduler_boundary_oracle_metadata.json"
    if not meta_path.exists():
        result = {"status": "BLOCKED", "reason": "scheduler oracle metadata missing"}
        write_json(OUT / "jittor_scheduler_boundary_alignment.json", result)
        print("TADSR_SCHEDULER_BOUNDARY_ALIGNMENT: BLOCKED")
        return 1

    meta = json.loads(meta_path.read_text())
    sched = TADSRSchedulerBoundaryTester.from_config(CONFIG)
    timesteps = sched.set_timesteps_for_alignment(int(meta["num_inference_steps"]))
    rows = []
    rows.append(compare_named("timesteps", timesteps, load_array("scheduler_timesteps.npy"), 0.0))

    sample = load_array("scheduler_sample.npy")
    timestep = int(load_array("scheduler_timesteps.npy").reshape(-1)[0])
    scaled = sched.scale_model_input_for_alignment(sample, timestep)
    rows.append(compare_named("scale_model_input", scaled, load_array("scheduler_scaled_model_input.npy"), 1e-8))

    step = sched.step_for_alignment(load_array("scheduler_model_output.npy"), timestep, sample)
    rows.append(compare_named("scheduler_prev_sample", step["prev_sample"], load_array("scheduler_prev_sample.npy"), 1e-4))
    rows.append(compare_named("scheduler_pred_original_sample", step["pred_original_sample"], load_array("scheduler_pred_original_sample.npy"), 1e-5))

    timesteps_status = "PASS" if rows[0]["pass"] else "FAIL"
    scale_status = "NOT_APPLICABLE_NOOP" if meta.get("scale_model_input_is_noop") and rows[1]["pass"] else ("PASS" if rows[1]["pass"] else "FAIL")
    step_status = status_from_rows(rows[2:])

    optional_status = "NOT_APPLICABLE"
    optional_rows: list[dict] = []
    if meta.get("optional_unet_scheduler_one_step_available"):
        unet_sample = load_array("unet_scheduler_sample.npy")
        unet_timestep = int(load_array("unet_scheduler_timestep.npy").reshape(-1)[0])
        unet_step = sched.step_for_alignment(load_array("unet_scheduler_model_output.npy"), unet_timestep, unet_sample)
        optional_rows.append(compare_named("unet_scheduler_prev_sample", unet_step["prev_sample"], load_array("unet_scheduler_prev_sample.npy"), 1e-4))
        optional_rows.append(compare_named("unet_scheduler_pred_original_sample", unet_step["pred_original_sample"], load_array("unet_scheduler_pred_original_sample.npy"), 1e-5))
        optional_status = status_from_rows(optional_rows)

    boundary_status = "PASS" if timesteps_status == "PASS" and scale_status in {"PASS", "NOT_APPLICABLE_NOOP"} and step_status == "PASS" and optional_status in {"PASS", "NOT_APPLICABLE"} else "FAIL"

    result = {
        "status": boundary_status,
        "tadsr_scheduler_timesteps_alignment": timesteps_status,
        "tadsr_scheduler_scale_model_input_alignment": scale_status,
        "tadsr_scheduler_step_alignment": step_status,
        "tadsr_unet_scheduler_one_step_alignment": optional_status,
        "tadsr_scheduler_boundary_alignment": boundary_status,
        "rows": rows,
        "optional_rows": optional_rows,
        "oracle_metadata": str(meta_path),
        "scheduler_loop_executed": False,
        "vae_decode_executed": False,
        "full_tadsr_inference_executed": False,
        "image_saved": False,
    }
    write_json(OUT / "jittor_scheduler_boundary_alignment.json", result)
    md = [
        "# Jittor Scheduler Boundary Alignment",
        "",
        f"Status: **{boundary_status}**",
        "",
        "| Check | Status | Max abs error | Tolerance |",
        "|---|---|---:|---:|",
    ]
    for r in rows + optional_rows:
        md.append(f"| `{r['name']}` | {'PASS' if r.get('pass') else 'FAIL'} | {r.get('max_abs_error')} | {r.get('tolerance')} |")
    md += [
        "",
        f"- `TADSR_SCHEDULER_TIMESTEPS_ALIGNMENT`: `{timesteps_status}`",
        f"- `TADSR_SCHEDULER_SCALE_MODEL_INPUT_ALIGNMENT`: `{scale_status}`",
        f"- `TADSR_SCHEDULER_STEP_ALIGNMENT`: `{step_status}`",
        f"- `TADSR_UNET_SCHEDULER_ONE_STEP_ALIGNMENT`: `{optional_status}`",
        f"- `TADSR_SCHEDULER_BOUNDARY_ALIGNMENT`: `{boundary_status}`",
        "",
        "No denoising loop, VAE decode, image save, or full TADSR inference was executed.",
    ]
    (OUT / "jittor_scheduler_boundary_alignment.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"TADSR_SCHEDULER_TIMESTEPS_ALIGNMENT: {timesteps_status}")
    print(f"TADSR_SCHEDULER_SCALE_MODEL_INPUT_ALIGNMENT: {scale_status}")
    print(f"TADSR_SCHEDULER_STEP_ALIGNMENT: {step_status}")
    print(f"TADSR_UNET_SCHEDULER_ONE_STEP_ALIGNMENT: {optional_status}")
    print(f"TADSR_SCHEDULER_BOUNDARY_ALIGNMENT: {boundary_status}")
    return 0 if boundary_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
