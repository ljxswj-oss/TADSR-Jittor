#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path


STRICT_PY = Path("/mnt/data/sj/venvs/tadsr_official_strict_cu118/bin/python")
OFFICIAL_REPO = Path("/mnt/data/sj/projects/TADSR_official_pytorch")
WEIGHTS_DIR = Path("/mnt/data/sj/checkpoints/TADSR/preset/weights")
OUT = Path("experiments/full_repro/scheduler_alignment/oracle_tensors_scheduler_boundary")
UNET_ORACLE = Path("experiments/full_repro/unet_alignment/oracle_tensors_unet_full_forward")


def reexec_into_strict_venv() -> None:
    if Path(sys.executable) != STRICT_PY:
        os.execv(str(STRICT_PY), [str(STRICT_PY), __file__, *sys.argv[1:]])


def to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else x


def summary(arr) -> dict:
    import numpy as np

    a = np.asarray(arr)
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "min": float(a.min()) if a.size else None,
        "max": float(a.max()) if a.size else None,
        "mean": float(a.mean()) if a.size else None,
        "std": float(a.std()) if a.size else None,
    }


def main() -> int:
    reexec_into_strict_venv()
    sys.path.insert(0, str(OFFICIAL_REPO))
    try:
        import huggingface_hub

        if not hasattr(huggingface_hub, "cached_download"):
            huggingface_hub.cached_download = huggingface_hub.hf_hub_download
    except Exception:
        pass
    import numpy as np
    import torch
    from diffusers import DDPMScheduler

    OUT.mkdir(parents=True, exist_ok=True)
    scheduler = DDPMScheduler.from_pretrained(str(WEIGHTS_DIR), subfolder="scheduler")
    scheduler.set_timesteps(1, device="cpu")
    timestep = scheduler.timesteps[0]

    sample = torch.linspace(-1.0, 1.0, 1 * 4 * 32 * 32, dtype=torch.float32).reshape(1, 4, 32, 32)
    model_output = torch.linspace(0.5, -0.5, 1 * 4 * 32 * 32, dtype=torch.float32).reshape(1, 4, 32, 32)
    scaled = scheduler.scale_model_input(sample, timestep)
    torch.manual_seed(123)
    step_out = scheduler.step(model_output, timestep, sample)

    np.save(OUT / "scheduler_timesteps.npy", to_numpy(scheduler.timesteps).astype("int64"))
    np.save(OUT / "scheduler_sample.npy", to_numpy(sample).astype("float32"))
    np.save(OUT / "scheduler_model_output.npy", to_numpy(model_output).astype("float32"))
    np.save(OUT / "scheduler_scaled_model_input.npy", to_numpy(scaled).astype("float32"))
    np.save(OUT / "scheduler_prev_sample.npy", to_numpy(step_out.prev_sample).astype("float32"))
    np.save(OUT / "scheduler_pred_original_sample.npy", to_numpy(step_out.pred_original_sample).astype("float32"))

    optional_available = False
    optional_reason = "existing UNet full-forward oracle tensors missing"
    optional_summaries = {}
    if (UNET_ORACLE / "full_forward_sample.npy").exists() and (UNET_ORACLE / "official_full_forward_output.npy").exists() and (UNET_ORACLE / "full_forward_timestep.npy").exists():
        unet_sample = torch.from_numpy(np.load(UNET_ORACLE / "full_forward_sample.npy")).float()
        unet_model_output = torch.from_numpy(np.load(UNET_ORACLE / "official_full_forward_output.npy")).float()
        unet_timestep_np = np.load(UNET_ORACLE / "full_forward_timestep.npy")
        unet_timestep = torch.tensor(int(np.asarray(unet_timestep_np).reshape(-1)[0]), dtype=torch.long)
        torch.manual_seed(123)
        unet_step = scheduler.step(unet_model_output, unet_timestep, unet_sample)
        np.save(OUT / "unet_scheduler_sample.npy", to_numpy(unet_sample).astype("float32"))
        np.save(OUT / "unet_scheduler_timestep.npy", np.asarray([int(unet_timestep.item())], dtype="int64"))
        np.save(OUT / "unet_scheduler_model_output.npy", to_numpy(unet_model_output).astype("float32"))
        np.save(OUT / "unet_scheduler_prev_sample.npy", to_numpy(unet_step.prev_sample).astype("float32"))
        np.save(OUT / "unet_scheduler_pred_original_sample.npy", to_numpy(unet_step.pred_original_sample).astype("float32"))
        optional_available = True
        optional_reason = "reused existing official UNet full-forward oracle tensors; no UNet or full pipeline run in this export"
        optional_summaries = {
            "unet_scheduler_sample": summary(to_numpy(unet_sample)),
            "unet_scheduler_model_output": summary(to_numpy(unet_model_output)),
            "unet_scheduler_prev_sample": summary(to_numpy(unet_step.prev_sample)),
            "unet_scheduler_pred_original_sample": summary(to_numpy(unet_step.pred_original_sample)),
        }

    meta = {
        "status": "PASS",
        "tadsr_scheduler_oracle_tensors": "PASS",
        "tadsr_unet_scheduler_one_step_oracle": "PASS" if optional_available else "NOT_APPLICABLE",
        "scheduler_class": scheduler.__class__.__name__,
        "config": dict(scheduler.config),
        "num_inference_steps": 1,
        "timesteps": to_numpy(scheduler.timesteps).astype(int).tolist(),
        "chosen_timestep": int(timestep.item()),
        "step_signature": str(inspect.signature(scheduler.step)),
        "scale_model_input_used": True,
        "scale_model_input_is_noop": bool(torch.max(torch.abs(scaled - sample)).item() == 0.0),
        "scheduler_only_sample": summary(to_numpy(sample)),
        "scheduler_only_model_output": summary(to_numpy(model_output)),
        "scheduler_scaled_model_input": summary(to_numpy(scaled)),
        "scheduler_prev_sample": summary(to_numpy(step_out.prev_sample)),
        "scheduler_pred_original_sample": summary(to_numpy(step_out.pred_original_sample)),
        "step_output_fields": list(step_out.keys()) if hasattr(step_out, "keys") else [],
        "optional_unet_scheduler_one_step_available": optional_available,
        "optional_unet_scheduler_one_step_reason": optional_reason,
        "optional_unet_scheduler_summaries": optional_summaries,
        "scheduler_loop_executed": False,
        "vae_decode_executed": False,
        "full_tadsr_inference_executed": False,
        "image_saved": False,
        "recommended_next_stage": "Run Jittor scheduler boundary alignment test; if PASS, plan minimal latent-only integration dry-run while keeping full inference NotImplemented.",
    }
    (OUT / "scheduler_boundary_oracle_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    summary_lines = [
        "# Scheduler Boundary Oracle",
        "",
        "Status: PASS",
        "",
        f"- Scheduler class: `{meta['scheduler_class']}`",
        f"- Timesteps: `{meta['timesteps']}`",
        f"- Chosen timestep: `{meta['chosen_timestep']}`",
        f"- Step fields: `{meta['step_output_fields']}`",
        f"- Optional UNet+scheduler one-step: `{meta['tadsr_unet_scheduler_one_step_oracle']}`",
        "",
        "No denoising loop, VAE decode, image save, or full TADSR inference was executed.",
    ]
    (OUT / "oracle_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("TADSR_SCHEDULER_ORACLE_TENSORS: PASS")
    print(f"TADSR_UNET_SCHEDULER_ONE_STEP_ORACLE: {meta['tadsr_unet_scheduler_one_step_oracle']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
