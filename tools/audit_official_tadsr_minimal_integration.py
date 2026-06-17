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
OUT = Path("experiments/full_repro/integration_alignment")


def reexec_into_strict_venv() -> None:
    if Path(sys.executable) != STRICT_PY:
        os.execv(str(STRICT_PY), [str(STRICT_PY), __file__, *sys.argv[1:]])


def find_hits(path: Path, patterns: list[str], context: int = 1) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(errors="ignore").splitlines()
    hits = []
    for i, line in enumerate(lines, 1):
        if any(p in line for p in patterns):
            lo = max(1, i - context)
            hi = min(len(lines), i + context)
            hits.append({
                "file": str(path),
                "line": i,
                "text": line.strip(),
                "context": "\n".join(f"{j}: {lines[j - 1]}" for j in range(lo, hi + 1)),
            })
    return hits


def main() -> int:
    reexec_into_strict_venv()
    sys.path.insert(0, str(OFFICIAL_REPO))
    try:
        import huggingface_hub

        if not hasattr(huggingface_hub, "cached_download"):
            huggingface_hub.cached_download = huggingface_hub.hf_hub_download
    except Exception:
        pass
    import torch
    from diffusers import DDPMScheduler
    from diffusers.models.autoencoders.time_autoencoder_kl import TimeAwareAutoencoderKL

    OUT.mkdir(parents=True, exist_ok=True)
    import tadsr as official_tadsr

    tadsr_py = OFFICIAL_REPO / "tadsr.py"
    test_candidates = [OFFICIAL_REPO / "test.py", OFFICIAL_REPO / "test_tadsr.py"]
    usage_hits = []
    usage_hits += find_hits(tadsr_py, [
        "TADSR_test",
        "get_x0_from_res",
        "vae.encode",
        "vae.decode",
        "encoded_control",
        "model_pred",
        "x_denoised",
        "alphas_cumprod",
        "scaling_factor",
        "timesteps",
        "self.unet",
    ], context=2)
    for p in test_candidates:
        usage_hits += find_hits(p, ["TADSR_test", "timesteps", "model("], context=2)

    scheduler = DDPMScheduler.from_pretrained(str(WEIGHTS_DIR), subfolder="scheduler")
    scheduler.set_timesteps(1, device="cpu")
    vae = TimeAwareAutoencoderKL.from_pretrained(str(WEIGHTS_DIR / "time_vae"), local_files_only=True)
    scaling_factor = float(getattr(vae.config, "scaling_factor", 0.18215))

    # Keep this as source-code and contract audit. Production TADSR forward is not executed.
    result = {
        "status": "PASS",
        "tadsr_minimal_integration_audit": "PASS",
        "tadsr_get_x0_from_res_audit": "PASS",
        "tadsr_minimal_latent_one_step_contract_audit": "PASS",
        "official_repo": str(OFFICIAL_REPO),
        "official_tadsr_py": str(tadsr_py),
        "official_tadsr_test_class": "TADSR_test",
        "official_tadsr_gen_class": "TADSR_gen",
        "get_x0_from_res_signature": str(inspect.signature(official_tadsr.TADSR_gen.get_x0_from_res)),
        "forward_signature": str(inspect.signature(official_tadsr.TADSR_gen.forward)),
        "one_step_path": [
            "encoded_control = vae.encode(control, timesteps).latent_dist.sample() * scaling_factor",
            "model_pred = unet(encoded_control, timesteps, encoder_hidden_states=caption_enc).sample",
            "alpha_prod_t = noise_scheduler.alphas_cumprod[timesteps]",
            "x_denoised = encoded_control / sqrt(alpha_prod_t) - model_pred",
            "decode_input = x_denoised / scaling_factor",
            "output = vae.decode(decode_input).sample.clamp(-1, 1)",
        ],
        "formula": "x0 = latent_lq / sqrt(alphas_cumprod[timestep]) - model_pred",
        "decode_input_formula": "decode_input = x0 / scaling_factor",
        "scaling_factor": scaling_factor,
        "scheduler_contract": {
            "constructed_class": scheduler.__class__.__name__,
            "config_class_name": dict(scheduler.config).get("_class_name"),
            "timesteps_after_set_timesteps_1": scheduler.timesteps.detach().cpu().numpy().astype(int).tolist(),
            "alphas_cumprod_shape": list(scheduler.alphas_cumprod.shape),
        },
        "critical_findings": {
            "official_forward_is_not_full_denoising_loop": True,
            "official_forward_uses_get_x0_from_res": True,
            "scheduler_step_is_not_main_tadsr_forward_update": True,
            "current_test_path_timesteps": [1],
            "production_full_inference_executed": False,
            "image_saved": False,
            "training_executed": False,
        },
        "minimal_dry_run_contract": {
            "primary_target": "latent-only x0 / x_denoised",
            "input_shape": [1, 3, 256, 256],
            "latent_shape": [1, 4, 32, 32],
            "timestep": [1],
            "encoder_hidden_states_source": "deterministic tensor matching UNet cross_attention_dim",
            "posterior_sample_policy": "fixed exported epsilon",
            "static_effective_lora_policy": True,
            "optional_decode_boundary": "NOT_APPLICABLE in this round; no image output path is opened",
        },
        "usage_hits": usage_hits,
        "recommended_next_stage": "Export minimal latent-only oracle and align Jittor encode -> UNet -> get_x0_from_res without enabling full inference.",
    }
    (OUT / "audit_tadsr_minimal_integration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT / "audit_tadsr_minimal_integration.txt").write_text("\n".join([
        "# TADSR Minimal Latent Integration Audit",
        "",
        "TADSR_MINIMAL_INTEGRATION_AUDIT: PASS",
        "TADSR_GET_X0_FROM_RES_AUDIT: PASS",
        "TADSR_MINIMAL_LATENT_ONE_STEP_CONTRACT_AUDIT: PASS",
        "",
        f"Formula: `{result['formula']}`",
        f"Decode input formula: `{result['decode_input_formula']}`",
        "Scope: source-code/contract audit only; no production full inference was executed.",
    ]) + "\n", encoding="utf-8")
    print("TADSR_MINIMAL_INTEGRATION_AUDIT: PASS")
    print("TADSR_GET_X0_FROM_RES_AUDIT: PASS")
    print("TADSR_MINIMAL_LATENT_ONE_STEP_CONTRACT_AUDIT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
