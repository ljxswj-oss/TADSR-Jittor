#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


STRICT_PY = Path("/mnt/data/sj/venvs/tadsr_official_strict_cu118/bin/python")
OFFICIAL_REPO = Path("/mnt/data/sj/projects/TADSR_official_pytorch")
WEIGHTS_DIR = Path("/mnt/data/sj/checkpoints/TADSR/preset/weights")
OUT = Path("experiments/full_repro/integration_alignment/oracle_tensors_minimal_latent_integration")


def reexec_into_strict_venv() -> None:
    if Path(sys.executable) != STRICT_PY:
        os.execv(str(STRICT_PY), [str(STRICT_PY), __file__, *sys.argv[1:]])


def to_np(x) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.float32) if hasattr(x, "detach") else np.asarray(x, dtype=np.float32)


def sha256_array(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def stat(arr: np.ndarray, source: str) -> dict:
    a = np.asarray(arr)
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "min": float(a.min()) if a.size else None,
        "max": float(a.max()) if a.size else None,
        "mean": float(a.mean()) if a.size else None,
        "std": float(a.std()) if a.size else None,
        "sha256": sha256_array(a),
        "nbytes": int(a.nbytes),
        "source": source,
    }


def save(saved: dict, name: str, arr, source: str, dtype=None) -> np.ndarray:
    out = np.asarray(to_np(arr) if hasattr(arr, "detach") else arr)
    if dtype is not None:
        out = out.astype(dtype)
    elif out.dtype.kind == "f":
        out = out.astype(np.float32)
    np.save(OUT / f"{name}.npy", out)
    saved[name] = stat(out, source)
    return out


def main() -> int:
    reexec_into_strict_venv()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
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
    from my_utils.vaehook import VAEHook
    import my_utils.devices as devices
    from export_tadsr_unet_downblock3_resnet0_oracle import load_unet

    torch.manual_seed(2026)
    np.random.seed(2026)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    devices.device = device
    devices.get_optimal_device = lambda: device
    devices.torch_gc = lambda: None

    vae = TimeAwareAutoencoderKL.from_pretrained(str(WEIGHTS_DIR / "time_vae"), local_files_only=True).to(device)
    vae.eval()
    if not hasattr(vae.encoder, "original_forward"):
        vae.encoder.original_forward = vae.encoder.forward
    if not hasattr(vae.decoder, "original_forward"):
        vae.decoder.original_forward = vae.decoder.forward
    raw_decoder_forward = vae.decoder.original_forward
    decoder_counter = {"original_forward_calls": 0, "vae_tile_forward_calls": 0}

    def counted_decoder_original_forward(*args, **kwargs):
        decoder_counter["original_forward_calls"] += 1
        return raw_decoder_forward(*args, **kwargs)

    vae.decoder.original_forward = counted_decoder_original_forward
    encoder_hook = VAEHook(vae.encoder, 16, is_decoder=False, fast_decoder=False, fast_encoder=False, color_fix=False, to_gpu=False, time_vae=True)
    vae.encoder.forward = encoder_hook
    decoder_hook = VAEHook(vae.decoder, 16, is_decoder=True, fast_decoder=False, fast_encoder=False, color_fix=False, to_gpu=False, time_vae=False)
    decoder_tile_forward = decoder_hook.vae_tile_forward

    def counted_decoder_tile_forward(*args, **kwargs):
        decoder_counter["vae_tile_forward_calls"] += 1
        return decoder_tile_forward(*args, **kwargs)

    decoder_hook.vae_tile_forward = counted_decoder_tile_forward
    vae.decoder.forward = decoder_hook

    unet, loaded_lora = load_unet()
    unet.eval()
    scheduler = DDPMScheduler.from_pretrained(str(WEIGHTS_DIR), subfolder="scheduler")
    scheduler.set_timesteps(1, device="cpu")

    scaling_factor = float(getattr(vae.config, "scaling_factor", 0.18215))
    timestep = torch.tensor([1], dtype=torch.long, device=device)
    minimal_input = torch.linspace(-1.0, 1.0, 1 * 3 * 256 * 256, dtype=torch.float32, device=device).reshape(1, 3, 256, 256)
    encoder_shape = [1, 77, int(unet.config.cross_attention_dim)]
    encoder_hidden_states = torch.linspace(-0.5, 0.5, steps=int(np.prod(encoder_shape)), dtype=torch.float32, device=device).reshape(*encoder_shape)

    with torch.no_grad():
        encoded = vae.encode(minimal_input, timestep)
        mean = encoded.latent_dist.mean
        std = encoded.latent_dist.std
        epsilon = torch.linspace(-0.5, 0.5, mean.numel(), dtype=mean.dtype, device=device).reshape_as(mean)
        posterior_sample = mean + std * epsilon
        scaled_latent = posterior_sample * scaling_factor
        model_pred = unet(scaled_latent, timestep, encoder_hidden_states=encoder_hidden_states, return_dict=True).sample
        alpha_prod_t = scheduler.alphas_cumprod[timestep].flatten().to(dtype=scaled_latent.dtype, device=device)
        sqrt_alpha_prod_t = alpha_prod_t.sqrt()
        x0 = scaled_latent / sqrt_alpha_prod_t[:, None, None, None] - model_pred
        decode_input = x0 / scaling_factor
        decoded_output = vae.decode(decode_input).sample
        final_clamped_output = decoded_output.clamp(-1, 1)

    saved = {}
    save(saved, "minimal_input", minimal_input, "deterministic low-quality/control input in [-1,1]")
    save(saved, "minimal_timestep", timestep.detach().cpu().numpy().astype(np.int64), "official one-step timestep", dtype=np.int64)
    save(saved, "minimal_encoder_hidden_states", encoder_hidden_states, "deterministic encoder_hidden_states matching UNet cross_attention_dim")
    save(saved, "minimal_sample_epsilon", epsilon, "fixed epsilon for deterministic posterior sample")
    save(saved, "minimal_posterior_sample", posterior_sample, "vae.encode(...).latent_dist mean + std * fixed epsilon")
    save(saved, "minimal_scaled_latent", scaled_latent, "posterior_sample * scaling_factor")
    save(saved, "minimal_unet_model_pred", model_pred, "official UNet full forward output from scaled latent")
    save(saved, "minimal_alpha_prod_t", alpha_prod_t.detach().cpu().numpy(), "scheduler.alphas_cumprod[timestep]")
    save(saved, "minimal_sqrt_alpha_prod_t", sqrt_alpha_prod_t.detach().cpu().numpy(), "sqrt(alpha_prod_t)")
    save(saved, "minimal_x0_from_res", x0, "scaled_latent / sqrt(alpha_prod_t) - model_pred")
    save(saved, "minimal_decode_input", decode_input, "x0 / scaling_factor; input to actual decoder original_forward boundary")
    save(saved, "minimal_decoded_output", decoded_output, "official TimeVAE actual decoder path: decoder hook installed but original_forward dispatch")
    save(saved, "minimal_final_clamped_output", final_clamped_output, "minimal decoded output clamped to [-1, 1]")

    metadata = {
        "status": "PASS",
        "tadsr_minimal_latent_one_step_oracle": "PASS",
        "tadsr_get_x0_from_res_oracle": "PASS",
        "tadsr_minimal_decode_boundary_oracle": "PASS",
        "official_repo": str(OFFICIAL_REPO),
        "weights_dir": str(WEIGHTS_DIR),
        "python": sys.executable,
        "input_shape": list(minimal_input.shape),
        "latent_shape": list(scaled_latent.shape),
        "model_pred_shape": list(model_pred.shape),
        "decode_input_shape": list(decode_input.shape),
        "decoded_output_shape": list(decoded_output.shape),
        "final_clamped_output_shape": list(final_clamped_output.shape),
        "final_clamped_output_range": [
            float(final_clamped_output.detach().cpu().min().item()),
            float(final_clamped_output.detach().cpu().max().item()),
        ],
        "timestep": [int(timestep.item())],
        "scaling_factor": scaling_factor,
        "alpha_prod_t": to_np(alpha_prod_t).astype(float).tolist(),
        "sqrt_alpha_prod_t": to_np(sqrt_alpha_prod_t).astype(float).tolist(),
        "formula": "x0 = scaled_latent / sqrt(alphas_cumprod[timestep]) - model_pred",
        "decode_input_formula": "decode_input = x0 / scaling_factor",
        "loaded_unet_lora_parameter_count": len(loaded_lora),
        "unet_full_forward_executed": True,
        "scheduler_step_executed": False,
        "full_denoising_loop_executed": False,
        "vae_decode_executed": True,
        "decoder_hook_installed": True,
        "decoder_actual_behavior": "original_forward_due_to_official_decoder_hook_time_vae_false",
        "decoder_original_forward_used": decoder_counter["original_forward_calls"] > 0,
        "decoder_original_forward_call_count": decoder_counter["original_forward_calls"],
        "decoder_tiled_path_triggered": decoder_counter["vae_tile_forward_calls"] > 0,
        "decoder_tiled_path_call_count": decoder_counter["vae_tile_forward_calls"],
        "image_saved": False,
        "full_tadsr_inference_executed": False,
        "production_cli_used": False,
        "runtime_lora_dynamic_loading": False,
        "static_effective_lora_policy": True,
        "optional_decode_boundary": "PASS_TENSOR_ONLY_DECODE_CLAMP_BOUNDARY",
        "tensor_stats": saved,
        "recommended_next_stage": "Run Jittor minimal one-step decode boundary alignment; if PASS, plan final packaging or a controlled production-CLI design audit without opening full inference.",
    }
    (OUT / "minimal_latent_integration_oracle_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (OUT / "oracle_summary.txt").write_text("\n".join([
        "# Minimal latent-only TADSR integration oracle",
        "",
        "TADSR_MINIMAL_LATENT_ONE_STEP_ORACLE: PASS",
        "TADSR_GET_X0_FROM_RES_ORACLE: PASS",
        "TADSR_MINIMAL_DECODE_BOUNDARY_ORACLE: PASS",
        "",
        f"Input shape: {metadata['input_shape']}",
        f"Latent shape: {metadata['latent_shape']}",
        f"Decoded output shape: {metadata['decoded_output_shape']}",
        f"Final clamped output range: {metadata['final_clamped_output_range']}",
        f"Formula: `{metadata['formula']}`",
        "",
        "No full denoising loop, image save, production CLI, or full TADSR inference was executed. Only tensor decode/clamp oracle values were saved.",
    ]) + "\n", encoding="utf-8")
    print("TADSR_MINIMAL_LATENT_ONE_STEP_ORACLE: PASS")
    print("TADSR_GET_X0_FROM_RES_ORACLE: PASS")
    print("TADSR_MINIMAL_DECODE_BOUNDARY_ORACLE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
