#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "experiments" / "production_completion" / "full_inference" / "one_step"


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(arr).tobytes()).hexdigest()


def to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def tensor_stats(arr: np.ndarray, source: str, local_path: str | None = None) -> dict:
    a = np.asarray(arr)
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "min": float(a.min()) if a.size else None,
        "max": float(a.max()) if a.size else None,
        "mean": float(a.mean()) if a.size else None,
        "std": float(a.std()) if a.size else None,
        "finite": bool(np.isfinite(a).all()) if a.size else True,
        "sha256": sha256_array(a),
        "nbytes": int(a.nbytes),
        "source": source,
        "local_tensor_file": local_path,
    }


def write_blocked(out_dir: Path, reason: str, args: argparse.Namespace, error: str | None = None) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status_marker": "TADSR_ONE_STEP_OFFICIAL_ORACLE_TENSORS",
        "status": "BLOCKED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "error": error,
        "official_repo": args.official_repo,
        "official_weights": args.official_weights,
        "official_python": args.official_python,
        "pythonpath_overlay": args.pythonpath_overlay,
        "metadata_only": bool(args.metadata_only),
        "save_local_tensors": bool(args.save_local_tensors),
        "full_inference_executed": False,
        "denoising_loop_executed": False,
        "production_cli_used": False,
        "image_or_video_generated": False,
        "raw_tensor_policy": "no raw tensor committed; local_tensors is ignored",
    }
    (out_dir / "official_one_step_oracle_metadata.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "official_one_step_oracle_metadata.md").write_text(
        "\n".join([
            "# Official one-step tensor oracle",
            "",
            "`TADSR_ONE_STEP_OFFICIAL_ORACLE_TENSORS: BLOCKED`",
            "",
            f"- reason: `{reason}`",
            f"- error: `{error}`" if error else "- error: `None`",
            "",
            "No full denoising loop, image/video generation, production full inference, or raw-tensor commit occurred.",
        ]) + "\n",
        encoding="utf-8",
    )
    print("TADSR_ONE_STEP_OFFICIAL_ORACLE_TENSORS: BLOCKED")
    return 1


def maybe_reexec(args: argparse.Namespace) -> None:
    official_python = Path(args.official_python)
    if not official_python.exists():
        return
    current = Path(sys.executable)
    try:
        if current.resolve() == official_python.resolve():
            return
    except Exception:
        if str(current) == str(official_python):
            return
    os.execv(str(official_python), [str(official_python), __file__, *sys.argv[1:]])


def save_stage(saved: dict, out_dir: Path, name: str, value, source: str, save_local: bool, dtype=None) -> np.ndarray:
    arr = to_numpy(value)
    if dtype is not None:
        arr = arr.astype(dtype)
    elif arr.dtype.kind == "f":
        arr = arr.astype(np.float32)
    rel_path = None
    if save_local:
        local = out_dir / "local_tensors"
        local.mkdir(parents=True, exist_ok=True)
        path = local / f"{name}.npy"
        np.save(path, arr)
        rel_path = path.relative_to(out_dir).as_posix()
    saved[name] = tensor_stats(arr, source, rel_path)
    return arr


def write_outputs(out_dir: Path, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "official_one_step_oracle_metadata.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = out_dir / "official_one_step_stage_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["stage", "shape", "dtype", "min", "max", "mean", "std", "finite", "nbytes", "sha256", "local_tensor_file"],
        )
        writer.writeheader()
        for stage, stats in payload["tensor_stats"].items():
            writer.writerow({
                "stage": stage,
                "shape": json.dumps(stats["shape"]),
                "dtype": stats["dtype"],
                "min": stats["min"],
                "max": stats["max"],
                "mean": stats["mean"],
                "std": stats["std"],
                "finite": stats["finite"],
                "nbytes": stats["nbytes"],
                "sha256": stats["sha256"],
                "local_tensor_file": stats.get("local_tensor_file"),
            })
    lines = [
        "# Official one-step tensor oracle",
        "",
        f"`TADSR_ONE_STEP_OFFICIAL_ORACLE_TENSORS: {payload['status']}`",
        "",
        "This file records a controlled one-step tensor oracle only.",
        "It does not run the production full-inference CLI, a full denoising loop, image generation, or video generation.",
        "",
        "## Stage summary",
        "",
        "| Stage | Shape | DType | Min | Max | Mean | Std | Local tensor |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for stage, stats in payload["tensor_stats"].items():
        lines.append(
            f"| `{stage}` | `{stats['shape']}` | `{stats['dtype']}` | `{stats['min']}` | `{stats['max']}` | "
            f"`{stats['mean']}` | `{stats['std']}` | `{stats.get('local_tensor_file')}` |"
        )
    lines += [
        "",
        "## Safety",
        "",
        f"- full_inference_executed: `{payload['full_inference_executed']}`",
        f"- denoising_loop_executed: `{payload['denoising_loop_executed']}`",
        f"- image_or_video_generated: `{payload['image_or_video_generated']}`",
        f"- production_cli_used: `{payload['production_cli_used']}`",
        f"- raw_tensor_policy: `{payload['raw_tensor_policy']}`",
    ]
    (out_dir / "official_one_step_oracle_metadata.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export official PyTorch controlled one-step TADSR tensor oracle.")
    parser.add_argument("--official-repo", default="/mnt/data/sj/projects/TADSR_official_pytorch")
    parser.add_argument("--official-weights", default="/mnt/data/sj/checkpoints/TADSR/preset/weights")
    parser.add_argument("--official-python", default="/mnt/data/sj/venvs/tadsr_official_strict_cu118/bin/python")
    parser.add_argument("--pythonpath-overlay", default="/mnt/data/sj/tmp/tadsr_official_dependency_overlay_diffusers")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--timestep", type=int, default=1)
    parser.add_argument("--metadata-only", type=int, default=0)
    parser.add_argument("--save-local-tensors", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    if not Path(args.official_python).exists():
        return write_blocked(out_dir, "official Python does not exist on this machine", args)
    if not Path(args.official_repo).exists():
        return write_blocked(out_dir, "official repo does not exist on this machine", args)
    if not Path(args.official_weights).exists():
        return write_blocked(out_dir, "official weights directory does not exist on this machine", args)

    maybe_reexec(args)

    sys.path.insert(0, str(Path(args.pythonpath_overlay)))
    sys.path.insert(0, str(Path(args.official_repo)))
    try:
        import huggingface_hub

        if not hasattr(huggingface_hub, "cached_download"):
            huggingface_hub.cached_download = huggingface_hub.hf_hub_download
    except Exception:
        pass

    try:
        import torch
        from diffusers import DDPMScheduler
        from diffusers.models.autoencoders.time_autoencoder_kl import TimeAwareAutoencoderKL
        from my_utils.vaehook import VAEHook
        import my_utils.devices as devices
        from export_tadsr_unet_downblock3_resnet0_oracle import load_unet
    except Exception as exc:
        return write_blocked(out_dir, "official imports failed", args, repr(exc))

    try:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        devices.device = device
        devices.get_optimal_device = lambda: device
        devices.torch_gc = lambda: None

        weights = Path(args.official_weights)
        vae = TimeAwareAutoencoderKL.from_pretrained(str(weights / "time_vae"), local_files_only=True).to(device)
        vae.eval()
        if not hasattr(vae.encoder, "original_forward"):
            vae.encoder.original_forward = vae.encoder.forward
        if not hasattr(vae.decoder, "original_forward"):
            vae.decoder.original_forward = vae.decoder.forward

        raw_decoder_forward = vae.decoder.original_forward
        decoder_counter = {"original_forward_calls": 0, "vae_tile_forward_calls": 0}

        def counted_decoder_original_forward(*inner_args, **inner_kwargs):
            decoder_counter["original_forward_calls"] += 1
            return raw_decoder_forward(*inner_args, **inner_kwargs)

        vae.decoder.original_forward = counted_decoder_original_forward
        encoder_hook = VAEHook(vae.encoder, 16, is_decoder=False, fast_decoder=False, fast_encoder=False, color_fix=False, to_gpu=False, time_vae=True)
        vae.encoder.forward = encoder_hook
        decoder_hook = VAEHook(vae.decoder, 16, is_decoder=True, fast_decoder=False, fast_encoder=False, color_fix=False, to_gpu=False, time_vae=False)
        decoder_tile_forward = decoder_hook.vae_tile_forward

        def counted_decoder_tile_forward(*inner_args, **inner_kwargs):
            decoder_counter["vae_tile_forward_calls"] += 1
            return decoder_tile_forward(*inner_args, **inner_kwargs)

        decoder_hook.vae_tile_forward = counted_decoder_tile_forward
        vae.decoder.forward = decoder_hook

        unet, loaded_lora = load_unet()
        unet = unet.to(device)
        unet.eval()
        scheduler = DDPMScheduler.from_pretrained(str(weights), subfolder="scheduler")
        scheduler.set_timesteps(1, device=device)
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)

        scaling_factor = float(getattr(vae.config, "scaling_factor", 0.18215))
        timestep = torch.tensor([args.timestep], dtype=torch.long, device=device)
        input_tensor = torch.linspace(
            -1.0,
            1.0,
            1 * 3 * args.height * args.width,
            dtype=torch.float32,
            device=device,
        ).reshape(1, 3, args.height, args.width)
        encoder_shape = [1, 77, int(unet.config.cross_attention_dim)]
        encoder_hidden_states = torch.linspace(
            -0.5,
            0.5,
            steps=int(np.prod(encoder_shape)),
            dtype=torch.float32,
            device=device,
        ).reshape(*encoder_shape)

        saved: dict[str, dict] = {}
        save_local = bool(args.save_local_tensors) and not bool(args.metadata_only)

        with torch.no_grad():
            encoded = vae.encode(input_tensor, timestep)
            mean = encoded.latent_dist.mean
            std = encoded.latent_dist.std
            epsilon = torch.linspace(-0.5, 0.5, mean.numel(), dtype=mean.dtype, device=device).reshape_as(mean)
            encoded_latent = mean + std * epsilon
            scaled_latent = encoded_latent * scaling_factor
            model_pred = unet(scaled_latent, timestep, encoder_hidden_states=encoder_hidden_states, return_dict=True).sample
            alpha_prod_t = scheduler.alphas_cumprod[timestep].flatten().to(dtype=scaled_latent.dtype, device=device)
            sqrt_alpha_prod_t = alpha_prod_t.sqrt()
            x0 = scaled_latent / sqrt_alpha_prod_t[:, None, None, None] - model_pred
            decode_input = x0 / scaling_factor
            decoded_output = vae.decode(decode_input).sample
            clamped_output = decoded_output.clamp(-1, 1)

        save_stage(saved, out_dir, "input_tensor", input_tensor, "deterministic low-resolution/control tensor", save_local)
        save_stage(saved, out_dir, "scheduler_timestep", timestep.detach().cpu().numpy().astype(np.int64), "single audited timestep", save_local, dtype=np.int64)
        save_stage(saved, out_dir, "encoder_hidden_states", encoder_hidden_states, "deterministic text/control conditioning tensor", save_local)
        save_stage(saved, out_dir, "sample_epsilon", epsilon, "fixed epsilon for deterministic posterior sample", save_local)
        save_stage(saved, out_dir, "encoded_latent", encoded_latent, "TimeVAE encode posterior mean + std * fixed epsilon", save_local)
        save_stage(saved, out_dir, "scaled_latent", scaled_latent, "encoded_latent * scaling_factor", save_local)
        save_stage(saved, out_dir, "unet_model_pred", model_pred, "official UNet full forward once", save_local)
        save_stage(saved, out_dir, "alpha_prod_t", alpha_prod_t, "scheduler alphas_cumprod[timestep]", save_local)
        save_stage(saved, out_dir, "sqrt_alpha_prod_t", sqrt_alpha_prod_t, "sqrt(alpha_prod_t)", save_local)
        save_stage(saved, out_dir, "x0_from_res", x0, "scaled_latent / sqrt(alpha_prod_t) - unet_model_pred", save_local)
        save_stage(saved, out_dir, "decode_input", decode_input, "x0_from_res / scaling_factor", save_local)
        save_stage(saved, out_dir, "decode_output", decoded_output, "TimeVAE decode tensor output", save_local)
        save_stage(saved, out_dir, "clamped_output", clamped_output, "decode_output clamped to [-1, 1]", save_local)

        payload = {
            "status_marker": "TADSR_ONE_STEP_OFFICIAL_ORACLE_TENSORS",
            "status": "PASS",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "official_repo": str(Path(args.official_repo)),
            "official_weights": str(Path(args.official_weights)),
            "official_python": sys.executable,
            "pythonpath_overlay": str(Path(args.pythonpath_overlay)),
            "device": str(device),
            "seed": args.seed,
            "height": args.height,
            "width": args.width,
            "timestep": [int(args.timestep)],
            "scaling_factor": scaling_factor,
            "loaded_unet_lora_parameter_count": len(loaded_lora),
            "tensor_stats": saved,
            "local_tensor_dir": "local_tensors" if save_local else None,
            "local_tensors_saved": save_local,
            "metadata_only": bool(args.metadata_only),
            "unet_full_forward_once_executed": True,
            "timevae_encode_executed": True,
            "timevae_decode_executed": True,
            "scheduler_step_executed": False,
            "full_inference_executed": False,
            "denoising_loop_executed": False,
            "production_cli_used": False,
            "image_or_video_generated": False,
            "image_saved": False,
            "runtime_lora_dynamic_loading": False,
            "decoder_hook_installed": True,
            "decoder_actual_behavior": "original_forward_due_to_official_decoder_hook_time_vae_false",
            "decoder_original_forward_call_count": decoder_counter["original_forward_calls"],
            "decoder_tiled_path_call_count": decoder_counter["vae_tile_forward_calls"],
            "raw_tensor_policy": "raw tensors are optional local_tensors only; do not stage or commit .npy/.npz",
            "must_remain_statuses": {
                "JITTOR_FULL_INFERENCE": "NOT_COMPLETE",
                "JITTOR_FULL_PORT": "PARTIAL",
                "TIME_VAE_FULL_ALIGNMENT": "NOT_COMPLETE",
                "TADSR_DYNAMIC_RUNTIME_LORA_IMPLEMENTATION": "NOT_IMPLEMENTED_BY_DESIGN",
            },
        }
        write_outputs(out_dir, payload)
        print("TADSR_ONE_STEP_OFFICIAL_ORACLE_TENSORS: PASS")
        return 0
    except Exception as exc:
        return write_blocked(out_dir, "official one-step tensor oracle execution failed", args, repr(exc))


if __name__ == "__main__":
    raise SystemExit(main())
