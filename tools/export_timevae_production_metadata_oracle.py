#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import textwrap
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        return data if isinstance(data, dict) else {"rows": data}
    except Exception as exc:
        return {"status": "FAIL", "error": repr(exc)}


def overlay_env(pythonpath_overlay: str | None = None, official_repo: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    paths: list[str] = []
    if official_repo:
        paths.append(official_repo)
    if pythonpath_overlay:
        paths.append(pythonpath_overlay)
    if paths:
        old = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(paths) + (os.pathsep + old if old else "")
    return env


def run_import(
    official_python: str,
    module: str,
    pythonpath_overlay: str | None = None,
    official_repo: str | None = None,
) -> dict[str, Any]:
    if not official_python or not Path(official_python).exists():
        return {"module": module, "status": "BLOCKED", "version": None, "error": "official_python_missing"}
    code = (
        "import importlib, json; "
        f"m=importlib.import_module({module!r}); "
        "print(json.dumps({'version': getattr(m, '__version__', 'unknown')}))"
    )
    proc = subprocess.run(
        [official_python, "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        env=overlay_env(pythonpath_overlay, official_repo),
    )
    if proc.returncode == 0:
        try:
            parsed = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception:
            parsed = {"version": "unknown"}
        return {"module": module, "status": "PASS", "version": parsed.get("version", "unknown"), "error": None}
    return {"module": module, "status": "MISSING", "version": None, "error": proc.stdout.strip()[-1000:]}


def run_live_timevae_metadata(
    official_python: str,
    official_repo: str,
    official_weights: str,
    pythonpath_overlay: str | None,
    height: int,
    width: int,
    seed: int,
    device: str,
    cpu_only: bool,
) -> dict[str, Any]:
    """Run a tightly scoped official TimeVAE encode/decode metadata pass.

    This subprocess saves no tensors and never calls the production TADSR CLI,
    UNet, scheduler loop, or image/video output path.
    """
    if not official_python or not Path(official_python).exists():
        return {"status": "BLOCKED", "error": "official_python_missing"}
    if not official_repo or not Path(official_repo).exists():
        return {"status": "BLOCKED", "error": "official_repo_missing"}
    if not official_weights or not Path(official_weights).exists():
        return {"status": "BLOCKED", "error": "official_weights_missing"}

    code = r"""
import json
import os
import traceback

payload = {
    "status": "BLOCKED",
    "full_inference_executed": False,
    "denoising_loop_executed": False,
    "unet_called": False,
    "scheduler_loop_called": False,
    "image_or_video_generated": False,
    "raw_tensors_committed": False,
    "save_local_tensors": False,
    "strict_env_modified": False,
}

def stats(t):
    x = t.detach().float().cpu()
    finite = bool(torch.isfinite(x).all().item())
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()) if x.numel() > 1 else 0.0,
        "finite": finite,
    }

try:
    import torch
    import diffusers
    from diffusers import TimeAwareAutoencoderKL

    official_weights = os.environ["TADSR_LIVE_OFFICIAL_WEIGHTS"]
    height = int(os.environ["TADSR_LIVE_HEIGHT"])
    width = int(os.environ["TADSR_LIVE_WIDTH"])
    seed = int(os.environ["TADSR_LIVE_SEED"])
    requested_device = os.environ.get("TADSR_LIVE_DEVICE", "auto")
    cpu_only = os.environ.get("TADSR_LIVE_CPU_ONLY", "0") == "1"

    if cpu_only or requested_device == "cpu":
        selected_device = "cpu"
    elif requested_device == "cuda":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    if selected_device == "cuda":
        torch.cuda.manual_seed_all(seed)

    vae = TimeAwareAutoencoderKL.from_pretrained(official_weights, subfolder="time_vae")
    vae.eval()
    vae.to(selected_device)
    for p in vae.parameters():
        p.requires_grad_(False)

    generator = torch.Generator(device=selected_device)
    generator.manual_seed(seed)
    encode_input = torch.rand((1, 3, height, width), generator=generator, device=selected_device) * 2.0 - 1.0

    with torch.no_grad():
        encoded = vae.encode(encode_input, timestep=1, return_dict=True)
        posterior = encoded.latent_dist
        posterior_mean = posterior.mean
        posterior_logvar = posterior.logvar
        latent = posterior.mode()
        scaling_factor = float(getattr(vae.config, "scaling_factor", 0.18215))
        scaled_latent = latent * scaling_factor
        decode_input = scaled_latent / scaling_factor
        decoded = vae.decode(decode_input, return_dict=True).sample
        clamped = decoded.clamp(-1.0, 1.0)

    encode_stats = stats(encode_input)
    mean_stats = stats(posterior_mean)
    logvar_stats = stats(posterior_logvar)
    latent_stats = stats(latent)
    scaled_stats = stats(scaled_latent)
    decode_input_stats = stats(decode_input)
    decoded_stats = stats(decoded)
    clamped_stats = stats(clamped)

    payload.update({
        "status": "PASS",
        "diffusers_available": True,
        "diffusers_version": getattr(diffusers, "__version__", "unknown"),
        "diffusers_file": getattr(diffusers, "__file__", "unknown"),
        "overlay_used": bool(os.environ.get("TADSR_LIVE_PYTHONPATH_OVERLAY")),
        "selected_device": selected_device,
        "torch_version": torch.__version__,

        "encode_input_shape": encode_stats["shape"],
        "encode_input_dtype": encode_stats["dtype"],
        "encode_input_min": encode_stats["min"],
        "encode_input_max": encode_stats["max"],
        "encode_input_mean": encode_stats["mean"],
        "encode_input_std": encode_stats["std"],
        "encode_input_finite": encode_stats["finite"],
        "encode_input_seed": seed,
        "encode_input_generation_policy": "torch.rand uniform in [-1, 1], in memory only, no tensor file saved",
        "encode_input_range": [encode_stats["min"], encode_stats["max"]],
        "encode_input_stats": encode_stats,

        "encoder_forward_called": True,
        "encoder_hook_used": bool(hasattr(getattr(vae, "encoder", None), "original_forward")),
        "encoder_tiled_used": bool(getattr(vae, "use_tiling", False)),
        "tiled_encoder_used": bool(getattr(vae, "use_tiling", False)),
        "quant_conv_used": hasattr(vae, "quant_conv") and vae.quant_conv is not None,
        "posterior_object_type": type(posterior).__name__,
        "latent_distribution_type": type(posterior).__name__,
        "posterior_mean_shape": mean_stats["shape"],
        "posterior_mean_dtype": mean_stats["dtype"],
        "posterior_mean_min": mean_stats["min"],
        "posterior_mean_max": mean_stats["max"],
        "posterior_mean_mean": mean_stats["mean"],
        "posterior_mean_std": mean_stats["std"],
        "posterior_logvar_shape": logvar_stats["shape"],
        "posterior_logvar_dtype": logvar_stats["dtype"],
        "posterior_logvar_min": logvar_stats["min"],
        "posterior_logvar_max": logvar_stats["max"],
        "posterior_logvar_mean": logvar_stats["mean"],
        "posterior_logvar_std": logvar_stats["std"],
        "posterior_sample_policy": "not used for decode; deterministic posterior.mode() used",
        "posterior_mode_policy": "posterior.mode() used to avoid stochastic tensor export",
        "fixed_epsilon_used": False,
        "posterior_stats": {"mean": mean_stats, "logvar": logvar_stats},
        "latent_shape": latent_stats["shape"],
        "latent_dtype": latent_stats["dtype"],
        "latent_min": latent_stats["min"],
        "latent_max": latent_stats["max"],
        "latent_mean": latent_stats["mean"],
        "latent_std": latent_stats["std"],
        "latent_finite": latent_stats["finite"],
        "latent_stats": latent_stats,

        "scaling_factor": scaling_factor,
        "scaled_latent_shape": scaled_stats["shape"],
        "scaled_latent_dtype": scaled_stats["dtype"],
        "scaled_latent_min": scaled_stats["min"],
        "scaled_latent_max": scaled_stats["max"],
        "scaled_latent_mean": scaled_stats["mean"],
        "scaled_latent_std": scaled_stats["std"],
        "scaled_latent_finite": scaled_stats["finite"],
        "scaled_latent_stats": scaled_stats,

        "decode_input_shape": decode_input_stats["shape"],
        "decode_input_dtype": decode_input_stats["dtype"],
        "post_quant_conv_used": hasattr(vae, "post_quant_conv") and vae.post_quant_conv is not None,
        "decoder_forward_called": True,
        "decoder_hook_used": bool(hasattr(getattr(vae, "decoder", None), "original_forward")),
        "decoder_tiled_used": bool(getattr(vae, "use_tiling", False)),
        "tiled_decoder_used": bool(getattr(vae, "use_tiling", False)),
        "decoder_original_forward_used": bool(hasattr(getattr(vae, "decoder", None), "original_forward")),
        "original_forward_used": bool(hasattr(getattr(vae, "decoder", None), "original_forward")),
        "decode_output_shape": decoded_stats["shape"],
        "decode_output_dtype": decoded_stats["dtype"],
        "decode_output_min": decoded_stats["min"],
        "decode_output_max": decoded_stats["max"],
        "decode_output_mean": decoded_stats["mean"],
        "decode_output_std": decoded_stats["std"],
        "decode_output_finite": decoded_stats["finite"],
        "decode_output_range": [decoded_stats["min"], decoded_stats["max"]],
        "decode_output_stats": decoded_stats,
        "clamp_policy": "torch.clamp(decoded, -1, 1), metadata only",
        "clamped_output_min": clamped_stats["min"],
        "clamped_output_max": clamped_stats["max"],
        "clamped_output_mean": clamped_stats["mean"],
        "clamped_output_std": clamped_stats["std"],
        "clamped_output_finite": clamped_stats["finite"],
        "summary_stats": {
            "encode_input": encode_stats,
            "posterior_mean": mean_stats,
            "posterior_logvar": logvar_stats,
            "latent": latent_stats,
            "scaled_latent": scaled_stats,
            "decode_input": decode_input_stats,
            "decode_output": decoded_stats,
            "clamped_output": clamped_stats,
        },
    })
except Exception as exc:
    payload.update({
        "status": "BLOCKED",
        "error": repr(exc),
        "traceback_tail": traceback.format_exc()[-4000:],
    })

print(json.dumps(payload, ensure_ascii=False))
"""
    env = overlay_env(pythonpath_overlay, official_repo)
    env.update(
        {
            "TADSR_LIVE_OFFICIAL_WEIGHTS": official_weights,
            "TADSR_LIVE_HEIGHT": str(height),
            "TADSR_LIVE_WIDTH": str(width),
            "TADSR_LIVE_SEED": str(seed),
            "TADSR_LIVE_DEVICE": device,
            "TADSR_LIVE_CPU_ONLY": "1" if cpu_only else "0",
            "TADSR_LIVE_PYTHONPATH_OVERLAY": pythonpath_overlay or "",
        }
    )
    proc = subprocess.run(
        [official_python, "-c", textwrap.dedent(code)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=300,
        env=env,
    )
    if proc.returncode != 0:
        return {"status": "BLOCKED", "error": proc.stdout[-4000:]}
    try:
        result = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {"status": "BLOCKED", "error": f"could_not_parse_live_metadata: {exc!r}", "stdout_tail": proc.stdout[-4000:]}
    return result


def find_candidates(root: Path, hints: list[str], limit: int = 30) -> list[str]:
    if not root.exists():
        return []
    out: list[str] = []
    for path in root.rglob("*"):
        if len(out) >= limit:
            break
        rel = str(path.relative_to(root)).replace("\\", "/")
        low = rel.lower()
        if any(hint in low for hint in hints):
            out.append(rel)
    return out


def blocked_value() -> str:
    return "blocked_or_not_executed"


def write(out_dir: Path, payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "timevae_production_oracle_metadata.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# TimeVAE production oracle metadata",
        "",
        f"`TADSR_TIMEVAE_PRODUCTION_ORACLE_METADATA: {payload['status']}`",
        "",
        f"- Metadata only: `{payload['metadata_only']}`",
        f"- Official env status: `{payload['official_env_status']}`",
        f"- Dependency status: `{payload['dependency_status']}`",
        f"- Official torch version: `{payload['official_torch_version']}`",
        f"- Diffusers available: `{payload['diffusers_available']}`",
        f"- Overlay used: `{payload.get('overlay_used')}`",
        f"- Strict env modified: `{payload.get('strict_env_modified')}`",
        f"- Forward executed: `{payload['forward_executed']}`",
        f"- Raw tensor saved: `{payload['raw_tensor_saved']}`",
        f"- Image/video output generated: `{payload['image_or_video_output_generated']}`",
        f"- Ready for TimeVAE preflight: `{payload['ready_for_timevae_preflight']}`",
        f"- Ready for one-step contract: `{payload['ready_for_one_step_contract']}`",
        "",
        "## Required metadata fields",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    for key in [
        "encode_input_shape",
        "encode_input_dtype",
        "encode_input_range",
        "posterior_mean_shape",
        "posterior_logvar_shape",
        "latent_shape",
        "scaling_factor",
        "scaled_latent_shape",
        "decode_input_shape",
        "decode_output_shape",
        "clamp_policy",
        "encoder_hook_path",
        "decoder_hook_path",
    ]:
        lines.append(f"| `{key}` | `{payload.get(key)}` |")
    if payload["dependency_blockers"]:
        lines += ["", "## Dependency blockers", ""]
        for blocker in payload["dependency_blockers"]:
            lines.append(f"- {blocker}")
    if payload["partial_reasons"]:
        lines += ["", "## Partial reasons", ""]
        for reason in payload["partial_reasons"]:
            lines.append(f"- {reason}")
    if payload["blockers"]:
        lines += ["", "## Blockers", ""]
        for blocker in payload["blockers"]:
            lines.append(f"- {blocker}")
    lines += [
        "",
        "## Safety policy",
        "",
        "- This metadata exporter did not run full TADSR inference.",
        "- This metadata exporter did not run the denoising loop.",
        "- This metadata exporter did not generate restored images or videos.",
        "- Raw tensor saving is disabled unless explicitly requested, and local tensor paths are git-ignored.",
    ]
    (out_dir / "timevae_production_oracle_metadata.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo")
    parser.add_argument("--official-weights")
    parser.add_argument("--official-python")
    parser.add_argument("--output-dir", default="experiments/production_completion/timevae_full")
    parser.add_argument("--metadata-only", default="1")
    parser.add_argument("--save-local-tensors", default="0")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cpu-only", default="0")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--pythonpath-overlay")
    parser.add_argument("--run-live-forward-metadata", default="0")
    args = parser.parse_args()

    env = load_json(ROOT / "experiments" / "production_completion" / "env" / "official_env_resolution.json")
    dep = load_json(ROOT / "experiments" / "production_completion" / "env" / "official_runtime_dependency_diagnosis.json")
    audit = load_json(ROOT / "experiments" / "production_completion" / "timevae_full" / "official_timevae_full_path_audit.json")
    gap = load_json(ROOT / "experiments" / "timevae_full_alignment_gap_analysis.json")

    official_repo = Path(args.official_repo or env.get("official_repo") or os.environ.get("TADSR_OFFICIAL_REPO", ""))
    official_weights = Path(args.official_weights or env.get("official_weights") or os.environ.get("TADSR_OFFICIAL_WEIGHTS", ""))
    official_python = Path(args.official_python or env.get("official_python") or os.environ.get("TADSR_OFFICIAL_PYTHON", ""))
    env_status = "PASS" if official_repo.exists() and official_weights.exists() and official_python.exists() else str(env.get("status", "BLOCKED"))

    overlay = args.pythonpath_overlay
    torch_import = run_import(str(official_python), "torch", overlay, str(official_repo))
    diffusers_import = run_import(str(official_python), "diffusers", overlay, str(official_repo))
    diffusers_plain_import = run_import(str(official_python), "diffusers")
    official_torch_version = torch_import.get("version") if torch_import.get("status") == "PASS" else None
    dependency_blockers: list[str] = []
    if env_status != "PASS":
        dependency_blockers.append("official_repo_or_weights_or_python_missing")
    if torch_import.get("status") != "PASS":
        dependency_blockers.append("official_python_cannot_import_torch")
    if diffusers_import.get("status") != "PASS":
        dependency_blockers.append("official_python_cannot_import_diffusers")
    for blocker in dep.get("dependency_blockers", []):
        if blocker == "official_python_cannot_import_diffusers" and diffusers_import.get("status") == "PASS":
            continue
        if blocker not in dependency_blockers:
            dependency_blockers.append(str(blocker))
    dependency_status = "PASS" if not dependency_blockers else ("PARTIAL" if torch_import.get("status") == "PASS" and env_status == "PASS" else "BLOCKED")

    repo_candidates = find_candidates(official_repo, ["time", "vae", "autoencoder", "tadsr"])
    weight_candidates = find_candidates(official_weights, ["vae", "time_vae", "unet", "lora", "scheduler"])
    forward_executed = False
    live_forward_requested = str(args.run_live_forward_metadata) == "1"
    save_local = str(args.save_local_tensors) == "1"
    metadata_only = str(args.metadata_only) == "1"
    cpu_only = str(args.cpu_only) == "1"
    live_metadata: dict[str, Any] = {}
    if live_forward_requested and not save_local:
        live_metadata = run_live_timevae_metadata(
            str(official_python),
            str(official_repo),
            str(official_weights),
            overlay,
            args.height,
            args.width,
            args.seed,
            args.device,
            cpu_only,
        )
        forward_executed = live_metadata.get("status") == "PASS"

    partial_reasons: list[str] = []
    blockers: list[str] = []
    if env_status != "PASS":
        blockers.append("official environment paths are not fully available")
    if diffusers_import.get("status") != "PASS":
        partial_reasons.append("official Python cannot import diffusers, so live production TimeVAE module instantiation is not attempted")
    if live_forward_requested and live_metadata.get("status") != "PASS":
        partial_reasons.append(f"controlled TimeVAE live encode/decode metadata forward did not pass: {live_metadata.get('error', live_metadata.get('status'))}")
    if not forward_executed:
        partial_reasons.append("metadata-only repair did not execute TimeVAE encode/decode forward")
    if not repo_candidates:
        blockers.append("no TimeVAE/TADSR source candidates found in official repo")
    if not weight_candidates:
        blockers.append("no VAE/UNet/LoRA/scheduler weight candidates found in official weights")

    # Conservative metadata: path/policy fields can be documented from audits, but
    # tensor shape/stat fields stay blocked until a controlled live forward export exists.
    field_blocked = blocked_value()
    ready_for_preflight = dependency_status in {"PASS", "PARTIAL"} and bool(audit)
    tensor_metadata_complete = bool(forward_executed)
    ready_for_one_step = bool(dependency_status == "PASS" and tensor_metadata_complete)
    status = "PASS" if ready_for_one_step else ("PARTIAL" if ready_for_preflight else "BLOCKED")

    payload: dict[str, Any] = {
        "status_marker": "TADSR_TIMEVAE_PRODUCTION_ORACLE_METADATA",
        "status": status,
        "official_env_status": env_status,
        "official_repo": str(official_repo),
        "official_weights": str(official_weights),
        "official_python": str(official_python),
        "pythonpath_overlay": overlay,
        "overlay_used": bool(overlay),
        "strict_env_modified": False,
        "official_torch_version": official_torch_version,
        "diffusers_available": diffusers_import.get("status") == "PASS",
        "diffusers_available_without_overlay": diffusers_plain_import.get("status") == "PASS",
        "dependency_status": dependency_status,
        "dependency_blockers": dependency_blockers,
        "dependency_imports": {
            "torch": torch_import,
            "diffusers": diffusers_import,
            "diffusers_without_overlay": diffusers_plain_import,
        },
        "official_repo_timevae_candidates": repo_candidates,
        "official_weight_candidates": weight_candidates,
        "metadata_only": metadata_only,
        "run_live_forward_metadata": live_forward_requested,
        "num_samples": args.num_samples,
        "height": args.height,
        "width": args.width,
        "seed": args.seed,
        "cpu_only": cpu_only,
        "device": args.device,
        "forward_executed": forward_executed,
        "full_inference_executed": False,
        "full_tadsr_inference_executed": False,
        "denoising_loop_executed": False,
        "raw_tensor_saved": False,
        "raw_tensors_committed": False,
        "image_or_video_output_generated": False,
        "image_or_video_generated": False,
        "local_tensor_dir_ignored": True,
        "encode_input_shape": field_blocked,
        "encode_input_dtype": field_blocked,
        "encode_input_range": field_blocked,
        "encode_input_stats": {},
        "encoder_hook_path": audit.get("encoder_uses_vae_hook", "requires live production metadata"),
        "encoder_tiled_used": audit.get("encoder_tiled_path_reachable", "requires live production metadata"),
        "tiled_encoder_used": audit.get("encoder_tiled_path_reachable", "requires live production metadata"),
        "quant_conv_used": audit.get("quant_conv_used", "requires live production metadata"),
        "latent_distribution_type": "requires live production metadata",
        "posterior_mean_shape": field_blocked,
        "posterior_logvar_shape": field_blocked,
        "posterior_sample_policy": audit.get("sampling_policy", gap.get("not_complete_subpaths", "requires live production metadata")),
        "fixed_epsilon_used": "requires live production metadata",
        "latent_shape": field_blocked,
        "latent_dtype": field_blocked,
        "latent_stats": {},
        "posterior_stats": {},
        "scaling_factor": audit.get("scaling_policy", "requires live production metadata"),
        "scaled_latent_shape": field_blocked,
        "scaled_latent_stats": {},
        "decode_input_shape": field_blocked,
        "decode_input_dtype": field_blocked,
        "post_quant_conv_used": audit.get("post_quant_conv_used", "requires live production metadata"),
        "decoder_hook_path": audit.get("decoder_uses_vae_hook", "requires live production metadata"),
        "decoder_tiled_used": audit.get("decoder_tiled_path_reachable", "requires live production metadata"),
        "tiled_decoder_used": audit.get("decoder_tiled_path_reachable", "requires live production metadata"),
        "decoder_original_forward_used": audit.get("original_forward_called", "requires live production metadata"),
        "original_forward_used": audit.get("original_forward_called", "requires live production metadata"),
        "decode_output_shape": field_blocked,
        "decode_output_dtype": field_blocked,
        "decode_output_range": field_blocked,
        "decode_output_stats": {},
        "clamp_policy": "requires live production metadata",
        "summary_stats": {},
        "sha256": {},
        "local_tensor_paths": [],
        "partial_reasons": partial_reasons,
        "blockers": blockers,
        "ready_for_timevae_preflight": ready_for_preflight,
        "ready_for_one_step_contract": ready_for_one_step,
        "tensor_metadata_complete": tensor_metadata_complete,
        "save_local_tensors_requested": save_local,
        "next_action": (
            "Install or select an official environment with required runtime dependencies, then run a controlled metadata-only "
            "TimeVAE encode/decode export. Keep raw tensors in ignored local_tensors if they are needed for a later one-step gate."
            if not ready_for_one_step
            else "Rerun TimeVAE preflight and full inference metadata contract."
        ),
    }
    if live_metadata:
        # Live fields intentionally override blocked placeholders. Safety fields
        # remain explicit so downstream validators can reject accidental scope creep.
        payload.update(live_metadata)
        payload.update(
            {
                "status_marker": "TADSR_TIMEVAE_PRODUCTION_ORACLE_METADATA",
                "status": status,
                "metadata_only": metadata_only,
                "run_live_forward_metadata": live_forward_requested,
                "save_local_tensors_requested": save_local,
                "forward_executed": forward_executed,
                "full_inference_executed": False,
                "full_tadsr_inference_executed": False,
                "denoising_loop_executed": False,
                "unet_called": False,
                "scheduler_loop_called": False,
                "raw_tensor_saved": False,
                "raw_tensors_committed": False,
                "image_or_video_output_generated": False,
                "image_or_video_generated": False,
                "local_tensor_paths": [],
                "partial_reasons": partial_reasons,
                "blockers": blockers,
                "ready_for_timevae_preflight": ready_for_preflight,
                "ready_for_one_step_contract": ready_for_one_step,
                "tensor_metadata_complete": tensor_metadata_complete,
                "dependency_status": dependency_status,
                "dependency_blockers": dependency_blockers,
            }
        )
    write(ROOT / args.output_dir, payload)
    print(f"TADSR_TIMEVAE_PRODUCTION_ORACLE_METADATA: {status}")
    return 0 if status in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
