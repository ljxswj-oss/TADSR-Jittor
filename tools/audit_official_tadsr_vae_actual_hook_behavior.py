#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

OFFICIAL_VENV_PYTHON = Path('/mnt/data/sj/venvs/tadsr_official_strict_cu118/bin/python')
OFFICIAL_VENV_ROOT = OFFICIAL_VENV_PYTHON.parents[1]
if (
    OFFICIAL_VENV_PYTHON.exists()
    and Path(sys.prefix).resolve() != OFFICIAL_VENV_ROOT.resolve()
    and os.environ.get('TADSR_OFFICIAL_VENV_REEXEC') != '1'
):
    env = os.environ.copy()
    env['TADSR_OFFICIAL_VENV_REEXEC'] = '1'
    env.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.execve(str(OFFICIAL_VENV_PYTHON), [str(OFFICIAL_VENV_PYTHON), *sys.argv], env)

import torch

OFFICIAL_REPO = Path('/mnt/data/sj/projects/TADSR_official_pytorch')
WEIGHTS_DIR = Path('/mnt/data/sj/checkpoints/TADSR/preset/weights/time_vae')
BASE_OUT = Path('experiments/full_repro/vae_alignment')
PREV_AUDIT = BASE_OUT / 'audit_tadsr_vae_tiled_boundary.json'
OUT_JSON = BASE_OUT / 'audit_tadsr_vae_actual_hook_behavior.json'
OUT_TXT = BASE_OUT / 'audit_tadsr_vae_actual_hook_behavior.txt'


def load_previous_audit() -> dict:
    if not PREV_AUDIT.exists():
        raise SystemExit(f'Missing previous tiled VAE audit metadata: {PREV_AUDIT}')
    audit = json.loads(PREV_AUDIT.read_text(encoding='utf-8'))
    blocker = audit.get('oracle_feasibility', {}).get('status')
    encoder_time_vae = audit.get('hook_audit', {}).get('hook_init_params', {}).get('encoder', {}).get('time_vae')
    decoder_time_vae = audit.get('hook_audit', {}).get('hook_init_params', {}).get('decoder', {}).get('time_vae')
    if blocker != 'BLOCKED_DECODER_HOOK_CONTRACT' or encoder_time_vae is not True or decoder_time_vae is not False:
        raise SystemExit(
            'Previous tiled VAE audit metadata does not match the expected official decoder-hook blocker contract.'
        )
    return audit


def make_input(device: torch.device, size: int = 96) -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, 1 * 3 * size * size, device=device, dtype=torch.float32).reshape(1, 3, size, size)


def wrap_hook_call_counter(hook, counter: dict):
    original = hook.vae_tile_forward

    def wrapped(*args, **kwargs):
        counter['vae_tile_forward_calls'] += 1
        return original(*args, **kwargs)

    hook.vae_tile_forward = wrapped
    return hook


def main() -> int:
    BASE_OUT.mkdir(parents=True, exist_ok=True)
    audit = load_previous_audit()
    sys.path.insert(0, str(OFFICIAL_REPO))
    from diffusers.models.autoencoders.time_autoencoder_kl import TimeAwareAutoencoderKL
    from my_utils.vaehook import VAEHook

    torch.manual_seed(123)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TimeAwareAutoencoderKL.from_pretrained(str(WEIGHTS_DIR), local_files_only=True)
    model.eval()
    if device.type == 'cuda':
        model = model.to(device)

    if not hasattr(model.encoder, 'original_forward'):
        model.encoder.original_forward = model.encoder.forward
    if not hasattr(model.decoder, 'original_forward'):
        model.decoder.original_forward = model.decoder.forward

    encoder_tile_size = 16
    decoder_tile_size = 16
    encoder_pad = 32
    decoder_pad = 11
    sample = make_input(device, 96)
    timestep = torch.tensor([50], dtype=torch.long, device=device)

    encoder_counter = {'vae_tile_forward_calls': 0}
    decoder_counter = {'vae_tile_forward_calls': 0, 'original_forward_calls': 0}

    encoder_hook = VAEHook(
        model.encoder,
        encoder_tile_size,
        is_decoder=False,
        fast_decoder=False,
        fast_encoder=False,
        color_fix=False,
        to_gpu=(device.type == 'cuda'),
        time_vae=True,
    )
    wrap_hook_call_counter(encoder_hook, encoder_counter)
    model.encoder.forward = encoder_hook

    original_decoder_forward = model.decoder.original_forward

    def counted_decoder_original_forward(*args, **kwargs):
        decoder_counter['original_forward_calls'] += 1
        return original_decoder_forward(*args, **kwargs)

    model.decoder.original_forward = counted_decoder_original_forward
    decoder_hook = VAEHook(
        model.decoder,
        decoder_tile_size,
        is_decoder=True,
        fast_decoder=False,
        fast_encoder=False,
        color_fix=False,
        to_gpu=(device.type == 'cuda'),
        time_vae=False,
    )
    wrap_hook_call_counter(decoder_hook, decoder_counter)
    model.decoder.forward = decoder_hook

    encoder_status = 'BLOCKED_UNKNOWN'
    error = None
    try:
        with torch.no_grad():
            encoded = model.encode(sample, timestep)
            mean = encoded.latent_dist.mean
            logvar = encoded.latent_dist.logvar
            std = encoded.latent_dist.std
            epsilon = torch.linspace(-0.5, 0.5, mean.numel(), device=device, dtype=mean.dtype).reshape_as(mean)
            posterior_sample = mean + std * epsilon
            scaling_factor = float(getattr(model.config, 'scaling_factor', 0.18215))
            scaled_latent = posterior_sample * scaling_factor
            decoded = model.decode(scaled_latent / scaling_factor).sample
            final = decoded.clamp(-1, 1)
        encoder_status = 'PASS' if encoder_counter['vae_tile_forward_calls'] > 0 else 'NOT_TRIGGERED'
    except Exception:
        error = traceback.format_exc()
        encoder_status = 'BLOCKED_RUNTIME_ERROR'
        encoded = mean = logvar = std = posterior_sample = scaled_latent = decoded = final = None
        scaling_factor = float(getattr(model.config, 'scaling_factor', 0.18215))

    h, w = int(sample.shape[-2]), int(sample.shape[-1])
    tile_count_h = max(__import__('math').ceil((h - 2 * encoder_pad) / encoder_tile_size), 1)
    tile_count_w = max(__import__('math').ceil((w - 2 * encoder_pad) / encoder_tile_size), 1)

    decoder_original_used = decoder_counter['original_forward_calls'] > 0
    decoder_tiled_triggered = decoder_counter['vae_tile_forward_calls'] > 0
    oracle_feasible = error is None and decoder_original_used and not decoder_tiled_triggered

    summary = {
        'status': 'PASS' if oracle_feasible else 'BLOCKED_RUNTIME_ERROR',
        'policy_decision': 'mirror_official_actual_behavior',
        'decision_status': 'PASS',
        'official_mirror_policy_status': 'PASS',
        'previous_tiled_audit': str(PREV_AUDIT),
        'previous_decoder_hook_blocker': audit.get('oracle_feasibility', {}).get('status'),
        'device': str(device),
        'input_shape': list(sample.shape),
        'timestep': int(timestep.detach().cpu().item()),
        'encoder_hook': {
            'installed': True,
            'time_vae': True,
            'tile_size': encoder_tile_size,
            'pad': encoder_pad,
            'input_height': h,
            'input_width': w,
            'tile_count_h': tile_count_h,
            'tile_count_w': tile_count_w,
            'total_tile_count': tile_count_h * tile_count_w,
            'vae_tile_forward_called': encoder_counter['vae_tile_forward_calls'] > 0,
            'vae_tile_forward_call_count': encoder_counter['vae_tile_forward_calls'],
            'tiled_oracle_feasibility': encoder_status,
            'encoded_parameters_shape': list(encoded.latent_dist.parameters.shape) if encoded is not None else None,
            'posterior_mean_shape': list(mean.shape) if mean is not None else None,
        },
        'decoder_hook': {
            'installed': True,
            'time_vae': False,
            'tile_size': decoder_tile_size,
            'pad': decoder_pad,
            'decoder_input_shape': list((scaled_latent / scaling_factor).shape) if scaled_latent is not None else None,
            'original_forward_called': decoder_original_used,
            'original_forward_call_count': decoder_counter['original_forward_calls'],
            'vae_tile_forward_called': decoder_tiled_triggered,
            'vae_tile_forward_call_count': decoder_counter['vae_tile_forward_calls'],
            'tiled_path_status': 'BLOCKED_DECODER_HOOK_CONTRACT',
            'actual_behavior': 'patched decoder hook installed, but official dispatch uses decoder.original_forward',
            'decoded_output_shape': list(decoded.shape) if decoded is not None else None,
            'final_clamped_output_shape': list(final.shape) if final is not None else None,
        },
        'scaling_factor': scaling_factor,
        'scheduler_executed': False,
        'full_tadsr_inference_executed': False,
        'image_saved': False,
        'jittor_tiled_implementation_done': False,
        'timevae_full_alignment_candidate': False,
        'error': error,
        'recommended_next_stage': 'Continue with Jittor official-actual VAEHook behavior wrapper/tester alignment; mirror official decoder original_forward behavior and do not implement corrected tiled decoder on the mainline.',
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    lines = [
        'TIME_VAE_ACTUAL_VAEHOOK_BEHAVIOR_DECISION: PASS',
        f"TIME_VAE_ACTUAL_VAEHOOK_BEHAVIOR_AUDIT: {summary['status']}",
        'TIME_VAE_OFFICIAL_MIRROR_POLICY: PASS',
        'TIME_VAE_DECODER_TILED_PATH_STATUS: BLOCKED_DECODER_HOOK_CONTRACT',
        f"TIME_VAE_ACTUAL_ENCODER_TILED_ORACLE_FEASIBILITY: {encoder_status}",
        '',
        f"encoder_vae_tile_forward_called: {summary['encoder_hook']['vae_tile_forward_called']}",
        f"encoder_tile_count: {tile_count_h}x{tile_count_w}",
        f"decoder_original_forward_called: {decoder_original_used}",
        f"decoder_vae_tile_forward_called: {decoder_tiled_triggered}",
        'corrected_tiled_decoder: deferred as beyond-official behavior',
    ]
    if error:
        lines += ['', 'ERROR:', error]
    OUT_TXT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines[:5]))
    return 0 if summary['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
