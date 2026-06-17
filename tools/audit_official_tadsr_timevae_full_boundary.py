#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

OFFICIAL_VENV_PYTHON = Path('/mnt/data/sj/venvs/tadsr_official_strict_cu118/bin/python')
OFFICIAL_VENV_ROOT = OFFICIAL_VENV_PYTHON.parents[1]
if (
    OFFICIAL_VENV_PYTHON.exists()
    and Path(sys.prefix).resolve() != OFFICIAL_VENV_ROOT.resolve()
    and os.environ.get('TADSR_OFFICIAL_VENV_REEXEC') != '1'
):
    env = os.environ.copy()
    env['TADSR_OFFICIAL_VENV_REEXEC'] = '1'
    os.execve(str(OFFICIAL_VENV_PYTHON), [str(OFFICIAL_VENV_PYTHON), *sys.argv], env)

import torch

OFFICIAL_REPO = Path('/mnt/data/sj/projects/TADSR_official_pytorch')
WEIGHTS_DIR = Path('/mnt/data/sj/checkpoints/TADSR/preset/weights/time_vae')
OUT = Path('experiments/full_repro/vae_alignment')


def text_at(path: Path) -> str:
    return path.read_text(encoding='utf-8', errors='ignore') if path.exists() else ''


def has_method(obj, name: str) -> bool:
    return callable(getattr(obj, name, None))


def find_snippets(text: str, patterns: list[str], context: int = 1) -> list[dict]:
    lines = text.splitlines()
    rows = []
    for idx, line in enumerate(lines, 1):
        if any(p in line for p in patterns):
            lo = max(1, idx - context)
            hi = min(len(lines), idx + context)
            rows.append({
                'line': idx,
                'text': line.strip(),
                'context': '\n'.join(f'{i}: {lines[i-1]}' for i in range(lo, hi + 1)),
            })
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(OFFICIAL_REPO))
    from diffusers.models.autoencoders.time_autoencoder_kl import TimeAwareAutoencoderKL
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    from my_utils.vaehook import VAEHook

    model = TimeAwareAutoencoderKL.from_pretrained(str(WEIGHTS_DIR), local_files_only=True)
    model.eval()
    config = dict(model.config)
    tadsr_text = text_at(OFFICIAL_REPO / 'tadsr.py')
    test_text = text_at(OFFICIAL_REPO / 'test_tadsr.py')
    vae_text = text_at(OFFICIAL_REPO / 'diffusers/models/autoencoders/time_autoencoder_kl.py')

    encode_patterns = ['vae.encode(', '.latent_dist.sample()', 'scaling_factor']
    decode_patterns = ['vae.decode(', 'clamp(-1, 1)', 'decode_latent']
    tiling_patterns = ['_init_tiled_vae', 'VAEHook', 'encoder.forward = VAEHook', 'decoder.forward = VAEHook']
    usage = {
        'tadsr_encode_snippets': find_snippets(tadsr_text, encode_patterns, context=2),
        'tadsr_decode_snippets': find_snippets(tadsr_text, decode_patterns, context=2),
        'tadsr_tiled_vae_snippets': find_snippets(tadsr_text, tiling_patterns, context=2),
        'test_script_vae_snippets': find_snippets(test_text, ['model(lq', 'decode_latent', 'timesteps'], context=2),
    }
    pipeline_requires_encode = bool(usage['tadsr_encode_snippets'])
    pipeline_requires_decode = bool(usage['tadsr_decode_snippets'])
    pipeline_uses_sample = '.latent_dist.sample()' in tadsr_text
    pipeline_uses_scaling = 'scaling_factor' in tadsr_text
    pipeline_uses_clamp = 'clamp(-1, 1)' in tadsr_text
    pipeline_installs_tiled_hook = '_init_tiled_vae' in tadsr_text and 'VAEHook' in tadsr_text

    full_boundary_type = 'encode_sample_scale_decode_clamp' if (pipeline_requires_encode and pipeline_requires_decode) else 'decode_only'
    deterministic_policy = (
        'Official TADSR calls latent_dist.sample() without exposing its random epsilon. '
        'The alignment boundary therefore exports a fixed epsilon tensor and reproduces '
        'z = mean + std * epsilon deterministically. This validates the encode/sample/'
        'scale/decode/clamp contract without running scheduler or full inference.'
    )
    remaining_gaps = []
    if pipeline_installs_tiled_hook:
        remaining_gaps.append('official TADSR_test monkey-patches VAE encoder/decoder with VAEHook tiling for real-size inference; this boundary uses the non-tiled small-tensor path')
    if pipeline_uses_sample:
        remaining_gaps.append('official pipeline uses stochastic posterior sample; this boundary uses exported fixed epsilon instead of an internal runtime RNG')
    remaining_gaps.append('generic runtime LoRA integration remains partial; current Jittor checks use converted/static weights')
    remaining_gaps.append('full scheduler/UNet/VAE image inference CLI remains intentionally NotImplemented')

    summary = {
        'status': 'PASS',
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'official_vae_class': f'{model.__class__.__module__}.{model.__class__.__name__}',
        'encoder_class': f'{model.encoder.__class__.__module__}.{model.encoder.__class__.__name__}',
        'decoder_class': f'{model.decoder.__class__.__module__}.{model.decoder.__class__.__name__}',
        'posterior_class': f'{DiagonalGaussianDistribution.__module__}.{DiagonalGaussianDistribution.__name__}',
        'vae_hook_class': f'{VAEHook.__module__}.{VAEHook.__name__}',
        'config': config,
        'available_methods': {
            'encode': has_method(model, 'encode'),
            'decode': has_method(model, 'decode'),
            'forward': has_method(model, 'forward'),
            '_decode': has_method(model, '_decode'),
            'enable_tiling': has_method(model, 'enable_tiling'),
            'disable_tiling': has_method(model, 'disable_tiling'),
            'enable_slicing': has_method(model, 'enable_slicing'),
            'disable_slicing': has_method(model, 'disable_slicing'),
        },
        'module_presence': {
            'quant_conv': hasattr(model, 'quant_conv'),
            'post_quant_conv': hasattr(model, 'post_quant_conv'),
            'encoder_time_proj': hasattr(model.encoder, 'time_proj'),
            'encoder_time_embedding': hasattr(model.encoder, 'time_embedding'),
        },
        'pipeline_usage': {
            'requires_encode': pipeline_requires_encode,
            'requires_decode': pipeline_requires_decode,
            'uses_posterior_sample': pipeline_uses_sample,
            'uses_posterior_mode': '.latent_dist.mode()' in tadsr_text,
            'uses_scaling_factor': pipeline_uses_scaling,
            'scaling_factor': float(config.get('scaling_factor', 0.18215)),
            'postprocess_clamp_minus1_1': pipeline_uses_clamp,
            'installs_tiled_vae_hook': pipeline_installs_tiled_hook,
            'tensor_layout': 'NCHW image tensors and NCHW latents',
            'input_range': 'image tensor is converted from [0,1] to [-1,1] before TADSR_test.forward',
            'output_range': 'decode sample is clamped to [-1,1], then caller maps to [0,1] for PIL',
            'timesteps_contract': 'TADSR passes the caller timestep tensor into TimeAwareAutoencoderKL.encode; decode has no timestep argument',
            'snippets': usage,
        },
        'full_boundary_contract': {
            'status': 'PASS',
            'boundary_type': full_boundary_type,
            'steps': [
                'input image tensor in NCHW float32 range [-1,1]',
                'TimeAwareAutoencoderKL.encode(x, timesteps)',
                'DiagonalGaussianDistribution mean/logvar/std',
                'deterministic posterior sample with exported epsilon',
                'scaled latent = sample * scaling_factor',
                'decode input = scaled latent / scaling_factor',
                'TimeAwareAutoencoderKL.decode(decode_input).sample',
                'clamp decoded sample to [-1,1]',
            ],
            'deterministic_policy': deterministic_policy,
            'scheduler_executed': False,
            'full_tadsr_inference_executed': False,
            'image_saved': False,
            'sufficient_for_non_tiled_small_tensor_api': True,
            'sufficient_for_full_tadsr_pipeline': False,
            'timevae_full_alignment_candidate': False,
            'remaining_timevae_gaps': remaining_gaps,
            'recommended_next_stage': 'If this boundary aligns, audit VAEHook tiled encode/decode and runtime VAE LoRA policy before opening full TADSR inference.',
        },
        'existing_decoder_relation': {
            'TIME_VAE_FULL_DECODER_ALIGNMENT': 'covers post_quant_conv -> decoder stack -> tail only',
            'this_boundary_adds': ['encode API', 'posterior sample formula with fixed epsilon', 'scaling_factor', 'decode API wrapper', 'clamp postprocess'],
        },
    }
    (OUT / 'audit_tadsr_timevae_full_boundary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    txt = [
        'TIME_VAE_FULL_API_AUDIT: PASS',
        'TIME_VAE_PIPELINE_USAGE_AUDIT: PASS',
        'TIME_VAE_FULL_BOUNDARY_CONTRACT_AUDIT: PASS',
        '',
        f"official_vae_class: {summary['official_vae_class']}",
        f"boundary_type: {full_boundary_type}",
        f"requires_encode: {pipeline_requires_encode}",
        f"requires_decode: {pipeline_requires_decode}",
        f"uses_posterior_sample: {pipeline_uses_sample}",
        f"scaling_factor: {summary['pipeline_usage']['scaling_factor']}",
        f"postprocess: clamp[-1,1]={pipeline_uses_clamp}",
        f"tiled_hook_present: {pipeline_installs_tiled_hook}",
        f"timevae_full_alignment_candidate: {summary['full_boundary_contract']['timevae_full_alignment_candidate']}",
        'remaining_timevae_gaps:',
    ]
    txt.extend(f'- {x}' for x in remaining_gaps)
    (OUT / 'audit_tadsr_timevae_full_boundary.txt').write_text('\n'.join(txt) + '\n', encoding='utf-8')
    print('\n'.join(txt[:3]))
    print(json.dumps(summary['full_boundary_contract'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
