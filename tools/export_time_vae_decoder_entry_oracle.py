#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

OFFICIAL_REPO = Path('/mnt/data/sj/projects/TADSR_official_pytorch')
WEIGHTS_DIR = Path('/mnt/data/sj/checkpoints/TADSR/preset/weights/time_vae')
BASE_OUT = Path('experiments/full_repro/time_vae_alignment')


def to_np(x):
    return x.detach().cpu().numpy().astype(np.float32)


def sha256_array(x: np.ndarray) -> str:
    arr = np.asarray(x)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def save_npy(out: Path, name: str, value: np.ndarray) -> None:
    np.save(out / f'{name}.npy', np.asarray(value, dtype=np.float32))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', type=int, default=32)
    parser.add_argument('--timestep', type=int, default=50)
    args = parser.parse_args()
    out = BASE_OUT / 'oracle_tensors_decoder_entry'
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(OFFICIAL_REPO))
    from diffusers.models.autoencoders.time_autoencoder_kl import TimeAwareAutoencoderKL
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

    torch.manual_seed(42)
    model = TimeAwareAutoencoderKL.from_pretrained(str(WEIGHTS_DIR), local_files_only=True)
    model.eval()
    enc = model.encoder
    blocks = enc.down_blocks
    mid = enc.mid_block

    with torch.no_grad():
        synthetic_latent_z = torch.linspace(-1.0, 1.0, steps=1 * 4 * 4 * 4, dtype=torch.float32).reshape(1, 4, 4, 4)
        synthetic_post_quant = model.post_quant_conv(synthetic_latent_z)
        synthetic_decoder_conv_in = model.decoder.conv_in(synthetic_post_quant)

        sample = torch.linspace(-1.0, 1.0, steps=1 * 3 * args.input_size * args.input_size, dtype=torch.float32).reshape(1, 3, args.input_size, args.input_size)
        timesteps = torch.tensor([args.timestep], dtype=torch.long)
        time_proj = enc.time_proj(timesteps)
        temb = enc.time_embedding(time_proj.to(dtype=list(enc.time_embedding.parameters())[0].dtype))
        hidden = enc.conv_in(sample)
        down_outputs = {}
        for i in range(4):
            hidden = blocks[i](hidden, temb=temb)
            down_outputs[f'downblock{i}_output'] = to_np(hidden)
        mid_out = mid(hidden, temb=temb)
        tail_norm = enc.conv_norm_out(mid_out)
        tail_act = enc.conv_act(tail_norm)
        tail_out = enc.conv_out(tail_act)
        moments = model.quant_conv(tail_out)
        manual_mean, manual_logvar_raw = torch.chunk(moments, 2, dim=1)
        manual_logvar = torch.clamp(manual_logvar_raw, -30.0, 20.0)
        latent_dist = DiagonalGaussianDistribution(moments)
        mode = latent_dist.mode()
        real_post_quant = model.post_quant_conv(mode)
        real_decoder_conv_in = model.decoder.conv_in(real_post_quant)

    arrays = {
        'synthetic_latent_z': to_np(synthetic_latent_z),
        'synthetic_post_quant_conv_output': to_np(synthetic_post_quant),
        'synthetic_decoder_conv_in_output': to_np(synthetic_decoder_conv_in),
        'real_synthetic_image_tensor': to_np(sample),
        'real_temb': to_np(temb),
        'real_encoder_mid_output': to_np(mid_out),
        'real_encoder_tail_output': to_np(tail_out),
        'real_moments': to_np(moments),
        'real_mean_manual': to_np(manual_mean),
        'real_logvar_raw_manual': to_np(manual_logvar_raw),
        'real_logvar_clamped_manual': to_np(manual_logvar),
        'real_latent_dist_mean': to_np(latent_dist.mean),
        'real_latent_dist_logvar': to_np(latent_dist.logvar),
        'real_latent_dist_mode': to_np(mode),
        'real_post_quant_conv_input': to_np(mode),
        'real_post_quant_conv_output': to_np(real_post_quant),
        'real_decoder_conv_in_output': to_np(real_decoder_conv_in),
    }
    arrays.update(down_outputs)
    for name, value in arrays.items():
        save_npy(out, name, value)

    np.savez_compressed(out / 'time_vae_decoder_entry_inputs.npz', **{
        'synthetic_latent_z': arrays['synthetic_latent_z'],
        'real_synthetic_image_tensor': arrays['real_synthetic_image_tensor'],
        'real_temb': arrays['real_temb'],
        'real_moments': arrays['real_moments'],
        'real_post_quant_conv_input': arrays['real_post_quant_conv_input'],
        'real_decoder_conv_in_input': arrays['real_post_quant_conv_output'],
    })
    np.savez_compressed(out / 'time_vae_decoder_entry_outputs.npz', **{
        'synthetic_post_quant_conv_output': arrays['synthetic_post_quant_conv_output'],
        'synthetic_decoder_conv_in_output': arrays['synthetic_decoder_conv_in_output'],
        'real_mean_manual': arrays['real_mean_manual'],
        'real_logvar_raw_manual': arrays['real_logvar_raw_manual'],
        'real_logvar_clamped_manual': arrays['real_logvar_clamped_manual'],
        'real_latent_dist_mean': arrays['real_latent_dist_mean'],
        'real_latent_dist_logvar': arrays['real_latent_dist_logvar'],
        'real_latent_dist_mode': arrays['real_latent_dist_mode'],
        'real_post_quant_conv_output': arrays['real_post_quant_conv_output'],
        'real_decoder_conv_in_output': arrays['real_decoder_conv_in_output'],
    })

    mean_diff = float(np.max(np.abs(arrays['real_mean_manual'] - arrays['real_latent_dist_mean'])))
    logvar_diff = float(np.max(np.abs(arrays['real_logvar_clamped_manual'] - arrays['real_latent_dist_logvar'])))
    mode_diff = float(np.max(np.abs(arrays['real_latent_dist_mean'] - arrays['real_latent_dist_mode'])))
    meta = {
        'status': 'PASS' if mean_diff == 0.0 and logvar_diff == 0.0 and mode_diff == 0.0 else 'FAIL',
        'official_class': f'{model.__class__.__module__}.{model.__class__.__name__}',
        'diagonal_gaussian_distribution_class': f'{DiagonalGaussianDistribution.__module__}.{DiagonalGaussianDistribution.__name__}',
        'sample_shape': list(sample.shape),
        'synthetic_latent_shape': list(synthetic_latent_z.shape),
        'timestep': int(timesteps.item()),
        'input_size': args.input_size,
        'split_rule': 'torch.chunk(moments, 2, dim=1)',
        'logvar_clamp_min': -30.0,
        'logvar_clamp_max': 20.0,
        'posterior_mode': 'mean',
        'uses_sampling': False,
        'path_a_order': ['synthetic_latent_z', 'post_quant_conv', 'decoder.conv_in'],
        'path_b_order': ['synthetic_image', 'encoder.conv_in', 'down_blocks.0..3', 'encoder.mid_block', 'encoder.conv_norm_out', 'encoder.conv_act', 'encoder.conv_out', 'quant_conv', 'split moments', 'mode(mean)', 'post_quant_conv', 'decoder.conv_in'],
        'post_quant_conv': {
            'weight_shape': list(model.post_quant_conv.weight.shape),
            'bias_shape': list(model.post_quant_conv.bias.shape),
            'kernel_size': list(model.post_quant_conv.kernel_size),
            'stride': list(model.post_quant_conv.stride),
            'padding': list(model.post_quant_conv.padding),
        },
        'decoder_conv_in': {
            'weight_shape': list(model.decoder.conv_in.weight.shape),
            'bias_shape': list(model.decoder.conv_in.bias.shape),
            'kernel_size': list(model.decoder.conv_in.kernel_size),
            'stride': list(model.decoder.conv_in.stride),
            'padding': list(model.decoder.conv_in.padding),
        },
        'consistency_checks': {
            'manual_mean_vs_distribution_mean_max_abs_diff': mean_diff,
            'manual_logvar_clamped_vs_distribution_logvar_max_abs_diff': logvar_diff,
            'mode_vs_mean_max_abs_diff': mode_diff,
        },
        'arrays': {name: {'shape': list(np.asarray(value).shape), 'sha256': sha256_array(value)} for name, value in arrays.items()},
    }
    (out / 'decoder_entry_oracle_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    md = [
        '# TimeAware VAE Decoder Entry Oracle Export',
        '',
        f"Status: **{meta['status']}**",
        '',
        '## Deterministic rules',
        '',
        '- Posterior moments are split with `torch.chunk(moments, 2, dim=1)`.',
        '- `logvar` is clamped to `[-30, 20]`.',
        '- Deterministic alignment uses `DiagonalGaussianDistribution.mode()`, which returns the mean.',
        '- No sampling is used in this oracle.',
        '',
        '## Saved paths',
        '',
        '- Path A: synthetic latent -> post_quant_conv -> decoder.conv_in.',
        '- Path B: synthetic image -> encoder quant moments -> posterior mode -> post_quant_conv -> decoder.conv_in.',
        '',
        '## Consistency checks',
        '',
        '| Check | Max abs diff |',
        '|---|---:|',
    ]
    for k, v in meta['consistency_checks'].items():
        md.append(f'| {k} | `{v}` |')
    (out / 'oracle_summary.txt').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
