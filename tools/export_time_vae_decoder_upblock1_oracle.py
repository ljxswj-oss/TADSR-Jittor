#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

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
BASE_OUT = Path('experiments/full_repro/time_vae_alignment')
PREV_UPBLOCK0_ORACLE = BASE_OUT / 'oracle_tensors_decoder_upblock0'


def to_np(x):
    return x.detach().cpu().numpy().astype(np.float32)


def sha256_array(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def tensor_stats(name: str, value: np.ndarray, source: str) -> dict:
    arr = np.asarray(value)
    return {
        'name': name,
        'shape': list(arr.shape),
        'dtype': str(arr.dtype),
        'min': float(arr.min()) if arr.size else None,
        'max': float(arr.max()) if arr.size else None,
        'mean': float(arr.mean()) if arr.size else None,
        'std': float(arr.std()) if arr.size else None,
        'sha256': sha256_array(arr),
        'source': source,
    }


def save(out: Path, arrays: dict[str, np.ndarray], name: str, value, source: str, stats: list[dict]) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    arrays[name] = arr
    np.save(out / f'{name}.npy', arr)
    stats.append(tensor_stats(name, arr, source))
    return arr


def run_upblock_parts(block, x, temb):
    hidden = x
    res_outputs = []
    for resnet in block.resnets:
        hidden = resnet(hidden, temb)
        res_outputs.append(hidden)
    pre_upsampler = hidden
    upsampler_outputs = []
    if getattr(block, 'upsamplers', None) is not None:
        for upsampler in block.upsamplers:
            hidden = upsampler(hidden)
            upsampler_outputs.append(hidden)
    direct = block(x, temb)
    return res_outputs, pre_upsampler, upsampler_outputs, direct


def export_upblock1_path(out, arrays, stats, prefix, source_label, upblock1, hidden, temb):
    res_outputs, pre_upsampler, upsampler_outputs, direct = run_upblock_parts(upblock1, hidden, temb)
    for i, value in enumerate(res_outputs):
        save(out, arrays, f'{prefix}_decoder_upblock1_resnet{i}_output', to_np(value), f'decoder.up_blocks.1.resnets.{i}({source_label})', stats)
    save(out, arrays, f'{prefix}_decoder_upblock1_pre_upsampler_output', to_np(pre_upsampler), f'decoder.up_blocks.1.pre_upsampler({source_label})', stats)
    for i, value in enumerate(upsampler_outputs):
        save(out, arrays, f'{prefix}_decoder_upblock1_upsampler{i}_output', to_np(value), f'decoder.up_blocks.1.upsamplers.{i}({source_label})', stats)
    save(out, arrays, f'{prefix}_decoder_upblock1_output', to_np(direct), f'decoder.up_blocks.1({source_label})', stats)
    return direct


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', type=int, default=32)
    parser.add_argument('--timestep', type=int, default=50)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    out = BASE_OUT / 'oracle_tensors_decoder_upblock1'
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(OFFICIAL_REPO))
    from diffusers.models.autoencoders.time_autoencoder_kl import TimeAwareAutoencoderKL
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

    model = TimeAwareAutoencoderKL.from_pretrained(str(WEIGHTS_DIR), local_files_only=True)
    model.eval()
    enc = model.encoder
    dec = model.decoder
    upblock0 = dec.up_blocks[0]
    upblock1 = dec.up_blocks[1]
    arrays: dict[str, np.ndarray] = {}
    stats: list[dict] = []

    with torch.no_grad():
        decoder_temb = None

        # Path A: isolated synthetic hidden -> decoder.up_blocks.1.
        synthetic_hidden_np = np.linspace(-1.0, 1.0, num=1 * 512 * 8 * 8, dtype=np.float32).reshape(1, 512, 8, 8)
        synthetic_hidden = torch.from_numpy(synthetic_hidden_np)
        save(out, arrays, 'synthetic_decoder_upblock1_input', synthetic_hidden_np, 'path_a.synthetic_linspace_hidden', stats)
        export_upblock1_path(out, arrays, stats, 'synthetic', 'synthetic', upblock1, synthetic_hidden, decoder_temb)

        # Path B: synthetic latent -> post_quant_conv -> decoder.conv_in -> decoder.mid_block -> upblock0 -> upblock1.
        prev_z = PREV_UPBLOCK0_ORACLE / 'entry_synthetic_latent_z.npy'
        if prev_z.exists():
            z_np = np.load(prev_z).astype(np.float32)
        else:
            z_np = np.linspace(-1.0, 1.0, num=1 * 4 * 4 * 4, dtype=np.float32).reshape(1, 4, 4, 4)
        z = torch.from_numpy(z_np)
        entry_post = model.post_quant_conv(z)
        entry_conv = dec.conv_in(entry_post)
        entry_mid = dec.mid_block(entry_conv, decoder_temb)
        entry_up0 = upblock0(entry_mid, decoder_temb)
        save(out, arrays, 'entry_synthetic_latent_z', z_np, 'path_b.synthetic_latent_z', stats)
        save(out, arrays, 'entry_post_quant_conv_output', to_np(entry_post), 'post_quant_conv(entry_z)', stats)
        save(out, arrays, 'entry_decoder_conv_in_output', to_np(entry_conv), 'decoder.conv_in(entry_post_quant)', stats)
        save(out, arrays, 'entry_decoder_midblock_output', to_np(entry_mid), 'decoder.mid_block(entry)', stats)
        save(out, arrays, 'entry_decoder_upblock0_output', to_np(entry_up0), 'decoder.up_blocks.0(entry)', stats)
        export_upblock1_path(out, arrays, stats, 'entry', 'entry', upblock1, entry_up0, decoder_temb)

        # Path C: real encoder deterministic path -> decoder.up_blocks.1.
        prev_img = PREV_UPBLOCK0_ORACLE / 'real_input_image.npy'
        if prev_img.exists():
            sample_np = np.load(prev_img).astype(np.float32)
        else:
            sample_np = np.linspace(-1.0, 1.0, num=1 * 3 * args.input_size * args.input_size, dtype=np.float32).reshape(1, 3, args.input_size, args.input_size)
        sample = torch.from_numpy(sample_np)
        timesteps = torch.tensor([args.timestep], dtype=torch.long)
        time_proj = enc.time_proj(timesteps)
        encoder_temb = enc.time_embedding(time_proj.to(dtype=list(enc.time_embedding.parameters())[0].dtype))
        hidden = enc.conv_in(sample)
        for i in range(4):
            hidden = enc.down_blocks[i](hidden, temb=encoder_temb)
        enc_mid = enc.mid_block(hidden, temb=encoder_temb)
        tail = enc.conv_out(enc.conv_act(enc.conv_norm_out(enc_mid)))
        moments = model.quant_conv(tail)
        latent_dist = DiagonalGaussianDistribution(moments)
        mode = latent_dist.mode()
        real_post = model.post_quant_conv(mode)
        real_conv = dec.conv_in(real_post)
        real_mid = dec.mid_block(real_conv, decoder_temb)
        real_up0 = upblock0(real_mid, decoder_temb)
        save(out, arrays, 'real_input_image', sample_np, 'path_c.synthetic_image', stats)
        save(out, arrays, 'real_encoder_temb', to_np(encoder_temb), 'encoder.time_embedding', stats)
        save(out, arrays, 'real_moments', to_np(moments), 'quant_conv(path_c)', stats)
        save(out, arrays, 'real_latent_dist_mode', to_np(mode), 'DiagonalGaussianDistribution.mode(path_c)', stats)
        save(out, arrays, 'real_post_quant_conv_output', to_np(real_post), 'post_quant_conv(path_c)', stats)
        save(out, arrays, 'real_decoder_conv_in_output', to_np(real_conv), 'decoder.conv_in(path_c)', stats)
        save(out, arrays, 'real_decoder_midblock_output', to_np(real_mid), 'decoder.mid_block(path_c)', stats)
        save(out, arrays, 'real_decoder_upblock0_output', to_np(real_up0), 'decoder.up_blocks.0(path_c)', stats)
        export_upblock1_path(out, arrays, stats, 'real', 'path_c', upblock1, real_up0, decoder_temb)

    prev_diff = None
    prev_real_up0 = PREV_UPBLOCK0_ORACLE / 'real_decoder_upblock0_output.npy'
    if prev_real_up0.exists():
        prev = np.load(prev_real_up0).astype(np.float32)
        prev_diff = float(np.max(np.abs(prev - arrays['real_decoder_upblock0_output'])))

    np.savez_compressed(out / 'time_vae_decoder_upblock1_inputs.npz', **{
        k: arrays[k] for k in [
            'synthetic_decoder_upblock1_input',
            'entry_synthetic_latent_z',
            'entry_decoder_upblock0_output',
            'real_input_image',
            'real_encoder_temb',
            'real_moments',
            'real_decoder_upblock0_output',
        ]
    })
    np.savez_compressed(out / 'time_vae_decoder_upblock1_outputs.npz', **{
        k: v for k, v in arrays.items() if k.endswith('_output') or k in {'real_latent_dist_mode'}
    })

    meta = {
        'status': 'PASS',
        'official_class': f'{model.__class__.__module__}.{model.__class__.__name__}',
        'decoder_class': f'{dec.__class__.__module__}.{dec.__class__.__name__}',
        'decoder_upblock1_class': f'{upblock1.__class__.__module__}.{upblock1.__class__.__name__}',
        'uses_sampling': False,
        'decoder_side_temb_is_none': True,
        'upblock1_resnet_count': len(upblock1.resnets),
        'upblock1_upsampler_count': len(upblock1.upsamplers or []),
        'upblock1_has_upsampler': bool(upblock1.upsamplers),
        'upblock1_module_order': [f'resnets.{i}' for i in range(len(upblock1.resnets))] + [f'upsamplers.{i}' for i in range(len(upblock1.upsamplers or []))],
        'upblock1_input_shape': list(arrays['real_decoder_upblock0_output'].shape),
        'upblock1_output_shape': list(arrays['real_decoder_upblock1_output'].shape),
        'real_input_image_shape': list(arrays['real_input_image'].shape),
        'real_moments_shape': list(arrays['real_moments'].shape),
        'real_posterior_mode_shape': list(arrays['real_latent_dist_mode'].shape),
        'real_decoder_midblock_output_shape': list(arrays['real_decoder_midblock_output'].shape),
        'real_decoder_upblock0_output_shape': list(arrays['real_decoder_upblock0_output'].shape),
        'real_decoder_upblock1_output_shape': list(arrays['real_decoder_upblock1_output'].shape),
        'max_abs_diff_between_real_decoder_upblock0_output_and_previous_decoder_upblock0_oracle': prev_diff,
        'tensor_stats': stats,
    }
    (out / 'decoder_upblock1_oracle_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    lines = [
        '# TimeAware VAE Decoder UpBlock1 Oracle Export',
        '',
        f"Status: **{meta['status']}**",
        '',
        f"Decoder upblock1 class: `{meta['decoder_upblock1_class']}`",
        f"Module order: `{meta['upblock1_module_order']}`",
        f"ResNet count: `{meta['upblock1_resnet_count']}`",
        f"Upsampler count: `{meta['upblock1_upsampler_count']}`",
        f"Decoder-side temb is None: `{meta['decoder_side_temb_is_none']}`",
        f"Previous decoder-upblock0 real output diff: `{prev_diff}`",
        '',
        '## Paths',
        '',
        '- Path A: synthetic hidden -> decoder.up_blocks.1.',
        '- Path B: synthetic latent -> post_quant_conv -> decoder.conv_in -> decoder.mid_block -> decoder.up_blocks.0 -> decoder.up_blocks.1.',
        '- Path C: synthetic image -> encoder -> quant_conv -> posterior mode -> post_quant_conv -> decoder.conv_in -> decoder.mid_block -> decoder.up_blocks.0 -> decoder.up_blocks.1.',
    ]
    (out / 'oracle_summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
