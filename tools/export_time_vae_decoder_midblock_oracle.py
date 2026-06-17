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
PREV_ORACLE = BASE_OUT / 'oracle_tensors_decoder_entry'


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


def save(out: Path, arrays: dict[str, np.ndarray], name: str, value: np.ndarray, source: str, stats: list[dict]) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    arrays[name] = arr
    np.save(out / f'{name}.npy', arr)
    stats.append(tensor_stats(name, arr, source))
    return arr


def run_midblock_parts(mid, x, temb):
    r0 = mid.resnets[0](x, temb)
    attn_out = r0
    attention_exists = getattr(mid, 'attentions', None) is not None and len(mid.attentions) > 0 and mid.attentions[0] is not None
    if attention_exists:
        attn_out = mid.attentions[0](r0, temb=temb)
    r1 = mid.resnets[1](attn_out, temb) if len(mid.resnets) > 1 else attn_out
    direct = mid(x, temb)
    return r0, attn_out, r1, direct, attention_exists


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

    out = BASE_OUT / 'oracle_tensors_decoder_midblock'
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(OFFICIAL_REPO))
    from diffusers.models.autoencoders.time_autoencoder_kl import TimeAwareAutoencoderKL
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

    model = TimeAwareAutoencoderKL.from_pretrained(str(WEIGHTS_DIR), local_files_only=True)
    model.eval()
    enc = model.encoder
    dec = model.decoder
    mid = dec.mid_block
    arrays: dict[str, np.ndarray] = {}
    stats: list[dict] = []

    with torch.no_grad():
        temb_for_decoder = None

        # Path A: isolated synthetic hidden -> decoder.mid_block.
        synthetic_hidden_np = np.linspace(-1.0, 1.0, num=1 * 512 * 4 * 4, dtype=np.float32).reshape(1, 512, 4, 4)
        synthetic_hidden = torch.from_numpy(synthetic_hidden_np)
        syn_r0, syn_attn, syn_r1, syn_direct, attention_exists = run_midblock_parts(mid, synthetic_hidden, temb_for_decoder)
        save(out, arrays, 'synthetic_decoder_midblock_input', synthetic_hidden_np, 'path_a.synthetic_linspace_hidden', stats)
        save(out, arrays, 'synthetic_decoder_midblock_resnet0_output', to_np(syn_r0), 'decoder.mid_block.resnets.0(synthetic)', stats)
        if attention_exists:
            save(out, arrays, 'synthetic_decoder_midblock_attention_output', to_np(syn_attn), 'decoder.mid_block.attentions.0(synthetic)', stats)
        save(out, arrays, 'synthetic_decoder_midblock_resnet1_output', to_np(syn_r1), 'decoder.mid_block.resnets.1(synthetic)', stats)
        save(out, arrays, 'synthetic_decoder_midblock_output', to_np(syn_direct), 'decoder.mid_block(synthetic)', stats)

        # Path B: synthetic latent -> post_quant_conv -> decoder.conv_in -> decoder.mid_block.
        prev_z = PREV_ORACLE / 'synthetic_latent_z.npy'
        if prev_z.exists():
            z_np = np.load(prev_z).astype(np.float32)
        else:
            z_np = np.linspace(-1.0, 1.0, num=1 * 4 * 4 * 4, dtype=np.float32).reshape(1, 4, 4, 4)
        z = torch.from_numpy(z_np)
        entry_post = model.post_quant_conv(z)
        entry_conv = dec.conv_in(entry_post)
        entry_r0, entry_attn, entry_r1, entry_direct, _ = run_midblock_parts(mid, entry_conv, temb_for_decoder)
        save(out, arrays, 'entry_synthetic_latent_z', z_np, 'path_b.synthetic_latent_z', stats)
        save(out, arrays, 'entry_post_quant_conv_output', to_np(entry_post), 'post_quant_conv(entry_z)', stats)
        save(out, arrays, 'entry_decoder_conv_in_output', to_np(entry_conv), 'decoder.conv_in(entry_post_quant)', stats)
        save(out, arrays, 'entry_decoder_midblock_resnet0_output', to_np(entry_r0), 'decoder.mid_block.resnets.0(entry)', stats)
        if attention_exists:
            save(out, arrays, 'entry_decoder_midblock_attention_output', to_np(entry_attn), 'decoder.mid_block.attentions.0(entry)', stats)
        save(out, arrays, 'entry_decoder_midblock_resnet1_output', to_np(entry_r1), 'decoder.mid_block.resnets.1(entry)', stats)
        save(out, arrays, 'entry_decoder_midblock_output', to_np(entry_direct), 'decoder.mid_block(entry)', stats)

        # Path C: real encoder deterministic path -> decoder.mid_block.
        prev_img = PREV_ORACLE / 'real_synthetic_image_tensor.npy'
        if prev_img.exists():
            sample_np = np.load(prev_img).astype(np.float32)
        else:
            sample_np = np.linspace(-1.0, 1.0, steps=1 * 3 * args.input_size * args.input_size, dtype=np.float32).reshape(1, 3, args.input_size, args.input_size)
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
        real_r0, real_attn, real_r1, real_direct, _ = run_midblock_parts(mid, real_conv, temb_for_decoder)
        save(out, arrays, 'real_input_image', sample_np, 'path_c.synthetic_image', stats)
        save(out, arrays, 'real_encoder_temb', to_np(encoder_temb), 'encoder.time_embedding', stats)
        save(out, arrays, 'real_moments', to_np(moments), 'quant_conv(path_c)', stats)
        save(out, arrays, 'real_latent_dist_mode', to_np(mode), 'DiagonalGaussianDistribution.mode(path_c)', stats)
        save(out, arrays, 'real_post_quant_conv_output', to_np(real_post), 'post_quant_conv(path_c)', stats)
        save(out, arrays, 'real_decoder_conv_in_output', to_np(real_conv), 'decoder.conv_in(path_c)', stats)
        save(out, arrays, 'real_decoder_midblock_resnet0_output', to_np(real_r0), 'decoder.mid_block.resnets.0(path_c)', stats)
        if attention_exists:
            save(out, arrays, 'real_decoder_midblock_attention_output', to_np(real_attn), 'decoder.mid_block.attentions.0(path_c)', stats)
        save(out, arrays, 'real_decoder_midblock_resnet1_output', to_np(real_r1), 'decoder.mid_block.resnets.1(path_c)', stats)
        save(out, arrays, 'real_decoder_midblock_output', to_np(real_direct), 'decoder.mid_block(path_c)', stats)

    prev_diff = None
    prev_real_conv = PREV_ORACLE / 'real_decoder_conv_in_output.npy'
    if prev_real_conv.exists():
        prev = np.load(prev_real_conv).astype(np.float32)
        prev_diff = float(np.max(np.abs(prev - arrays['real_decoder_conv_in_output'])))

    np.savez_compressed(out / 'time_vae_decoder_midblock_inputs.npz', **{
        k: arrays[k] for k in [
            'synthetic_decoder_midblock_input',
            'entry_synthetic_latent_z',
            'real_input_image',
            'real_encoder_temb',
            'real_moments',
            'real_decoder_conv_in_output',
        ]
    })
    np.savez_compressed(out / 'time_vae_decoder_midblock_outputs.npz', **{
        k: v for k, v in arrays.items() if k.endswith('_output') or k in {'real_latent_dist_mode'}
    })

    meta = {
        'status': 'PASS',
        'official_class': f'{model.__class__.__module__}.{model.__class__.__name__}',
        'decoder_class': f'{dec.__class__.__module__}.{dec.__class__.__name__}',
        'decoder_mid_block_class': f'{mid.__class__.__module__}.{mid.__class__.__name__}',
        'uses_sampling': False,
        'temb_is_none': True,
        'temb_shape': None,
        'decoder_midblock_accepts_temb_but_default_decode_uses_none': True,
        'moments_shape': list(arrays['real_moments'].shape),
        'mode_shape': list(arrays['real_latent_dist_mode'].shape),
        'post_quant_conv_output_shape': list(arrays['real_post_quant_conv_output'].shape),
        'decoder_conv_in_output_shape': list(arrays['real_decoder_conv_in_output'].shape),
        'decoder_midblock_input_shape': list(arrays['real_decoder_conv_in_output'].shape),
        'decoder_midblock_output_shape': list(arrays['real_decoder_midblock_output'].shape),
        'module_order': ['resnets.0'] + (['attentions.0'] if attention_exists else []) + ['resnets.1'],
        'attention_exists': bool(attention_exists),
        'resnet_count': len(mid.resnets),
        'attention_count': 1 if attention_exists else 0,
        'max_abs_diff_between_real_decoder_conv_in_output_and_previous_decoder_entry_oracle': prev_diff,
        'tensor_stats': stats,
    }
    (out / 'decoder_midblock_oracle_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    (out / 'real_temb_metadata.json').write_text(json.dumps({'decoder_midblock_temb_is_none': True, 'encoder_temb_shape': list(arrays['real_encoder_temb'].shape), 'note': 'real_encoder_temb is used only for the encoder-side path; decoder.mid_block receives None.'}, indent=2), encoding='utf-8')
    lines = [
        '# TimeAware VAE Decoder MidBlock Oracle Export',
        '',
        f"Status: **{meta['status']}**",
        '',
        f"Decoder mid_block class: `{meta['decoder_mid_block_class']}`",
        f"Module order: `{meta['module_order']}`",
        f"Attention exists: `{meta['attention_exists']}`",
        f"Decoder mid_block temb is None: `{meta['temb_is_none']}`",
        f"Previous decoder-entry real conv_in diff: `{prev_diff}`",
        '',
        '## Paths',
        '',
        '- Path A: synthetic hidden -> decoder.mid_block.',
        '- Path B: synthetic latent -> post_quant_conv -> decoder.conv_in -> decoder.mid_block.',
        '- Path C: synthetic image -> encoder -> quant_conv -> posterior mode -> post_quant_conv -> decoder.conv_in -> decoder.mid_block.',
    ]
    (out / 'oracle_summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
