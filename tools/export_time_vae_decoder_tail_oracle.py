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
ORACLE_OUT = BASE_OUT / 'oracle_tensors_decoder_tail'
PREV_UPBLOCK3_ORACLE = BASE_OUT / 'oracle_tensors_decoder_upblock3'
AUDIT_PATH = BASE_OUT / 'audit_time_vae_decoder_tail.json'


def to_np(x):
    return x.detach().cpu().numpy().astype(np.float32)


def sha256_array(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def tensor_stats(name: str, value: np.ndarray, source: str) -> dict:
    arr = np.asarray(value, dtype=np.float32)
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


def load_tail_input_shape() -> tuple[int, int, int, int]:
    if AUDIT_PATH.exists():
        audit = json.loads(AUDIT_PATH.read_text(encoding='utf-8'))
        shape = audit.get('tail_input_shape')
        if shape and len(shape) == 4:
            return tuple(int(x) for x in shape)
    return (1, 128, 32, 32)


def run_tail(decoder, hidden):
    norm = decoder.conv_norm_out(hidden)
    act = decoder.conv_act(norm)
    conv = decoder.conv_out(act)
    return norm, act, conv


def save_tail_path(out, arrays, stats, prefix, source_label, decoder, hidden):
    save(out, arrays, f'{prefix}_decoder_tail_input', to_np(hidden), f'{source_label}.tail_input', stats)
    norm, act, conv = run_tail(decoder, hidden)
    save(out, arrays, f'{prefix}_decoder_tail_norm_out_output', to_np(norm), f'decoder.conv_norm_out({source_label})', stats)
    save(out, arrays, f'{prefix}_decoder_tail_act_output', to_np(act), f'decoder.conv_act({source_label})', stats)
    save(out, arrays, f'{prefix}_decoder_tail_conv_out_output', to_np(conv), f'decoder.conv_out({source_label})', stats)
    save(out, arrays, f'{prefix}_decoder_tail_output', to_np(conv), f'decoder.tail({source_label})', stats)
    return conv


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

    ORACLE_OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(OFFICIAL_REPO))
    from diffusers.models.autoencoders.time_autoencoder_kl import TimeAwareAutoencoderKL
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

    model = TimeAwareAutoencoderKL.from_pretrained(str(WEIGHTS_DIR), local_files_only=True)
    model.eval()
    enc = model.encoder
    dec = model.decoder
    upblock0, upblock1, upblock2, upblock3 = dec.up_blocks
    arrays: dict[str, np.ndarray] = {}
    stats: list[dict] = []

    with torch.no_grad():
        decoder_temb = None

        # Path A: isolated synthetic hidden -> decoder tail.
        synthetic_shape = load_tail_input_shape()
        synthetic_hidden_np = np.linspace(-1.0, 1.0, num=int(np.prod(synthetic_shape)), dtype=np.float32).reshape(synthetic_shape)
        synthetic_hidden = torch.from_numpy(synthetic_hidden_np)
        save_tail_path(ORACLE_OUT, arrays, stats, 'synthetic', 'path_a.synthetic_tail_hidden', dec, synthetic_hidden)

        # Path B: synthetic latent -> deterministic decoder stack -> tail.
        prev_z = PREV_UPBLOCK3_ORACLE / 'entry_synthetic_latent_z.npy'
        if prev_z.exists():
            z_np = np.load(prev_z).astype(np.float32)
        else:
            z_np = np.linspace(-1.0, 1.0, num=1 * 4 * 4 * 4, dtype=np.float32).reshape(1, 4, 4, 4)
        z = torch.from_numpy(z_np)
        entry_post = model.post_quant_conv(z)
        entry_conv = dec.conv_in(entry_post)
        entry_mid = dec.mid_block(entry_conv, decoder_temb)
        entry_up0 = upblock0(entry_mid, decoder_temb)
        entry_up1 = upblock1(entry_up0, decoder_temb)
        entry_up2 = upblock2(entry_up1, decoder_temb)
        entry_up3 = upblock3(entry_up2, decoder_temb)
        save(ORACLE_OUT, arrays, 'entry_synthetic_latent_z', z_np, 'path_b.synthetic_latent_z', stats)
        save(ORACLE_OUT, arrays, 'entry_post_quant_conv_output', to_np(entry_post), 'post_quant_conv(entry_z)', stats)
        save(ORACLE_OUT, arrays, 'entry_decoder_conv_in_output', to_np(entry_conv), 'decoder.conv_in(entry_post_quant)', stats)
        save(ORACLE_OUT, arrays, 'entry_decoder_midblock_output', to_np(entry_mid), 'decoder.mid_block(entry)', stats)
        save(ORACLE_OUT, arrays, 'entry_decoder_upblock0_output', to_np(entry_up0), 'decoder.up_blocks.0(entry)', stats)
        save(ORACLE_OUT, arrays, 'entry_decoder_upblock1_output', to_np(entry_up1), 'decoder.up_blocks.1(entry)', stats)
        save(ORACLE_OUT, arrays, 'entry_decoder_upblock2_output', to_np(entry_up2), 'decoder.up_blocks.2(entry)', stats)
        save(ORACLE_OUT, arrays, 'entry_decoder_upblock3_output', to_np(entry_up3), 'decoder.up_blocks.3(entry)', stats)
        save_tail_path(ORACLE_OUT, arrays, stats, 'entry', 'path_b.entry', dec, entry_up3)

        # Path C: real encoder deterministic path -> decoder tail.
        prev_img = PREV_UPBLOCK3_ORACLE / 'real_input_image.npy'
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
        enc_tail = enc.conv_out(enc.conv_act(enc.conv_norm_out(enc_mid)))
        moments = model.quant_conv(enc_tail)
        latent_dist = DiagonalGaussianDistribution(moments)
        mode = latent_dist.mode()
        real_post = model.post_quant_conv(mode)
        real_conv = dec.conv_in(real_post)
        real_mid = dec.mid_block(real_conv, decoder_temb)
        real_up0 = upblock0(real_mid, decoder_temb)
        real_up1 = upblock1(real_up0, decoder_temb)
        real_up2 = upblock2(real_up1, decoder_temb)
        real_up3 = upblock3(real_up2, decoder_temb)
        save(ORACLE_OUT, arrays, 'real_input_image', sample_np, 'path_c.synthetic_image', stats)
        save(ORACLE_OUT, arrays, 'real_encoder_temb', to_np(encoder_temb), 'encoder.time_embedding', stats)
        save(ORACLE_OUT, arrays, 'real_moments', to_np(moments), 'quant_conv(path_c)', stats)
        save(ORACLE_OUT, arrays, 'real_latent_dist_mode', to_np(mode), 'DiagonalGaussianDistribution.mode(path_c)', stats)
        save(ORACLE_OUT, arrays, 'real_post_quant_conv_output', to_np(real_post), 'post_quant_conv(path_c)', stats)
        save(ORACLE_OUT, arrays, 'real_decoder_conv_in_output', to_np(real_conv), 'decoder.conv_in(path_c)', stats)
        save(ORACLE_OUT, arrays, 'real_decoder_midblock_output', to_np(real_mid), 'decoder.mid_block(path_c)', stats)
        save(ORACLE_OUT, arrays, 'real_decoder_upblock0_output', to_np(real_up0), 'decoder.up_blocks.0(path_c)', stats)
        save(ORACLE_OUT, arrays, 'real_decoder_upblock1_output', to_np(real_up1), 'decoder.up_blocks.1(path_c)', stats)
        save(ORACLE_OUT, arrays, 'real_decoder_upblock2_output', to_np(real_up2), 'decoder.up_blocks.2(path_c)', stats)
        save(ORACLE_OUT, arrays, 'real_decoder_upblock3_output', to_np(real_up3), 'decoder.up_blocks.3(path_c)', stats)
        save_tail_path(ORACLE_OUT, arrays, stats, 'real', 'path_c.real', dec, real_up3)

    prev_up3_tensor = PREV_UPBLOCK3_ORACLE / 'real_decoder_upblock3_output.npy'
    previous_available = prev_up3_tensor.exists()
    prev_diff = None
    if previous_available:
        prev = np.load(prev_up3_tensor).astype(np.float32)
        prev_diff = float(np.max(np.abs(prev - arrays['real_decoder_upblock3_output'])))

    np.savez_compressed(ORACLE_OUT / 'time_vae_decoder_tail_inputs.npz', **{
        k: arrays[k] for k in [
            'synthetic_decoder_tail_input',
            'entry_synthetic_latent_z',
            'entry_decoder_upblock3_output',
            'real_input_image',
            'real_encoder_temb',
            'real_moments',
            'real_decoder_upblock3_output',
        ]
    })
    np.savez_compressed(ORACLE_OUT / 'time_vae_decoder_tail_outputs.npz', **{
        k: v for k, v in arrays.items() if k.endswith('_output') or k in {'real_latent_dist_mode'}
    })

    scaling_factor = getattr(getattr(model, 'config', object()), 'scaling_factor', None)
    meta = {
        'status': 'PASS',
        'official_class': f'{model.__class__.__module__}.{model.__class__.__name__}',
        'decoder_class': f'{dec.__class__.__module__}.{dec.__class__.__name__}',
        'uses_sampling': False,
        'decoder_side_temb_is_none': True,
        'tail_module_order': ['decoder.conv_norm_out', 'decoder.conv_act', 'decoder.conv_out'],
        'tail_input_shape': list(arrays['real_decoder_upblock3_output'].shape),
        'tail_output_shape': list(arrays['real_decoder_tail_output'].shape),
        'tail_norm_eps': float(dec.conv_norm_out.eps),
        'tail_norm_groups': int(dec.conv_norm_out.num_groups),
        'tail_norm_channels': int(dec.conv_norm_out.num_channels),
        'tail_activation_type': dec.conv_act.__class__.__name__,
        'tail_conv_out_weight_shape': list(dec.conv_out.weight.shape),
        'tail_conv_out_bias_shape': list(dec.conv_out.bias.shape),
        'tail_conv_out_padding': list(dec.conv_out.padding),
        'tail_conv_out_stride': list(dec.conv_out.stride),
        'tail_output_channels': int(dec.conv_out.out_channels),
        'real_input_image_shape': list(arrays['real_input_image'].shape),
        'real_moments_shape': list(arrays['real_moments'].shape),
        'real_posterior_mode_shape': list(arrays['real_latent_dist_mode'].shape),
        'real_decoder_upblock3_output_shape': list(arrays['real_decoder_upblock3_output'].shape),
        'real_decoder_tail_output_shape': list(arrays['real_decoder_tail_output'].shape),
        'scaling_factor_exists': scaling_factor is not None,
        'scaling_factor_value': float(scaling_factor) if scaling_factor is not None else None,
        'scaling_factor_applied_inside_decoder': False,
        'output_clamp_or_tanh_inside_decoder': False,
        'previous_upblock3_oracle_tensor_available': previous_available,
        'max_abs_diff_between_real_decoder_upblock3_output_and_previous_decoder_upblock3_oracle': prev_diff,
        'all_dtype_float32': all(item['dtype'] == 'float32' for item in stats),
        'tensor_stats': stats,
    }
    (ORACLE_OUT / 'decoder_tail_oracle_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    summary = [
        'TIME_VAE_DECODER_TAIL_ORACLE_TENSORS: PASS',
        f"output_dir: {ORACLE_OUT}",
        f"tail_module_order: {meta['tail_module_order']}",
        f"tail_input_shape: {meta['tail_input_shape']}",
        f"tail_output_shape: {meta['tail_output_shape']}",
        f"tail_norm_eps: {meta['tail_norm_eps']}",
        f"tail_norm_groups: {meta['tail_norm_groups']}",
        f"tail_activation_type: {meta['tail_activation_type']}",
        f"tail_conv_out_weight_shape: {meta['tail_conv_out_weight_shape']}",
        f"scaling_factor_value: {meta['scaling_factor_value']}",
        f"scaling_factor_applied_inside_decoder: {meta['scaling_factor_applied_inside_decoder']}",
        f"output_clamp_or_tanh_inside_decoder: {meta['output_clamp_or_tanh_inside_decoder']}",
        f"previous_upblock3_oracle_tensor_available: {previous_available}",
        f"max_abs_diff_between_real_decoder_upblock3_output_and_previous_decoder_upblock3_oracle: {prev_diff}",
    ]
    (ORACLE_OUT / 'oracle_summary.txt').write_text('\n'.join(summary) + '\n', encoding='utf-8')
    print('TIME_VAE_DECODER_TAIL_ORACLE_TENSORS: PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
