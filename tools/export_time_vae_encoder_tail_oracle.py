#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', type=int, default=32)
    parser.add_argument('--timestep', type=int, default=50)
    args = parser.parse_args()
    out = BASE_OUT / 'oracle_tensors_encoder_tail'
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(OFFICIAL_REPO))
    from diffusers.models.autoencoders.time_autoencoder_kl import TimeAwareAutoencoderKL

    torch.manual_seed(42)
    model = TimeAwareAutoencoderKL.from_pretrained(str(WEIGHTS_DIR), local_files_only=True)
    model.eval()
    enc = model.encoder
    blocks = enc.down_blocks
    mid = enc.mid_block
    sample = torch.linspace(-1.0, 1.0, steps=1 * 3 * args.input_size * args.input_size, dtype=torch.float32).reshape(1, 3, args.input_size, args.input_size)
    timesteps = torch.tensor([args.timestep], dtype=torch.long)
    hooks = []
    failures = []
    previous_outputs = {}
    with torch.no_grad():
        time_proj = enc.time_proj(timesteps)
        temb = enc.time_embedding(time_proj.to(dtype=list(enc.time_embedding.parameters())[0].dtype))
        conv_in = enc.conv_in(sample)
        hidden = conv_in
        for i in range(4):
            hidden = blocks[i](hidden, temb=temb)
            previous_outputs[f'downblock{i}_output'] = to_np(hidden)
        mid_out = mid(hidden, temb=temb)
        norm_in = mid_out
        norm_out = enc.conv_norm_out(norm_in)
        act_in = norm_out
        act_out = enc.conv_act(act_in)
        conv_out_in = act_out
        conv_out = enc.conv_out(conv_out_in)
        tail_out = conv_out
        quant_in = tail_out
        quant_out = model.quant_conv(quant_in)

    hooks += [
        {'name': 'encoder.time_proj', 'class': enc.time_proj.__class__.__name__, 'input_shape': list(timesteps.shape), 'output_shape': list(time_proj.shape), 'status': 'PASS'},
        {'name': 'encoder.time_embedding', 'class': enc.time_embedding.__class__.__name__, 'input_shape': list(time_proj.shape), 'output_shape': list(temb.shape), 'status': 'PASS'},
        {'name': 'encoder.conv_in', 'class': enc.conv_in.__class__.__name__, 'input_shape': list(sample.shape), 'output_shape': list(conv_in.shape), 'status': 'PASS'},
    ]
    for i, arr in previous_outputs.items():
        idx = int(i[len('downblock'):-len('_output')])
        hooks.append({'name': f'encoder.down_blocks.{idx}', 'class': blocks[idx].__class__.__name__, 'output_shape': list(arr.shape), 'status': 'PASS'})
    hooks += [
        {'name': 'encoder.mid_block', 'class': mid.__class__.__name__, 'input_shape': list(hidden.shape), 'temb_shape': list(temb.shape), 'output_shape': list(mid_out.shape), 'status': 'PASS', 'input_spatial_size': list(hidden.shape[-2:]), 'output_spatial_size': list(mid_out.shape[-2:])},
        {'name': 'encoder.conv_norm_out', 'class': enc.conv_norm_out.__class__.__name__, 'input_shape': list(norm_in.shape), 'output_shape': list(norm_out.shape), 'status': 'PASS', 'input_spatial_size': list(norm_in.shape[-2:]), 'output_spatial_size': list(norm_out.shape[-2:])},
        {'name': 'encoder.conv_act', 'class': enc.conv_act.__class__.__name__, 'input_shape': list(act_in.shape), 'output_shape': list(act_out.shape), 'status': 'PASS'},
        {'name': 'encoder.conv_out', 'class': enc.conv_out.__class__.__name__, 'input_shape': list(conv_out_in.shape), 'output_shape': list(conv_out.shape), 'status': 'PASS', 'input_spatial_size': list(conv_out_in.shape[-2:]), 'output_spatial_size': list(conv_out.shape[-2:])},
        {'name': 'quant_conv', 'class': model.quant_conv.__class__.__name__, 'input_shape': list(quant_in.shape), 'output_shape': list(quant_out.shape), 'status': 'PASS', 'input_spatial_size': list(quant_in.shape[-2:]), 'output_spatial_size': list(quant_out.shape[-2:])},
    ]
    inputs = {
        'synthetic_image_tensor': to_np(sample),
        'timestep': timesteps.cpu().numpy().astype(np.int64),
        'encoder_conv_in_input': to_np(sample),
        'stage0123_mid_tail_input': to_np(sample),
        'stage0123_mid_tail_quant_input': to_np(sample),
        'conv_norm_out_input': to_np(norm_in),
        'conv_act_input': to_np(act_in),
        'conv_out_input': to_np(conv_out_in),
        'quant_conv_input': to_np(quant_in),
        'tail_temb': to_np(temb),
    }
    outputs = {
        'encoder_time_proj_output': to_np(time_proj),
        'encoder_time_embedding_output': to_np(temb),
        'encoder_conv_in_output': to_np(conv_in),
        'midblock_output': to_np(mid_out),
        'conv_norm_out_output': to_np(norm_out),
        'conv_act_output': to_np(act_out),
        'conv_out_output': to_np(conv_out),
        'encoder_tail_output': to_np(tail_out),
        'quant_conv_output': to_np(quant_out),
        'stage0123_mid_tail_output': to_np(tail_out),
        'stage0123_mid_tail_quant_output': to_np(quant_out),
    }
    outputs.update(previous_outputs)
    np.savez_compressed(out / 'time_vae_encoder_tail_inputs.npz', **inputs)
    np.savez_compressed(out / 'time_vae_encoder_tail_outputs.npz', **outputs)
    meta = {
        'status': 'PASS',
        'official_class': f'{model.__class__.__module__}.{model.__class__.__name__}',
        'encoder_class': f'{enc.__class__.__module__}.{enc.__class__.__name__}',
        'sample_shape': list(sample.shape),
        'timestep': int(timesteps.item()),
        'input_size': args.input_size,
        'module_order': ['encoder.conv_in'] + [f'encoder.down_blocks.{i}' for i in range(4)] + ['encoder.mid_block', 'encoder.conv_norm_out', 'encoder.conv_act', 'encoder.conv_out', 'quant_conv'],
        'hook_targets': hooks,
        'hook_failures': failures,
        'quant_conv_channel_mapping': f'{model.quant_conv.in_channels}->{model.quant_conv.out_channels}',
        'post_quant_conv_exists': hasattr(model, 'post_quant_conv'),
        'post_quant_conv_skipped': True,
        'moments_note': 'quant_conv output is deterministic moments tensor; no DiagonalGaussianDistribution sampling is performed in this export.',
        'input_spatial_size': list(sample.shape[-2:]),
        'tail_output_spatial_size': list(tail_out.shape[-2:]),
        'quant_output_spatial_size': list(quant_out.shape[-2:]),
    }
    (out / 'time_vae_encoder_tail_hook_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    md = ['# TimeAware VAE Encoder Tail + QuantConv Oracle Export', '', f"Status: **{meta['status']}**", '', f"Input shape: `{meta['sample_shape']}`", f"Timestep: `{meta['timestep']}`", f"Quant conv channels: `{meta['quant_conv_channel_mapping']}`", f"Post quant conv exists but skipped: `{meta['post_quant_conv_exists']}`", '', '| Target | Class | Input shape | Output shape | Status |', '|---|---|---:|---:|---|']
    for row in hooks:
        inp = row.get('input_shape')
        if row.get('temb_shape'):
            inp = f"{inp}, temb={row['temb_shape']}"
        md.append(f"| `{row['name']}` | `{row.get('class')}` | `{inp}` | `{row.get('output_shape')}` | {row['status']} |")
    if failures:
        md += ['', '## Hook failures']
        for row in failures:
            md.append(f"- `{row['name']}`: {row.get('status')} {row.get('reason', '')}")
    md += ['', '## Sampling note', '', meta['moments_note']]
    (out / 'time_vae_encoder_tail_hook_report.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
