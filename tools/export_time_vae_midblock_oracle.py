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
    out = BASE_OUT / 'oracle_tensors_midblock'
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
        mid_input = hidden
        r0_in = mid_input
        r0_out = mid.resnets[0](r0_in, temb=temb)
        attention_count = 0
        attn_in = r0_out
        attn_out = r0_out
        if getattr(mid, 'attentions', None) is not None and len(mid.attentions) > 0 and mid.attentions[0] is not None:
            attention_count = 1
            attn_out = mid.attentions[0](attn_in, temb=temb)
            hooks.append({'name': 'encoder.mid_block.attentions.0', 'class': mid.attentions[0].__class__.__name__, 'input_shape': list(attn_in.shape), 'output_shape': list(attn_out.shape), 'status': 'PASS', 'input_spatial_size': list(attn_in.shape[-2:]), 'output_spatial_size': list(attn_out.shape[-2:])})
        else:
            failures.append({'name': 'encoder.mid_block.attentions.0', 'status': 'NOT_APPLICABLE', 'reason': 'official mid_block has no attention'})
        if len(mid.resnets) > 1:
            r1_in = attn_out
            r1_out = mid.resnets[1](r1_in, temb=temb)
            final_out = r1_out
            hooks.append({'name': 'encoder.mid_block.resnets.1', 'class': mid.resnets[1].__class__.__name__, 'input_shape': list(r1_in.shape), 'temb_shape': list(temb.shape), 'output_shape': list(r1_out.shape), 'status': 'PASS', 'input_spatial_size': list(r1_in.shape[-2:]), 'output_spatial_size': list(r1_out.shape[-2:])})
        else:
            r1_in = None
            r1_out = None
            final_out = attn_out
            failures.append({'name': 'encoder.mid_block.resnets.1', 'status': 'NOT_APPLICABLE', 'reason': 'official mid_block has no resnets.1'})
        direct_final = mid(mid_input, temb=temb)
    hooks = [
        {'name': 'encoder.time_proj', 'class': enc.time_proj.__class__.__name__, 'input_shape': list(timesteps.shape), 'output_shape': list(time_proj.shape), 'status': 'PASS'},
        {'name': 'encoder.time_embedding', 'class': enc.time_embedding.__class__.__name__, 'input_shape': list(time_proj.shape), 'output_shape': list(temb.shape), 'status': 'PASS'},
        {'name': 'encoder.conv_in', 'class': enc.conv_in.__class__.__name__, 'input_shape': list(sample.shape), 'output_shape': list(conv_in.shape), 'status': 'PASS'},
    ] + hooks
    for i, arr in previous_outputs.items():
        idx = int(i[len('downblock'):-len('_output')])
        hooks.append({'name': f'encoder.down_blocks.{idx}', 'class': blocks[idx].__class__.__name__, 'output_shape': list(arr.shape), 'status': 'PASS'})
    hooks.append({'name': 'encoder.mid_block.resnets.0', 'class': mid.resnets[0].__class__.__name__, 'input_shape': list(r0_in.shape), 'temb_shape': list(temb.shape), 'output_shape': list(r0_out.shape), 'status': 'PASS', 'input_spatial_size': list(r0_in.shape[-2:]), 'output_spatial_size': list(r0_out.shape[-2:])})
    hooks.append({'name': 'encoder.mid_block', 'class': mid.__class__.__name__, 'input_shape': list(mid_input.shape), 'temb_shape': list(temb.shape), 'output_shape': list(final_out.shape), 'status': 'PASS', 'input_spatial_size': list(mid_input.shape[-2:]), 'output_spatial_size': list(final_out.shape[-2:])})

    inputs = {
        'synthetic_image_tensor': to_np(sample),
        'timestep': timesteps.cpu().numpy().astype(np.int64),
        'encoder_conv_in_input': to_np(sample),
        'stage0123_mid_input': to_np(sample),
        'midblock_input': to_np(mid_input),
        'resnet0_input': to_np(r0_in),
        'attention0_input': to_np(attn_in),
        'midblock_temb': to_np(temb),
    }
    outputs = {
        'encoder_time_proj_output': to_np(time_proj),
        'encoder_time_embedding_output': to_np(temb),
        'encoder_conv_in_output': to_np(conv_in),
        'resnet0_output': to_np(r0_out),
        'attention0_output': to_np(attn_out),
        'midblock_output': to_np(final_out),
        'midblock_direct_output': to_np(direct_final),
        'stage0123_mid_output': to_np(final_out),
    }
    outputs.update(previous_outputs)
    if r1_in is not None:
        inputs['resnet1_input'] = to_np(r1_in)
        outputs['resnet1_output'] = to_np(r1_out)
    np.savez_compressed(out / 'time_vae_midblock_inputs.npz', **inputs)
    np.savez_compressed(out / 'time_vae_midblock_outputs.npz', **outputs)
    meta = {
        'status': 'PASS',
        'official_class': f'{model.__class__.__module__}.{model.__class__.__name__}',
        'encoder_class': f'{enc.__class__.__module__}.{enc.__class__.__name__}',
        'mid_block_class': f'{mid.__class__.__module__}.{mid.__class__.__name__}',
        'sample_shape': list(sample.shape),
        'timestep': int(timesteps.item()),
        'input_size': args.input_size,
        'attention_exists': attention_count > 0,
        'attention_count': attention_count,
        'resnet_count': len(mid.resnets),
        'module_order': ['encoder.conv_in'] + [f'encoder.down_blocks.{i}' for i in range(4)] + ['encoder.mid_block.resnets.0'] + (['encoder.mid_block.attentions.0'] if attention_count else []) + (['encoder.mid_block.resnets.1'] if len(mid.resnets) > 1 else []),
        'hook_targets': hooks,
        'hook_failures': failures,
        'input_spatial_size': list(mid_input.shape[-2:]),
        'output_spatial_size': list(final_out.shape[-2:]),
    }
    (out / 'time_vae_midblock_hook_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    md = ['# TimeAware VAE MidBlock Oracle Export', '', f"Status: **{meta['status']}**", '', f"Input shape: `{meta['sample_shape']}`", f"Timestep: `{meta['timestep']}`", f"Attention exists: `{meta['attention_exists']}`", '', '| Target | Class | Input shape | Output shape | Status |', '|---|---|---:|---:|---|']
    for row in hooks:
        inp = row.get('input_shape')
        if row.get('temb_shape'):
            inp = f"{inp}, temb={row['temb_shape']}"
        md.append(f"| `{row['name']}` | `{row.get('class')}` | `{inp}` | `{row.get('output_shape')}` | {row['status']} |")
    if failures:
        md += ['', '## Hook failures / not applicable']
        for row in failures:
            md.append(f"- `{row['name']}`: {row.get('status')} {row.get('reason', '')}")
    (out / 'time_vae_midblock_hook_report.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
