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
    parser.add_argument('--block-index', type=int, required=True)
    parser.add_argument('--input-size', type=int, default=32)
    parser.add_argument('--timestep', type=int, default=50)
    args = parser.parse_args()
    block_index = args.block_index
    out = BASE_OUT / f'oracle_tensors_downblock{block_index}'
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(OFFICIAL_REPO))
    from diffusers.models.autoencoders.time_autoencoder_kl import TimeAwareAutoencoderKL

    torch.manual_seed(42)
    model = TimeAwareAutoencoderKL.from_pretrained(str(WEIGHTS_DIR), local_files_only=True)
    model.eval()
    enc = model.encoder
    blocks = enc.down_blocks
    if block_index < 0 or block_index >= len(blocks):
        raise IndexError(f'encoder.down_blocks.{block_index} out of range; model has {len(blocks)} blocks')
    block = blocks[block_index]
    sample = torch.linspace(-1.0, 1.0, steps=1 * 3 * args.input_size * args.input_size, dtype=torch.float32).reshape(1, 3, args.input_size, args.input_size)
    timesteps = torch.tensor([args.timestep], dtype=torch.long)
    hooks = []
    failures = []
    with torch.no_grad():
        time_proj = enc.time_proj(timesteps)
        temb = enc.time_embedding(time_proj.to(dtype=list(enc.time_embedding.parameters())[0].dtype))
        conv_in = enc.conv_in(sample)
        prev = conv_in
        previous_outputs = {}
        for i in range(block_index):
            prev = blocks[i](prev, temb=temb)
            previous_outputs[f'downblock{i}_output'] = to_np(prev)
        block_input = prev
        r0_in = block_input
        r0_out = block.resnets[0](r0_in, temb=temb)
        r1_in = r0_out
        r1_out = block.resnets[1](r1_in, temb=temb)
        ds_in = r1_out
        if getattr(block, 'downsamplers', None) is not None and len(block.downsamplers) > 0:
            ds_out = block.downsamplers[0](ds_in)
            final_out = ds_out
            hooks.append({'name': f'encoder.down_blocks.{block_index}.downsamplers.0', 'class': block.downsamplers[0].__class__.__name__, 'input_shape': list(ds_in.shape), 'output_shape': list(ds_out.shape), 'status': 'PASS', 'input_spatial_size': list(ds_in.shape[-2:]), 'output_spatial_size': list(ds_out.shape[-2:])})
        else:
            ds_out = None
            final_out = r1_out
            failures.append({'name': f'encoder.down_blocks.{block_index}.downsamplers.0', 'status': 'NOT_APPLICABLE', 'reason': 'official block has no downsampler'})
        direct_final = block(block_input, temb=temb)
    hooks += [
        {'name': 'encoder.time_proj', 'class': enc.time_proj.__class__.__name__, 'input_shape': list(timesteps.shape), 'output_shape': list(time_proj.shape), 'status': 'PASS'},
        {'name': 'encoder.time_embedding', 'class': enc.time_embedding.__class__.__name__, 'input_shape': list(time_proj.shape), 'output_shape': list(temb.shape), 'status': 'PASS'},
        {'name': 'encoder.conv_in', 'class': enc.conv_in.__class__.__name__, 'input_shape': list(sample.shape), 'output_shape': list(conv_in.shape), 'status': 'PASS'},
    ]
    for i, arr in previous_outputs.items():
        hooks.append({'name': f'encoder.down_blocks.{i[len("downblock"):-len("_output")]}', 'class': blocks[int(i[len("downblock"):-len("_output")])].__class__.__name__, 'output_shape': list(arr.shape), 'status': 'PASS'})
    hooks += [
        {'name': f'encoder.down_blocks.{block_index}.resnets.0', 'class': block.resnets[0].__class__.__name__, 'input_shape': list(r0_in.shape), 'temb_shape': list(temb.shape), 'output_shape': list(r0_out.shape), 'status': 'PASS', 'channel_change': r0_in.shape[1] != r0_out.shape[1], 'input_spatial_size': list(r0_in.shape[-2:]), 'output_spatial_size': list(r0_out.shape[-2:])},
        {'name': f'encoder.down_blocks.{block_index}.resnets.1', 'class': block.resnets[1].__class__.__name__, 'input_shape': list(r1_in.shape), 'temb_shape': list(temb.shape), 'output_shape': list(r1_out.shape), 'status': 'PASS', 'channel_change': r1_in.shape[1] != r1_out.shape[1], 'input_spatial_size': list(r1_in.shape[-2:]), 'output_spatial_size': list(r1_out.shape[-2:])},
        {'name': f'encoder.down_blocks.{block_index}', 'class': block.__class__.__name__, 'input_shape': list(block_input.shape), 'temb_shape': list(temb.shape), 'output_shape': list(final_out.shape), 'status': 'PASS', 'input_spatial_size': list(block_input.shape[-2:]), 'output_spatial_size': list(final_out.shape[-2:])},
    ]
    stage_key = 'stage' + ''.join(str(i) for i in range(block_index + 1))
    inputs = {
        'synthetic_image_tensor': to_np(sample),
        'timestep': timesteps.cpu().numpy().astype(np.int64),
        'encoder_conv_in_input': to_np(sample),
        f'{stage_key}_input': to_np(sample),
        f'downblock{block_index}_input': to_np(block_input),
        'resnet0_input': to_np(r0_in),
        'resnet1_input': to_np(r1_in),
        f'downblock{block_index}_temb': to_np(temb),
    }
    outputs = {
        'encoder_time_proj_output': to_np(time_proj),
        'encoder_time_embedding_output': to_np(temb),
        'encoder_conv_in_output': to_np(conv_in),
        'resnet0_output': to_np(r0_out),
        'resnet1_output': to_np(r1_out),
        f'downblock{block_index}_output': to_np(final_out),
        f'downblock{block_index}_direct_output': to_np(direct_final),
        f'{stage_key}_output': to_np(final_out),
    }
    outputs.update(previous_outputs)
    if ds_out is not None:
        inputs['downsampler0_input'] = to_np(ds_in)
        outputs['downsampler0_output'] = to_np(ds_out)
    np.savez_compressed(out / f'time_vae_downblock{block_index}_inputs.npz', **inputs)
    np.savez_compressed(out / f'time_vae_downblock{block_index}_outputs.npz', **outputs)
    meta = {
        'status': 'PASS',
        'block_index': block_index,
        'official_class': f'{model.__class__.__module__}.{model.__class__.__name__}',
        'encoder_class': f'{enc.__class__.__module__}.{enc.__class__.__name__}',
        'sample_shape': list(sample.shape),
        'timestep': int(timesteps.item()),
        'input_size': args.input_size,
        'module_order': ['encoder.conv_in'] + [f'encoder.down_blocks.{i}' for i in range(block_index)] + [f'encoder.down_blocks.{block_index}.resnets.0', f'encoder.down_blocks.{block_index}.resnets.1', f'encoder.down_blocks.{block_index}.downsamplers.0'],
        'channel_change_happens': r0_in.shape[1] != r0_out.shape[1] or r1_in.shape[1] != r1_out.shape[1],
        'has_downsampler': ds_out is not None,
        'hook_targets': hooks,
        'hook_failures': failures,
    }
    (out / f'time_vae_downblock{block_index}_hook_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    md = [f'# TimeAware VAE DownBlock{block_index} Oracle Export', '', f"Status: **{meta['status']}**", '', f"Input shape: `{meta['sample_shape']}`", f"Timestep: `{meta['timestep']}`", f"Channel change happens: `{meta['channel_change_happens']}`", '', '| Target | Class | Input shape | Output shape | Status | Channel change |', '|---|---|---:|---:|---|---|']
    for row in hooks:
        inp = row.get('input_shape')
        if row.get('temb_shape'):
            inp = f"{inp}, temb={row['temb_shape']}"
        md.append(f"| `{row['name']}` | `{row.get('class')}` | `{inp}` | `{row.get('output_shape')}` | {row['status']} | `{row.get('channel_change', '')}` |")
    if failures:
        md += ['', '## Hook failures / not applicable']
        for row in failures:
            md.append(f"- `{row['name']}`: {row.get('status')} {row.get('reason', '')}")
    (out / f'time_vae_downblock{block_index}_hook_report.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())