#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

OFFICIAL_REPO = Path('/mnt/data/sj/projects/TADSR_official_pytorch')
WEIGHTS_DIR = Path('/mnt/data/sj/checkpoints/TADSR/preset/weights/time_vae')
OUT = Path('experiments/full_repro/time_vae_alignment/oracle_tensors')


def to_np(x):
    return x.detach().cpu().numpy().astype(np.float32)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(OFFICIAL_REPO))
    from diffusers.models.autoencoders.time_autoencoder_kl import TimeAwareAutoencoderKL

    torch.manual_seed(42)
    model = TimeAwareAutoencoderKL.from_pretrained(str(WEIGHTS_DIR), local_files_only=True)
    model.eval()
    encoder = model.encoder

    sample = torch.linspace(-1.0, 1.0, steps=1 * 3 * 32 * 32, dtype=torch.float32).reshape(1, 3, 32, 32)
    timesteps = torch.tensor([50], dtype=torch.long)

    metadata = {
        'status': 'PASS',
        'official_class': f'{model.__class__.__module__}.{model.__class__.__name__}',
        'encoder_class': f'{encoder.__class__.__module__}.{encoder.__class__.__name__}',
        'weights_dir': str(WEIGHTS_DIR),
        'sample_shape': list(sample.shape),
        'timestep': int(timesteps.item()),
        'hook_targets': [],
        'hook_failures': [],
    }

    with torch.no_grad():
        time_proj_output = encoder.time_proj(timesteps)
        time_embedding_output = encoder.time_embedding(time_proj_output.to(dtype=list(encoder.time_embedding.parameters())[0].dtype))
        conv_in_output = encoder.conv_in(sample)
        first_resnet = encoder.down_blocks[0].resnets[0]
        first_resnet_output = first_resnet(conv_in_output, temb=time_embedding_output)
        try:
            first_down_block_output = encoder.down_blocks[0](conv_in_output, temb=time_embedding_output)
            metadata['hook_targets'].append({'name': 'encoder.down_blocks.0', 'input_shape': list(conv_in_output.shape), 'output_shape': list(first_down_block_output.shape), 'status': 'PASS'})
        except Exception as exc:
            first_down_block_output = None
            metadata['hook_failures'].append({'name': 'encoder.down_blocks.0', 'error': repr(exc)})

    inputs = {
        'synthetic_image_tensor': to_np(sample),
        'timestep': timesteps.detach().cpu().numpy().astype(np.int64),
        'encoder_conv_in_input': to_np(sample),
        'first_resnet_input': to_np(conv_in_output),
        'first_resnet_temb': to_np(time_embedding_output),
    }
    outputs = {
        'encoder_time_proj_output': to_np(time_proj_output),
        'encoder_time_embedding_output': to_np(time_embedding_output),
        'encoder_conv_in_output': to_np(conv_in_output),
        'first_resnet_output': to_np(first_resnet_output),
    }
    if first_down_block_output is not None:
        outputs['first_down_block_output'] = to_np(first_down_block_output)

    np.savez_compressed(OUT / 'time_vae_oracle_inputs.npz', **inputs)
    np.savez_compressed(OUT / 'time_vae_oracle_outputs.npz', **outputs)

    metadata['hook_targets'] += [
        {'name': 'encoder.time_proj', 'input_shape': list(timesteps.shape), 'output_shape': list(time_proj_output.shape), 'status': 'PASS'},
        {'name': 'encoder.time_embedding', 'input_shape': list(time_proj_output.shape), 'output_shape': list(time_embedding_output.shape), 'status': 'PASS'},
        {'name': 'encoder.conv_in', 'input_shape': list(sample.shape), 'output_shape': list(conv_in_output.shape), 'status': 'PASS'},
        {'name': 'encoder.down_blocks.0.resnets.0', 'input_shape': list(conv_in_output.shape), 'temb_shape': list(time_embedding_output.shape), 'output_shape': list(first_resnet_output.shape), 'status': 'PASS'},
    ]
    (OUT / 'time_vae_hook_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    md = ['# TimeAware VAE PyTorch Oracle Tensor Export', '', f"Status: **{metadata['status']}**", '', f"Official class: `{metadata['official_class']}`", f"Input shape: `{metadata['sample_shape']}`", f"Timestep: `{metadata['timestep']}`", '', '| Target | Input shape | Output shape | Status |', '|---|---:|---:|---|']
    for row in metadata['hook_targets']:
        inp = row.get('input_shape')
        if 'temb_shape' in row:
            inp = f"{inp}, temb={row['temb_shape']}"
        md.append(f"| `{row['name']}` | `{inp}` | `{row.get('output_shape')}` | {row['status']} |")
    if metadata['hook_failures']:
        md += ['', '## Hook failures']
        for row in metadata['hook_failures']:
            md.append(f"- `{row['name']}`: {row['error']}")
    (OUT / 'time_vae_hook_report.md').write_text('\n'.join(md) + '\n', encoding='utf-8')

    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
