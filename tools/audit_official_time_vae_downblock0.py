#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

OFFICIAL_REPO = Path('/mnt/data/sj/projects/TADSR_official_pytorch')
WEIGHTS_DIR = Path('/mnt/data/sj/checkpoints/TADSR/preset/weights/time_vae')
OUT = Path('experiments/full_repro/time_vae_alignment')


def module_tree(module, prefix='encoder.down_blocks.0'):
    rows = []
    for name, child in module.named_modules():
        full = prefix if not name else f'{prefix}.{name}'
        row = {
            'name': full,
            'class': child.__class__.__name__,
            'module': f'{child.__class__.__module__}.{child.__class__.__name__}',
            'child_count': len(list(child.children())),
        }
        for attr in ['channels', 'out_channels', 'use_conv', 'padding', 'name', 'stride', 'kernel_size']:
            if hasattr(child, attr):
                value = getattr(child, attr)
                row[attr] = str(value) if not isinstance(value, (int, float, str, bool, type(None))) else value
        rows.append(row)
    return rows


def state_rows(model):
    rows = []
    for key, tensor in model.state_dict().items():
        if key.startswith('encoder.down_blocks.0.'):
            rows.append({'key': key, 'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'numel': int(tensor.numel())})
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(OFFICIAL_REPO))
    from diffusers.models.autoencoders.time_autoencoder_kl import TimeAwareAutoencoderKL

    model = TimeAwareAutoencoderKL.from_pretrained(str(WEIGHTS_DIR), local_files_only=True)
    model.eval()
    block = model.encoder.down_blocks[0]
    rows = module_tree(block)
    states = state_rows(model)
    has_resnet1 = hasattr(block, 'resnets') and len(block.resnets) > 1
    has_downsampler = getattr(block, 'downsamplers', None) is not None and len(block.downsamplers) > 0
    downsampler_info = None
    if has_downsampler:
        ds = block.downsamplers[0]
        conv = getattr(ds, 'conv', None)
        downsampler_info = {
            'module_name': 'encoder.down_blocks.0.downsamplers.0',
            'class_name': ds.__class__.__name__,
            'class_module': ds.__class__.__module__,
            'use_conv': getattr(ds, 'use_conv', None),
            'padding': getattr(ds, 'padding', None),
            'channels': getattr(ds, 'channels', None),
            'out_channels': getattr(ds, 'out_channels', None),
            'conv_class': conv.__class__.__name__ if conv is not None else None,
            'conv_kernel_size': list(conv.kernel_size) if conv is not None and hasattr(conv, 'kernel_size') else None,
            'conv_stride': list(conv.stride) if conv is not None and hasattr(conv, 'stride') else None,
            'conv_padding': list(conv.padding) if conv is not None and hasattr(conv, 'padding') else None,
            'asymmetric_padding_when_padding_zero': getattr(ds, 'padding', None) == 0,
            'asymmetric_padding_note': 'PyTorch Downsample2D applies F.pad(hidden_states, (0, 1, 0, 1)) before conv when use_conv=True and padding=0.' if getattr(ds, 'padding', None) == 0 else '',
        }
    mapping = []
    for row in states:
        pt_key = row['key']
        jt_key = pt_key.replace('.', '__')
        mapping.append({'pytorch_key': pt_key, 'jittor_npz_key': jt_key, 'shape': row['shape']})
    result = {
        'status': 'PASS' if has_resnet1 else 'PARTIAL',
        'block_class': block.__class__.__name__,
        'block_module': block.__class__.__module__,
        'resnet_count': len(block.resnets) if hasattr(block, 'resnets') else 0,
        'has_resnet1': has_resnet1,
        'has_downsampler': has_downsampler,
        'downsampler_info': downsampler_info,
        'module_tree': rows,
        'state_keys': states,
        'key_mapping': mapping,
    }
    (OUT / 'downblock0_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (OUT / 'downblock0_key_mapping.json').write_text(json.dumps({'status': result['status'], 'rows': mapping}, indent=2), encoding='utf-8')

    md = ['# Official TimeAware VAE DownBlock0 Audit', '', f"Status: **{result['status']}**", '', f"Block class: `{result['block_module']}.{result['block_class']}`", f"ResNet count: `{result['resnet_count']}`", f"Has downsampler: `{result['has_downsampler']}`"]
    if downsampler_info:
        md += ['', '## Downsampler', '', '| Field | Value |', '|---|---|']
        for k, v in downsampler_info.items():
            md.append(f'| {k} | `{v}` |')
    md += ['', '## Module Tree', '', '| Name | Class | Children |', '|---|---|---:|']
    for row in rows:
        md.append(f"| `{row['name']}` | `{row['class']}` | {row['child_count']} |")
    md += ['', '## State Keys', '', '| Key | Shape | DType |', '|---|---:|---|']
    for row in states:
        md.append(f"| `{row['key']}` | `{row['shape']}` | `{row['dtype']}` |")
    (OUT / 'downblock0_audit.md').write_text('\n'.join(md) + '\n', encoding='utf-8')

    map_md = ['# DownBlock0 PyTorch-to-NPZ Key Mapping', '', '| PyTorch key | Jittor NPZ key | Shape |', '|---|---|---:|']
    for row in mapping:
        map_md.append(f"| `{row['pytorch_key']}` | `{row['jittor_npz_key']}` | `{row['shape']}` |")
    (OUT / 'downblock0_key_mapping.md').write_text('\n'.join(map_md) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['status', 'block_class', 'resnet_count', 'has_downsampler', 'downsampler_info']}, indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
