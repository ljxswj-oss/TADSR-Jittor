#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

OFFICIAL_REPO = Path('/mnt/data/sj/projects/TADSR_official_pytorch')
WEIGHTS_DIR = Path('/mnt/data/sj/checkpoints/TADSR/preset/weights/time_vae')
OUT = Path('experiments/full_repro/time_vae_alignment')


def module_tree(module, prefix: str):
    rows = []
    for name, child in module.named_modules():
        full = prefix if not name else f'{prefix}.{name}'
        row = {
            'name': full,
            'class': child.__class__.__name__,
            'module': f'{child.__class__.__module__}.{child.__class__.__name__}',
            'child_count': len(list(child.children())),
        }
        for attr in ['channels', 'out_channels', 'use_conv', 'padding', 'name', 'time_embedding_norm', 'output_scale_factor', 'in_channels']:
            if hasattr(child, attr):
                value = getattr(child, attr)
                row[attr] = value if isinstance(value, (int, float, str, bool, type(None))) else str(value)
        rows.append(row)
    return rows


def state_rows(model, block_index: int):
    rows = []
    prefix = f'encoder.down_blocks.{block_index}.'
    for key, tensor in model.state_dict().items():
        if key.startswith(prefix):
            rows.append({'key': key, 'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'numel': int(tensor.numel())})
    return rows


def resnet_info(block, block_index: int, idx: int):
    r = block.resnets[idx]
    conv_shortcut = getattr(r, 'conv_shortcut', None)
    nin_shortcut = getattr(r, 'nin_shortcut', None)
    tp = getattr(r, 'time_emb_proj', None)
    return {
        'module_name': f'encoder.down_blocks.{block_index}.resnets.{idx}',
        'class_name': r.__class__.__name__,
        'class_module': r.__class__.__module__,
        'in_channels': getattr(r, 'in_channels', None),
        'out_channels': getattr(r, 'out_channels', None),
        'time_embedding_norm': getattr(r, 'time_embedding_norm', None),
        'output_scale_factor': getattr(r, 'output_scale_factor', None),
        'conv1_weight_shape': list(r.conv1.weight.shape),
        'conv2_weight_shape': list(r.conv2.weight.shape),
        'time_emb_proj_weight_shape': list(tp.weight.shape) if tp is not None else None,
        'time_emb_proj_out_features': int(tp.weight.shape[0]) if tp is not None else None,
        'expected_scale_shift_channels': int(r.conv2.weight.shape[0] * 2),
        'has_conv_shortcut': conv_shortcut is not None,
        'conv_shortcut_weight_shape': list(conv_shortcut.weight.shape) if conv_shortcut is not None else None,
        'has_nin_shortcut': nin_shortcut is not None,
        'nin_shortcut_weight_shape': list(nin_shortcut.weight.shape) if nin_shortcut is not None and hasattr(nin_shortcut, 'weight') else None,
    }


def downsampler_info(block, block_index: int):
    has_down = getattr(block, 'downsamplers', None) is not None and len(block.downsamplers) > 0
    if not has_down:
        return None
    ds = block.downsamplers[0]
    conv = getattr(ds, 'conv', None)
    return {
        'module_name': f'encoder.down_blocks.{block_index}.downsamplers.0',
        'class_name': ds.__class__.__name__,
        'class_module': ds.__class__.__module__,
        'use_conv': getattr(ds, 'use_conv', None),
        'padding': getattr(ds, 'padding', None),
        'channels': getattr(ds, 'channels', None),
        'out_channels': getattr(ds, 'out_channels', None),
        'conv_class': conv.__class__.__name__ if conv is not None else None,
        'conv_key': f'encoder.down_blocks.{block_index}.downsamplers.0.conv',
        'conv_kernel_size': list(conv.kernel_size) if conv is not None else None,
        'conv_stride': list(conv.stride) if conv is not None else None,
        'conv_padding': list(conv.padding) if conv is not None else None,
        'asymmetric_padding_when_padding_zero': getattr(ds, 'padding', None) == 0,
        'asymmetric_padding_note': 'PyTorch Downsample2D applies F.pad(hidden_states, (0, 1, 0, 1)) before conv when use_conv=True and padding=0.' if getattr(ds, 'padding', None) == 0 else '',
    }


def write_markdown(result: dict, block_index: int):
    md = [
        f'# Official TimeAware VAE DownBlock{block_index} Audit',
        '',
        f"Status: **{result['status']}**",
        '',
        f"Block class: `{result['block_module']}.{result['block_class']}`",
        f"ResNet count: `{result['resnet_count']}`",
        f"Has downsampler: `{result['has_downsampler']}`",
        f"Channel change detected: `{result['channel_change_detected']}`",
        '',
        '## ResNet modules',
        '',
        '| Module | In | Out | Time norm | Time proj shape | Conv shortcut | Shortcut shape | NIN shortcut |',
        '|---|---:|---:|---|---:|---|---:|---|',
    ]
    for info in result['resnet_info']:
        md.append(f"| `{info['module_name']}` | {info.get('in_channels')} | {info.get('out_channels')} | `{info.get('time_embedding_norm')}` | `{info.get('time_emb_proj_weight_shape')}` | {info.get('has_conv_shortcut')} | `{info.get('conv_shortcut_weight_shape')}` | {info.get('has_nin_shortcut')} |")
    if result.get('downsampler_info'):
        md += ['', '## Downsampler', '', '| Field | Value |', '|---|---|']
        for k, v in result['downsampler_info'].items():
            md.append(f'| {k} | `{v}` |')
    md += ['', '## State Keys', '', '| Key | Shape | DType |', '|---|---:|---|']
    for row in result['state_keys']:
        md.append(f"| `{row['key']}` | `{row['shape']}` | `{row['dtype']}` |")
    (OUT / f'downblock{block_index}_audit.md').write_text('\n'.join(md) + '\n', encoding='utf-8')

    map_md = [f'# DownBlock{block_index} PyTorch-to-NPZ Key Mapping', '', '| PyTorch key | Jittor NPZ key | Shape |', '|---|---|---:|']
    for row in result['key_mapping']:
        map_md.append(f"| `{row['pytorch_key']}` | `{row['jittor_npz_key']}` | `{row['shape']}` |")
    (OUT / f'downblock{block_index}_key_mapping.md').write_text('\n'.join(map_md) + '\n', encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--block-index', type=int, required=True)
    args = parser.parse_args()
    block_index = args.block_index
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(OFFICIAL_REPO))
    from diffusers.models.autoencoders.time_autoencoder_kl import TimeAwareAutoencoderKL

    model = TimeAwareAutoencoderKL.from_pretrained(str(WEIGHTS_DIR), local_files_only=True)
    model.eval()
    blocks = model.encoder.down_blocks
    if block_index < 0 or block_index >= len(blocks):
        raise IndexError(f'encoder.down_blocks.{block_index} out of range; model has {len(blocks)} blocks')
    block = blocks[block_index]
    rows = module_tree(block, f'encoder.down_blocks.{block_index}')
    states = state_rows(model, block_index)
    resnets = [resnet_info(block, block_index, i) for i in range(len(block.resnets))]
    ds_info = downsampler_info(block, block_index)
    mapping = [{'pytorch_key': r['key'], 'jittor_npz_key': r['key'].replace('.', '__'), 'shape': r['shape']} for r in states]
    result = {
        'status': 'PASS',
        'block_index': block_index,
        'block_class': block.__class__.__name__,
        'block_module': block.__class__.__module__,
        'resnet_count': len(block.resnets),
        'has_resnet0': len(block.resnets) > 0,
        'has_resnet1': len(block.resnets) > 1,
        'has_downsampler': ds_info is not None,
        'resnet_info': resnets,
        'downsampler_info': ds_info,
        'channel_change_detected': any(x.get('in_channels') != x.get('out_channels') for x in resnets),
        'module_tree': rows,
        'state_keys': states,
        'key_mapping': mapping,
    }
    (OUT / f'downblock{block_index}_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (OUT / f'downblock{block_index}_key_mapping.json').write_text(json.dumps({'status': result['status'], 'rows': mapping}, indent=2), encoding='utf-8')
    write_markdown(result, block_index)
    print(json.dumps({k: result[k] for k in ['status', 'block_index', 'resnet_count', 'has_downsampler', 'channel_change_detected', 'resnet_info', 'downsampler_info']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())