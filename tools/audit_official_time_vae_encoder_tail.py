#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

OFFICIAL_REPO = Path('/mnt/data/sj/projects/TADSR_official_pytorch')
WEIGHTS_DIR = Path('/mnt/data/sj/checkpoints/TADSR/preset/weights/time_vae')
OUT = Path('experiments/full_repro/time_vae_alignment')


def conv_info(name: str, mod):
    return {
        'module_name': name,
        'class_name': mod.__class__.__name__,
        'class_module': mod.__class__.__module__,
        'weight_shape': list(mod.weight.shape),
        'bias_shape': list(mod.bias.shape) if mod.bias is not None else None,
        'kernel_size': list(mod.kernel_size),
        'stride': list(mod.stride),
        'padding': list(mod.padding),
        'in_channels': int(mod.in_channels),
        'out_channels': int(mod.out_channels),
    }


def state_rows(model):
    rows = []
    prefixes = ('encoder.conv_norm_out.', 'encoder.conv_out.', 'quant_conv.', 'post_quant_conv.')
    for key, tensor in model.state_dict().items():
        if key.startswith(prefixes):
            rows.append({'key': key, 'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'numel': int(tensor.numel())})
    return rows


def write_markdown(result: dict):
    md = ['# Official TimeAware VAE Encoder Tail + QuantConv Audit', '', f"Status: **{result['status']}**", '', '## Encoder tail', '', '| Module | Class | Key facts |', '|---|---|---|']
    norm = result['conv_norm_out']
    md.append(f"| `encoder.conv_norm_out` | `{norm['class_module']}.{norm['class_name']}` | groups={norm['num_groups']}, channels={norm['num_channels']}, eps={norm['eps']}, affine={norm['affine']} |")
    act = result['conv_act']
    md.append(f"| `encoder.conv_act` | `{act['class_module']}.{act['class_name']}` | activation={act['class_name']} |")
    conv = result['conv_out']
    md.append(f"| `encoder.conv_out` | `{conv['class_module']}.{conv['class_name']}` | weight={conv['weight_shape']}, stride={conv['stride']}, padding={conv['padding']}, out_channels={conv['out_channels']} |")
    md += ['', '## Quant conv', '', '| Module | Class | Weight | Kernel | Stride | Padding | Channels |', '|---|---|---:|---:|---:|---:|---|']
    q = result['quant_conv']
    md.append(f"| `quant_conv` | `{q['class_module']}.{q['class_name']}` | `{q['weight_shape']}` | `{q['kernel_size']}` | `{q['stride']}` | `{q['padding']}` | {q['in_channels']} -> {q['out_channels']} |")
    if result.get('post_quant_conv'):
        pq = result['post_quant_conv']
        md.append(f"| `post_quant_conv` | `{pq['class_module']}.{pq['class_name']}` | `{pq['weight_shape']}` | `{pq['kernel_size']}` | `{pq['stride']}` | `{pq['padding']}` | {pq['in_channels']} -> {pq['out_channels']} |")
    md += ['', '## State Keys', '', '| Key | Shape | DType |', '|---|---:|---|']
    for row in result['state_keys']:
        md.append(f"| `{row['key']}` | `{row['shape']}` | `{row['dtype']}` |")
    (OUT / 'encoder_tail_audit.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    map_md = ['# Encoder Tail + QuantConv PyTorch-to-NPZ Key Mapping', '', '| PyTorch key | Jittor NPZ key | Shape |', '|---|---|---:|']
    for row in result['key_mapping']:
        map_md.append(f"| `{row['pytorch_key']}` | `{row['jittor_npz_key']}` | `{row['shape']}` |")
    (OUT / 'encoder_tail_key_mapping.md').write_text('\n'.join(map_md) + '\n', encoding='utf-8')


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(OFFICIAL_REPO))
    from diffusers.models.autoencoders.time_autoencoder_kl import TimeAwareAutoencoderKL

    model = TimeAwareAutoencoderKL.from_pretrained(str(WEIGHTS_DIR), local_files_only=True)
    model.eval()
    enc = model.encoder
    norm = enc.conv_norm_out
    conv_out = enc.conv_out
    quant = model.quant_conv
    post = getattr(model, 'post_quant_conv', None)
    states = state_rows(model)
    result = {
        'status': 'PASS',
        'conv_norm_out': {
            'module_name': 'encoder.conv_norm_out',
            'class_name': norm.__class__.__name__,
            'class_module': norm.__class__.__module__,
            'num_groups': int(norm.num_groups),
            'num_channels': int(norm.num_channels),
            'eps': float(norm.eps),
            'affine': bool(norm.affine),
            'weight_key': 'encoder.conv_norm_out.weight',
            'bias_key': 'encoder.conv_norm_out.bias',
            'weight_shape': list(norm.weight.shape),
            'bias_shape': list(norm.bias.shape),
        },
        'conv_act': {
            'module_name': 'encoder.conv_act',
            'class_name': enc.conv_act.__class__.__name__,
            'class_module': enc.conv_act.__class__.__module__,
        },
        'conv_out': conv_info('encoder.conv_out', conv_out),
        'conv_out_outputs_double_z': int(conv_out.out_channels) % 2 == 0,
        'quant_conv': conv_info('quant_conv', quant),
        'quant_conv_channel_mapping': f'{int(quant.in_channels)}->{int(quant.out_channels)}',
        'post_quant_conv_exists': post is not None,
        'post_quant_conv': conv_info('post_quant_conv', post) if post is not None else None,
        'post_quant_conv_note': 'Recorded but not ported in this stage.',
        'state_keys': states,
        'key_mapping': [{'pytorch_key': row['key'], 'jittor_npz_key': row['key'].replace('.', '__'), 'shape': row['shape']} for row in states if not row['key'].startswith('post_quant_conv.')],
    }
    (OUT / 'encoder_tail_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (OUT / 'encoder_tail_key_mapping.json').write_text(json.dumps({'status': 'PASS', 'rows': result['key_mapping']}, indent=2), encoding='utf-8')
    write_markdown(result)
    print(json.dumps({k: result[k] for k in ['status', 'conv_norm_out', 'conv_act', 'conv_out', 'quant_conv', 'post_quant_conv_exists', 'post_quant_conv']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
