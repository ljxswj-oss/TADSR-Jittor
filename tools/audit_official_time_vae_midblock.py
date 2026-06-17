#!/usr/bin/env python3
from __future__ import annotations

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
        for attr in ['add_attention', 'time_embedding_norm', 'output_scale_factor', 'rescale_output_factor', 'heads', 'scale', 'scale_qk', 'upcast_attention', 'upcast_softmax', 'residual_connection', 'sliceable_head_dim']:
            if hasattr(child, attr):
                value = getattr(child, attr)
                row[attr] = value if isinstance(value, (int, float, str, bool, type(None))) else str(value)
        rows.append(row)
    return rows


def state_rows(model):
    rows = []
    for key, tensor in model.state_dict().items():
        if key.startswith('encoder.mid_block.'):
            rows.append({'key': key, 'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'numel': int(tensor.numel())})
    return rows


def resnet_info(block, idx: int):
    r = block.resnets[idx]
    conv_shortcut = getattr(r, 'conv_shortcut', None)
    nin_shortcut = getattr(r, 'nin_shortcut', None)
    tp = getattr(r, 'time_emb_proj', None)
    return {
        'module_name': f'encoder.mid_block.resnets.{idx}',
        'class_name': r.__class__.__name__,
        'class_module': r.__class__.__module__,
        'in_channels': getattr(r, 'in_channels', None),
        'out_channels': getattr(r, 'out_channels', None),
        'time_embedding_norm': getattr(r, 'time_embedding_norm', None),
        'output_scale_factor': getattr(r, 'output_scale_factor', None),
        'norm1_channels': int(r.norm1.weight.shape[0]),
        'norm2_channels': int(r.norm2.weight.shape[0]),
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


def attention_info(block, idx: int):
    attentions = getattr(block, 'attentions', None)
    if attentions is None or idx >= len(attentions) or attentions[idx] is None:
        return None
    a = attentions[idx]
    info = {
        'module_name': f'encoder.mid_block.attentions.{idx}',
        'class_name': a.__class__.__name__,
        'class_module': a.__class__.__module__,
        'has_group_norm': getattr(a, 'group_norm', None) is not None,
        'has_spatial_norm': getattr(a, 'spatial_norm', None) is not None,
        'heads': getattr(a, 'heads', None),
        'scale': getattr(a, 'scale', None),
        'scale_qk': getattr(a, 'scale_qk', None),
        'upcast_attention': getattr(a, 'upcast_attention', None),
        'upcast_softmax': getattr(a, 'upcast_softmax', None),
        'residual_connection': getattr(a, 'residual_connection', None),
        'rescale_output_factor': getattr(a, 'rescale_output_factor', None),
        'processor_class': getattr(a, 'processor', None).__class__.__name__ if getattr(a, 'processor', None) is not None else None,
        'to_q_weight_shape': list(a.to_q.weight.shape),
        'to_k_weight_shape': list(a.to_k.weight.shape),
        'to_v_weight_shape': list(a.to_v.weight.shape),
        'to_out_0_weight_shape': list(a.to_out[0].weight.shape),
        'group_norm_channels': int(a.group_norm.weight.shape[0]) if getattr(a, 'group_norm', None) is not None else None,
        'group_norm_groups': int(a.group_norm.num_groups) if getattr(a, 'group_norm', None) is not None else None,
        'group_norm_eps': float(a.group_norm.eps) if getattr(a, 'group_norm', None) is not None else None,
    }
    inner_dim = int(a.to_k.weight.shape[0])
    heads = int(info['heads']) if info['heads'] is not None else 1
    info['inner_dim'] = inner_dim
    info['head_dim'] = inner_dim // heads
    return info


def npz_key_for_state_key(key: str) -> str:
    key = key.replace('encoder.mid_block.attentions.0.to_q.', 'encoder.mid_block.attentions.0.query.')
    key = key.replace('encoder.mid_block.attentions.0.to_k.', 'encoder.mid_block.attentions.0.key.')
    key = key.replace('encoder.mid_block.attentions.0.to_v.', 'encoder.mid_block.attentions.0.value.')
    key = key.replace('encoder.mid_block.attentions.0.to_out.0.', 'encoder.mid_block.attentions.0.proj_attn.')
    return key.replace('.', '__')


def write_markdown(result: dict):
    md = [
        '# Official TimeAware VAE MidBlock Audit', '',
        f"Status: **{result['status']}**", '',
        f"MidBlock class: `{result['mid_block_module']}.{result['mid_block_class']}`",
        f"ResNet count: `{result['resnet_count']}`",
        f"Attention count: `{result['attention_count']}`", '',
        '## ResNet modules', '',
        '| Module | In | Out | Time norm | Time proj shape | Conv shortcut | Shortcut shape | NIN shortcut |',
        '|---|---:|---:|---|---:|---|---:|---|',
    ]
    for info in result['resnet_info']:
        md.append(f"| `{info['module_name']}` | {info.get('in_channels')} | {info.get('out_channels')} | `{info.get('time_embedding_norm')}` | `{info.get('time_emb_proj_weight_shape')}` | {info.get('has_conv_shortcut')} | `{info.get('conv_shortcut_weight_shape')}` | {info.get('has_nin_shortcut')} |")
    md += ['', '## Attention modules', '']
    if result['attention_info']:
        md += ['| Module | Heads | Head dim | GroupNorm | Q/K/V shape | Out shape | Residual | Rescale |', '|---|---:|---:|---|---|---|---|---:|']
        for info in result['attention_info']:
            md.append(f"| `{info['module_name']}` | {info.get('heads')} | {info.get('head_dim')} | {info.get('has_group_norm')} | `{info.get('to_q_weight_shape')}` / `{info.get('to_k_weight_shape')}` / `{info.get('to_v_weight_shape')}` | `{info.get('to_out_0_weight_shape')}` | {info.get('residual_connection')} | `{info.get('rescale_output_factor')}` |")
    else:
        md.append('Official `encoder.mid_block` has no attention modules.')
    md += ['', '## State Keys', '', '| Key | Shape | DType |', '|---|---:|---|']
    for row in result['state_keys']:
        md.append(f"| `{row['key']}` | `{row['shape']}` | `{row['dtype']}` |")
    (OUT / 'midblock_audit.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    map_md = ['# MidBlock PyTorch-to-NPZ Key Mapping', '', '| PyTorch key | Jittor NPZ key | Shape |', '|---|---|---:|']
    for row in result['key_mapping']:
        map_md.append(f"| `{row['pytorch_key']}` | `{row['jittor_npz_key']}` | `{row['shape']}` |")
    (OUT / 'midblock_key_mapping.md').write_text('\n'.join(map_md) + '\n', encoding='utf-8')


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(OFFICIAL_REPO))
    from diffusers.models.autoencoders.time_autoencoder_kl import TimeAwareAutoencoderKL
    model = TimeAwareAutoencoderKL.from_pretrained(str(WEIGHTS_DIR), local_files_only=True)
    model.eval()
    block = model.encoder.mid_block
    states = state_rows(model)
    attention_infos = []
    attentions = getattr(block, 'attentions', None)
    if attentions is not None:
        for i in range(len(attentions)):
            info = attention_info(block, i)
            if info is not None:
                attention_infos.append(info)
    mapping = [{'pytorch_key': r['key'], 'jittor_npz_key': npz_key_for_state_key(r['key']), 'shape': r['shape']} for r in states]
    result = {
        'status': 'PASS',
        'mid_block_class': block.__class__.__name__,
        'mid_block_module': block.__class__.__module__,
        'add_attention': getattr(block, 'add_attention', None),
        'resnet_count': len(block.resnets),
        'attention_count': len(attention_infos),
        'has_attention0': len(attention_infos) > 0,
        'has_resnet1': len(block.resnets) > 1,
        'resnet_info': [resnet_info(block, i) for i in range(len(block.resnets))],
        'attention_info': attention_infos,
        'module_tree': module_tree(block, 'encoder.mid_block'),
        'state_keys': states,
        'key_mapping': mapping,
    }
    (OUT / 'midblock_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (OUT / 'midblock_key_mapping.json').write_text(json.dumps({'status': result['status'], 'rows': mapping}, indent=2), encoding='utf-8')
    write_markdown(result)
    print(json.dumps({k: result[k] for k in ['status', 'mid_block_class', 'resnet_count', 'attention_count', 'resnet_info', 'attention_info']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
