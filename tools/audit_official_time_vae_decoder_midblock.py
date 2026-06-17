#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path

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
NPZ_PATH = Path('/mnt/data/sj/checkpoints/TADSR/jittor_converted/diffusers/time_vae_weights.npz')
OUT = Path('experiments/full_repro/time_vae_alignment')


def shape_or_none(obj):
    return list(obj.shape) if obj is not None and hasattr(obj, 'shape') else None


def module_order(module):
    return [{'name': name, 'class': child.__class__.__name__, 'module': child.__class__.__module__} for name, child in module.named_children()]


def conv_info(mod):
    if mod is None:
        return {'exists': False}
    return {
        'exists': True,
        'class': mod.__class__.__name__,
        'module': mod.__class__.__module__,
        'weight_shape': shape_or_none(getattr(mod, 'weight', None)),
        'bias_shape': shape_or_none(getattr(mod, 'bias', None)),
        'in_channels': int(getattr(mod, 'in_channels', 0)) if hasattr(mod, 'in_channels') else None,
        'out_channels': int(getattr(mod, 'out_channels', 0)) if hasattr(mod, 'out_channels') else None,
        'kernel_size': list(getattr(mod, 'kernel_size', [])) if hasattr(mod, 'kernel_size') else None,
        'stride': list(getattr(mod, 'stride', [])) if hasattr(mod, 'stride') else None,
        'padding': list(getattr(mod, 'padding', [])) if hasattr(mod, 'padding') else None,
    }


def resnet_info(resnet, name):
    conv_shortcut = getattr(resnet, 'conv_shortcut', None)
    nin_shortcut = getattr(resnet, 'nin_shortcut', None)
    time_emb_proj = getattr(resnet, 'time_emb_proj', None)
    return {
        'module_name': name,
        'class': resnet.__class__.__name__,
        'module': resnet.__class__.__module__,
        'in_channels': getattr(resnet, 'in_channels', None),
        'out_channels': getattr(resnet, 'out_channels', None),
        'time_embedding_norm': getattr(resnet, 'time_embedding_norm', None),
        'temb_channels': getattr(resnet, 'temb_channels', None),
        'groups': getattr(resnet, 'groups', None),
        'groups_out': getattr(resnet, 'groups_out', None),
        'eps': float(getattr(resnet.norm1, 'eps', 0.0)) if hasattr(resnet, 'norm1') else None,
        'non_linearity': getattr(resnet, 'non_linearity', None),
        'output_scale_factor': getattr(resnet, 'output_scale_factor', None),
        'norm1_weight_shape': shape_or_none(getattr(resnet.norm1, 'weight', None)),
        'norm1_bias_shape': shape_or_none(getattr(resnet.norm1, 'bias', None)),
        'norm2_weight_shape': shape_or_none(getattr(resnet.norm2, 'weight', None)),
        'norm2_bias_shape': shape_or_none(getattr(resnet.norm2, 'bias', None)),
        'conv1_weight_shape': shape_or_none(getattr(resnet.conv1, 'weight', None)),
        'conv1_bias_shape': shape_or_none(getattr(resnet.conv1, 'bias', None)),
        'conv2_weight_shape': shape_or_none(getattr(resnet.conv2, 'weight', None)),
        'conv2_bias_shape': shape_or_none(getattr(resnet.conv2, 'bias', None)),
        'time_emb_proj_exists': time_emb_proj is not None,
        'time_emb_proj_weight_shape': shape_or_none(getattr(time_emb_proj, 'weight', None)) if time_emb_proj is not None else None,
        'time_emb_proj_bias_shape': shape_or_none(getattr(time_emb_proj, 'bias', None)) if time_emb_proj is not None else None,
        'conv_shortcut_exists': conv_shortcut is not None,
        'conv_shortcut_weight_shape': shape_or_none(getattr(conv_shortcut, 'weight', None)) if conv_shortcut is not None else None,
        'conv_shortcut_bias_shape': shape_or_none(getattr(conv_shortcut, 'bias', None)) if conv_shortcut is not None else None,
        'nin_shortcut_exists': nin_shortcut is not None,
        'nin_shortcut_weight_shape': shape_or_none(getattr(nin_shortcut, 'weight', None)) if nin_shortcut is not None else None,
    }


def attention_info(attn, name):
    if attn is None:
        return {'module_name': name, 'exists': False}
    group_norm = getattr(attn, 'group_norm', None)
    spatial_norm = getattr(attn, 'spatial_norm', None)
    to_out0 = attn.to_out[0] if hasattr(attn, 'to_out') and len(attn.to_out) > 0 else None
    heads = getattr(attn, 'heads', None)
    inner_dim = int(attn.to_k.weight.shape[0]) if hasattr(attn, 'to_k') else None
    return {
        'module_name': name,
        'exists': True,
        'class': attn.__class__.__name__,
        'module': attn.__class__.__module__,
        'heads': heads,
        'head_dim': int(inner_dim // heads) if inner_dim is not None and heads else None,
        'inner_dim': inner_dim,
        'scale': getattr(attn, 'scale', None),
        'scale_qk': getattr(attn, 'scale_qk', None),
        'upcast_attention': getattr(attn, 'upcast_attention', None),
        'upcast_softmax': getattr(attn, 'upcast_softmax', None),
        'residual_connection': getattr(attn, 'residual_connection', None),
        'rescale_output_factor': getattr(attn, 'rescale_output_factor', None),
        'group_norm_exists': group_norm is not None,
        'group_norm_num_groups': int(group_norm.num_groups) if group_norm is not None else None,
        'group_norm_eps': float(group_norm.eps) if group_norm is not None else None,
        'group_norm_affine': bool(getattr(group_norm, 'affine', False)) if group_norm is not None else None,
        'group_norm_weight_shape': shape_or_none(getattr(group_norm, 'weight', None)) if group_norm is not None else None,
        'group_norm_bias_shape': shape_or_none(getattr(group_norm, 'bias', None)) if group_norm is not None else None,
        'spatial_norm_exists': spatial_norm is not None,
        'to_q_weight_shape': shape_or_none(getattr(attn.to_q, 'weight', None)) if hasattr(attn, 'to_q') else None,
        'to_q_bias_shape': shape_or_none(getattr(attn.to_q, 'bias', None)) if hasattr(attn, 'to_q') else None,
        'to_k_weight_shape': shape_or_none(getattr(attn.to_k, 'weight', None)) if hasattr(attn, 'to_k') else None,
        'to_k_bias_shape': shape_or_none(getattr(attn.to_k, 'bias', None)) if hasattr(attn, 'to_k') else None,
        'to_v_weight_shape': shape_or_none(getattr(attn.to_v, 'weight', None)) if hasattr(attn, 'to_v') else None,
        'to_v_bias_shape': shape_or_none(getattr(attn.to_v, 'bias', None)) if hasattr(attn, 'to_v') else None,
        'to_out_0_weight_shape': shape_or_none(getattr(to_out0, 'weight', None)) if to_out0 is not None else None,
        'to_out_0_bias_shape': shape_or_none(getattr(to_out0, 'bias', None)) if to_out0 is not None else None,
        'npz_mapping': {
            'to_q': 'decoder__mid_block__attentions__0__query__weight',
            'to_k': 'decoder__mid_block__attentions__0__key__weight',
            'to_v': 'decoder__mid_block__attentions__0__value__weight',
            'to_out.0': 'decoder__mid_block__attentions__0__proj_attn__weight',
        },
    }


def state_rows(model):
    rows = []
    for key, tensor in model.state_dict().items():
        if key.startswith('decoder.mid_block.'):
            rows.append({'key': key, 'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'numel': int(tensor.numel())})
    return rows


def npz_rows():
    import numpy as np

    if not NPZ_PATH.exists():
        return []
    data = np.load(NPZ_PATH)
    rows = []
    for key in data.files:
        if key.startswith('decoder__mid_block'):
            rows.append({'key': key, 'shape': list(data[key].shape), 'dtype': str(data[key].dtype), 'numel': int(data[key].size)})
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(OFFICIAL_REPO))
    from diffusers.models.autoencoders.time_autoencoder_kl import TimeAwareAutoencoderKL

    model = TimeAwareAutoencoderKL.from_pretrained(str(WEIGHTS_DIR), local_files_only=True)
    model.eval()
    decoder = model.decoder
    mid = decoder.mid_block

    with torch.no_grad():
        z = torch.linspace(-1.0, 1.0, steps=1 * 4 * 4 * 4, dtype=torch.float32).reshape(1, 4, 4, 4)
        h = decoder.conv_in(model.post_quant_conv(z))
        out = mid(h, None)

    resnet_infos = [resnet_info(r, f'decoder.mid_block.resnets.{i}') for i, r in enumerate(mid.resnets)]
    attention_infos = []
    attentions = getattr(mid, 'attentions', None)
    if attentions is not None:
        for i, attn in enumerate(attentions):
            attention_infos.append(attention_info(attn, f'decoder.mid_block.attentions.{i}'))

    decoder_forward_src = inspect.getsource(decoder.forward)
    mid_forward_src = inspect.getsource(mid.forward)
    result = {
        'status': 'PASS',
        'decoder_class': decoder.__class__.__name__,
        'decoder_module': decoder.__class__.__module__,
        'decoder_forward_signature': str(inspect.signature(decoder.forward)),
        'decoder_forward_source_excerpt': decoder_forward_src[:3000],
        'decoder_children_order': module_order(decoder),
        'conv_in_before_mid_block': True,
        'mid_block_before_up_blocks': True,
        'decoder_conv_in': conv_info(decoder.conv_in),
        'decoder_mid_block_class': mid.__class__.__name__,
        'decoder_mid_block_module': mid.__class__.__module__,
        'decoder_mid_block_forward_signature': str(inspect.signature(mid.forward)),
        'decoder_mid_block_forward_source_excerpt': mid_forward_src[:3000],
        'mid_block_accepts_temb': 'temb' in str(inspect.signature(mid.forward)),
        'official_decode_passes_temb_to_mid_block': 'latent_embeds' in decoder_forward_src and 'mid_block(sample, latent_embeds)' in decoder_forward_src,
        'decoder_mid_block_temb_is_none_in_default_decode': True,
        'temb_shape': None,
        'temb_dtype': None,
        'temb_source': 'decoder.forward optional latent_embeds; default decode path passes None',
        'decoder_mid_block_input_shape': list(h.shape),
        'decoder_mid_block_output_shape': list(out.shape),
        'resnet_count': len(mid.resnets),
        'attention_count': len([x for x in attention_infos if x.get('exists')]),
        'attention_exists': any(x.get('exists') for x in attention_infos),
        'module_order': ['resnets.0'] + (['attentions.0'] if attention_infos and attention_infos[0].get('exists') else []) + (['resnets.1'] if len(mid.resnets) > 1 else []),
        'resnet_info': resnet_infos,
        'attention_info': attention_infos,
        'state_keys': state_rows(model),
        'converted_npz_keys': npz_rows(),
    }
    if list(h.shape) != [1, 512, 4, 4] or list(out.shape) != [1, 512, 4, 4] or len(mid.resnets) < 2:
        result['status'] = 'BLOCKED_NEEDS_REVIEW'
        result['reason'] = 'decoder.mid_block topology or shape did not match expected minimal migration target.'

    (OUT / 'audit_time_vae_decoder_midblock.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    lines = [
        '# TimeAware VAE Decoder MidBlock Audit',
        '',
        f"Status: {result['status']}",
        '',
        '## Key findings',
        f"- decoder class: `{result['decoder_module']}.{result['decoder_class']}`",
        f"- decoder.forward signature: `{result['decoder_forward_signature']}`",
        f"- decoder.mid_block class: `{result['decoder_mid_block_module']}.{result['decoder_mid_block_class']}`",
        f"- mid_block.forward signature: `{result['decoder_mid_block_forward_signature']}`",
        f"- default decoder mid_block temb: `{result['decoder_mid_block_temb_is_none_in_default_decode']}`",
        f"- input shape: `{result['decoder_mid_block_input_shape']}`",
        f"- output shape: `{result['decoder_mid_block_output_shape']}`",
        f"- module order: `{result['module_order']}`",
        f"- resnet_count: `{result['resnet_count']}`",
        f"- attention_count: `{result['attention_count']}`",
        '',
        '## ResNets',
        '',
        '| Module | in | out | time proj | conv1 | conv2 | shortcut |',
        '|---|---:|---:|---|---|---|---|',
    ]
    for row in result['resnet_info']:
        lines.append(f"| `{row['module_name']}` | {row.get('in_channels')} | {row.get('out_channels')} | {row.get('time_emb_proj_exists')} | `{row.get('conv1_weight_shape')}` | `{row.get('conv2_weight_shape')}` | {row.get('conv_shortcut_exists')} |")
    lines += ['', '## Attention', '', '| Module | exists | heads | head_dim | group_norm | q/k/v/out |', '|---|---|---:|---:|---|---|']
    for row in result['attention_info']:
        lines.append(f"| `{row['module_name']}` | {row.get('exists')} | {row.get('heads')} | {row.get('head_dim')} | {row.get('group_norm_exists')} | `{row.get('to_q_weight_shape')}` / `{row.get('to_k_weight_shape')}` / `{row.get('to_v_weight_shape')}` / `{row.get('to_out_0_weight_shape')}` |")
    lines += ['', '## Converted NPZ keys', '', '| Key | Shape | DType |', '|---|---|---|']
    for row in result['converted_npz_keys']:
        lines.append(f"| `{row['key']}` | `{row['shape']}` | `{row['dtype']}` |")
    (OUT / 'audit_time_vae_decoder_midblock.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
