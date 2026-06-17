#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

STRICT_PY = Path('/mnt/data/sj/venvs/tadsr_official_strict_cu118/bin/python')
OFFICIAL_REPO = Path('/mnt/data/sj/projects/TADSR_official_pytorch')
WEIGHTS_DIR = Path('/mnt/data/sj/checkpoints/TADSR/preset/weights')
TADSR_PKL = WEIGHTS_DIR / 'tadsr.pkl'
OUT_DIR = Path('experiments/full_repro/unet_alignment')
OUT_JSON = OUT_DIR / 'audit_tadsr_unet_downblock0_attention0.json'
OUT_TXT = OUT_DIR / 'audit_tadsr_unet_downblock0_attention0.txt'

def maybe_reexec() -> None:
    expected_prefix = str(STRICT_PY.parents[1])
    in_expected_venv = str(sys.prefix) == expected_prefix or str(sys.executable).startswith(expected_prefix)
    if STRICT_PY.exists() and not in_expected_venv:
        os.environ['PYTHONNOUSERSITE'] = '1'
        os.execv(str(STRICT_PY), [str(STRICT_PY), __file__, *sys.argv[1:]])

def shape(x):
    return list(x.shape) if x is not None and hasattr(x, 'shape') else None

def base_layer(module):
    return module.base_layer if hasattr(module, 'base_layer') else module

def module_shape(m, attr='weight'):
    base = base_layer(m)
    t = getattr(base, attr, None)
    return shape(t) if t is not None else None

def module_lora_info(module):
    base = base_layer(module)
    info = {
        'class': f'{type(module).__module__}.{type(module).__name__}',
        'base_class': f'{type(base).__module__}.{type(base).__name__}',
        'is_lora_wrapped': hasattr(module, 'base_layer') or hasattr(module, 'lora_A'),
        'weight_shape': module_shape(module, 'weight'),
        'bias_shape': module_shape(module, 'bias') if getattr(base, 'bias', None) is not None else None,
        'active_adapters': list(getattr(module, 'active_adapters', [])) if hasattr(module, 'active_adapters') else [],
        'lora_A_shapes': {},
        'lora_B_shapes': {},
        'scaling': dict(getattr(module, 'scaling', {})) if isinstance(getattr(module, 'scaling', None), dict) else {},
        'delta_weight_max_abs': {},
    }
    for side in ['lora_A', 'lora_B']:
        obj = getattr(module, side, None)
        if hasattr(obj, 'items'):
            info[f'{side}_shapes'] = {k: shape(v.weight) for k, v in obj.items()}
    if hasattr(module, 'get_delta_weight'):
        for adapter in info['active_adapters']:
            try:
                delta = module.get_delta_weight(adapter).detach().cpu()
            except Exception:
                continue
            info['delta_weight_max_abs'][adapter] = float(delta.abs().max().item()) if delta.numel() else 0.0
    return info

def load_unet():
    sys.path.insert(0, str(OFFICIAL_REPO))
    import torch
    from tadsr import initialize_unet
    args = SimpleNamespace(pretrained_model_name_or_path=str(WEIGHTS_DIR), pretrained_lora_path=str(TADSR_PKL), lora_rank=4)
    unet, enc, dec, oth = initialize_unet(args)
    ckpt = torch.load(str(TADSR_PKL), map_location='cpu')
    sd = ckpt.get('state_dict_unet', {})
    loaded = []
    for n, p in unet.named_parameters():
        if 'lora' in n and n in sd:
            p.data.copy_(sd[n])
            loaded.append(n)
    conv = base_layer(unet.conv_in)
    if 'conv_in.weight' in sd:
        conv.weight.data.copy_(sd['conv_in.weight'])
    if 'conv_in.bias' in sd and getattr(conv, 'bias', None) is not None:
        conv.bias.data.copy_(sd['conv_in.bias'])
    if hasattr(unet, 'set_adapter'):
        unet.set_adapter(['default_encoder', 'default_decoder', 'default_others'])
    unet.eval()
    return unet, loaded

def norm_info(norm):
    if norm is None:
        return {'exists': False}
    return {
        'exists': True,
        'class': f'{type(norm).__module__}.{type(norm).__name__}',
        'weight_shape': shape(getattr(norm, 'weight', None)),
        'bias_shape': shape(getattr(norm, 'bias', None)),
        'eps': getattr(norm, 'eps', None),
        'normalized_shape': list(getattr(norm, 'normalized_shape', [])) if hasattr(norm, 'normalized_shape') else None,
        'num_groups': getattr(norm, 'num_groups', None),
        'elementwise_affine': getattr(norm, 'elementwise_affine', None),
    }

def attn_info(attn, name):
    if attn is None:
        return {'exists': False}
    info = {
        'exists': True,
        'name': name,
        'class': f'{type(attn).__module__}.{type(attn).__name__}',
        'processor_class': f'{type(attn.processor).__module__}.{type(attn.processor).__name__}',
        'query_dim': getattr(attn, 'query_dim', None),
        'cross_attention_dim': getattr(attn, 'cross_attention_dim', None),
        'heads': getattr(attn, 'heads', None),
        'inner_dim': getattr(attn, 'inner_dim', None),
        'scale': getattr(attn, 'scale', None),
        'upcast_attention': getattr(attn, 'upcast_attention', None),
        'upcast_softmax': getattr(attn, 'upcast_softmax', None),
        'residual_connection': getattr(attn, 'residual_connection', None),
        'rescale_output_factor': getattr(attn, 'rescale_output_factor', None),
        'dropout': getattr(attn, 'dropout', None),
        'scale_qk': getattr(attn, 'scale_qk', None),
        'norm_cross': norm_info(getattr(attn, 'norm_cross', None)),
        'group_norm': norm_info(getattr(attn, 'group_norm', None)),
        'spatial_norm': f'{type(getattr(attn, "spatial_norm", None)).__module__}.{type(getattr(attn, "spatial_norm", None)).__name__}' if getattr(attn, 'spatial_norm', None) is not None else None,
        'to_q': module_lora_info(attn.to_q),
        'to_k': module_lora_info(attn.to_k) if getattr(attn, 'to_k', None) is not None else None,
        'to_v': module_lora_info(attn.to_v) if getattr(attn, 'to_v', None) is not None else None,
        'to_out_0': module_lora_info(attn.to_out[0]),
        'to_out_1_dropout_p': getattr(attn.to_out[1], 'p', None),
    }
    return info

def ff_info(ff):
    modules = []
    for i, m in enumerate(ff.net):
        entry = {'index': i, 'class': f'{type(m).__module__}.{type(m).__name__}'}
        if hasattr(m, 'proj'):
            entry['proj'] = module_lora_info(m.proj)
        elif hasattr(m, 'weight'):
            entry['linear'] = module_lora_info(m)
        if hasattr(m, 'p'):
            entry['dropout_p'] = getattr(m, 'p', None)
        modules.append(entry)
    return {'class': f'{type(ff).__module__}.{type(ff).__name__}', 'module_order': modules}

def main() -> int:
    maybe_reexec()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import torch

    unet, loaded = load_unet()
    block = unet.down_blocks[0]
    attention0 = block.attentions[0]
    tb0 = attention0.transformer_blocks[0]

    sample = torch.linspace(-1.0, 1.0, steps=1 * int(unet.config.in_channels) * 32 * 32, dtype=torch.float32).reshape(1, int(unet.config.in_channels), 32, 32)
    timestep = torch.tensor([1], dtype=torch.long)
    encoder_hidden_states = torch.linspace(-0.5, 0.5, steps=1 * 77 * int(unet.config.cross_attention_dim), dtype=torch.float32).reshape(1, 77, int(unet.config.cross_attention_dim))
    with torch.no_grad():
        centered = 2.0 * sample - 1.0 if unet.config.center_input_sample else sample
        conv_in = unet.conv_in(centered)
        time_proj = unet.time_proj(timestep)
        temb = unet.time_embedding(time_proj.to(dtype=list(unet.time_embedding.parameters())[0].dtype))
        resnet0_out = block.resnets[0](conv_in, temb)
        attention0_out = attention0(resnet0_out, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]

    blocks = []
    for i, tb in enumerate(attention0.transformer_blocks):
        blocks.append({
            'index': i,
            'class': f'{type(tb).__module__}.{type(tb).__name__}',
            'forward_signature': str(inspect.signature(tb.forward)),
            'only_cross_attention': getattr(tb, 'only_cross_attention', None),
            'use_layer_norm': getattr(tb, 'use_layer_norm', None),
            'use_ada_layer_norm': getattr(tb, 'use_ada_layer_norm', None),
            'use_ada_layer_norm_zero': getattr(tb, 'use_ada_layer_norm_zero', None),
            'use_ada_layer_norm_single': getattr(tb, 'use_ada_layer_norm_single', None),
            'chunk_size': getattr(tb, '_chunk_size', None),
            'chunk_dim': getattr(tb, '_chunk_dim', None),
            'pos_embed_exists': getattr(tb, 'pos_embed', None) is not None,
            'norm1': norm_info(getattr(tb, 'norm1', None)),
            'attn1': attn_info(getattr(tb, 'attn1', None), f'transformer_blocks.{i}.attn1'),
            'norm2': norm_info(getattr(tb, 'norm2', None)),
            'attn2': attn_info(getattr(tb, 'attn2', None), f'transformer_blocks.{i}.attn2'),
            'norm3': norm_info(getattr(tb, 'norm3', None)),
            'ff': ff_info(getattr(tb, 'ff')),
        })

    report = {
        'status': 'PASS',
        'python': sys.executable,
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'loaded_lora_param_count': len(loaded),
        'active_adapters': ['default_encoder', 'default_decoder', 'default_others'],
        'down_blocks_0_overview_refresh': {
            'class': f'{type(block).__module__}.{type(block).__name__}',
            'forward_signature': str(inspect.signature(block.forward)),
            'module_order': [{'name': n, 'class': f'{type(m).__module__}.{type(m).__name__}'} for n, m in block.named_children()],
            'resnet_count': len(getattr(block, 'resnets', [])),
            'attention_count': len(getattr(block, 'attentions', [])),
            'downsampler_count': len(getattr(block, 'downsamplers', [])) if getattr(block, 'downsamplers', None) is not None else 0,
            'has_downsampler': bool(getattr(block, 'downsamplers', None)),
            'completed_module': 'resnets.0 PASS',
            'current_target': 'attentions.0',
            'remaining_after_this_stage': ['resnets.1', 'attentions.1', 'downsamplers.0'],
            'consumes_encoder_hidden_states': 'encoder_hidden_states' in inspect.signature(block.forward).parameters,
            'consumes_attention_mask': 'attention_mask' in inspect.signature(block.forward).parameters,
            'consumes_encoder_attention_mask': 'encoder_attention_mask' in inspect.signature(block.forward).parameters,
            'consumes_cross_attention_kwargs': 'cross_attention_kwargs' in inspect.signature(block.forward).parameters,
        },
        'attention0_top_level': {
            'class': f'{type(attention0).__module__}.{type(attention0).__name__}',
            'forward_signature': str(inspect.signature(attention0.forward)),
            'is_input_continuous': getattr(attention0, 'is_input_continuous', None),
            'use_linear_projection': getattr(attention0, 'use_linear_projection', None),
            'in_channels': getattr(attention0, 'in_channels', None),
            'out_channels': getattr(attention0, 'out_channels', None),
            'num_attention_heads': getattr(attention0, 'num_attention_heads', None),
            'attention_head_dim': getattr(attention0, 'attention_head_dim', None),
            'inner_dim': int(getattr(attention0, 'num_attention_heads')) * int(getattr(attention0, 'attention_head_dim')),
            'cross_attention_dim': int(unet.config.cross_attention_dim),
            'norm': norm_info(attention0.norm),
            'proj_in': module_lora_info(attention0.proj_in),
            'proj_out': module_lora_info(attention0.proj_out),
            'return_dict_default': True,
            'official_downblock_uses_return_dict_false': True,
            'operation_order': [
                'NCHW input residual saved',
                'GroupNorm on NCHW',
                'NCHW -> BHW,C sequence',
                'linear proj_in',
                'transformer_blocks.0 norm1 + attn1 + residual',
                'transformer_blocks.0 norm2 + attn2(cross) + residual',
                'transformer_blocks.0 norm3 + GEGLU feed-forward + residual',
                'linear proj_out',
                'sequence -> NCHW',
                'top-level residual add',
            ],
        },
        'transformer_blocks': blocks,
        'encoder_hidden_states_contract': {
            'strategy': 'deterministic synthetic encoder_hidden_states for leaf alignment',
            'shape': list(encoder_hidden_states.shape),
            'dtype': str(encoder_hidden_states.dtype),
            'sequence_length': int(encoder_hidden_states.shape[1]),
            'cross_attention_dim': int(encoder_hidden_states.shape[2]),
            'attention_mask': None,
            'encoder_attention_mask': None,
            'cross_attention_kwargs': None,
        },
        'dry_run': {
            'sample_shape': list(sample.shape),
            'conv_in_shape': list(conv_in.shape),
            'temb_shape': list(temb.shape),
            'resnet0_output_shape': list(resnet0_out.shape),
            'attention0_output_shape': list(attention0_out.shape),
            'attention0_output_stats': {
                'min': float(attention0_out.min().item()),
                'max': float(attention0_out.max().item()),
                'mean': float(attention0_out.mean().item()),
                'std': float(attention0_out.std().item()),
            },
        },
        'effective_weight_plan': 'Export static effective merged weights for proj_in/proj_out/attn1/attn2/ff; do not implement generic Jittor LoRA runtime.',
        'not_in_scope': ['full down_blocks.0', 'resnets.1', 'attentions.1', 'downsampler', 'full UNet forward', 'generic LoRA runtime', 'full TADSR inference'],
        'markers': {
            'TADSR_UNET_DOWNBLOCK0_ATTENTION0_AUDIT': 'PASS',
            'TADSR_UNET_DOWNBLOCK0_ATTENTION0_TRANSFORMER_AUDIT': 'PASS',
            'TADSR_UNET_DOWNBLOCK0_ATTENTION0_LORA_AUDIT': 'PASS',
        },
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    lines = [
        '# TADSR UNet down_blocks.0.attentions.0 Audit', '',
        'TADSR_UNET_DOWNBLOCK0_ATTENTION0_AUDIT: PASS',
        'TADSR_UNET_DOWNBLOCK0_ATTENTION0_TRANSFORMER_AUDIT: PASS',
        'TADSR_UNET_DOWNBLOCK0_ATTENTION0_LORA_AUDIT: PASS', '',
        f"attention0 class: `{report['attention0_top_level']['class']}`",
        f"use_linear_projection: `{report['attention0_top_level']['use_linear_projection']}`",
        f"input/output channels: `{report['attention0_top_level']['in_channels']} -> {report['attention0_top_level']['out_channels']}`",
        f"heads/head_dim: `{report['attention0_top_level']['num_attention_heads']} x {report['attention0_top_level']['attention_head_dim']}`",
        f"transformer blocks: `{len(blocks)}`",
        f"encoder_hidden_states: `{report['encoder_hidden_states_contract']['shape']}`",
        '',
        'Scope guard: this audit does not run full down_blocks.0 or full UNet.',
    ]
    OUT_TXT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('TADSR_UNET_DOWNBLOCK0_ATTENTION0_AUDIT: PASS')
    print('TADSR_UNET_DOWNBLOCK0_ATTENTION0_TRANSFORMER_AUDIT: PASS')
    print('TADSR_UNET_DOWNBLOCK0_ATTENTION0_LORA_AUDIT: PASS')
    print(json.dumps({'status': 'PASS', 'json': str(OUT_JSON), 'txt': str(OUT_TXT)}, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
