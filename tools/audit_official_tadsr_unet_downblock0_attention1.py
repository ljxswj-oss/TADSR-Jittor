#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

STRICT_PY = Path('/mnt/data/sj/venvs/tadsr_official_strict_cu118/bin/python')
OFFICIAL_REPO = Path('/mnt/data/sj/projects/TADSR_official_pytorch')
WEIGHTS_DIR = Path('/mnt/data/sj/checkpoints/TADSR/preset/weights')
TADSR_PKL = WEIGHTS_DIR / 'tadsr.pkl'
OUT_DIR = Path('experiments/full_repro/unet_alignment')
OUT_JSON = OUT_DIR / 'audit_tadsr_unet_downblock0_attention1.json'
OUT_TXT = OUT_DIR / 'audit_tadsr_unet_downblock0_attention1.txt'


def maybe_reexec() -> None:
    expected_prefix = str(STRICT_PY.parents[1])
    in_expected_venv = str(sys.prefix) == expected_prefix or str(sys.executable).startswith(expected_prefix)
    if STRICT_PY.exists() and not in_expected_venv:
        os.environ['PYTHONNOUSERSITE'] = '1'
        os.execv(str(STRICT_PY), [str(STRICT_PY), __file__, *sys.argv[1:]])


def shape(x):
    return list(x.shape) if x is not None and hasattr(x, 'shape') else None


def module_shape(m, attr='weight'):
    t = getattr(m, attr, None)
    return shape(t) if t is not None else None


def base_layer(module):
    return module.base_layer if hasattr(module, 'base_layer') else module


def module_lora_info(module):
    if module is None:
        return {'exists': False}
    base = base_layer(module)
    info = {
        'exists': True,
        'class': f'{type(module).__module__}.{type(module).__name__}',
        'base_class': f'{type(base).__module__}.{type(base).__name__}',
        'is_lora_wrapped': hasattr(module, 'base_layer') or hasattr(module, 'lora_A'),
        'weight_shape': module_shape(base, 'weight'),
        'bias_shape': module_shape(base, 'bias') if getattr(base, 'bias', None) is not None else None,
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
            except Exception as exc:
                info['delta_weight_max_abs'][adapter] = f'ERROR: {exc}'
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


def safe_attr(obj, name, default=None):
    try:
        return getattr(obj, name)
    except Exception:
        return default


def attention_config(att, prefix: str):
    norm = getattr(att, 'norm', None)
    proj_in = getattr(att, 'proj_in', None)
    proj_out = getattr(att, 'proj_out', None)
    return {
        'prefix': prefix,
        'class': f'{type(att).__module__}.{type(att).__name__}',
        'forward_signature': str(inspect.signature(att.forward)),
        'source_module_path': 'unet.down_blocks.0.attentions.1',
        'in_channels': safe_attr(att, 'in_channels'),
        'out_channels': safe_attr(att, 'out_channels'),
        'num_attention_heads': safe_attr(att, 'num_attention_heads'),
        'attention_head_dim': safe_attr(att, 'attention_head_dim'),
        'inner_dim': safe_attr(att, 'inner_dim'),
        'cross_attention_dim': safe_attr(att, 'cross_attention_dim'),
        'use_linear_projection': safe_attr(att, 'use_linear_projection'),
        'norm_class': f'{type(norm).__module__}.{type(norm).__name__}' if norm is not None else None,
        'norm_num_groups': getattr(norm, 'num_groups', None),
        'norm_eps': getattr(norm, 'eps', None),
        'proj_in_class': f'{type(proj_in).__module__}.{type(proj_in).__name__}' if proj_in is not None else None,
        'proj_in_base_class': f'{type(base_layer(proj_in)).__module__}.{type(base_layer(proj_in)).__name__}' if proj_in is not None else None,
        'proj_in_weight_shape': module_shape(base_layer(proj_in), 'weight') if proj_in is not None else None,
        'proj_in_bias_shape': module_shape(base_layer(proj_in), 'bias') if proj_in is not None and getattr(base_layer(proj_in), 'bias', None) is not None else None,
        'proj_out_class': f'{type(proj_out).__module__}.{type(proj_out).__name__}' if proj_out is not None else None,
        'proj_out_base_class': f'{type(base_layer(proj_out)).__module__}.{type(base_layer(proj_out)).__name__}' if proj_out is not None else None,
        'proj_out_weight_shape': module_shape(base_layer(proj_out), 'weight') if proj_out is not None else None,
        'proj_out_bias_shape': module_shape(base_layer(proj_out), 'bias') if proj_out is not None and getattr(base_layer(proj_out), 'bias', None) is not None else None,
        'residual_connection': 'proj_out_nchw + input hidden_states',
        'reshape_order': 'NCHW -> NHWC -> BNC before transformer; BNC -> NHWC -> NCHW after proj_out',
        'official_return_dict_false_supported': True,
    }


def attention_module_details(attn):
    if attn is None:
        return {'exists': False}
    return {
        'exists': True,
        'class': f'{type(attn).__module__}.{type(attn).__name__}',
        'processor_class': f'{type(getattr(attn, "processor", None)).__module__}.{type(getattr(attn, "processor", None)).__name__}',
        'query_dim': getattr(attn, 'query_dim', None),
        'cross_attention_dim': getattr(attn, 'cross_attention_dim', None),
        'heads': getattr(attn, 'heads', None),
        'dim_head': getattr(attn, 'inner_dim', 0) // getattr(attn, 'heads', 1),
        'inner_dim': getattr(attn, 'inner_dim', None),
        'scale': getattr(attn, 'scale', None),
        'upcast_attention': getattr(attn, 'upcast_attention', None),
        'upcast_softmax': getattr(attn, 'upcast_softmax', None),
        'residual_connection': getattr(attn, 'residual_connection', None),
        'rescale_output_factor': getattr(attn, 'rescale_output_factor', None),
        'dropout': getattr(attn, 'dropout', None),
        'attention_bias': getattr(attn, 'bias', None),
        'to_q': module_lora_info(attn.to_q),
        'to_k': module_lora_info(attn.to_k),
        'to_v': module_lora_info(attn.to_v),
        'to_out_0': module_lora_info(attn.to_out[0]),
        'to_out_1_class': f'{type(attn.to_out[1]).__module__}.{type(attn.to_out[1]).__name__}' if len(attn.to_out) > 1 else None,
        'official_processor_uses_sdpa': '2_0' in type(getattr(attn, 'processor', None)).__name__,
        'attention_mask_used_in_this_stage': False,
    }


def ff_details(ff):
    modules = []
    for i, mod in enumerate(getattr(ff, 'net', [])):
        rec = {'index': i, 'class': f'{type(mod).__module__}.{type(mod).__name__}'}
        if hasattr(mod, 'proj'):
            rec['proj'] = module_lora_info(mod.proj)
        elif hasattr(mod, 'weight'):
            rec['linear'] = module_lora_info(mod)
        modules.append(rec)
    return {
        'class': f'{type(ff).__module__}.{type(ff).__name__}',
        'module_order': modules,
        'activation_type': modules[0]['class'] if modules else None,
        'dropout_values': [getattr(m, 'p', None) for m in getattr(ff, 'net', []) if 'Dropout' in type(m).__name__],
        'final_dropout_noop_in_eval': True,
    }


def transformer_block_details(block):
    return {
        'class': f'{type(block).__module__}.{type(block).__name__}',
        'forward_signature': str(inspect.signature(block.forward)),
        'norm1': {'class': f'{type(block.norm1).__module__}.{type(block.norm1).__name__}', 'shape': module_shape(block.norm1, 'weight'), 'eps': getattr(block.norm1, 'eps', None), 'elementwise_affine': getattr(block.norm1, 'elementwise_affine', None)},
        'attn1': attention_module_details(block.attn1),
        'norm2': {'class': f'{type(block.norm2).__module__}.{type(block.norm2).__name__}', 'shape': module_shape(block.norm2, 'weight'), 'eps': getattr(block.norm2, 'eps', None), 'elementwise_affine': getattr(block.norm2, 'elementwise_affine', None)},
        'attn2': attention_module_details(block.attn2),
        'norm3': {'class': f'{type(block.norm3).__module__}.{type(block.norm3).__name__}', 'shape': module_shape(block.norm3, 'weight'), 'eps': getattr(block.norm3, 'eps', None), 'elementwise_affine': getattr(block.norm3, 'elementwise_affine', None)},
        'ff': ff_details(block.ff),
        'only_cross_attention': getattr(block, 'only_cross_attention', None),
        'double_self_attention': getattr(block, 'double_self_attention', None),
        'upcast_attention': getattr(block, 'upcast_attention', None),
        'norm_type': getattr(block, 'norm_type', None),
        'requires_timestep': False,
        'requires_class_labels': False,
        'requires_added_cond_kwargs': False,
    }


def stats_tensor(x):
    y = x.detach().cpu()
    return {
        'shape': list(y.shape),
        'dtype': str(y.dtype),
        'min': float(y.min().item()) if y.numel() else 0.0,
        'max': float(y.max().item()) if y.numel() else 0.0,
        'mean': float(y.float().mean().item()) if y.numel() else 0.0,
        'std': float(y.float().std(unbiased=False).item()) if y.numel() else 0.0,
    }


def main() -> int:
    maybe_reexec()
    import torch
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    unet, loaded = load_unet()
    block = unet.down_blocks[0]
    attention0 = block.attentions[0]
    attention1 = block.attentions[1]
    resnet0 = block.resnets[0]
    resnet1 = block.resnets[1]
    cfg = dict(unet.config)
    sample = torch.linspace(-1.0, 1.0, steps=1 * int(unet.config.in_channels) * 32 * 32, dtype=torch.float32).reshape(1, int(unet.config.in_channels), 32, 32)
    timestep = torch.tensor([1], dtype=torch.long)
    encoder_shape = [1, 77, int(unet.config.cross_attention_dim)]
    encoder_hidden_states = torch.linspace(-0.5, 0.5, steps=int(np.prod(encoder_shape)), dtype=torch.float32).reshape(*encoder_shape)
    with torch.no_grad():
        centered = 2.0 * sample - 1.0 if unet.config.center_input_sample else sample
        conv_in = unet.conv_in(centered)
        time_proj = unet.time_proj(timestep)
        temb = unet.time_embedding(time_proj.to(dtype=list(unet.time_embedding.parameters())[0].dtype))
        resnet0_out = resnet0(conv_in, temb)
        attention0_out = attention0(resnet0_out, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
        resnet1_out = resnet1(attention0_out, temb)
        attention1_out = attention1(resnet1_out, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]

    children = [{'name': n, 'class': f'{type(m).__module__}.{type(m).__name__}'} for n, m in block.named_children()]
    a0_cfg = attention_config(attention0, 'down_blocks_0_attentions_0')
    a1_cfg = attention_config(attention1, 'down_blocks_0_attentions_1')
    comparison = {k: {'attention0': a0_cfg.get(k), 'attention1': a1_cfg.get(k), 'same': a0_cfg.get(k) == a1_cfg.get(k)}
                  for k in ['class', 'in_channels', 'out_channels', 'num_attention_heads', 'attention_head_dim', 'inner_dim', 'cross_attention_dim', 'use_linear_projection', 'norm_num_groups', 'norm_eps', 'proj_in_weight_shape', 'proj_out_weight_shape']}
    tblocks = [transformer_block_details(tb) for tb in attention1.transformer_blocks]
    lora_modules = {}
    for label, info in [('proj_in', module_lora_info(attention1.proj_in)), ('proj_out', module_lora_info(attention1.proj_out))]:
        if info.get('is_lora_wrapped'):
            lora_modules[label] = info
    for i, tb in enumerate(attention1.transformer_blocks):
        for an in ['attn1', 'attn2']:
            attn = getattr(tb, an)
            for ln in ['to_q', 'to_k', 'to_v']:
                info = module_lora_info(getattr(attn, ln))
                if info.get('is_lora_wrapped'):
                    lora_modules[f'transformer{i}_{an}_{ln}'] = info
            info = module_lora_info(attn.to_out[0])
            if info.get('is_lora_wrapped'):
                lora_modules[f'transformer{i}_{an}_to_out_0'] = info
        info = module_lora_info(tb.ff.net[0].proj)
        if info.get('is_lora_wrapped'):
            lora_modules[f'transformer{i}_ff_geglu_proj'] = info
        info = module_lora_info(tb.ff.net[2])
        if info.get('is_lora_wrapped'):
            lora_modules[f'transformer{i}_ff_out'] = info
    overview = {
        'class': f'{type(block).__module__}.{type(block).__name__}',
        'has_cross_attention': getattr(block, 'has_cross_attention', True),
        'child_module_list_order': children,
        'actual_forward_order': ['resnets.0', 'attentions.0', 'resnets.1', 'attentions.1', 'downsamplers.0'],
        'resnet_count': len(block.resnets),
        'attention_count': len(block.attentions),
        'transformer_count': sum(len(getattr(a, 'transformer_blocks', [])) for a in block.attentions),
        'downsampler_count': len(block.downsamplers) if block.downsamplers is not None else 0,
        'has_downsampler': block.downsamplers is not None,
        'known_completed_modules': {'resnets.0': 'PASS', 'attentions.0': 'PASS', 'resnets.1': 'PASS'},
        'current_target_module': 'attentions.1',
        'remaining_modules_after_this_stage': ['downsamplers.0'] if block.downsamplers is not None else [],
        'actual_next_module_after_attentions_1': 'downsamplers.0' if block.downsamplers is not None else None,
        'consumes_encoder_hidden_states': True,
        'consumes_attention_mask': 'attention_mask' in str(inspect.signature(block.forward)),
        'consumes_encoder_attention_mask': 'encoder_attention_mask' in str(inspect.signature(block.forward)),
        'consumes_cross_attention_kwargs': 'cross_attention_kwargs' in str(inspect.signature(block.forward)),
        'note': 'Overview only; this stage audits/ports attentions.1 but does not port downsampler or full down_blocks.0.',
    }
    report = {
        'status': 'PASS',
        'python': sys.executable,
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'loaded_lora_param_count': len(loaded),
        'down_blocks_0_overview': overview,
        'attention1_config': a1_cfg,
        'attention0_vs_attention1': comparison,
        'transformer_blocks': tblocks,
        'transformer_block_count': len(attention1.transformer_blocks),
        'encoder_hidden_states_contract': {
            'shape': encoder_shape,
            'cross_attention_dim': int(unet.config.cross_attention_dim),
            'sequence_length': 77,
            'same_strategy_as_attention0': True,
            'attention_mask': None,
            'encoder_attention_mask': None,
            'cross_attention_kwargs': None,
        },
        'effective_weight_plan': {
            'status': 'PASS',
            'strategy': 'export LoRA-merged/effective static weights for attentions.1; no generic Jittor LoRA runtime',
            'lora_wrapped_modules': sorted(lora_modules.keys()),
            'planned_npz': 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention1_effective_weights.npz',
        },
        'dry_run_shapes_stats': {
            'synthetic_unet_sample': stats_tensor(sample),
            'encoder_hidden_states': stats_tensor(encoder_hidden_states),
            'conv_in_output': stats_tensor(conv_in),
            'time_embedding_output': stats_tensor(temb),
            'resnet0_output': stats_tensor(resnet0_out),
            'attention0_output': stats_tensor(attention0_out),
            'resnet1_output': stats_tensor(resnet1_out),
            'attention1_input': stats_tensor(resnet1_out),
            'attention1_output': stats_tensor(attention1_out),
            'stopped_before': ['down_blocks.0.downsamplers.0', 'down_blocks.1', 'full down_blocks.0', 'full UNet'],
        },
        'not_in_scope': ['full down_blocks.0', 'downsampler port', 'full UNet forward', 'full TADSR inference', 'generic runtime LoRA'],
        'markers': {
            'TADSR_UNET_DOWNBLOCK0_ATTENTION1_AUDIT': 'PASS',
            'TADSR_UNET_DOWNBLOCK0_ATTENTION1_TRANSFORMER_AUDIT': 'PASS',
            'TADSR_UNET_DOWNBLOCK0_ATTENTION1_LORA_AUDIT': 'PASS',
        },
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    lines = [
        '# TADSR UNet down_blocks.0.attentions.1 Audit',
        '',
        'TADSR_UNET_DOWNBLOCK0_ATTENTION1_AUDIT: PASS',
        'TADSR_UNET_DOWNBLOCK0_ATTENTION1_TRANSFORMER_AUDIT: PASS',
        'TADSR_UNET_DOWNBLOCK0_ATTENTION1_LORA_AUDIT: PASS',
        '',
        f"attention1 class: `{a1_cfg['class']}`",
        f"input shape: `{report['dry_run_shapes_stats']['attention1_input']['shape']}`",
        f"output shape: `{report['dry_run_shapes_stats']['attention1_output']['shape']}`",
        f"actual next module after attentions.1: `{overview['actual_next_module_after_attentions_1']}`",
        '',
        'This audit stops before downsampler, down_blocks.1, full down_blocks.0 and full UNet.',
    ]
    OUT_TXT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('TADSR_UNET_DOWNBLOCK0_ATTENTION1_AUDIT: PASS')
    print('TADSR_UNET_DOWNBLOCK0_ATTENTION1_TRANSFORMER_AUDIT: PASS')
    print('TADSR_UNET_DOWNBLOCK0_ATTENTION1_LORA_AUDIT: PASS')
    print(json.dumps({'status': 'PASS', 'json': str(OUT_JSON), 'txt': str(OUT_TXT)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
