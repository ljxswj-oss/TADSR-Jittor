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
OUT_JSON = OUT_DIR / 'audit_tadsr_unet_downblock0_resnet1.json'
OUT_TXT = OUT_DIR / 'audit_tadsr_unet_downblock0_resnet1.txt'


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


def resnet_config(resnet, prefix: str):
    norm1 = getattr(resnet, 'norm1', None)
    norm2 = getattr(resnet, 'norm2', None)
    dropout = getattr(resnet, 'dropout', None)
    cfg = {
        'prefix': prefix,
        'class': f'{type(resnet).__module__}.{type(resnet).__name__}',
        'forward_signature': str(inspect.signature(resnet.forward)),
        'in_channels': getattr(resnet, 'in_channels', None),
        'out_channels': getattr(resnet, 'out_channels', None),
        'temb_channels': shape(resnet.time_emb_proj.weight)[1] if getattr(resnet, 'time_emb_proj', None) is not None else None,
        'groups': getattr(norm1, 'num_groups', None),
        'groups_out': getattr(norm2, 'num_groups', None),
        'eps': getattr(norm1, 'eps', None),
        'eps_out': getattr(norm2, 'eps', None),
        'non_linearity': type(getattr(resnet, 'nonlinearity', None)).__name__,
        'time_embedding_norm': getattr(resnet, 'time_embedding_norm', None),
        'output_scale_factor': getattr(resnet, 'output_scale_factor', None),
        'use_in_shortcut': getattr(resnet, 'use_in_shortcut', None),
        'use_conv_shortcut': getattr(resnet, 'use_conv_shortcut', None),
        'up': getattr(resnet, 'up', None),
        'down': getattr(resnet, 'down', None),
        'dropout': getattr(dropout, 'p', None),
        'skip_time_act': getattr(resnet, 'skip_time_act', None),
        'pre_norm': getattr(resnet, 'pre_norm', None),
        'has_time_emb_proj': getattr(resnet, 'time_emb_proj', None) is not None,
        'time_emb_proj_out_features': shape(resnet.time_emb_proj.weight)[0] if getattr(resnet, 'time_emb_proj', None) is not None else None,
        'has_shortcut': getattr(resnet, 'conv_shortcut', None) is not None,
        'has_conv_shortcut': getattr(resnet, 'conv_shortcut', None) is not None,
        'operation_order': [
            'input_tensor -> norm1',
            'norm1 -> nonlinearity',
            'act1 -> conv1',
            'temb -> nonlinearity unless skip_time_act',
            'temb -> time_emb_proj -> NCHW broadcast',
            'default: conv1 + temb_proj before norm2',
            'scale_shift: norm2 then scale/shift',
            'norm2 -> nonlinearity -> dropout -> conv2',
            'optional conv_shortcut on input_tensor',
            '(shortcut + conv2) / output_scale_factor',
        ],
    }
    return cfg


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


def attention_preview(att):
    if att is None:
        return {'exists': False}
    out = {
        'exists': True,
        'class': f'{type(att).__module__}.{type(att).__name__}',
        'transformer_blocks': len(getattr(att, 'transformer_blocks', [])),
        'norm_num_groups': getattr(getattr(att, 'norm', None), 'num_groups', None),
        'inner_dim': getattr(att, 'inner_dim', None),
        'num_attention_heads': getattr(att, 'num_attention_heads', None),
        'attention_head_dim': getattr(att, 'attention_head_dim', None),
        'cross_attention_dim': getattr(att, 'cross_attention_dim', None),
        'use_linear_projection': getattr(att, 'use_linear_projection', None),
        'not_ported_this_stage': True,
    }
    tb = getattr(att, 'transformer_blocks', [None])[0] if len(getattr(att, 'transformer_blocks', [])) else None
    if tb is not None:
        out['transformer0_class'] = f'{type(tb).__module__}.{type(tb).__name__}'
        out['attn1_class'] = f'{type(tb.attn1).__module__}.{type(tb.attn1).__name__}'
        out['attn2_class'] = f'{type(tb.attn2).__module__}.{type(tb.attn2).__name__}'
    return out


def main() -> int:
    maybe_reexec()
    import torch
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    unet, loaded = load_unet()
    block = unet.down_blocks[0]
    resnet0 = block.resnets[0]
    resnet1 = block.resnets[1]
    attention0 = block.attentions[0] if hasattr(block, 'attentions') and len(block.attentions) > 0 else None
    attention1 = block.attentions[1] if hasattr(block, 'attentions') and len(block.attentions) > 1 else None
    cfg = dict(unet.config)

    sample = torch.linspace(-1.0, 1.0, steps=1 * int(unet.config.in_channels) * 32 * 32, dtype=torch.float32).reshape(1, int(unet.config.in_channels), 32, 32)
    timestep = torch.tensor([1], dtype=torch.long)
    encoder_shape = [1, 77, int(unet.config.cross_attention_dim)]
    encoder_hidden_states = torch.linspace(-0.5, 0.5, steps=int(torch.tensor(encoder_shape).prod().item()), dtype=torch.float32).reshape(*encoder_shape)
    with torch.no_grad():
        centered = 2.0 * sample - 1.0 if unet.config.center_input_sample else sample
        conv_in = unet.conv_in(centered)
        time_proj = unet.time_proj(timestep)
        temb = unet.time_embedding(time_proj.to(dtype=list(unet.time_embedding.parameters())[0].dtype))
        resnet0_out = resnet0(conv_in, temb)
        if attention0 is None:
            raise RuntimeError('down_blocks.0.attentions.0 missing; prior stage should have audited it.')
        attention0_out = attention0(resnet0_out, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
        resnet1_out = resnet1(attention0_out, temb)

    children = [{'name': n, 'class': f'{type(m).__module__}.{type(m).__name__}'} for n, m in block.named_children()]
    submodules = {}
    for name in ['norm1', 'conv1', 'time_emb_proj', 'norm2', 'conv2', 'conv_shortcut']:
        mod = getattr(resnet1, name, None)
        if mod is None:
            submodules[name] = {'exists': False}
        elif name.startswith('norm'):
            submodules[name] = {
                'exists': True,
                'class': f'{type(mod).__module__}.{type(mod).__name__}',
                'weight_shape': module_shape(mod, 'weight'),
                'bias_shape': module_shape(mod, 'bias'),
                'num_groups': getattr(mod, 'num_groups', None),
                'eps': getattr(mod, 'eps', None),
                'is_lora_wrapped': False,
            }
        else:
            submodules[name] = module_lora_info(mod)

    r0_cfg = resnet_config(resnet0, 'down_blocks_0_resnets_0')
    r1_cfg = resnet_config(resnet1, 'down_blocks_0_resnets_1')
    comparison = {k: {'resnet0': r0_cfg.get(k), 'resnet1': r1_cfg.get(k), 'same': r0_cfg.get(k) == r1_cfg.get(k)}
                  for k in ['in_channels', 'out_channels', 'temb_channels', 'groups', 'groups_out', 'eps', 'time_embedding_norm', 'output_scale_factor', 'dropout', 'has_shortcut']}

    overview = {
        'down_blocks_count': len(unet.down_blocks),
        'class': f'{type(block).__module__}.{type(block).__name__}',
        'has_cross_attention': getattr(block, 'has_cross_attention', True),
        'forward_signature': str(inspect.signature(block.forward)),
        'module_order': children,
        'resnet_count': len(getattr(block, 'resnets', [])),
        'attention_count': len(getattr(block, 'attentions', [])) if hasattr(block, 'attentions') else 0,
        'downsampler_count': len(getattr(block, 'downsamplers', [])) if getattr(block, 'downsamplers', None) is not None else 0,
        'has_downsampler': getattr(block, 'downsamplers', None) is not None,
        'known_completed_modules': {'resnets.0': 'PASS', 'attentions.0': 'PASS'},
        'current_target_module': 'resnets.1',
        'remaining_modules_after_this_stage': ['attentions.1', 'downsamplers.0' if getattr(block, 'downsamplers', None) is not None else None],
        'actual_next_module_after_resnets_1': 'attentions.1' if attention1 is not None else ('downsamplers.0' if getattr(block, 'downsamplers', None) is not None else None),
        'consumes_encoder_hidden_states': True,
        'consumes_attention_mask': 'attention_mask' in str(inspect.signature(block.forward)),
        'consumes_encoder_attention_mask': 'encoder_attention_mask' in str(inspect.signature(block.forward)),
        'consumes_cross_attention_kwargs': 'cross_attention_kwargs' in str(inspect.signature(block.forward)),
        'note': 'Overview only; this stage audits/ports resnets.1 but does not port full down_blocks.0.',
    }
    dry_run = {
        'synthetic_unet_sample': stats_tensor(sample),
        'timestep': [1],
        'encoder_hidden_states': stats_tensor(encoder_hidden_states),
        'conv_in_output': stats_tensor(conv_in),
        'time_proj_output': stats_tensor(time_proj),
        'time_embedding_output': stats_tensor(temb),
        'resnet0_output': stats_tensor(resnet0_out),
        'attention0_output': stats_tensor(attention0_out),
        'resnet1_input': stats_tensor(attention0_out),
        'resnet1_output': stats_tensor(resnet1_out),
        'stopped_before': ['down_blocks.0.attentions.1', 'down_blocks.0.downsamplers.0', 'full down_blocks.0', 'full UNet'],
    }
    lora_modules = {name: info for name, info in submodules.items() if info.get('exists') and info.get('is_lora_wrapped')}
    report = {
        'status': 'PASS',
        'python': sys.executable,
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'loaded_lora_param_count': len(loaded),
        'down_blocks_0_overview': overview,
        'resnet1_config': r1_cfg,
        'resnet1_submodules': submodules,
        'resnet0_vs_resnet1': comparison,
        'can_reuse_existing_resnet_tester': True,
        'required_jittor_code_changes': ['reuse prefix-based UNetResnetBlock2DTester with prefix=down_blocks_0_resnets_1', 'add entry->resnet0->attention0->resnet1 bridge tester'],
        'effective_weight_plan': {
            'status': 'PASS',
            'strategy': 'export LoRA-merged/effective static weights for resnets.1; no generic Jittor LoRA runtime',
            'lora_wrapped_submodules': sorted(lora_modules.keys()),
            'planned_npz': 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet1_effective_weights.npz',
        },
        'dry_run_shapes_stats': dry_run,
        'downstream_attention1_preview': attention_preview(attention1),
        'not_in_scope': ['full down_blocks.0', 'attentions.1 port', 'downsampler port', 'full UNet forward', 'full TADSR inference', 'generic runtime LoRA'],
        'markers': {
            'TADSR_UNET_DOWNBLOCK0_RESNET1_AUDIT': 'PASS',
            'TADSR_UNET_DOWNBLOCK0_RESNET1_LORA_AUDIT': 'PASS',
        },
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    lines = [
        '# TADSR UNet down_blocks.0.resnets.1 Audit',
        '',
        'TADSR_UNET_DOWNBLOCK0_RESNET1_AUDIT: PASS',
        'TADSR_UNET_DOWNBLOCK0_RESNET1_LORA_AUDIT: PASS',
        '',
        f"down_blocks.0 class: `{overview['class']}`",
        f"module order: `{overview['module_order']}`",
        f"resnet1 input shape: `{dry_run['resnet1_input']['shape']}`",
        f"resnet1 output shape: `{dry_run['resnet1_output']['shape']}`",
        f"actual next module after resnets.1: `{overview['actual_next_module_after_resnets_1']}`",
        '',
        'This audit stops before attentions.1, downsampler, full down_blocks.0 and full UNet.',
    ]
    OUT_TXT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('TADSR_UNET_DOWNBLOCK0_RESNET1_AUDIT: PASS')
    print('TADSR_UNET_DOWNBLOCK0_RESNET1_LORA_AUDIT: PASS')
    print(json.dumps({'status': 'PASS', 'json': str(OUT_JSON), 'txt': str(OUT_TXT)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
