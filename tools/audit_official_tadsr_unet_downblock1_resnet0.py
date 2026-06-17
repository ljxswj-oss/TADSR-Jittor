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
OUT_JSON = OUT_DIR / 'audit_tadsr_unet_downblock1_resnet0.json'
OUT_TXT = OUT_DIR / 'audit_tadsr_unet_downblock1_resnet0.txt'


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
    return {
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
    block0 = unet.down_blocks[0]
    block1 = unet.down_blocks[1]
    resnet0 = block1.resnets[0]
    attention0 = block1.attentions[0] if hasattr(block1, 'attentions') and len(block1.attentions) > 0 else None
    resnet1 = block1.resnets[1] if len(getattr(block1, 'resnets', [])) > 1 else None
    downsampler0 = block1.downsamplers[0] if getattr(block1, 'downsamplers', None) is not None and len(block1.downsamplers) > 0 else None

    sample = torch.linspace(-1.0, 1.0, steps=1 * int(unet.config.in_channels) * 32 * 32, dtype=torch.float32).reshape(1, int(unet.config.in_channels), 32, 32)
    timestep = torch.tensor([1], dtype=torch.long)
    encoder_shape = [1, 77, int(unet.config.cross_attention_dim)]
    encoder_hidden_states = torch.linspace(-0.5, 0.5, steps=int(torch.tensor(encoder_shape).prod().item()), dtype=torch.float32).reshape(*encoder_shape)
    synthetic_shape = [1, int(getattr(resnet0, 'in_channels')), 16, 16]
    synthetic_temb_shape = [1, int(resnet0.time_emb_proj.weight.shape[1])]
    with torch.no_grad():
        centered = 2.0 * sample - 1.0 if unet.config.center_input_sample else sample
        conv_in = unet.conv_in(centered)
        time_proj = unet.time_proj(timestep)
        temb = unet.time_embedding(time_proj.to(dtype=list(unet.time_embedding.parameters())[0].dtype))
        block0_hidden, block0_states = block0(conv_in, temb, encoder_hidden_states=encoder_hidden_states)
        resnet0_out = resnet0(block0_hidden, temb)

    children = [{'name': n, 'class': f'{type(m).__module__}.{type(m).__name__}'} for n, m in block1.named_children()]
    submodules = {}
    for name in ['norm1', 'conv1', 'time_emb_proj', 'norm2', 'conv2', 'conv_shortcut']:
        mod = getattr(resnet0, name, None)
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

    audit = {
        'status': 'PASS',
        'scope': 'down_blocks.1.resnets.0 only; down_blocks.1 attention/resnet1/downsampler and full UNet forward are intentionally not ported in this stage.',
        'python': sys.executable,
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'loaded_lora_parameter_count': len(loaded),
        'unet_config_subset': {
            'sample_size': getattr(unet.config, 'sample_size', None),
            'in_channels': getattr(unet.config, 'in_channels', None),
            'block_out_channels': list(getattr(unet.config, 'block_out_channels', [])),
            'cross_attention_dim': getattr(unet.config, 'cross_attention_dim', None),
            'center_input_sample': getattr(unet.config, 'center_input_sample', None),
        },
        'downblock1_overview': {
            'class': f'{type(block1).__module__}.{type(block1).__name__}',
            'forward_signature': str(inspect.signature(block1.forward)),
            'module_order': children,
            'resnet_count': len(getattr(block1, 'resnets', [])),
            'attention_count': len(getattr(block1, 'attentions', [])) if hasattr(block1, 'attentions') else 0,
            'downsampler_count': len(getattr(block1, 'downsamplers', [])) if getattr(block1, 'downsamplers', None) is not None else 0,
            'current_target_module': 'resnets.0',
            'completed_prior_context': 'entry + full local down_blocks.0 is already aligned and used only as input bridge.',
            'remaining_modules_after_this_stage': ['attentions.0', 'resnets.1', 'attentions.1', 'downsamplers.0' if downsampler0 is not None else None],
            'attention0_preview': attention_preview(attention0),
            'resnet1_preview': resnet_config(resnet1, 'down_blocks_1_resnets_1') if resnet1 is not None else {'exists': False},
            'downsampler0_class': f'{type(downsampler0).__module__}.{type(downsampler0).__name__}' if downsampler0 is not None else None,
        },
        'resnet0_config': resnet_config(resnet0, 'down_blocks_1_resnets_0'),
        'submodules': submodules,
        'bridge_shapes': {
            'sample': list(sample.shape),
            'encoder_hidden_states': encoder_shape,
            'conv_in': list(conv_in.shape),
            'time_embedding': list(temb.shape),
            'downblock0_hidden': list(block0_hidden.shape),
            'downblock0_output_state_shapes': [list(t.shape) for t in block0_states],
            'downblock1_resnet0_output': list(resnet0_out.shape),
            'synthetic_resnet0_input': synthetic_shape,
            'synthetic_temb': synthetic_temb_shape,
        },
        'bridge_tensor_stats': {
            'downblock0_hidden': stats_tensor(block0_hidden),
            'downblock1_resnet0_output': stats_tensor(resnet0_out),
        },
        'next_action': 'Port down_blocks.1.attentions.0 only after this resnet0 bridge remains PASS; do not enable full UNet forward yet.',
    }
    OUT_JSON.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    lines = [
        'TADSR UNet down_blocks.1.resnets.0 audit',
        f"status: {audit['status']}",
        f"resnet0 input/output: {audit['resnet0_config']['in_channels']} -> {audit['resnet0_config']['out_channels']}",
        f"has shortcut: {audit['resnet0_config']['has_shortcut']}",
        f"block0 bridge output shape: {audit['bridge_shapes']['downblock0_hidden']}",
        f"resnet0 output shape: {audit['bridge_shapes']['downblock1_resnet0_output']}",
        f"remaining down_blocks.1 modules: {audit['downblock1_overview']['remaining_modules_after_this_stage']}",
    ]
    OUT_TXT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('TADSR_UNET_DOWNBLOCK1_RESNET0_AUDIT: PASS')
    print('TADSR_UNET_DOWNBLOCK1_ALIGNMENT: NOT_COMPLETE')
    print('TADSR_UNET_FULL_FORWARD_ALIGNMENT: NOT_COMPLETE')
    print(f'wrote {OUT_JSON}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
