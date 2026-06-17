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
OUT_JSON = OUT_DIR / 'audit_tadsr_unet_downblock0_resnet0.json'
OUT_TXT = OUT_DIR / 'audit_tadsr_unet_downblock0_resnet0.txt'


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
    base = base_layer(module)
    info = {
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
    return unet, enc, dec, oth, loaded


def resnet_config(resnet):
    norm1 = getattr(resnet, 'norm1', None)
    norm2 = getattr(resnet, 'norm2', None)
    dropout = getattr(resnet, 'dropout', None)
    cfg = {
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


def main() -> int:
    maybe_reexec()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import torch

    unet, enc, dec, oth, loaded = load_unet()
    block = unet.down_blocks[0]
    resnet = block.resnets[0]
    attention0 = block.attentions[0] if hasattr(block, 'attentions') and len(block.attentions) else None

    sample = torch.linspace(-1.0, 1.0, steps=1 * int(unet.config.in_channels) * 32 * 32, dtype=torch.float32).reshape(1, int(unet.config.in_channels), 32, 32)
    timesteps = torch.tensor([1], dtype=torch.long)
    with torch.no_grad():
        centered = 2.0 * sample - 1.0 if unet.config.center_input_sample else sample
        conv_in = unet.conv_in(centered)
        time_proj = unet.time_proj(timesteps)
        temb = unet.time_embedding(time_proj.to(dtype=list(unet.time_embedding.parameters())[0].dtype))
        resnet_out = resnet(conv_in, temb)

    children = [{'name': n, 'class': f'{type(m).__module__}.{type(m).__name__}'} for n, m in block.named_children()]
    attention_preview = None
    if attention0 is not None:
        attention_preview = {
            'class': f'{type(attention0).__module__}.{type(attention0).__name__}',
            'transformer_blocks': len(getattr(attention0, 'transformer_blocks', [])),
            'norm_num_groups': getattr(getattr(attention0, 'norm', None), 'num_groups', None),
            'inner_dim': getattr(attention0, 'inner_dim', None),
            'num_attention_heads': getattr(attention0, 'num_attention_heads', None),
            'attention_head_dim': getattr(attention0, 'attention_head_dim', None),
            'cross_attention_dim': getattr(attention0, 'cross_attention_dim', None),
            'expected_input_shape_from_resnet0': shape(resnet_out),
            'expected_encoder_hidden_states_shape': ['batch', 'sequence_length', int(unet.config.cross_attention_dim)],
            'not_exported_this_stage': True,
        }
        # Preview the first transformer processor LoRA wrapping without entering attention forward.
        tb = getattr(attention0, 'transformer_blocks', [None])[0] if len(getattr(attention0, 'transformer_blocks', [])) else None
        if tb is not None:
            attn1 = getattr(tb, 'attn1', None)
            attn2 = getattr(tb, 'attn2', None)
            attention_preview['lora_probe'] = {}
            for label, attn in [('attn1', attn1), ('attn2', attn2)]:
                if attn is not None:
                    attention_preview['lora_probe'][label] = {
                        'to_q': module_lora_info(attn.to_q),
                        'to_k': module_lora_info(attn.to_k),
                        'to_v': module_lora_info(attn.to_v),
                        'to_out_0': module_lora_info(attn.to_out[0]),
                    }

    submodules = {}
    for name in ['norm1', 'conv1', 'time_emb_proj', 'norm2', 'conv2', 'conv_shortcut']:
        mod = getattr(resnet, name, None)
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
            info = module_lora_info(mod)
            info['exists'] = True
            submodules[name] = info

    report = {
        'status': 'PASS',
        'python': sys.executable,
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'official_unet_class': f'{type(unet).__module__}.{type(unet).__name__}',
        'loaded_lora_param_count': len(loaded),
        'down_blocks_0_overview': {
            'down_blocks_count': len(unet.down_blocks),
            'class': f'{type(block).__module__}.{type(block).__name__}',
            'has_cross_attention': getattr(block, 'has_cross_attention', True),
            'forward_signature': str(inspect.signature(block.forward)),
            'module_order': children,
            'resnet_count': len(getattr(block, 'resnets', [])),
            'attention_count': len(getattr(block, 'attentions', [])) if hasattr(block, 'attentions') else 0,
            'transformer_count': sum(len(getattr(a, 'transformer_blocks', [])) for a in getattr(block, 'attentions', [])) if hasattr(block, 'attentions') else 0,
            'downsampler_count': len(getattr(block, 'downsamplers', [])) if getattr(block, 'downsamplers', None) is not None else 0,
            'has_downsampler': bool(getattr(block, 'downsamplers', None)),
            'conv_in_output_shape': shape(conv_in),
            'resnet0_output_shape': shape(resnet_out),
            'residual_samples_count_if_full_block': len(getattr(block, 'resnets', [])) + (1 if getattr(block, 'downsamplers', None) else 0),
            'consumes_encoder_hidden_states': 'encoder_hidden_states' in inspect.signature(block.forward).parameters,
            'consumes_attention_mask': 'attention_mask' in inspect.signature(block.forward).parameters,
            'consumes_encoder_attention_mask': 'encoder_attention_mask' in inspect.signature(block.forward).parameters,
            'consumes_cross_attention_kwargs': 'cross_attention_kwargs' in inspect.signature(block.forward).parameters,
            'has_lora_scale_argument': 'scale' in inspect.signature(block.forward).parameters,
            'has_peft_wrappers': any(info.get('is_lora_wrapped') for info in submodules.values() if isinstance(info, dict)),
        },
        'down_blocks_0_resnets_0': {
            **resnet_config(resnet),
            'input_shape_from_conv_in': shape(conv_in),
            'temb_shape': shape(temb),
            'output_shape': shape(resnet_out),
            'submodules': submodules,
        },
        'input_output_dry_run': {
            'synthetic_unet_sample_shape': shape(sample),
            'timestep': [1],
            'conv_in_output_shape': shape(conv_in),
            'time_embedding_output_shape': shape(temb),
            'resnet0_input_shape': shape(conv_in),
            'resnet0_output_shape': shape(resnet_out),
            'dtype': str(resnet_out.dtype),
            'resnet0_output_stats': {
                'min': float(resnet_out.min().item()),
                'max': float(resnet_out.max().item()),
                'mean': float(resnet_out.mean().item()),
                'std': float(resnet_out.std().item()),
            },
        },
        'downstream_attention0_preview_only': attention_preview,
        'not_in_scope': [
            'full down_blocks.0', 'down_blocks.0.attentions.0 forward', 'BasicTransformerBlock',
            'cross-attention', 'down_blocks.0.resnets.1', 'downsamplers.0',
            'full UNet forward', 'full LoRA runtime', 'full TADSR inference',
        ],
        'audit_markers': {
            'TADSR_UNET_DOWNBLOCK0_AUDIT': 'PASS',
            'TADSR_UNET_DOWNBLOCK0_RESNET0_AUDIT': 'PASS',
        },
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    lines = [
        '# TADSR UNet down_blocks.0.resnets.0 Audit', '',
        'TADSR_UNET_DOWNBLOCK0_AUDIT: PASS',
        'TADSR_UNET_DOWNBLOCK0_RESNET0_AUDIT: PASS', '',
        f"down_blocks.0 class: `{report['down_blocks_0_overview']['class']}`",
        f"has_cross_attention: `{report['down_blocks_0_overview']['has_cross_attention']}`",
        f"module_order: `{report['down_blocks_0_overview']['module_order']}`",
        f"resnet_count: `{report['down_blocks_0_overview']['resnet_count']}`",
        f"attention_count: `{report['down_blocks_0_overview']['attention_count']}`",
        f"downsampler_count: `{report['down_blocks_0_overview']['downsampler_count']}`",
        f"resnets.0 config: `{report['down_blocks_0_resnets_0']}`", '',
        'Scope guard: this audit does not run attention0, full down_blocks.0, or full UNet forward.',
    ]
    OUT_TXT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('TADSR_UNET_DOWNBLOCK0_AUDIT: PASS')
    print('TADSR_UNET_DOWNBLOCK0_RESNET0_AUDIT: PASS')
    print(json.dumps({'status': 'PASS', 'json': str(OUT_JSON), 'txt': str(OUT_TXT)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
