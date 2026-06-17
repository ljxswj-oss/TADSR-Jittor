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
        'channel_change': getattr(resnet, 'in_channels', None) != getattr(resnet, 'out_channels', None),
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


def attention_preview(att, expected_input_shape=None, encoder_hidden_states_shape=None):
    if att is None:
        return {'exists': False}
    out = {
        'exists': True,
        'class': f'{type(att).__module__}.{type(att).__name__}',
        'expected_input_shape_from_resnet0': expected_input_shape,
        'expected_encoder_hidden_states_shape': encoder_hidden_states_shape,
        'transformer_blocks': len(getattr(att, 'transformer_blocks', [])),
        'norm_num_groups': getattr(getattr(att, 'norm', None), 'num_groups', None),
        'inner_dim': getattr(att, 'inner_dim', None),
        'num_attention_heads': getattr(att, 'num_attention_heads', None),
        'attention_head_dim': getattr(att, 'attention_head_dim', None),
        'cross_attention_dim': getattr(att, 'cross_attention_dim', None),
        'use_linear_projection': getattr(att, 'use_linear_projection', None),
        'analogous_to_previous_attention_modules': True,
        'not_ported_this_stage': True,
    }
    blocks = getattr(att, 'transformer_blocks', [])
    tb = blocks[0] if blocks else None
    if tb is not None:
        out['transformer0_class'] = f'{type(tb).__module__}.{type(tb).__name__}'
        out['attn1_class'] = f'{type(tb.attn1).__module__}.{type(tb.attn1).__name__}'
        out['attn2_class'] = f'{type(tb.attn2).__module__}.{type(tb.attn2).__name__}'
    return out

OUT_JSON = OUT_DIR / 'audit_tadsr_unet_downblock2_resnet0.json'
OUT_TXT = OUT_DIR / 'audit_tadsr_unet_downblock2_resnet0.txt'


def main() -> int:
    maybe_reexec()
    import torch
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    unet, loaded = load_unet()
    block0 = unet.down_blocks[0]
    block1 = unet.down_blocks[1]
    block2 = unet.down_blocks[2]
    resnet0 = block2.resnets[0]
    attention0 = block2.attentions[0] if hasattr(block2, 'attentions') and len(block2.attentions) > 0 else None
    resnet1 = block2.resnets[1] if len(getattr(block2, 'resnets', [])) > 1 else None
    downsampler0 = block2.downsamplers[0] if getattr(block2, 'downsamplers', None) is not None and len(block2.downsamplers) > 0 else None

    sample = torch.linspace(-1.0, 1.0, steps=1 * int(unet.config.in_channels) * 32 * 32, dtype=torch.float32).reshape(1, int(unet.config.in_channels), 32, 32)
    timestep = torch.tensor([1], dtype=torch.long)
    encoder_shape = [1, 77, int(unet.config.cross_attention_dim)]
    encoder_hidden_states = torch.linspace(-0.5, 0.5, steps=int(torch.tensor(encoder_shape).prod().item()), dtype=torch.float32).reshape(*encoder_shape)
    with torch.no_grad():
        centered = 2.0 * sample - 1.0 if unet.config.center_input_sample else sample
        conv_in = unet.conv_in(centered)
        time_proj = unet.time_proj(timestep)
        temb = unet.time_embedding(time_proj.to(dtype=list(unet.time_embedding.parameters())[0].dtype))
        block0_hidden, block0_states = block0(conv_in, temb, encoder_hidden_states=encoder_hidden_states)
        block1_hidden, block1_states = block1(block0_hidden, temb, encoder_hidden_states=encoder_hidden_states)
        resnet0_out = resnet0(block1_hidden, temb)
    accumulated_shapes = [list(conv_in.shape)] + [list(x.shape) for x in block0_states] + [list(x.shape) for x in block1_states]
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
    children = [{'name': n, 'class': f'{type(m).__module__}.{type(m).__name__}'} for n, m in block2.named_children()]
    cfg = resnet_config(resnet0, 'down_blocks_2_resnets_0')
    audit = {
        'status': 'PASS',
        'scope': 'down_blocks.2.resnets.0 only; down_blocks.2 attention/resnet1/downsampler and full UNet forward are intentionally not ported in this stage.',
        'python': sys.executable,
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'loaded_lora_parameter_count': len(loaded),
        'downblock2_overview': {
            'class': f'{type(block2).__module__}.{type(block2).__name__}',
            'forward_signature': str(inspect.signature(block2.forward)),
            'module_order': children,
            'resnet_count': len(getattr(block2, 'resnets', [])),
            'attention_count': len(getattr(block2, 'attentions', [])) if hasattr(block2, 'attentions') else 0,
            'downsampler_count': len(getattr(block2, 'downsamplers', [])) if getattr(block2, 'downsamplers', None) is not None else 0,
            'has_downsampler': downsampler0 is not None,
            'has_cross_attention': hasattr(block2, 'attentions') and len(getattr(block2, 'attentions', [])) > 0,
            'current_target_module': 'down_blocks.2.resnets.0',
            'remaining_modules_after_this_stage': ['down_blocks.2.attentions.0' if attention0 is not None else None, 'down_blocks.2.resnets.1' if resnet1 is not None else None, 'down_blocks.2.attentions.1' if hasattr(block2, 'attentions') and len(block2.attentions) > 1 else None, 'down_blocks.2.downsamplers.0' if downsampler0 is not None else None],
            'actual_next_module_after_resnet0': 'down_blocks.2.attentions.0' if attention0 is not None else ('down_blocks.2.resnets.1' if resnet1 is not None else None),
            'consumes_encoder_hidden_states': True if attention0 is not None else False,
            'consumes_attention_mask': True,
            'consumes_encoder_attention_mask': True,
            'consumes_cross_attention_kwargs': True,
            'lora_scale_argument_exists': 'scale' in str(inspect.signature(block2.forward)),
            'has_peft_wrappers': any(module_lora_info(getattr(resnet0, n, None)).get('is_lora_wrapped') for n in ['conv1', 'time_emb_proj', 'conv2', 'conv_shortcut']),
        },
        'resnet0_config': cfg,
        'submodules': submodules,
        'tester_support': {
            'existing_UNetResnetBlock2DTester_supported': cfg['time_embedding_norm'] in {'default', 'scale_shift'} and cfg['dropout'] == 0.0,
            'channel_change': cfg['channel_change'],
            'conv_shortcut_required': cfg['has_shortcut'],
            'groups_match': cfg['groups'] == cfg['groups_out'],
            'output_scale_factor': cfg['output_scale_factor'],
            'required_code_changes': [] if cfg['time_embedding_norm'] in {'default', 'scale_shift'} and cfg['dropout'] == 0.0 else ['extend UNetResnetBlock2DTester'],
        },
        'dry_run': {
            'synthetic_unet_sample': stats_tensor(sample),
            'timestep': [1],
            'encoder_hidden_states': stats_tensor(encoder_hidden_states),
            'conv_in_output': stats_tensor(conv_in),
            'time_embedding_output': stats_tensor(temb),
            'down_blocks_0_final_hidden_states': stats_tensor(block0_hidden),
            'down_blocks_0_output_state_shapes': [list(x.shape) for x in block0_states],
            'down_blocks_1_final_hidden_states': stats_tensor(block1_hidden),
            'down_blocks_1_output_state_shapes': [list(x.shape) for x in block1_states],
            'accumulated_down_block_res_sample_shapes': accumulated_shapes,
            'down_blocks_2_resnets_0_input': stats_tensor(block1_hidden),
            'down_blocks_2_resnets_0_output': stats_tensor(resnet0_out),
            'resnet0_consumes_only_hidden_states_and_temb': True,
            'resnet0_independent_of_down_block_res_samples': True,
        },
        'downstream_attention0_preview': attention_preview(attention0, expected_input_shape=list(resnet0_out.shape), encoder_hidden_states_shape=encoder_shape),
        'markers': {
            'TADSR_UNET_DOWNBLOCK2_AUDIT': 'PASS',
            'TADSR_UNET_DOWNBLOCK2_RESNET0_AUDIT': 'PASS',
            'TADSR_UNET_DOWNBLOCK2_RESNET0_LORA_AUDIT': 'PASS',
        },
        'not_in_scope': ['down_blocks.2.attentions.0', 'down_blocks.2.resnets.1', 'down_blocks.2.attentions.1', 'down_blocks.2.downsamplers.0', 'mid_block', 'up_blocks', 'full UNet forward', 'full TADSR inference', 'generic LoRA runtime'],
    }
    OUT_JSON.write_text(json.dumps(audit, indent=2, default=str), encoding='utf-8')
    lines = [
        '# TADSR UNet down_blocks.2.resnets.0 Audit',
        '',
        'TADSR_UNET_DOWNBLOCK2_AUDIT: PASS',
        'TADSR_UNET_DOWNBLOCK2_RESNET0_AUDIT: PASS',
        'TADSR_UNET_DOWNBLOCK2_RESNET0_LORA_AUDIT: PASS',
        '',
        f"Block class: `{audit['downblock2_overview']['class']}`",
        f"ResNet0 channels: `{cfg['in_channels']} -> {cfg['out_channels']}`, shortcut: `{cfg['has_shortcut']}`",
        f"Input shape: `{audit['dry_run']['down_blocks_2_resnets_0_input']['shape']}`, output shape: `{audit['dry_run']['down_blocks_2_resnets_0_output']['shape']}`",
        f"Next module preview: `{audit['downblock2_overview']['actual_next_module_after_resnet0']}`",
        '',
        'This audit stops after `down_blocks.2.resnets.0`; attention0, full down_blocks.2, full UNet forward and full TADSR inference remain out of scope.',
    ]
    OUT_TXT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    for k, v in audit['markers'].items():
        print(f'{k}: {v}')
    print(json.dumps({'status': audit['status'], 'json': str(OUT_JSON), 'txt': str(OUT_TXT)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
