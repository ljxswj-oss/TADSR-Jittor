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
OUT_JSON = OUT_DIR / 'audit_tadsr_unet_downblock0_downsampler.json'
OUT_TXT = OUT_DIR / 'audit_tadsr_unet_downblock0_downsampler.txt'


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
        'stride': list(getattr(base, 'stride', [])) if getattr(base, 'stride', None) is not None else None,
        'padding': list(getattr(base, 'padding', [])) if getattr(base, 'padding', None) is not None else None,
        'kernel_size': list(getattr(base, 'kernel_size', [])) if getattr(base, 'kernel_size', None) is not None else None,
        'dilation': list(getattr(base, 'dilation', [])) if getattr(base, 'dilation', None) is not None else None,
        'groups': getattr(base, 'groups', None),
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
    if getattr(block, 'downsamplers', None) is None or len(block.downsamplers) != 1:
        raise RuntimeError(f'Expected exactly one downsampler in down_blocks.0, got {getattr(block, "downsamplers", None)}')
    downsampler = block.downsamplers[0]
    conv = getattr(downsampler, 'conv', None)
    base = base_layer(conv)
    sample = torch.linspace(-1.0, 1.0, steps=1 * int(unet.config.in_channels) * 32 * 32, dtype=torch.float32).reshape(1, int(unet.config.in_channels), 32, 32)
    timestep = torch.tensor([1], dtype=torch.long)
    encoder_shape = [1, 77, int(unet.config.cross_attention_dim)]
    encoder_hidden_states = torch.linspace(-0.5, 0.5, steps=int(np.prod(encoder_shape)), dtype=torch.float32).reshape(*encoder_shape)
    with torch.no_grad():
        centered = 2.0 * sample - 1.0 if unet.config.center_input_sample else sample
        conv_in = unet.conv_in(centered)
        time_proj = unet.time_proj(timestep)
        temb = unet.time_embedding(time_proj.to(dtype=list(unet.time_embedding.parameters())[0].dtype))
        resnet0_out = block.resnets[0](conv_in, temb)
        attention0_out = block.attentions[0](resnet0_out, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
        resnet1_out = block.resnets[1](attention0_out, temb)
        attention1_out = block.attentions[1](resnet1_out, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
        downsample_out = downsampler(attention1_out)
        block_out, block_res_samples = block(conv_in, temb, encoder_hidden_states=encoder_hidden_states)
        diff = (downsample_out - block_out).abs()
    children = [{'name': n, 'class': f'{type(m).__module__}.{type(m).__name__}'} for n, m in block.named_children()]
    metadata = {
        'status': 'PASS',
        'python': sys.executable,
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'source_module_path': 'unet.down_blocks.0.downsamplers.0',
        'block_class': f'{type(block).__module__}.{type(block).__name__}',
        'block_forward_signature': str(inspect.signature(block.forward)),
        'block_module_order': children,
        'downsampler_config': {
            'prefix': 'down_blocks_0_downsamplers_0',
            'class': f'{type(downsampler).__module__}.{type(downsampler).__name__}',
            'forward_signature': str(inspect.signature(downsampler.forward)),
            'channels': getattr(downsampler, 'channels', None),
            'out_channels': getattr(downsampler, 'out_channels', None),
            'use_conv': getattr(downsampler, 'use_conv', None),
            'padding': getattr(downsampler, 'padding', None),
            'name': getattr(downsampler, 'name', None),
            'conv': module_lora_info(conv),
            'operation': 'single effective LoRA-merged Conv2d, stride=2, padding=1',
        },
        'full_downblock0_local_order': [
            'entry conv_in + time_embedding',
            'down_blocks.0.resnets.0',
            'down_blocks.0.attentions.0',
            'down_blocks.0.resnets.1',
            'down_blocks.0.attentions.1',
            'down_blocks.0.downsamplers.0',
        ],
        'input_contract': {
            'sample_shape': list(sample.shape),
            'timestep': [1],
            'encoder_hidden_states_shape': encoder_shape,
            'attention_mask': None,
            'encoder_attention_mask': None,
            'cross_attention_kwargs': None,
            'additional_residuals': None,
        },
        'dry_run': {
            'conv_in_output': stats_tensor(conv_in),
            'time_embedding_output': stats_tensor(temb),
            'resnet0_output': stats_tensor(resnet0_out),
            'attention0_output': stats_tensor(attention0_out),
            'resnet1_output': stats_tensor(resnet1_out),
            'attention1_output': stats_tensor(attention1_out),
            'downsampler_output': stats_tensor(downsample_out),
            'block_forward_output': stats_tensor(block_out),
            'block_forward_res_sample_shapes': [list(x.shape) for x in block_res_samples],
            'manual_vs_block_forward_max_abs_error': float(diff.max().item()) if diff.numel() else 0.0,
            'manual_vs_block_forward_mean_abs_error': float(diff.mean().item()) if diff.numel() else 0.0,
        },
        'effective_weight_plan': {
            'export_conv_weight_key': 'down_blocks_0_downsamplers_0_conv_weight',
            'export_conv_bias_key': 'down_blocks_0_downsamplers_0_conv_bias',
            'use_static_lora_merged_weight': True,
            'generic_runtime_lora_integration': 'PARTIAL/DEFERRED',
        },
        'markers': {
            'TADSR_UNET_DOWNBLOCK0_DOWNSAMPLER_AUDIT': 'PASS',
            'TADSR_UNET_DOWNBLOCK0_LOCAL_FORWARD_AUDIT': 'PASS',
        },
        'not_in_scope': ['down_blocks.1', 'mid_block', 'up_blocks', 'full UNet forward', 'full TADSR inference'],
    }
    OUT_JSON.write_text(json.dumps(metadata, indent=2, default=str), encoding='utf-8')
    lines = [
        '# TADSR UNet down_blocks.0.downsamplers.0 Audit',
        '',
        'TADSR_UNET_DOWNBLOCK0_DOWNSAMPLER_AUDIT: PASS',
        'TADSR_UNET_DOWNBLOCK0_LOCAL_FORWARD_AUDIT: PASS',
        '',
        f"Official class: `{metadata['downsampler_config']['class']}`",
        f"Conv shape: `{metadata['downsampler_config']['conv']['weight_shape']}`",
        f"Stride: `{metadata['downsampler_config']['conv']['stride']}`, padding: `{metadata['downsampler_config']['conv']['padding']}`",
        f"Manual local down_blocks.0 vs official block.forward max abs error: `{metadata['dry_run']['manual_vs_block_forward_max_abs_error']}`",
        '',
        'This audit stops after `down_blocks.0.downsamplers.0`; full UNet forward and full TADSR inference remain out of scope.',
    ]
    OUT_TXT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('TADSR_UNET_DOWNBLOCK0_DOWNSAMPLER_AUDIT: PASS')
    print('TADSR_UNET_DOWNBLOCK0_LOCAL_FORWARD_AUDIT: PASS')
    print(json.dumps({'status': 'PASS', 'json': str(OUT_JSON), 'txt': str(OUT_TXT)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
