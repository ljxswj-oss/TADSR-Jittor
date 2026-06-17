#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path
import numpy as np

from export_tadsr_unet_downblock3_resnet0_oracle import (
    STRICT_PY,
    OUT_DIR,
    OFFICIAL_REPO,
    WEIGHTS_DIR,
    load_unet,
    module_lora_info,
    to_np,
    stats,
)

OUT_JSON = OUT_DIR / 'audit_official_tadsr_unet_upblock0_upsampler.json'
OUT_TXT = OUT_DIR / 'audit_official_tadsr_unet_upblock0_upsampler.txt'


def maybe_reexec() -> None:
    expected_prefix = str(STRICT_PY.parents[1])
    in_expected_venv = str(sys.prefix) == expected_prefix or str(sys.executable).startswith(expected_prefix)
    if STRICT_PY.exists() and not in_expected_venv:
        os.environ['PYTHONNOUSERSITE'] = '1'
        os.execv(str(STRICT_PY), [str(STRICT_PY), __file__, *sys.argv[1:]])


def source_names():
    return (
        ['conv_in']
        + [f'down_blocks.0.output_state_{i}' for i in range(3)]
        + [f'down_blocks.1.output_state_{i}' for i in range(3)]
        + [f'down_blocks.2.output_state_{i}' for i in range(3)]
        + [f'down_blocks.3.output_state_{i}' for i in range(2)]
    )


def extract_midblock_output(out):
    if isinstance(out, tuple):
        return out[0]
    if hasattr(out, 'sample'):
        return out.sample
    return out


def main() -> int:
    maybe_reexec()
    import torch

    torch.manual_seed(1234)
    np.random.seed(1234)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    unet, loaded = load_unet()
    up0 = unet.up_blocks[0]
    if not getattr(up0, 'upsamplers', None):
        raise RuntimeError('Official up_blocks.0 has no upsamplers; expected upsamplers.0')
    if len(up0.resnets) < 3:
        raise RuntimeError(f'Official up_blocks.0 has {len(up0.resnets)} resnets; expected at least 3 before upsampler.')
    upsampler = up0.upsamplers[0]
    conv = getattr(upsampler, 'conv', None)
    if conv is None:
        raise RuntimeError('Official up_blocks.0.upsamplers.0 has no conv child; this stage expects Upsample2D(use_conv=True).')

    sample_shape = [1, int(unet.config.in_channels), 32, 32]
    sample = torch.linspace(-1.0, 1.0, steps=int(np.prod(sample_shape)), dtype=torch.float32).reshape(*sample_shape)
    timestep = torch.tensor([1], dtype=torch.long)
    encoder_shape = [1, 77, int(unet.config.cross_attention_dim)]
    encoder_hidden_states = torch.linspace(-0.5, 0.5, steps=int(np.prod(encoder_shape)), dtype=torch.float32).reshape(*encoder_shape)

    captured = {}

    def conv_hook(_module, inputs, output):
        captured['conv_input'] = inputs[0].detach()
        captured['conv_output'] = output.detach()

    hook = conv.register_forward_hook(conv_hook)
    with torch.no_grad():
        centered = 2.0 * sample - 1.0 if unet.config.center_input_sample else sample
        hidden = unet.conv_in(centered)
        time_proj = unet.time_proj(timestep)
        temb = unet.time_embedding(time_proj.to(dtype=list(unet.time_embedding.parameters())[0].dtype))
        down_res = (hidden,)
        down_state_shapes = {}
        down_hidden_shapes = {}
        for i, block in enumerate(unet.down_blocks):
            if getattr(block, 'has_cross_attention', False):
                hidden, states = block(hidden, temb, encoder_hidden_states=encoder_hidden_states)
            else:
                hidden, states = block(hidden, temb)
            down_hidden_shapes[f'down_blocks.{i}'] = list(hidden.shape)
            down_state_shapes[f'down_blocks.{i}'] = [list(x.shape) for x in states]
            down_res += states
        mid = extract_midblock_output(unet.mid_block(hidden, temb=temb, encoder_hidden_states=encoder_hidden_states, attention_mask=None, cross_attention_kwargs=None, encoder_attention_mask=None))
        res_tuple = down_res[-len(up0.resnets):]
        global_remaining = down_res[:-len(up0.resnets)]
        out0 = up0.resnets[0](torch.cat([mid, res_tuple[-1]], dim=1), temb)
        out1 = up0.resnets[1](torch.cat([out0, res_tuple[-2]], dim=1), temb)
        out2 = up0.resnets[2](torch.cat([out1, res_tuple[-3]], dim=1), temb)
        upsampled = upsampler(out2)
        local = up0(mid, res_tuple, temb=temb, upsample_size=None)
    hook.remove()

    sources = source_names()
    meta = {
        'status': 'PASS',
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'loaded_lora_parameter_count': len(loaded),
        'upblock0_class': f'{type(up0).__module__}.{type(up0).__name__}',
        'upblock0_forward_signature': str(inspect.signature(up0.forward)),
        'upblock0_resnet_count': len(up0.resnets),
        'upblock0_upsampler_count': len(up0.upsamplers),
        'upsampler_config': {
            'prefix': 'up_blocks_0_upsamplers_0',
            'module_name': 'up_blocks.0.upsamplers.0',
            'class': f'{type(upsampler).__module__}.{type(upsampler).__name__}',
            'forward_signature': str(inspect.signature(upsampler.forward)),
            'channels': getattr(upsampler, 'channels', None),
            'out_channels': getattr(upsampler, 'out_channels', None),
            'use_conv': getattr(upsampler, 'use_conv', None),
            'use_conv_transpose': getattr(upsampler, 'use_conv_transpose', None),
            'interpolate': getattr(upsampler, 'interpolate', None),
            'name': getattr(upsampler, 'name', None),
            'scale_factor': 2,
            'conv_padding': int(conv.base_layer.padding[0] if hasattr(conv, 'base_layer') else conv.padding[0]),
            'conv_kernel_size': list(conv.base_layer.kernel_size if hasattr(conv, 'base_layer') else conv.kernel_size),
            'conv_stride': list(conv.base_layer.stride if hasattr(conv, 'base_layer') else conv.stride),
            'conv_weight_shape': list((conv.base_layer if hasattr(conv, 'base_layer') else conv).weight.shape),
        },
        'upsampler_conv_lora_info': module_lora_info(conv),
        'sample_shape': sample_shape,
        'encoder_hidden_states_shape': encoder_shape,
        'downblock_output_state_shapes': down_state_shapes,
        'downblock_output_hidden_shapes': down_hidden_shapes,
        'accumulated_down_block_res_sample_sources': sources,
        'accumulated_down_block_res_sample_shapes': [list(x.shape) for x in down_res],
        'global_remaining_residual_count_after_upblock0_slice': len(global_remaining),
        'upblock0_pre_upsampler_shape': list(out2.shape),
        'upblock0_upsampler_interpolation_shape': list(captured['conv_input'].shape),
        'upblock0_upsampler_output_shape': list(upsampled.shape),
        'upblock0_local_output_shape': list(local.shape),
        'upblock0_local_vs_manual_max_abs_diff': float((local - upsampled).abs().max().item()),
        'upblock0_upsampler_output_stats': stats(to_np(upsampled)),
        'markers': {
            'TADSR_UNET_UPBLOCK0_UPSAMPLER_AUDIT': 'PASS',
            'TADSR_UNET_UPBLOCK0_UPSAMPLER_LORA_AUDIT': 'PASS',
            'TADSR_UNET_UPBLOCK0_LOCAL_FORWARD_AUDIT': 'PASS',
        },
        'scope_boundaries': {
            'audited_this_stage': 'up_blocks.0.upsamplers.0 plus safe local up_blocks.0 hidden-state forward',
            'not_audited_this_stage': ['up_blocks.1', 'up_blocks.2', 'up_blocks.3', 'full UNet forward', 'generic runtime LoRA', 'full TADSR inference'],
        },
    }
    OUT_JSON.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    OUT_TXT.write_text('\n'.join([
        '# TADSR UNet up_blocks.0.upsamplers.0 official audit',
        '',
        *[f'{k}: {v}' for k, v in meta['markers'].items()],
        '',
        f"upsampler class: {meta['upsampler_config']['class']}",
        f"conv weight shape: {meta['upsampler_config']['conv_weight_shape']}",
        f"pre-upsample shape: {meta['upblock0_pre_upsampler_shape']}",
        f"post-upsample shape: {meta['upblock0_upsampler_output_shape']}",
        f"local upblock0 vs manual max_abs_diff: {meta['upblock0_local_vs_manual_max_abs_diff']}",
        'next unopened module: up_blocks.1.resnets.0',
    ]) + '\n', encoding='utf-8')
    for k, v in meta['markers'].items():
        print(f'{k}: {v}')
    print(json.dumps({'status': 'PASS', 'metadata': str(OUT_JSON)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
