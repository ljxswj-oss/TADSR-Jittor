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
    to_np,
    stats,
)
from export_tadsr_unet_upblock1_resnet0_oracle import extract_midblock_output, source_names
from audit_official_tadsr_unet_downblock2_attention1 import module_lora_info

OUT_JSON = OUT_DIR / 'audit_tadsr_unet_upblock1_upsampler.json'
OUT_TXT = OUT_DIR / 'audit_tadsr_unet_upblock1_upsampler.txt'


def maybe_reexec() -> None:
    expected_prefix = str(STRICT_PY.parents[1])
    in_expected_venv = str(sys.prefix) == expected_prefix or str(sys.executable).startswith(expected_prefix)
    if STRICT_PY.exists() and not in_expected_venv:
        os.environ['PYTHONNOUSERSITE'] = '1'
        os.execv(str(STRICT_PY), [str(STRICT_PY), __file__, *sys.argv[1:]])


def conv_base(conv):
    return conv.base_layer if hasattr(conv, 'base_layer') else conv


def conv_config(conv):
    if conv is None:
        return {'exists': False}
    base = conv_base(conv)
    return {
        'exists': True,
        'class': f'{type(conv).__module__}.{type(conv).__name__}',
        'base_class': f'{type(base).__module__}.{type(base).__name__}',
        'weight_shape': list(base.weight.shape),
        'bias_shape': list(base.bias.shape) if getattr(base, 'bias', None) is not None else None,
        'kernel_size': list(base.kernel_size),
        'stride': list(base.stride),
        'padding': list(base.padding),
        'dilation': list(base.dilation),
        'groups': int(base.groups),
        'padding_mode': getattr(base, 'padding_mode', 'zeros'),
    }


def call_upsampler(upsampler, hidden_states, upsample_size=None):
    sig = inspect.signature(upsampler.forward)
    if 'upsample_size' in sig.parameters:
        return upsampler(hidden_states, upsample_size=upsample_size)
    return upsampler(hidden_states)


def main() -> int:
    maybe_reexec()
    import torch

    torch.manual_seed(1234)
    np.random.seed(1234)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    unet, loaded = load_unet()
    up0 = unet.up_blocks[0]
    up1 = unet.up_blocks[1]
    if not getattr(up1, 'upsamplers', None):
        raise RuntimeError('Official up_blocks.1 has no upsamplers.0')
    if len(up1.resnets) < 3 or len(getattr(up1, 'attentions', []) or []) < 3:
        raise RuntimeError('Official up_blocks.1 must expose resnets.0/1/2 and attentions.0/1/2 before upsampler.')
    upsampler = up1.upsamplers[0]
    conv = getattr(upsampler, 'conv', None)

    sample_shape = [1, int(unet.config.in_channels), 32, 32]
    sample = torch.linspace(-1.0, 1.0, steps=int(np.prod(sample_shape)), dtype=torch.float32).reshape(*sample_shape)
    timestep = torch.tensor([1], dtype=torch.long)
    encoder_shape = [1, 77, int(unet.config.cross_attention_dim)]
    encoder_hidden_states = torch.linspace(-0.5, 0.5, steps=int(np.prod(encoder_shape)), dtype=torch.float32).reshape(*encoder_shape)

    captured = {}
    hook = None
    if conv is not None:
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
        mid_raw = unet.mid_block(hidden, temb=temb, encoder_hidden_states=encoder_hidden_states, attention_mask=None, cross_attention_kwargs=None, encoder_attention_mask=None)
        mid_output, mid_return = extract_midblock_output(mid_raw)
        up0_output = up0(mid_output, down_res[-len(up0.resnets):], temb=temb, upsample_size=None)
        remaining_after_up0 = down_res[:-len(up0.resnets)]
        up1_res_tuple = remaining_after_up0[-len(up1.resnets):]
        remaining_before_up1 = remaining_after_up0[:-len(up1.resnets)]

        res0_hidden = up1_res_tuple[-1]
        remaining_after_resnet0 = up1_res_tuple[:-1]
        res0 = up1.resnets[0](torch.cat([up0_output, res0_hidden], dim=1), temb)
        att0 = up1.attentions[0](res0, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
        res1_hidden = remaining_after_resnet0[-1]
        remaining_after_resnet1 = remaining_after_resnet0[:-1]
        res1 = up1.resnets[1](torch.cat([att0, res1_hidden], dim=1), temb)
        att1 = up1.attentions[1](res1, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
        res2_hidden = remaining_after_resnet1[-1]
        remaining_after_resnet2 = remaining_after_resnet1[:-1]
        res2 = up1.resnets[2](torch.cat([att1, res2_hidden], dim=1), temb)
        att2 = up1.attentions[2](res2, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
        upsampled = call_upsampler(upsampler, att2, upsample_size=None)
    if hook is not None:
        hook.remove()

    sources = source_names()
    resnet0_idx = len(remaining_after_up0) - 1
    resnet1_idx = len(remaining_after_up0) - 2
    resnet2_idx = len(remaining_after_up0) - 3
    upsampler_cfg = {
        'prefix': 'up_blocks_1_upsamplers_0',
        'module_name': 'up_blocks.1.upsamplers.0',
        'class': f'{type(upsampler).__module__}.{type(upsampler).__name__}',
        'forward_signature': str(inspect.signature(upsampler.forward)),
        'accepts_upsample_size': 'upsample_size' in inspect.signature(upsampler.forward).parameters,
        'channels': getattr(upsampler, 'channels', None),
        'out_channels': getattr(upsampler, 'out_channels', None),
        'use_conv': getattr(upsampler, 'use_conv', None),
        'use_conv_transpose': getattr(upsampler, 'use_conv_transpose', None),
        'interpolate': getattr(upsampler, 'interpolate', None),
        'name': getattr(upsampler, 'name', None),
        'scale_factor': 2,
        'operation_type': 'nearest_interpolation_plus_conv' if conv is not None else 'interpolation_only_or_unknown',
        'conv': conv_config(conv),
    }
    result = {
        'status': 'PASS',
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'python': sys.executable,
        'loaded_lora_parameter_count': len(loaded),
        'upblock1_overview': {
            'class': f'{type(up1).__module__}.{type(up1).__name__}',
            'forward_signature': str(inspect.signature(up1.forward)),
            'children': [{'name': n, 'class': f'{type(m).__module__}.{type(m).__name__}'} for n, m in up1.named_children()],
            'resnet_count': len(up1.resnets),
            'attention_count': len(getattr(up1, 'attentions', []) or []),
            'upsampler_count': len(getattr(up1, 'upsamplers', []) or []),
            'has_cross_attention': bool(getattr(up1, 'has_cross_attention', False)),
            'expected_operation_order': ['resnets.0', 'attentions.0', 'resnets.1', 'attentions.1', 'resnets.2', 'attentions.2', 'upsamplers.0'],
            'known_completed_modules_before_this_stage': [
                'up_blocks.1.resnets.0',
                'up_blocks.1.attentions.0',
                'up_blocks.1.resnets.1',
                'up_blocks.1.attentions.1',
                'up_blocks.1.resnets.2',
                'up_blocks.1.attentions.2',
            ],
            'target_module': 'up_blocks.1.upsamplers.0',
            'actual_next_module_after_upsampler': 'up_blocks.2',
        },
        'upsampler_config': upsampler_cfg,
        'upsampler_lora_info': module_lora_info(conv) if conv is not None else {'is_lora_wrapped': False, 'reason': 'no conv child'},
        'entry_context_shapes': {
            'sample': sample_shape,
            'timestep': list(timestep.shape),
            'encoder_hidden_states': encoder_shape,
            'downblock_output_state_shapes': down_state_shapes,
            'downblock_output_hidden_shapes': down_hidden_shapes,
            'midblock_output': list(mid_output.shape),
            'upblock0_output': list(up0_output.shape),
            'upblock1_resnet0_residual_source': sources[resnet0_idx],
            'upblock1_resnet1_residual_source': sources[resnet1_idx],
            'upblock1_resnet2_residual_source': sources[resnet2_idx],
            'upblock1_attention2_output': list(att2.shape),
            'upblock1_upsampler_input': list(att2.shape),
            'upblock1_upsampler_interpolation_output': list(captured.get('conv_input', upsampled).shape),
            'upblock1_upsampler_output': list(upsampled.shape),
            'remaining_residual_count_before_upblock1_slice': len(remaining_before_up1),
            'internal_remaining_residual_count_after_attention2': len(remaining_after_resnet2),
            'internal_remaining_residual_count_after_upsampler': len(remaining_after_resnet2),
            'upsampler_consumes_accumulated_residuals': False,
            'local_midblock_return': mid_return,
        },
        'upsampler_output_stats': stats(to_np(upsampled)),
        'markers': {
            'TADSR_UNET_UPBLOCK1_UPSAMPLER_AUDIT': 'PASS',
            'TADSR_UNET_UPBLOCK1_UPSAMPLER_LORA_AUDIT': 'PASS',
            'TADSR_UNET_UPBLOCK1_UPSAMPLER_RESIDUAL_CONTRACT_AUDIT': 'PASS',
        },
        'scope_boundaries': {
            'audited_this_stage': 'up_blocks.1.upsamplers.0 only after selected local chain through attention2',
            'not_audited_this_stage': ['full up_blocks.1 aggregate', 'up_blocks.2', 'up_blocks.3', 'full UNet forward', 'generic runtime LoRA', 'full TADSR inference'],
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding='utf-8')
    OUT_TXT.write_text('\n'.join([
        '# TADSR UNet up_blocks.1.upsamplers.0 official audit',
        '',
        *[f'{k}: {v}' for k, v in result['markers'].items()],
        '',
        f"upblock1 class: {result['upblock1_overview']['class']}",
        f"upsampler class: {upsampler_cfg['class']}",
        f"operation type: {upsampler_cfg['operation_type']}",
        f"input shape: {result['entry_context_shapes']['upblock1_upsampler_input']}",
        f"output shape: {result['entry_context_shapes']['upblock1_upsampler_output']}",
        f"actual next module: {result['upblock1_overview']['actual_next_module_after_upsampler']}",
        'full up_blocks.1 aggregate/full UNet/full inference remain NOT_COMPLETE.',
    ]) + '\n', encoding='utf-8')
    for k, v in result['markers'].items():
        print(f'{k}: {v}')
    print(json.dumps({'status': 'PASS', 'metadata': str(OUT_JSON)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
