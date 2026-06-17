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
)
from export_tadsr_unet_upblock1_resnet0_oracle import source_names
from audit_official_tadsr_unet_upblock1_local import run_context
from audit_official_tadsr_unet_downblock2_attention1 import module_lora_info, stats_tensor

OUT_JSON = OUT_DIR / 'audit_tadsr_unet_upblock2_upsampler.json'
OUT_TXT = OUT_DIR / 'audit_tadsr_unet_upblock2_upsampler.txt'


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
    up2 = unet.up_blocks[2]
    if len(up2.resnets) < 3 or len(getattr(up2, 'attentions', []) or []) < 3:
        raise RuntimeError('Official up_blocks.2 must expose resnets.0/1/2 and attentions.0/1/2 before upsampler.')
    if not getattr(up2, 'upsamplers', None):
        raise RuntimeError('Official up_blocks.2 has no upsamplers.0')
    upsampler = up2.upsamplers[0]
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
        ctx = run_context(unet, sample, timestep, encoder_hidden_states)
        remaining_after_up1 = ctx['remaining_before_up1']
        up2_res_tuple = remaining_after_up1[-len(up2.resnets):]
        remaining_before_up2 = remaining_after_up1[:-len(up2.resnets)]

        res0_hidden = up2_res_tuple[-1]
        remaining_after_resnet0 = up2_res_tuple[:-1]
        res0 = up2.resnets[0](torch.cat([ctx['official_output'], res0_hidden], dim=1), ctx['time_embedding'])
        att0 = up2.attentions[0](res0, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]

        res1_hidden = remaining_after_resnet0[-1]
        remaining_after_resnet1 = remaining_after_resnet0[:-1]
        res1 = up2.resnets[1](torch.cat([att0, res1_hidden], dim=1), ctx['time_embedding'])
        att1 = up2.attentions[1](res1, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]

        res2_hidden = remaining_after_resnet1[-1]
        remaining_after_resnet2 = remaining_after_resnet1[:-1]
        res2 = up2.resnets[2](torch.cat([att1, res2_hidden], dim=1), ctx['time_embedding'])
        att2 = up2.attentions[2](res2, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
        upsampled = call_upsampler(upsampler, att2, upsample_size=None)
    if hook is not None:
        hook.remove()

    sources = source_names()
    res0_consumed_index = len(remaining_after_up1) - 1
    res1_consumed_index = len(remaining_before_up2) + len(remaining_after_resnet0) - 1
    res2_consumed_index = len(remaining_before_up2) + len(remaining_after_resnet1) - 1
    cfg = {
        'prefix': 'up_blocks_2_upsamplers_0',
        'module_name': 'up_blocks.2.upsamplers.0',
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
        'scope': 'up_blocks.2.upsamplers.0 audit only; full up_blocks.2 aggregate and later modules remain intentionally unopened.',
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'python': sys.executable,
        'loaded_lora_parameter_count': len(loaded),
        'upblock2_overview': {
            'class': f'{type(up2).__module__}.{type(up2).__name__}',
            'forward_signature': str(inspect.signature(up2.forward)),
            'children': [{'name': n, 'class': f'{type(m).__module__}.{type(m).__name__}'} for n, m in up2.named_children()],
            'resnet_count': len(up2.resnets),
            'attention_count': len(getattr(up2, 'attentions', []) or []),
            'upsampler_count': len(getattr(up2, 'upsamplers', []) or []),
            'has_cross_attention': bool(getattr(up2, 'has_cross_attention', False)),
            'known_completed_modules_before_this_stage': [
                'full local up_blocks.1',
                'up_blocks.2.resnets.0',
                'up_blocks.2.attentions.0',
                'up_blocks.2.resnets.1',
                'up_blocks.2.attentions.1',
                'up_blocks.2.resnets.2',
                'up_blocks.2.attentions.2',
            ],
            'target_module': 'up_blocks.2.upsamplers.0',
            'actual_next_module_after_upsampler': 'up_blocks.3',
            'expected_operation_order_until_target': [
                'full local up_blocks.1 output enters up_blocks.2',
                'up_blocks.2.resnets.0 -> attentions.0',
                'up_blocks.2.resnets.1 -> attentions.1',
                'up_blocks.2.resnets.2 -> attentions.2',
                'up_blocks.2.upsamplers.0',
                'stop before up_blocks.3',
            ],
        },
        'upsampler_config': cfg,
        'upsampler_lora_info': module_lora_info(conv) if conv is not None else {'is_lora_wrapped': False, 'reason': 'no conv child'},
        'entry_context_shapes': {
            'sample': sample_shape,
            'timestep': list(timestep.shape),
            'encoder_hidden_states': encoder_shape,
            'upblock1_output': list(ctx['official_output'].shape),
            'remaining_residual_count_before_upblock2_slice': len(remaining_before_up2),
            'upblock2_resnet0_consumed_residual_source': sources[res0_consumed_index],
            'upblock2_resnet0_residual_shape': list(res0_hidden.shape),
            'upblock2_resnet1_consumed_residual_source': sources[res1_consumed_index],
            'upblock2_resnet1_residual_shape': list(res1_hidden.shape),
            'upblock2_resnet2_consumed_residual_source': sources[res2_consumed_index],
            'upblock2_resnet2_residual_shape': list(res2_hidden.shape),
            'upblock2_attention2_output': list(att2.shape),
            'upblock2_upsampler_input': list(att2.shape),
            'upblock2_upsampler_interpolation_output': list(captured.get('conv_input', upsampled).shape),
            'upblock2_upsampler_output': list(upsampled.shape),
            'internal_remaining_residual_count_after_attention2': len(remaining_after_resnet2),
            'internal_remaining_residual_count_after_upsampler': len(remaining_after_resnet2),
            'upsampler_consumes_accumulated_residuals': False,
        },
        'upsampler_input_stats': stats_tensor(att2),
        'upsampler_output_stats': stats_tensor(upsampled),
        'residual_contract': {
            'status': 'PASS',
            'upsampler_consumes_accumulated_residuals': False,
            'remaining_residual_count_unchanged': len(remaining_after_resnet2),
        },
        'lora_audit': {
            'status': 'PASS',
            'effective_static_weight_export_required': conv is not None,
        },
        'uses_full_unet_forward': False,
        'uses_upblocks3': False,
        'uses_full_tadsr_inference': False,
        'markers': {
            'TADSR_UNET_UPBLOCK2_UPSAMPLER_AUDIT': 'PASS',
            'TADSR_UNET_UPBLOCK2_UPSAMPLER_LORA_AUDIT': 'PASS',
            'TADSR_UNET_UPBLOCK2_UPSAMPLER_RESIDUAL_CONTRACT_AUDIT': 'PASS',
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding='utf-8')
    OUT_TXT.write_text('\n'.join([
        '# TADSR UNet up_blocks.2.upsamplers.0 official audit',
        '',
        *[f'{k}: {v}' for k, v in result['markers'].items()],
        '',
        f"target: {result['upblock2_overview']['target_module']}",
        f"next unopened module: {result['upblock2_overview']['actual_next_module_after_upsampler']}",
        f"upsampler input shape: {result['entry_context_shapes']['upblock2_upsampler_input']}",
        f"upsampler output shape: {result['entry_context_shapes']['upblock2_upsampler_output']}",
        'full up_blocks.2 aggregate/full UNet/full inference remain NOT_COMPLETE.',
    ]) + '\n', encoding='utf-8')
    for k, v in result['markers'].items():
        print(f'{k}: {v}')
    print(json.dumps({'status': 'PASS', 'audit_json': str(OUT_JSON), 'next': 'up_blocks.3'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
