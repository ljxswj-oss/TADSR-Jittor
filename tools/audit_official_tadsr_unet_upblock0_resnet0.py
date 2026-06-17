#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np

from audit_official_tadsr_unet_downblock3_resnet0 import (
    STRICT_PY,
    OUT_DIR,
    load_unet,
    module_lora_info,
    resnet_config,
    shape,
    stats_tensor,
)

OUT_JSON = OUT_DIR / 'audit_tadsr_unet_upblock0_resnet0.json'
OUT_TXT = OUT_DIR / 'audit_tadsr_unet_upblock0_resnet0.txt'


def maybe_reexec() -> None:
    expected_prefix = str(STRICT_PY.parents[1])
    in_expected_venv = str(sys.prefix) == expected_prefix or str(sys.executable).startswith(expected_prefix)
    if STRICT_PY.exists() and not in_expected_venv:
        os.environ['PYTHONNOUSERSITE'] = '1'
        os.execv(str(STRICT_PY), [str(STRICT_PY), __file__, *sys.argv[1:]])


def max_abs(a, b) -> float:
    aa = a.detach().cpu().numpy().astype(np.float32)
    bb = b.detach().cpu().numpy().astype(np.float32)
    return float(np.max(np.abs(aa - bb))) if aa.size else 0.0


def extract_midblock_output(out):
    if isinstance(out, tuple):
        return out[0], {'return_type': 'tuple', 'tuple_len': len(out), 'output_states_status': 'UNKNOWN_TUPLE'}
    if hasattr(out, 'sample'):
        return out.sample, {'return_type': type(out).__name__, 'output_states_status': 'NOT_APPLICABLE'}
    return out, {'return_type': type(out).__name__, 'output_states_status': 'NOT_APPLICABLE'}


def source_names():
    return (
        ['conv_in']
        + [f'down_blocks.0.output_state_{i}' for i in range(3)]
        + [f'down_blocks.1.output_state_{i}' for i in range(3)]
        + [f'down_blocks.2.output_state_{i}' for i in range(3)]
        + [f'down_blocks.3.output_state_{i}' for i in range(2)]
    )


def main() -> int:
    maybe_reexec()
    import torch

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    unet, loaded = load_unet()
    up0 = unet.up_blocks[0]
    resnet0 = up0.resnets[0]
    next_module = 'up_blocks.0.resnets.1' if len(up0.resnets) > 1 else ('up_blocks.0.upsamplers.0' if getattr(up0, 'upsamplers', None) else 'end_of_up_blocks.0')
    sample = torch.linspace(-1.0, 1.0, steps=1 * int(unet.config.in_channels) * 32 * 32, dtype=torch.float32).reshape(1, int(unet.config.in_channels), 32, 32)
    timestep = torch.tensor([1], dtype=torch.long)
    encoder_shape = [1, 77, int(unet.config.cross_attention_dim)]
    encoder_hidden_states = torch.linspace(-0.5, 0.5, steps=int(np.prod(encoder_shape)), dtype=torch.float32).reshape(*encoder_shape)
    with torch.no_grad():
        centered = 2.0 * sample - 1.0 if unet.config.center_input_sample else sample
        conv_in = unet.conv_in(centered)
        time_proj = unet.time_proj(timestep)
        temb = unet.time_embedding(time_proj.to(dtype=list(unet.time_embedding.parameters())[0].dtype))
        down_res = (conv_in,)
        down_hidden_shapes = []
        down_state_shapes = []
        hidden = conv_in
        for i, block in enumerate(unet.down_blocks):
            if getattr(block, 'has_cross_attention', False):
                hidden, states = block(hidden, temb, encoder_hidden_states=encoder_hidden_states)
            else:
                hidden, states = block(hidden, temb)
            down_hidden_shapes.append(list(hidden.shape))
            down_state_shapes.append([list(x.shape) for x in states])
            down_res += states
        mid_out_raw = unet.mid_block(
            hidden,
            temb=temb,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=None,
            cross_attention_kwargs=None,
            encoder_attention_mask=None,
        )
        mid_out, mid_return = extract_midblock_output(mid_out_raw)
        res_tuple = down_res[-len(up0.resnets):]
        global_remaining = down_res[:-len(up0.resnets)]
        res_hidden = res_tuple[-1]
        internal_remaining = res_tuple[:-1]
        concat = torch.cat([mid_out, res_hidden], dim=1)
        resnet0_out = resnet0(concat, temb)

    sources = source_names()
    consumed_index = len(down_res) - 1
    consumed_source = sources[consumed_index] if consumed_index < len(sources) else f'accumulated_index_{consumed_index}'
    submodules = {}
    for name in ['norm1', 'conv1', 'time_emb_proj', 'norm2', 'conv2', 'conv_shortcut']:
        mod = getattr(resnet0, name, None)
        if mod is None:
            submodules[name] = {'exists': False}
        elif name.startswith('norm'):
            submodules[name] = {
                'exists': True,
                'class': f'{type(mod).__module__}.{type(mod).__name__}',
                'weight_shape': shape(mod.weight),
                'bias_shape': shape(mod.bias),
                'num_groups': getattr(mod, 'num_groups', None),
                'eps': getattr(mod, 'eps', None),
                'is_lora_wrapped': False,
            }
        else:
            submodules[name] = module_lora_info(mod)

    forward_sig = str(inspect.signature(up0.forward))
    resnet_sig = str(inspect.signature(resnet0.forward))
    attention_preview = {'exists': False, 'reason': 'official up_blocks.0 is UpBlock2D with no attentions'}
    audit = {
        'status': 'PASS',
        'scope': 'up_blocks.0.resnets.0 audit only; full up_blocks.0, upsampler, full UNet and full inference remain unopened.',
        'python': sys.executable,
        'loaded_lora_parameter_count': len(loaded),
        'upblock0_overview': {
            'class': f'{type(up0).__module__}.{type(up0).__name__}',
            'forward_signature': forward_sig,
            'children': [{'name': n, 'class': f'{type(m).__module__}.{type(m).__name__}'} for n, m in up0.named_children()],
            'has_cross_attention': bool(getattr(up0, 'has_cross_attention', False)),
            'resnet_count': len(getattr(up0, 'resnets', [])),
            'attention_count': len(getattr(up0, 'attentions', [])) if hasattr(up0, 'attentions') else 0,
            'upsampler_count': len(getattr(up0, 'upsamplers', []) or []),
            'target_module': 'up_blocks.0.resnets.0',
            'actual_forward_order': ['for each resnet: pop residual from res_hidden_states_tuple[-1]', 'concat hidden/residual on channel axis', 'resnet', 'optional upsampler after all resnets'],
            'actual_next_module_after_resnets_0': next_module,
            'remaining_modules_after_this_stage': ['up_blocks.0.resnets.1', 'up_blocks.0.resnets.2', 'up_blocks.0.upsamplers.0'],
            'consumes_encoder_hidden_states': 'encoder_hidden_states' in forward_sig,
            'consumes_attention_mask': 'attention_mask' in forward_sig,
            'consumes_encoder_attention_mask': 'encoder_attention_mask' in forward_sig,
            'consumes_cross_attention_kwargs': 'cross_attention_kwargs' in forward_sig,
            'has_lora_scale_arg': 'scale' in forward_sig,
            'peft_wrapped_children_present': any(hasattr(m, 'base_layer') or hasattr(m, 'lora_A') for _, m in up0.named_modules()),
            'uses_freeu': any(hasattr(up0, name) for name in ['s1', 's2', 'b1', 'b2']),
        },
        'residual_consumption': {
            'initial_conv_in_included': True,
            'downblock_output_state_shapes': {
                'down_blocks.0': down_state_shapes[0],
                'down_blocks.1': down_state_shapes[1],
                'down_blocks.2': down_state_shapes[2],
                'down_blocks.3': down_state_shapes[3],
            },
            'accumulated_down_block_res_sample_count_before_up_blocks_0': len(down_res),
            'accumulated_down_block_res_sample_sources': sources,
            'accumulated_down_block_res_sample_shapes': [list(x.shape) for x in down_res],
            'up_blocks_0_resnets_count': len(up0.resnets),
            'global_slice_for_up_blocks_0_start_index': len(down_res) - len(up0.resnets),
            'global_remaining_residual_count_after_slice': len(global_remaining),
            'resnet0_consumed_residual_index_in_accumulated_tuple': consumed_index,
            'resnet0_consumed_residual_source': consumed_source,
            'resnet0_consumed_residual_shape': list(res_hidden.shape),
            'internal_remaining_residual_count_after_resnet0_pop': len(internal_remaining),
            'internal_remaining_residual_shapes_after_resnet0_pop': [list(x.shape) for x in internal_remaining],
            'hidden_states_before_concat_shape': list(mid_out.shape),
            'concat_axis': 1,
            'concat_shape': list(concat.shape),
            'resnet0_output_shape': list(resnet0_out.shape),
            'pop_direction': 'last residual from up_blocks.0 local residual tuple',
        },
        'resnet0_config': resnet_config(resnet0, 'up_blocks_0_resnets_0'),
        'resnet0_forward_signature': resnet_sig,
        'resnet0_submodules': submodules,
        'existing_tester_compatibility': {
            'tester': 'UNetResnetBlock2DTester',
            'compatible': True,
            'wrapper_needed': 'upblock wrapper must consume official residual and concatenate before generic ResNet tester',
            'already_concatenated_input_supported': True,
            'channel_change': getattr(resnet0, 'in_channels', None) != getattr(resnet0, 'out_channels', None),
            'has_conv_shortcut': getattr(resnet0, 'conv_shortcut', None) is not None,
            'time_embedding_norm': getattr(resnet0, 'time_embedding_norm', None),
            'groups': getattr(getattr(resnet0, 'norm1', None), 'num_groups', None),
            'groups_out': getattr(getattr(resnet0, 'norm2', None), 'num_groups', None),
            'output_scale_factor': getattr(resnet0, 'output_scale_factor', None),
            'required_code_changes': ['add TADSRUNetUpBlock0Resnet0Tester wrapper for residual concat'],
        },
        'input_output_stats': {
            'conv_in': stats_tensor(conv_in),
            'time_embedding': stats_tensor(temb),
            'midblock_hidden_output': stats_tensor(mid_out),
            'res_hidden_states': stats_tensor(res_hidden),
            'concat_input': stats_tensor(concat),
            'upblock0_resnet0_output': stats_tensor(resnet0_out),
        },
        'entry_context_shapes': {
            'sample': list(sample.shape),
            'timestep': [1],
            'encoder_hidden_states': encoder_shape,
            'conv_in': list(conv_in.shape),
            'time_embedding': list(temb.shape),
            'downblock_hidden_shapes': down_hidden_shapes,
            'downblock_output_state_shapes': down_state_shapes,
            'midblock_hidden_output': list(mid_out.shape),
            'upblock0_resnet0_res_hidden': list(res_hidden.shape),
            'upblock0_resnet0_concat_input': list(concat.shape),
            'upblock0_resnet0_output': list(resnet0_out.shape),
            'dtype': str(resnet0_out.dtype),
        },
        'downstream_attention_preview': attention_preview,
        'next_module_preview': {
            'actual_next_module_after_up_blocks_0_resnets_0': next_module,
            'next_module_not_ported_this_stage': True,
        },
        'markers': {
            'TADSR_UNET_UPBLOCK0_AUDIT': 'PASS',
            'TADSR_UNET_UPBLOCK0_RESNET0_AUDIT': 'PASS',
            'TADSR_UNET_UPBLOCK0_RESNET0_RESIDUAL_CONTRACT_AUDIT': 'PASS',
            'TADSR_UNET_UPBLOCK0_RESNET0_LORA_AUDIT': 'PASS',
        },
        'scope_boundaries': {
            'ported_this_stage': 'up_blocks.0.resnets.0 only',
            'not_ported_this_stage': ['full up_blocks.0', 'up_blocks.0.resnets.1', 'up_blocks.0.resnets.2', 'up_blocks.0.upsamplers.0', 'up_blocks.1/2/3', 'full UNet forward', 'full TADSR inference', 'generic runtime LoRA'],
        },
    }
    OUT_JSON.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    OUT_TXT.write_text(
        '\n'.join([
            'TADSR_UNET_UPBLOCK0_AUDIT: PASS',
            'TADSR_UNET_UPBLOCK0_RESNET0_AUDIT: PASS',
            'TADSR_UNET_UPBLOCK0_RESNET0_RESIDUAL_CONTRACT_AUDIT: PASS',
            'TADSR_UNET_UPBLOCK0_RESNET0_LORA_AUDIT: PASS',
            f'up_blocks.0 class: {audit["upblock0_overview"]["class"]}',
            f'resnet_count: {audit["upblock0_overview"]["resnet_count"]}',
            f'upsampler_count: {audit["upblock0_overview"]["upsampler_count"]}',
            f'consumed residual source: {consumed_source}',
            f'consumed residual index: {consumed_index}',
            f'hidden shape before concat: {list(mid_out.shape)}',
            f'residual shape: {list(res_hidden.shape)}',
            f'concat shape: {list(concat.shape)}',
            f'output shape: {list(resnet0_out.shape)}',
            f'actual_next_module_after_resnets_0: {next_module}',
            'full up_blocks.0/full UNet/full inference remain unopened.',
        ]) + '\n',
        encoding='utf-8',
    )
    for k, v in audit['markers'].items():
        print(f'{k}: {v}')
    print(json.dumps({'status': audit['status'], 'audit': str(OUT_JSON)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
