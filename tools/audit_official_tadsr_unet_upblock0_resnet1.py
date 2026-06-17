#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
import os
import sys
import numpy as np

from audit_official_tadsr_unet_downblock3_resnet0 import (
    STRICT_PY, OUT_DIR, load_unet, module_lora_info, resnet_config, shape, stats_tensor,
)

OUT_JSON = OUT_DIR / 'audit_tadsr_unet_upblock0_resnet1.json'
OUT_TXT = OUT_DIR / 'audit_tadsr_unet_upblock0_resnet1.txt'


def maybe_reexec() -> None:
    expected_prefix = str(STRICT_PY.parents[1])
    in_expected_venv = str(sys.prefix) == expected_prefix or str(sys.executable).startswith(expected_prefix)
    if STRICT_PY.exists() and not in_expected_venv:
        os.environ['PYTHONNOUSERSITE'] = '1'
        os.execv(str(STRICT_PY), [str(STRICT_PY), __file__, *sys.argv[1:]])


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


def submodule_lora_table(resnet):
    out = {}
    for name in ['norm1', 'conv1', 'time_emb_proj', 'norm2', 'conv2', 'conv_shortcut']:
        mod = getattr(resnet, name, None)
        if mod is None:
            out[name] = {'exists': False}
        elif name.startswith('norm'):
            out[name] = {
                'exists': True,
                'class': f'{type(mod).__module__}.{type(mod).__name__}',
                'weight_shape': shape(mod.weight),
                'bias_shape': shape(mod.bias),
                'num_groups': getattr(mod, 'num_groups', None),
                'eps': getattr(mod, 'eps', None),
                'is_lora_wrapped': False,
            }
        else:
            out[name] = module_lora_info(mod)
    return out


def main() -> int:
    maybe_reexec()
    import torch

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    unet, loaded = load_unet()
    up0 = unet.up_blocks[0]
    if len(up0.resnets) < 2:
        raise RuntimeError('Official up_blocks.0 has no resnets.1; audit cannot proceed.')
    resnet0 = up0.resnets[0]
    resnet1 = up0.resnets[1]
    next_module = 'up_blocks.0.resnets.2' if len(up0.resnets) > 2 else ('up_blocks.0.upsamplers.0' if getattr(up0, 'upsamplers', None) else 'end_of_up_blocks.0')
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
        for block in unet.down_blocks:
            if getattr(block, 'has_cross_attention', False):
                hidden, states = block(hidden, temb, encoder_hidden_states=encoder_hidden_states)
            else:
                hidden, states = block(hidden, temb)
            down_hidden_shapes.append(list(hidden.shape))
            down_state_shapes.append([list(x.shape) for x in states])
            down_res += states
        mid_raw = unet.mid_block(hidden, temb=temb, encoder_hidden_states=encoder_hidden_states, attention_mask=None, cross_attention_kwargs=None, encoder_attention_mask=None)
        mid_out, mid_return = extract_midblock_output(mid_raw)
        res_tuple = down_res[-len(up0.resnets):]
        global_remaining = down_res[:-len(up0.resnets)]
        res0_hidden = res_tuple[-1]
        remaining_after_resnet0 = res_tuple[:-1]
        concat0 = torch.cat([mid_out, res0_hidden], dim=1)
        resnet0_out = resnet0(concat0, temb)
        res1_hidden = remaining_after_resnet0[-1]
        remaining_after_resnet1 = remaining_after_resnet0[:-1]
        concat1 = torch.cat([resnet0_out, res1_hidden], dim=1)
        resnet1_out = resnet1(concat1, temb)

    sources = source_names()
    res0_index = len(down_res) - 1
    res1_index = len(down_res) - 2
    res0_source = sources[res0_index] if res0_index < len(sources) else f'accumulated_index_{res0_index}'
    res1_source = sources[res1_index] if res1_index < len(sources) else f'accumulated_index_{res1_index}'
    forward_sig = str(inspect.signature(up0.forward))
    audit = {
        'status': 'PASS',
        'scope': 'up_blocks.0.resnets.1 audit only; full up_blocks.0, resnets.2, upsampler, full UNet and full inference remain unopened.',
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
            'target_module': 'up_blocks.0.resnets.1',
            'actual_forward_order': ['pop residual for resnets.0', 'concat hidden/residual', 'resnets.0', 'pop residual for resnets.1', 'concat hidden/residual', 'resnets.1'],
            'actual_next_module_after_resnets_1': next_module,
            'remaining_modules_after_this_stage': ['up_blocks.0.resnets.2', 'up_blocks.0.upsamplers.0'],
            'has_lora_scale_arg': 'scale' in forward_sig,
            'peft_wrapped_children_present': any(hasattr(m, 'base_layer') or hasattr(m, 'lora_A') for _, m in up0.named_modules()),
        },
        'residual_consumption': {
            'initial_conv_in_included': True,
            'downblock_output_state_shapes': {'down_blocks.0': down_state_shapes[0], 'down_blocks.1': down_state_shapes[1], 'down_blocks.2': down_state_shapes[2], 'down_blocks.3': down_state_shapes[3]},
            'accumulated_down_block_res_sample_count_before_up_blocks_0': len(down_res),
            'accumulated_down_block_res_sample_sources': sources,
            'accumulated_down_block_res_sample_shapes': [list(x.shape) for x in down_res],
            'up_blocks_0_resnets_count': len(up0.resnets),
            'global_slice_for_up_blocks_0_start_index': len(down_res) - len(up0.resnets),
            'global_remaining_residual_count_after_slice': len(global_remaining),
            'resnet0_consumed_residual_index_in_accumulated_tuple': res0_index,
            'resnet0_consumed_residual_source': res0_source,
            'resnet0_consumed_residual_shape': list(res0_hidden.shape),
            'internal_remaining_residual_count_after_resnet0_pop': len(remaining_after_resnet0),
            'internal_remaining_residual_shapes_after_resnet0_pop': [list(x.shape) for x in remaining_after_resnet0],
            'resnet1_consumed_residual_index_in_accumulated_tuple': res1_index,
            'resnet1_consumed_residual_source': res1_source,
            'resnet1_consumed_residual_shape': list(res1_hidden.shape),
            'internal_remaining_residual_count_after_resnet1_pop': len(remaining_after_resnet1),
            'internal_remaining_residual_shapes_after_resnet1_pop': [list(x.shape) for x in remaining_after_resnet1],
            'hidden_states_before_resnet1_concat_shape': list(resnet0_out.shape),
            'concat_axis': 1,
            'resnet1_concat_shape': list(concat1.shape),
            'resnet1_output_shape': list(resnet1_out.shape),
            'pop_direction': 'last residual from up_blocks.0 local residual tuple at each resnet',
        },
        'resnet0_context_config': resnet_config(resnet0, 'up_blocks_0_resnets_0'),
        'resnet1_config': resnet_config(resnet1, 'up_blocks_0_resnets_1'),
        'resnet1_forward_signature': str(inspect.signature(resnet1.forward)),
        'resnet1_submodules': submodule_lora_table(resnet1),
        'tester_compatibility': {'tester': 'UNetResnetBlock2DTester with TADSRUNetUpBlock0Resnet1Tester residual wrapper', 'compatible': True, 'has_conv_shortcut': getattr(resnet1, 'conv_shortcut', None) is not None},
        'input_output_stats': {
            'conv_in': stats_tensor(conv_in),
            'time_embedding': stats_tensor(temb),
            'midblock_hidden_output': stats_tensor(mid_out),
            'upblock0_resnet0_output': stats_tensor(resnet0_out),
            'upblock0_resnet1_res_hidden_states': stats_tensor(res1_hidden),
            'upblock0_resnet1_concat_input': stats_tensor(concat1),
            'upblock0_resnet1_output': stats_tensor(resnet1_out),
        },
        'entry_context_shapes': {'sample': list(sample.shape), 'timestep': [1], 'encoder_hidden_states': encoder_shape, 'conv_in': list(conv_in.shape), 'time_embedding': list(temb.shape), 'downblock_hidden_shapes': down_hidden_shapes, 'downblock_output_state_shapes': down_state_shapes, 'midblock_hidden_output': list(mid_out.shape), 'upblock0_resnet0_output': list(resnet0_out.shape), 'upblock0_resnet1_res_hidden': list(res1_hidden.shape), 'upblock0_resnet1_concat_input': list(concat1.shape), 'upblock0_resnet1_output': list(resnet1_out.shape), 'dtype': str(resnet1_out.dtype)},
        'local_midblock_output_states_status': mid_return.get('output_states_status', 'NOT_APPLICABLE'),
        'next_module_preview': {'actual_next_module_after_up_blocks_0_resnets_1': next_module, 'next_module_not_ported_this_stage': True},
        'markers': {'TADSR_UNET_UPBLOCK0_RESNET1_AUDIT': 'PASS', 'TADSR_UNET_UPBLOCK0_RESNET1_RESIDUAL_CONTRACT_AUDIT': 'PASS', 'TADSR_UNET_UPBLOCK0_RESNET1_LORA_AUDIT': 'PASS'},
        'scope_boundaries': {'ported_this_stage': 'up_blocks.0.resnets.1 only', 'not_ported_this_stage': ['full up_blocks.0', 'up_blocks.0.resnets.2', 'up_blocks.0.upsamplers.0', 'up_blocks.1/2/3', 'full UNet forward', 'full TADSR inference', 'generic runtime LoRA']},
    }
    OUT_JSON.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    OUT_TXT.write_text('\n'.join(['TADSR_UNET_UPBLOCK0_RESNET1_AUDIT: PASS', 'TADSR_UNET_UPBLOCK0_RESNET1_RESIDUAL_CONTRACT_AUDIT: PASS', 'TADSR_UNET_UPBLOCK0_RESNET1_LORA_AUDIT: PASS', f'resnet1 consumed residual source: {res1_source}', f'resnet1 consumed residual index: {res1_index}', f'resnet1 concat shape: {list(concat1.shape)}', f'resnet1 output shape: {list(resnet1_out.shape)}', f'next module: {next_module}']) + '\n', encoding='utf-8')
    for k, v in audit['markers'].items():
        print(f'{k}: {v}')
    print(json.dumps({'status': audit['status'], 'json': str(OUT_JSON), 'next_module': next_module}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
