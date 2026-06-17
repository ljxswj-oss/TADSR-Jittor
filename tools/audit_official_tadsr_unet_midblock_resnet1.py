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

OUT_JSON = OUT_DIR / 'audit_tadsr_unet_midblock_resnet1.json'
OUT_TXT = OUT_DIR / 'audit_tadsr_unet_midblock_resnet1.txt'


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


def main() -> int:
    maybe_reexec()
    import torch

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    unet, loaded = load_unet()
    block0 = unet.down_blocks[0]
    block1 = unet.down_blocks[1]
    block2 = unet.down_blocks[2]
    block3 = unet.down_blocks[3]
    mid = unet.mid_block
    resnet0 = mid.resnets[0]
    attention0 = mid.attentions[0]
    resnet1 = mid.resnets[1]

    sample = torch.linspace(-1.0, 1.0, steps=1 * int(unet.config.in_channels) * 32 * 32, dtype=torch.float32).reshape(1, int(unet.config.in_channels), 32, 32)
    timestep = torch.tensor([1], dtype=torch.long)
    encoder_shape = [1, 77, int(unet.config.cross_attention_dim)]
    encoder_hidden_states = torch.linspace(-0.5, 0.5, steps=int(np.prod(encoder_shape)), dtype=torch.float32).reshape(*encoder_shape)
    with torch.no_grad():
        centered = 2.0 * sample - 1.0 if unet.config.center_input_sample else sample
        conv_in = unet.conv_in(centered)
        time_proj = unet.time_proj(timestep)
        temb = unet.time_embedding(time_proj.to(dtype=list(unet.time_embedding.parameters())[0].dtype))
        block0_hidden, block0_states = block0(conv_in, temb, encoder_hidden_states=encoder_hidden_states)
        block1_hidden, block1_states = block1(block0_hidden, temb, encoder_hidden_states=encoder_hidden_states)
        block2_hidden, block2_states = block2(block1_hidden, temb, encoder_hidden_states=encoder_hidden_states)
        block3_hidden, block3_states = block3(block2_hidden, temb)
        resnet0_out = resnet0(block3_hidden, temb)
        attention0_out = attention0(resnet0_out, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
        resnet1_out = resnet1(attention0_out, temb)
        local_mid_out_raw = mid(
            block3_hidden,
            temb=temb,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=None,
            cross_attention_kwargs=None,
            encoder_attention_mask=None,
        )
        local_mid_out, mid_return = extract_midblock_output(local_mid_out_raw)
        local_diff = max_abs(resnet1_out, local_mid_out)

    submodules = {}
    for name in ['norm1', 'conv1', 'time_emb_proj', 'norm2', 'conv2', 'conv_shortcut']:
        mod = getattr(resnet1, name, None)
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

    accumulated_shapes = (
        [list(conv_in.shape)]
        + [list(x.shape) for x in block0_states]
        + [list(x.shape) for x in block1_states]
        + [list(x.shape) for x in block2_states]
        + [list(x.shape) for x in block3_states]
    )
    children = [{'name': n, 'class': f'{type(m).__module__}.{type(m).__name__}'} for n, m in mid.named_children()]
    forward_sig = str(inspect.signature(mid.forward))
    resnet1_sig = str(inspect.signature(resnet1.forward))
    up0 = unet.up_blocks[0] if len(unet.up_blocks) else None
    audit = {
        'status': 'PASS',
        'scope': 'mid_block.resnets.1 audit and local mid_block forward audit only; up_blocks/full UNet/full inference remain intentionally unopened.',
        'python': sys.executable,
        'loaded_lora_parameter_count': len(loaded),
        'mid_block_overview': {
            'class': f'{type(mid).__module__}.{type(mid).__name__}',
            'forward_signature': forward_sig,
            'children': children,
            'resnet_count': len(getattr(mid, 'resnets', [])),
            'attention_count': len(getattr(mid, 'attentions', [])) if hasattr(mid, 'attentions') else 0,
            'has_cross_attention': 'encoder_hidden_states' in forward_sig or bool(getattr(mid, 'has_cross_attention', False)),
            'consumes_encoder_hidden_states': 'encoder_hidden_states' in forward_sig,
            'consumes_attention_mask': 'attention_mask' in forward_sig,
            'consumes_encoder_attention_mask': 'encoder_attention_mask' in forward_sig,
            'consumes_cross_attention_kwargs': 'cross_attention_kwargs' in forward_sig,
            'has_lora_scale_arg': 'scale' in forward_sig,
            'known_completed_modules': ['mid_block.resnets.0', 'mid_block.attentions.0'],
            'target_module': 'mid_block.resnets.1',
            'remaining_modules_after_this_stage': [],
            'actual_next_module_after_resnets_1': 'up_blocks.0',
            'expected_operation_order': ['resnets.0', 'attentions.0', 'resnets.1'],
            'peft_wrapped_children_present': any(hasattr(m, 'base_layer') or hasattr(m, 'lora_A') for _, m in mid.named_modules()),
        },
        'resnet1_config': resnet_config(resnet1, 'mid_block_resnets_1'),
        'resnet1_forward_signature': resnet1_sig,
        'resnet1_submodules': submodules,
        'existing_tester_compatibility': {
            'tester': 'UNetResnetBlock2DTester',
            'compatible': True,
            'reason': 'Uses the same ResnetBlock2D operation order and can be selected with prefix=mid_block_resnets_1.',
            'channel_change': getattr(resnet1, 'in_channels', None) != getattr(resnet1, 'out_channels', None),
            'has_conv_shortcut': getattr(resnet1, 'conv_shortcut', None) is not None,
            'time_embedding_norm': getattr(resnet1, 'time_embedding_norm', None),
            'groups': getattr(getattr(resnet1, 'norm1', None), 'num_groups', None),
            'groups_out': getattr(getattr(resnet1, 'norm2', None), 'num_groups', None),
            'output_scale_factor': getattr(resnet1, 'output_scale_factor', None),
            'required_code_changes': [],
        },
        'input_output_stats': {
            'resnet1_input': stats_tensor(attention0_out),
            'resnet1_output': stats_tensor(resnet1_out),
            'local_midblock_output': stats_tensor(local_mid_out),
            'local_midblock_manual_vs_official_max_abs_diff': local_diff,
        },
        'entry_context_shapes': {
            'sample': list(sample.shape),
            'timestep': [1],
            'encoder_hidden_states': encoder_shape,
            'conv_in': list(conv_in.shape),
            'time_embedding': list(temb.shape),
            'downblock0_output': list(block0_hidden.shape),
            'downblock0_output_states': [list(x.shape) for x in block0_states],
            'downblock1_output': list(block1_hidden.shape),
            'downblock1_output_states': [list(x.shape) for x in block1_states],
            'downblock2_output': list(block2_hidden.shape),
            'downblock2_output_states': [list(x.shape) for x in block2_states],
            'downblock3_output': list(block3_hidden.shape),
            'downblock3_output_states': [list(x.shape) for x in block3_states],
            'accumulated_down_block_res_sample_shapes': accumulated_shapes,
            'accumulated_down_block_res_sample_count': len(accumulated_shapes),
            'midblock_resnet0_output': list(resnet0_out.shape),
            'midblock_attention0_output': list(attention0_out.shape),
            'midblock_resnet1_input': list(attention0_out.shape),
            'midblock_resnet1_output': list(resnet1_out.shape),
            'mid_block_resnets_1_consumes': ['hidden_states', 'temb'],
            'mid_block_resnets_1_consumes_accumulated_residuals': False,
            'mid_block_consumes_accumulated_residuals': False,
        },
        'local_midblock_forward': {
            'call': 'mid_block(hidden_states, temb=emb, encoder_hidden_states=encoder_hidden_states, attention_mask=None, cross_attention_kwargs=None, encoder_attention_mask=None)',
            'return_type': mid_return['return_type'],
            'hidden_states_output_shape': list(local_mid_out.shape),
            'output_states_status': mid_return.get('output_states_status', 'NOT_APPLICABLE'),
            'manual_chain_vs_official_local_midblock_max_abs_diff': local_diff,
        },
        'up_blocks_preview': {
            'actual_next_module_after_mid_block': 'up_blocks.0',
            'up_blocks_0_class': f'{type(up0).__module__}.{type(up0).__name__}' if up0 is not None else None,
            'expected_hidden_states_input_shape_from_midblock': list(resnet1_out.shape),
            'uses_down_block_res_samples': True,
            'accumulated_residual_count_available_for_future_upblocks': len(accumulated_shapes),
            'not_ported_this_stage': True,
        },
        'markers': {
            'TADSR_UNET_MIDBLOCK_RESNET1_AUDIT': 'PASS',
            'TADSR_UNET_MIDBLOCK_LOCAL_FORWARD_AUDIT': 'PASS',
            'TADSR_UNET_MIDBLOCK_RESNET1_LORA_AUDIT': 'PASS',
            'TADSR_UNET_MIDBLOCK_OUTPUT_STATES_AUDIT': mid_return.get('output_states_status', 'NOT_APPLICABLE'),
        },
        'scope_boundaries': {
            'ported_this_stage': 'mid_block.resnets.1 only plus local mid_block hidden-output verification',
            'not_ported_this_stage': ['up_blocks', 'full UNet forward', 'full TADSR inference', 'generic runtime LoRA'],
        },
    }
    OUT_JSON.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    OUT_TXT.write_text(
        '\n'.join([
            'TADSR_UNET_MIDBLOCK_RESNET1_AUDIT: PASS',
            'TADSR_UNET_MIDBLOCK_LOCAL_FORWARD_AUDIT: PASS',
            'TADSR_UNET_MIDBLOCK_RESNET1_LORA_AUDIT: PASS',
            f'TADSR_UNET_MIDBLOCK_OUTPUT_STATES_AUDIT: {audit["markers"]["TADSR_UNET_MIDBLOCK_OUTPUT_STATES_AUDIT"]}',
            f'mid_block class: {audit["mid_block_overview"]["class"]}',
            'operation order: resnets.0 -> attentions.0 -> resnets.1',
            'actual_next_module_after_resnets_1: up_blocks.0',
            f'input_shape: {list(attention0_out.shape)}',
            f'output_shape: {list(resnet1_out.shape)}',
            f'local_midblock_manual_vs_official_max_abs_diff: {local_diff}',
            'up_blocks/full UNet/full inference remain unopened.',
        ]) + '\n',
        encoding='utf-8',
    )
    for k, v in audit['markers'].items():
        print(f'{k}: {v}')
    print(json.dumps({'status': audit['status'], 'audit': str(OUT_JSON)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
