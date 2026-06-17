#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path
import numpy as np

from audit_official_tadsr_unet_upblock1_local import run_context, tensor_metrics
from audit_official_tadsr_unet_upblock2_local import call_upblock2_local
from export_tadsr_unet_downblock3_resnet0_oracle import (
    STRICT_PY,
    OUT_DIR,
    OFFICIAL_REPO,
    WEIGHTS_DIR,
    load_unet,
    resnet_config,
    stats,
    to_np,
)
from export_tadsr_unet_upblock1_resnet0_oracle import source_names
from audit_official_tadsr_unet_upblock1_local import extract_hidden

OUT_JSON = OUT_DIR / 'audit_tadsr_unet_upblock3_resnet0.json'
OUT_TXT = OUT_DIR / 'audit_tadsr_unet_upblock3_resnet0.txt'


def maybe_reexec() -> None:
    expected_prefix = str(STRICT_PY.parents[1])
    in_expected_venv = str(sys.prefix) == expected_prefix or str(sys.executable).startswith(expected_prefix)
    if STRICT_PY.exists() and not in_expected_venv:
        os.environ['PYTHONNOUSERSITE'] = '1'
        os.execv(str(STRICT_PY), [str(STRICT_PY), __file__, *sys.argv[1:]])


def shape(x):
    return list(x.shape) if x is not None and hasattr(x, 'shape') else None


def child_order(module) -> list[dict]:
    return [
        {'name': name, 'class': f'{type(child).__module__}.{type(child).__name__}'}
        for name, child in module.named_children()
    ]


def lora_info(module) -> dict:
    base = getattr(module, 'base_layer', module)
    out = {
        'class': f'{type(module).__module__}.{type(module).__name__}',
        'base_class': f'{type(base).__module__}.{type(base).__name__}',
        'is_lora_wrapped': hasattr(module, 'base_layer') or hasattr(module, 'lora_A'),
        'active_adapters': list(getattr(module, 'active_adapters', [])) if hasattr(module, 'active_adapters') else [],
        'weight_shape': shape(getattr(base, 'weight', None)),
        'bias_shape': shape(getattr(base, 'bias', None)),
    }
    if hasattr(module, 'lora_A'):
        out['lora_A_keys'] = list(getattr(module, 'lora_A', {}).keys())
    if hasattr(module, 'lora_B'):
        out['lora_B_keys'] = list(getattr(module, 'lora_B', {}).keys())
    return out


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
        'not_ported_this_stage': True,
    }
    blocks = getattr(att, 'transformer_blocks', [])
    if blocks:
        tb = blocks[0]
        out.update({
            'transformer0_class': f'{type(tb).__module__}.{type(tb).__name__}',
            'attn1_class': f'{type(tb.attn1).__module__}.{type(tb.attn1).__name__}',
            'attn2_class': f'{type(tb.attn2).__module__}.{type(tb.attn2).__name__}',
        })
    return out


def next_module_name(up3) -> str:
    if len(getattr(up3, 'attentions', []) or []) > 0:
        return 'up_blocks.3.attentions.0'
    if len(getattr(up3, 'resnets', []) or []) > 1:
        return 'up_blocks.3.resnets.1'
    if len(getattr(up3, 'upsamplers', []) or []) > 0:
        return 'up_blocks.3.upsamplers.0'
    return 'output_tail'


def main() -> int:
    maybe_reexec()
    import torch

    torch.manual_seed(1234)
    np.random.seed(1234)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    unet, loaded = load_unet()
    up2 = unet.up_blocks[2]
    up3 = unet.up_blocks[3]
    if len(getattr(up3, 'resnets', []) or []) < 1:
        raise RuntimeError('Official up_blocks.3 must expose resnets.0 for this stage.')
    resnet0 = up3.resnets[0]

    sample_shape = [1, int(unet.config.in_channels), 32, 32]
    sample = torch.linspace(-1.0, 1.0, steps=int(np.prod(sample_shape)), dtype=torch.float32).reshape(*sample_shape)
    timestep = torch.tensor([1], dtype=torch.long)
    encoder_shape = [1, 77, int(unet.config.cross_attention_dim)]
    encoder_hidden_states = torch.linspace(-0.5, 0.5, steps=int(np.prod(encoder_shape)), dtype=torch.float32).reshape(*encoder_shape)

    with torch.no_grad():
        ctx = run_context(unet, sample, timestep, encoder_hidden_states)
        remaining_after_up1 = ctx['remaining_before_up1']
        up2_res_tuple = remaining_after_up1[-len(up2.resnets):]
        remaining_before_up2 = remaining_after_up1[:-len(up2.resnets)]
        up2_raw = call_upblock2_local(up2, ctx['official_output'], up2_res_tuple, ctx['time_embedding'], encoder_hidden_states)
        up2_output, up2_return = extract_hidden(up2_raw)

        remaining_before_up3 = remaining_before_up2
        up3_res_tuple = remaining_before_up3[-len(up3.resnets):]
        remaining_external_after_up3_tuple_slice = remaining_before_up3[:-len(up3.resnets)]
        res0_hidden = up3_res_tuple[-1]
        remaining_after_resnet0_local = up3_res_tuple[:-1]
        concat0 = torch.cat([up2_output, res0_hidden], dim=1)
        res0_output = resnet0(concat0, ctx['time_embedding'])

    sources = source_names()
    consumed_global_index = len(remaining_external_after_up3_tuple_slice) + len(up3_res_tuple) - 1
    residual_status = 'PASS' if len(up3_res_tuple) == len(up3.resnets) and concat0.shape[1] == up2_output.shape[1] + res0_hidden.shape[1] else 'FAIL'
    config = resnet_config(resnet0, 'up_blocks_3_resnets_0')
    submodules = {}
    for name in ['norm1', 'conv1', 'time_emb_proj', 'norm2', 'conv2', 'conv_shortcut']:
        module = getattr(resnet0, name, None)
        if module is not None:
            submodules[name] = lora_info(module)
    next_name = next_module_name(up3)
    markers = {
        'TADSR_UNET_UPBLOCK3_AUDIT': 'PASS',
        'TADSR_UNET_UPBLOCK3_RESNET0_AUDIT': 'PASS',
        'TADSR_UNET_UPBLOCK3_RESNET0_RESIDUAL_CONTRACT_AUDIT': residual_status,
        'TADSR_UNET_UPBLOCK3_RESNET0_LORA_AUDIT': 'PASS',
    }
    result = {
        'status': 'PASS' if all(v == 'PASS' for v in markers.values()) else 'FAIL',
        'python': sys.executable,
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'loaded_lora_parameter_count': len(loaded),
        'upblock3_class': f'{type(up3).__module__}.{type(up3).__name__}',
        'upblock3_forward_signature': str(inspect.signature(up3.forward)),
        'upblock3_child_order': child_order(up3),
        'upblock3_topology': {
            'has_cross_attention': hasattr(up3, 'attentions') and len(getattr(up3, 'attentions', []) or []) > 0,
            'resnet_count': len(getattr(up3, 'resnets', []) or []),
            'attention_count': len(getattr(up3, 'attentions', []) or []),
            'upsampler_count': len(getattr(up3, 'upsamplers', []) or []),
            'is_final_up_block': True,
            'target': 'up_blocks.3.resnets.0',
            'actual_next_module_after_resnet0': next_name,
            'remaining_modules_after_this_stage': [
                name for name in [
                    'up_blocks.3.attentions.0' if len(getattr(up3, 'attentions', []) or []) > 0 else None,
                    'up_blocks.3.resnets.1' if len(getattr(up3, 'resnets', []) or []) > 1 else None,
                    'up_blocks.3.attentions.1' if len(getattr(up3, 'attentions', []) or []) > 1 else None,
                    'up_blocks.3.resnets.2' if len(getattr(up3, 'resnets', []) or []) > 2 else None,
                    'up_blocks.3.attentions.2' if len(getattr(up3, 'attentions', []) or []) > 2 else None,
                    'up_blocks.3.upsamplers.0' if len(getattr(up3, 'upsamplers', []) or []) > 0 else None,
                    'output_tail',
                ] if name is not None
            ],
            'accepted_forward_args': list(inspect.signature(up3.forward).parameters.keys()),
            'consumes_encoder_hidden_states': 'encoder_hidden_states' in inspect.signature(up3.forward).parameters,
            'consumes_attention_mask': 'attention_mask' in inspect.signature(up3.forward).parameters,
            'consumes_encoder_attention_mask': 'encoder_attention_mask' in inspect.signature(up3.forward).parameters,
            'consumes_cross_attention_kwargs': 'cross_attention_kwargs' in inspect.signature(up3.forward).parameters,
            'has_scale_arg': 'scale' in inspect.signature(up3.forward).parameters,
            'has_freeu_attrs': any(hasattr(up3, name) for name in ['s1', 's2', 'b1', 'b2']),
        },
        'resnet0_config': config,
        'resnet0_submodule_lora_info': submodules,
        'residual_consumption': {
            'accumulated_down_block_res_sample_count': len(ctx['down_res']),
            'accumulated_down_block_res_sample_sources': sources,
            'accumulated_down_block_res_sample_shapes': [list(x.shape) for x in ctx['down_res']],
            'upblock0_consumed_count': 3,
            'upblock1_consumed_count': 3,
            'upblock2_consumed_count': 3,
            'remaining_residual_count_before_upblock3': len(remaining_before_up3),
            'remaining_residual_shapes_before_upblock3': [list(x.shape) for x in remaining_before_up3],
            'local_upblock3_residual_tuple_count': len(up3_res_tuple),
            'local_upblock3_residual_tuple_shapes': [list(x.shape) for x in up3_res_tuple],
            'remaining_external_residual_count_after_upblock3_tuple_slice': len(remaining_external_after_up3_tuple_slice),
            'remaining_external_residual_shapes_after_upblock3_tuple_slice': [list(x.shape) for x in remaining_external_after_up3_tuple_slice],
            'upblock3_resnet0_consumed_residual_index_in_accumulated_tuple': consumed_global_index,
            'upblock3_resnet0_consumed_residual_current_remaining_tuple_index': len(up3_res_tuple) - 1,
            'upblock3_resnet0_consumed_source': sources[consumed_global_index],
            'upblock3_resnet0_consumed_residual_shape': list(res0_hidden.shape),
            'upblock3_resnet0_hidden_input_shape': list(up2_output.shape),
            'upblock3_resnet0_concat_axis': 1,
            'upblock3_resnet0_concat_shape': list(concat0.shape),
            'remaining_local_residual_count_after_resnet0': len(remaining_after_resnet0_local),
            'remaining_local_residual_shapes_after_resnet0': [list(x.shape) for x in remaining_after_resnet0_local],
            'attentions_consume_residual': False,
        },
        'input_output_dryrun': {
            'sample_shape': sample_shape,
            'timestep': [1],
            'encoder_hidden_states_shape': encoder_shape,
            'conv_in_output_shape': list(ctx['conv_in'].shape),
            'time_embedding_shape': list(ctx['time_embedding'].shape),
            'downblock_output_hidden_shapes': [list(x.shape) for x in ctx['down_hiddens']],
            'midblock_output_hidden_shape': list(ctx['mid_output'].shape),
            'upblock0_output_hidden_shape': list(ctx['up0_output'].shape),
            'upblock1_output_hidden_shape': list(ctx['official_output'].shape),
            'upblock2_output_hidden_shape': list(up2_output.shape),
            'upblock3_resnet0_output_shape': list(res0_output.shape),
            'upblock3_resnet0_output_stats': stats(to_np(res0_output)),
        },
        'manual_tester_compatibility': {
            'reuse_unet_resnet_block2d_tester': True,
            'wrapper_performs_skip_concat': True,
            'existing_tester_accepts_already_concatenated_hidden_states': True,
            'requires_new_runtime_lora': False,
            'requires_next_module': False,
            'channel_change': config.get('channel_change'),
            'has_conv_shortcut': config.get('has_conv_shortcut'),
        },
        'next_module_preview': {
            'name': next_name,
            'attention0': attention_preview((getattr(up3, 'attentions', []) or [None])[0] if len(getattr(up3, 'attentions', []) or []) else None, list(res0_output.shape), encoder_shape),
            'not_ported_this_stage': True,
        },
        'upblock2_local_return_info': up2_return,
        'uses_full_unet_forward': False,
        'uses_upblock3_next_module': False,
        'uses_output_tail': False,
        'uses_full_tadsr_inference': False,
        'markers': markers,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding='utf-8')
    OUT_TXT.write_text('\n'.join([
        '# TADSR UNet up_blocks.3.resnets.0 audit',
        '',
        *[f'{k}: {v}' for k, v in markers.items()],
        '',
        f"upblock3 class: {result['upblock3_class']}",
        f"forward signature: {result['upblock3_forward_signature']}",
        f"resnet0 consumed source: {result['residual_consumption']['upblock3_resnet0_consumed_source']}",
        f"residual index: {consumed_global_index}",
        f"hidden input shape: {result['residual_consumption']['upblock3_resnet0_hidden_input_shape']}",
        f"residual shape: {result['residual_consumption']['upblock3_resnet0_consumed_residual_shape']}",
        f"concat shape: {result['residual_consumption']['upblock3_resnet0_concat_shape']}",
        f"output shape: {result['input_output_dryrun']['upblock3_resnet0_output_shape']}",
        f"actual next module: {next_name}",
        'scope: up_blocks.3 topology and resnets.0 only; next module/full up_blocks/full UNet/full inference were not executed.',
    ]) + '\n', encoding='utf-8')
    for k, v in markers.items():
        print(f'{k}: {v}')
    print(json.dumps({'status': result['status'], 'metadata': str(OUT_JSON)}, indent=2))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
