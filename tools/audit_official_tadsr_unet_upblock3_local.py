#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path
import numpy as np

from audit_official_tadsr_unet_upblock1_local import run_context, tensor_metrics, extract_hidden
from audit_official_tadsr_unet_upblock2_local import call_upblock2_local
from export_tadsr_unet_downblock3_resnet0_oracle import (
    STRICT_PY,
    OUT_DIR,
    OFFICIAL_REPO,
    WEIGHTS_DIR,
    load_unet,
    to_np,
    stats,
)
from export_tadsr_unet_upblock1_resnet0_oracle import source_names

OUT_JSON = OUT_DIR / 'audit_tadsr_unet_upblock3_local.json'
OUT_TXT = OUT_DIR / 'audit_tadsr_unet_upblock3_local.txt'


def maybe_reexec() -> None:
    expected_prefix = str(STRICT_PY.parents[1])
    in_expected_venv = str(sys.prefix) == expected_prefix or str(sys.executable).startswith(expected_prefix)
    if STRICT_PY.exists() and not in_expected_venv:
        os.environ['PYTHONNOUSERSITE'] = '1'
        os.execv(str(STRICT_PY), [str(STRICT_PY), __file__, *sys.argv[1:]])


def call_upblock3_local(up3, hidden_states, res_tuple, temb, encoder_hidden_states):
    sig = inspect.signature(up3.forward)
    params = sig.parameters
    kwargs = {}
    if 'temb' in params:
        kwargs['temb'] = temb
    if 'encoder_hidden_states' in params:
        kwargs['encoder_hidden_states'] = encoder_hidden_states
    if 'cross_attention_kwargs' in params:
        kwargs['cross_attention_kwargs'] = None
    if 'upsample_size' in params:
        kwargs['upsample_size'] = None
    if 'output_size' in params:
        kwargs['output_size'] = None
    if 'attention_mask' in params:
        kwargs['attention_mask'] = None
    if 'encoder_attention_mask' in params:
        kwargs['encoder_attention_mask'] = None
    if 'scale' in params:
        kwargs['scale'] = 1.0
    return up3(hidden_states, res_tuple, **kwargs)


def manual_upblock3_chain(up3, hidden_states, up3_res_tuple, temb, encoder_hidden_states):
    import torch

    res0_hidden = up3_res_tuple[-1]
    remaining_after_resnet0 = up3_res_tuple[:-1]
    res0 = up3.resnets[0](torch.cat([hidden_states, res0_hidden], dim=1), temb)
    att0 = up3.attentions[0](res0, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]

    res1_hidden = remaining_after_resnet0[-1]
    remaining_after_resnet1 = remaining_after_resnet0[:-1]
    res1 = up3.resnets[1](torch.cat([att0, res1_hidden], dim=1), temb)
    att1 = up3.attentions[1](res1, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]

    res2_hidden = remaining_after_resnet1[-1]
    remaining_after_resnet2 = remaining_after_resnet1[:-1]
    res2 = up3.resnets[2](torch.cat([att1, res2_hidden], dim=1), temb)
    att2 = up3.attentions[2](res2, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]

    return {
        'resnet0': res0,
        'attention0': att0,
        'resnet1': res1,
        'attention1': att1,
        'resnet2': res2,
        'attention2': att2,
        'remaining_after_manual_upblock3': remaining_after_resnet2,
        'resnet0_residual': res0_hidden,
        'resnet1_residual': res1_hidden,
        'resnet2_residual': res2_hidden,
    }


def tail_boundary_preview(unet, upblock3_output):
    tail = []
    for name in ['conv_norm_out', 'conv_act', 'conv_out']:
        mod = getattr(unet, name, None)
        if mod is None:
            tail.append({'name': name, 'present': False})
            continue
        item = {
            'name': name,
            'present': True,
            'class': f'{type(mod).__module__}.{type(mod).__name__}',
            'has_lora_layer_attr': hasattr(mod, 'lora_layer'),
            'has_lora_linear_layer_attr': hasattr(mod, 'lora_linear_layer'),
        }
        for attr in ['num_groups', 'eps', 'in_channels', 'out_channels', 'kernel_size', 'stride', 'padding']:
            if hasattr(mod, attr):
                value = getattr(mod, attr)
                if isinstance(value, tuple):
                    value = list(value)
                elif not isinstance(value, (str, int, float, bool, type(None))):
                    value = repr(value)
                item[attr] = value
        tail.append(item)
    return {
        'status': 'PASS',
        'actual_next_logical_stage_after_upblock3': 'output_tail',
        'tail_module_names': [x['name'] for x in tail if x.get('present')],
        'tail_modules_preview': tail,
        'tail_input_shape': list(upblock3_output.shape),
        'tail_execution_performed': False,
        'recommended_next_stage': 'TADSR UNet output tail audit/export/port after full local up_blocks.3 aggregate alignment',
    }


def main() -> int:
    maybe_reexec()
    import torch

    torch.manual_seed(1234)
    np.random.seed(1234)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    unet, loaded = load_unet()
    up2 = unet.up_blocks[2]
    up3 = unet.up_blocks[3]

    if len(up3.resnets) != 3 or len(getattr(up3, 'attentions', []) or []) != 3:
        raise RuntimeError('This local audit expects up_blocks.3 to have 3 resnets, 3 attentions, and no required upsampler.')
    if len(getattr(up3, 'upsamplers', []) or []) != 0:
        raise RuntimeError('This local audit expects final up_blocks.3 to have no upsampler.')

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
        official_raw = call_upblock3_local(up3, up2_output, up3_res_tuple, ctx['time_embedding'], encoder_hidden_states)
        official_output, official_return = extract_hidden(official_raw)
        manual = manual_upblock3_chain(up3, up2_output, up3_res_tuple, ctx['time_embedding'], encoder_hidden_states)

    local_diff = tensor_metrics(manual['attention2'], official_output)
    status = 'PASS' if local_diff['shape_match'] and local_diff['max_abs_error'] <= 1e-5 else 'FAIL'
    residual_status = 'PASS' if len(manual['remaining_after_manual_upblock3']) == 0 and len(remaining_external_after_up3_tuple_slice) == 0 else 'FAIL'
    output_tail_boundary = tail_boundary_preview(unet, official_output)
    sources = source_names()
    res0_index = len(remaining_external_after_up3_tuple_slice) + len(up3_res_tuple) - 1
    res1_index = len(remaining_external_after_up3_tuple_slice) + len(up3_res_tuple[:-1]) - 1
    res2_index = len(remaining_external_after_up3_tuple_slice) + len(up3_res_tuple[:-1][:-1]) - 1

    result = {
        'status': status,
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'python': sys.executable,
        'loaded_lora_parameter_count': len(loaded),
        'upblock3_class': f'{type(up3).__module__}.{type(up3).__name__}',
        'upblock3_forward_signature': str(inspect.signature(up3.forward)),
        'accepted_forward_args': list(inspect.signature(up3.forward).parameters.keys()),
        'return_info': official_return,
        'output_states_status': official_return.get('output_states_status', 'NOT_APPLICABLE'),
        'upblock2_return_info': up2_return,
        'residual_contract': {
            'accumulated_down_block_res_sample_count': len(ctx['down_res']),
            'accumulated_down_block_res_sample_sources': sources,
            'accumulated_down_block_res_sample_shapes': [list(x.shape) for x in ctx['down_res']],
            'remaining_residual_count_before_upblock3': len(remaining_before_up3),
            'remaining_residual_shapes_before_upblock3': [list(x.shape) for x in remaining_before_up3],
            'local_upblock3_residual_tuple_count': len(up3_res_tuple),
            'local_upblock3_residual_tuple_shapes': [list(x.shape) for x in up3_res_tuple],
            'upblock3_resnet0_consumed_source': sources[res0_index],
            'upblock3_resnet1_consumed_source': sources[res1_index],
            'upblock3_resnet2_consumed_source': sources[res2_index],
            'upblock3_resnet0_consumed_shape': list(manual['resnet0_residual'].shape),
            'upblock3_resnet1_consumed_shape': list(manual['resnet1_residual'].shape),
            'upblock3_resnet2_consumed_shape': list(manual['resnet2_residual'].shape),
            'attention0_consumes_residual': False,
            'attention1_consumes_residual': False,
            'attention2_consumes_residual': False,
            'upsampler_count': len(getattr(up3, 'upsamplers', []) or []),
            'remaining_internal_residual_count_after_upblock3': len(manual['remaining_after_manual_upblock3']),
            'remaining_external_residual_count_after_upblock3': len(remaining_external_after_up3_tuple_slice),
            'remaining_external_residual_shapes_after_upblock3': [list(x.shape) for x in remaining_external_after_up3_tuple_slice],
        },
        'manual_chain_vs_official_local': local_diff,
        'local_output_shape': list(official_output.shape),
        'local_output_stats': stats(to_np(official_output)),
        'output_tail_boundary_preview': output_tail_boundary,
        'uses_full_unet_forward': False,
        'uses_output_tail': False,
        'uses_full_tadsr_inference': False,
        'markers': {
            'TADSR_UNET_UPBLOCK3_LOCAL_FORWARD_AUDIT': status,
            'TADSR_UNET_UPBLOCK3_RESIDUAL_CONTRACT_AUDIT': residual_status,
            'TADSR_UNET_OUTPUT_TAIL_BOUNDARY_AUDIT': output_tail_boundary['status'],
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding='utf-8')
    OUT_TXT.write_text('\n'.join([
        '# TADSR UNet up_blocks.3 local forward audit',
        '',
        *[f'{k}: {v}' for k, v in result['markers'].items()],
        '',
        f"upblock3 class: {result['upblock3_class']}",
        f"forward signature: {result['upblock3_forward_signature']}",
        f"manual vs official max_abs_error: {local_diff['max_abs_error']}",
        f"manual vs official mean_abs_error: {local_diff['mean_abs_error']}",
        f"manual vs official cosine: {local_diff['cosine_similarity']}",
        f"output shape: {result['local_output_shape']}",
        f"output-tail modules preview: {', '.join(output_tail_boundary['tail_module_names'])}",
        'scope: local up_blocks.3 only; output tail/full UNet/full inference were not executed.',
    ]) + '\n', encoding='utf-8')
    for k, v in result['markers'].items():
        print(f'{k}: {v}')
    print(json.dumps({'status': status, 'metadata': str(OUT_JSON)}, indent=2))
    return 0 if status == 'PASS' and residual_status == 'PASS' and output_tail_boundary['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
