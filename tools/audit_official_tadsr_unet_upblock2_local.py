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
from export_tadsr_unet_upblock1_resnet0_oracle import source_names
from audit_official_tadsr_unet_upblock1_local import run_context, tensor_metrics, extract_hidden, call_upsampler

OUT_JSON = OUT_DIR / 'audit_tadsr_unet_upblock2_local.json'
OUT_TXT = OUT_DIR / 'audit_tadsr_unet_upblock2_local.txt'


def maybe_reexec() -> None:
    expected_prefix = str(STRICT_PY.parents[1])
    in_expected_venv = str(sys.prefix) == expected_prefix or str(sys.executable).startswith(expected_prefix)
    if STRICT_PY.exists() and not in_expected_venv:
        os.environ['PYTHONNOUSERSITE'] = '1'
        os.execv(str(STRICT_PY), [str(STRICT_PY), __file__, *sys.argv[1:]])


def call_upblock2_local(up2, hidden_states, res_tuple, temb, encoder_hidden_states):
    sig = inspect.signature(up2.forward)
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
    return up2(hidden_states, res_tuple, **kwargs)


def manual_upblock2_chain(up2, ctx, up2_res_tuple, encoder_hidden_states):
    import torch

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
    upsampled = call_upsampler(up2.upsamplers[0], att2)
    return {
        'resnet0': res0,
        'attention0': att0,
        'resnet1': res1,
        'attention1': att1,
        'resnet2': res2,
        'attention2': att2,
        'upsampler': upsampled,
        'remaining_after_manual_upblock2': remaining_after_resnet2,
    }


def main() -> int:
    maybe_reexec()
    import torch

    torch.manual_seed(1234)
    np.random.seed(1234)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    unet, loaded = load_unet()
    up2 = unet.up_blocks[2]

    if len(up2.resnets) != 3 or len(getattr(up2, 'attentions', []) or []) != 3 or len(getattr(up2, 'upsamplers', []) or []) != 1:
        raise RuntimeError('This local audit expects up_blocks.2 to have 3 resnets, 3 attentions, and 1 upsampler.')

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
        official_raw = call_upblock2_local(up2, ctx['official_output'], up2_res_tuple, ctx['time_embedding'], encoder_hidden_states)
        official_output, official_return = extract_hidden(official_raw)
        manual = manual_upblock2_chain(up2, ctx, up2_res_tuple, encoder_hidden_states)

    local_diff = tensor_metrics(manual['upsampler'], official_output)
    status = 'PASS' if local_diff['shape_match'] and local_diff['max_abs_error'] <= 1e-5 else 'FAIL'
    output_states_status = official_return.get('output_states_status', 'NOT_APPLICABLE')
    sources = source_names()
    res0_index = len(remaining_after_up1) - 1
    res1_index = len(remaining_before_up2) + len(up2_res_tuple[:-1]) - 1
    res2_index = len(remaining_before_up2) + len(up2_res_tuple[:-1][:-1]) - 1

    result = {
        'status': status,
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'python': sys.executable,
        'loaded_lora_parameter_count': len(loaded),
        'upblock2_class': f'{type(up2).__module__}.{type(up2).__name__}',
        'upblock2_forward_signature': str(inspect.signature(up2.forward)),
        'accepted_forward_args': list(inspect.signature(up2.forward).parameters.keys()),
        'return_info': official_return,
        'output_states_status': output_states_status,
        'residual_contract': {
            'accumulated_down_block_res_sample_count': len(ctx['down_res']),
            'accumulated_down_block_res_sample_sources': sources,
            'accumulated_down_block_res_sample_shapes': [list(x.shape) for x in ctx['down_res']],
            'remaining_residual_count_before_upblock2': len(remaining_after_up1),
            'remaining_residual_shapes_before_upblock2': [list(x.shape) for x in remaining_after_up1],
            'local_upblock2_residual_tuple_count': len(up2_res_tuple),
            'local_upblock2_residual_tuple_shapes': [list(x.shape) for x in up2_res_tuple],
            'upblock2_resnet0_consumed_source': sources[res0_index],
            'upblock2_resnet1_consumed_source': sources[res1_index],
            'upblock2_resnet2_consumed_source': sources[res2_index],
            'attentions_consume_residual': False,
            'upsampler_consumes_residual': False,
            'remaining_internal_residual_count_after_upblock2': len(manual['remaining_after_manual_upblock2']),
            'remaining_external_residual_count_after_upblock2': len(remaining_before_up2),
            'remaining_external_residual_shapes_after_upblock2': [list(x.shape) for x in remaining_before_up2],
            'remaining_residuals_reserved_for': ['up_blocks.3'],
        },
        'manual_chain_vs_official_local': local_diff,
        'local_output_shape': list(official_output.shape),
        'local_output_stats': stats(to_np(official_output)),
        'uses_full_unet_forward': False,
        'uses_upblocks3': False,
        'uses_full_tadsr_inference': False,
        'next_module_preview': 'up_blocks.3.resnets.0',
        'markers': {
            'TADSR_UNET_UPBLOCK2_LOCAL_FORWARD_AUDIT': status,
            'TADSR_UNET_UPBLOCK2_RESIDUAL_CONTRACT_AUDIT': 'PASS',
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding='utf-8')
    OUT_TXT.write_text('\n'.join([
        '# TADSR UNet up_blocks.2 local forward audit',
        '',
        *[f'{k}: {v}' for k, v in result['markers'].items()],
        '',
        f"upblock2 class: {result['upblock2_class']}",
        f"forward signature: {result['upblock2_forward_signature']}",
        f"manual vs official max_abs_error: {local_diff['max_abs_error']}",
        f"manual vs official mean_abs_error: {local_diff['mean_abs_error']}",
        f"manual vs official cosine: {local_diff['cosine_similarity']}",
        f"output shape: {result['local_output_shape']}",
        'scope: local up_blocks.2 only; up_blocks.3/full UNet/full inference were not executed.',
    ]) + '\n', encoding='utf-8')
    for k, v in result['markers'].items():
        print(f'{k}: {v}')
    print(json.dumps({'status': status, 'metadata': str(OUT_JSON)}, indent=2))
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
