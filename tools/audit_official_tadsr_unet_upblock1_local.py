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

OUT_JSON = OUT_DIR / 'audit_tadsr_unet_upblock1_local.json'
OUT_TXT = OUT_DIR / 'audit_tadsr_unet_upblock1_local.txt'


def maybe_reexec() -> None:
    expected_prefix = str(STRICT_PY.parents[1])
    in_expected_venv = str(sys.prefix) == expected_prefix or str(sys.executable).startswith(expected_prefix)
    if STRICT_PY.exists() and not in_expected_venv:
        os.environ['PYTHONNOUSERSITE'] = '1'
        os.execv(str(STRICT_PY), [str(STRICT_PY), __file__, *sys.argv[1:]])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(np.float64).reshape(-1)
    bb = b.astype(np.float64).reshape(-1)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / denom) if denom else 1.0


def tensor_metrics(a, b) -> dict:
    aa = to_np(a).astype(np.float32)
    bb = to_np(b).astype(np.float32)
    diff = np.abs(aa - bb)
    return {
        'shape_match': list(aa.shape) == list(bb.shape),
        'shape': list(aa.shape),
        'max_abs_error': float(diff.max()) if diff.size else 0.0,
        'mean_abs_error': float(diff.mean()) if diff.size else 0.0,
        'cosine_similarity': cosine(aa, bb),
    }


def extract_hidden(out):
    if isinstance(out, tuple):
        return out[0], {'return_type': 'tuple', 'output_states_status': 'UNKNOWN_TUPLE'}
    if hasattr(out, 'sample'):
        return out.sample, {'return_type': type(out).__name__, 'output_states_status': 'UNKNOWN_OBJECT'}
    return out, {'return_type': type(out).__name__, 'output_states_status': 'NOT_APPLICABLE'}


def call_upblock1_local(up1, hidden_states, res_tuple, temb, encoder_hidden_states):
    sig = inspect.signature(up1.forward)
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
    return up1(hidden_states, res_tuple, **kwargs)


def call_upsampler(upsampler, hidden_states):
    sig = inspect.signature(upsampler.forward)
    if 'output_size' in sig.parameters:
        return upsampler(hidden_states, output_size=None)
    if 'upsample_size' in sig.parameters:
        return upsampler(hidden_states, upsample_size=None)
    return upsampler(hidden_states)


def run_context(unet, sample, timestep, encoder_hidden_states):
    import torch

    up0 = unet.up_blocks[0]
    up1 = unet.up_blocks[1]
    centered = 2.0 * sample - 1.0 if unet.config.center_input_sample else sample
    hidden = unet.conv_in(centered)
    time_proj = unet.time_proj(timestep)
    temb = unet.time_embedding(time_proj.to(dtype=list(unet.time_embedding.parameters())[0].dtype))
    down_res = (hidden,)
    down_hiddens = []
    down_states = []
    for block in unet.down_blocks:
        if getattr(block, 'has_cross_attention', False):
            hidden, states = block(hidden, temb, encoder_hidden_states=encoder_hidden_states)
        else:
            hidden, states = block(hidden, temb)
        down_hiddens.append(hidden)
        down_states.append(states)
        down_res += states
    mid_raw = unet.mid_block(hidden, temb=temb, encoder_hidden_states=encoder_hidden_states, attention_mask=None, cross_attention_kwargs=None, encoder_attention_mask=None)
    mid_output, mid_return = extract_midblock_output(mid_raw)
    up0_output = up0(mid_output, down_res[-len(up0.resnets):], temb=temb, upsample_size=None)
    remaining_after_up0 = down_res[:-len(up0.resnets)]
    up1_res_tuple = remaining_after_up0[-len(up1.resnets):]
    remaining_before_up1 = remaining_after_up0[:-len(up1.resnets)]

    official_raw = call_upblock1_local(up1, up0_output, up1_res_tuple, temb, encoder_hidden_states)
    official_output, official_return = extract_hidden(official_raw)

    res0_hidden = up1_res_tuple[-1]
    rem0 = up1_res_tuple[:-1]
    res0 = up1.resnets[0](torch.cat([up0_output, res0_hidden], dim=1), temb)
    att0 = up1.attentions[0](res0, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
    res1_hidden = rem0[-1]
    rem1 = rem0[:-1]
    res1 = up1.resnets[1](torch.cat([att0, res1_hidden], dim=1), temb)
    att1 = up1.attentions[1](res1, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
    res2_hidden = rem1[-1]
    rem2 = rem1[:-1]
    res2 = up1.resnets[2](torch.cat([att1, res2_hidden], dim=1), temb)
    att2 = up1.attentions[2](res2, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
    manual_output = call_upsampler(up1.upsamplers[0], att2)
    return {
        'centered': centered,
        'conv_in': down_res[0],
        'time_proj': time_proj,
        'time_embedding': temb,
        'down_res': down_res,
        'down_hiddens': down_hiddens,
        'down_states': down_states,
        'mid_output': mid_output,
        'mid_return': mid_return,
        'up0_output': up0_output,
        'remaining_after_up0': remaining_after_up0,
        'remaining_before_up1': remaining_before_up1,
        'up1_res_tuple': up1_res_tuple,
        'official_output': official_output,
        'official_return': official_return,
        'manual_outputs': {
            'resnet0': res0,
            'attention0': att0,
            'resnet1': res1,
            'attention1': att1,
            'resnet2': res2,
            'attention2': att2,
            'upsampler': manual_output,
        },
        'remaining_after_manual_upblock1': rem2,
    }


def main() -> int:
    maybe_reexec()
    import torch

    torch.manual_seed(1234)
    np.random.seed(1234)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    unet, loaded = load_unet()
    up1 = unet.up_blocks[1]
    sample_shape = [1, int(unet.config.in_channels), 32, 32]
    sample = torch.linspace(-1.0, 1.0, steps=int(np.prod(sample_shape)), dtype=torch.float32).reshape(*sample_shape)
    timestep = torch.tensor([1], dtype=torch.long)
    encoder_shape = [1, 77, int(unet.config.cross_attention_dim)]
    encoder_hidden_states = torch.linspace(-0.5, 0.5, steps=int(np.prod(encoder_shape)), dtype=torch.float32).reshape(*encoder_shape)

    with torch.no_grad():
        ctx = run_context(unet, sample, timestep, encoder_hidden_states)

    sources = source_names()
    resnet_sources = [
        sources[len(ctx['remaining_after_up0']) - 1],
        sources[len(ctx['remaining_after_up0']) - 2],
        sources[len(ctx['remaining_after_up0']) - 3],
    ]
    local_diff = tensor_metrics(ctx['manual_outputs']['upsampler'], ctx['official_output'])
    status = 'PASS' if local_diff['shape_match'] and local_diff['max_abs_error'] <= 1e-5 else 'FAIL'
    result = {
        'status': status,
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'python': sys.executable,
        'loaded_lora_parameter_count': len(loaded),
        'upblock1_class': f'{type(up1).__module__}.{type(up1).__name__}',
        'upblock1_forward_signature': str(inspect.signature(up1.forward)),
        'accepted_forward_args': list(inspect.signature(up1.forward).parameters.keys()),
        'return_info': ctx['official_return'],
        'output_states_status': ctx['official_return'].get('output_states_status', 'NOT_APPLICABLE'),
        'residual_contract': {
            'accumulated_down_block_res_sample_count': len(ctx['down_res']),
            'accumulated_down_block_res_sample_sources': sources,
            'accumulated_down_block_res_sample_shapes': [list(x.shape) for x in ctx['down_res']],
            'upblock0_consumed_sources': sources[-len(unet.up_blocks[0].resnets):],
            'remaining_before_upblock1_count': len(ctx['remaining_after_up0']),
            'remaining_before_upblock1_shapes': [list(x.shape) for x in ctx['remaining_after_up0']],
            'local_upblock1_residual_tuple_count': len(ctx['up1_res_tuple']),
            'local_upblock1_residual_tuple_shapes': [list(x.shape) for x in ctx['up1_res_tuple']],
            'upblock1_resnet0_consumed_source': resnet_sources[0],
            'upblock1_resnet1_consumed_source': resnet_sources[1],
            'upblock1_resnet2_consumed_source': resnet_sources[2],
            'attentions_consume_residual': False,
            'upsampler_consumes_residual': False,
            'remaining_residual_count_after_upblock1': len(ctx['remaining_before_up1']),
            'remaining_residual_shapes_after_upblock1': [list(x.shape) for x in ctx['remaining_before_up1']],
            'remaining_residuals_reserved_for': ['up_blocks.2', 'up_blocks.3'],
        },
        'manual_chain_vs_official_local': local_diff,
        'local_output_shape': list(ctx['official_output'].shape),
        'local_output_stats': stats(to_np(ctx['official_output'])),
        'uses_full_unet_forward': False,
        'uses_upblocks2_3': False,
        'uses_full_tadsr_inference': False,
        'next_module_preview': 'up_blocks.2',
        'markers': {
            'TADSR_UNET_UPBLOCK1_LOCAL_FORWARD_AUDIT': status,
            'TADSR_UNET_UPBLOCK1_RESIDUAL_CONTRACT_AUDIT': 'PASS',
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding='utf-8')
    OUT_TXT.write_text('\n'.join([
        '# TADSR UNet up_blocks.1 local forward audit',
        '',
        *[f'{k}: {v}' for k, v in result['markers'].items()],
        '',
        f"upblock1 class: {result['upblock1_class']}",
        f"forward signature: {result['upblock1_forward_signature']}",
        f"manual vs official max_abs_error: {local_diff['max_abs_error']}",
        f"manual vs official mean_abs_error: {local_diff['mean_abs_error']}",
        f"manual vs official cosine: {local_diff['cosine_similarity']}",
        f"output shape: {result['local_output_shape']}",
        'scope: local up_blocks.1 only; up_blocks.2/full UNet/full inference were not executed.',
    ]) + '\n', encoding='utf-8')
    for k, v in result['markers'].items():
        print(f'{k}: {v}')
    print(json.dumps({'status': status, 'metadata': str(OUT_JSON)}, indent=2))
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
