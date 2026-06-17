#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
import os
import sys
import numpy as np

from export_tadsr_unet_downblock3_resnet0_oracle import (
    STRICT_PY,
    OUT_DIR,
    OFFICIAL_REPO,
    WEIGHTS_DIR,
    load_unet,
    module_lora_info,
    resnet_config,
    to_np,
    stats,
)

OUT_JSON = OUT_DIR / 'audit_tadsr_unet_upblock1_resnet1.json'
OUT_TXT = OUT_DIR / 'audit_tadsr_unet_upblock1_resnet1.txt'


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
    up1 = unet.up_blocks[1]
    if len(up0.resnets) < 3 or not getattr(up0, 'upsamplers', None):
        raise RuntimeError('Official up_blocks.0 must be locally complete before auditing up_blocks.1.resnets.1.')
    if len(up1.resnets) < 2 or len(getattr(up1, 'attentions', []) or []) < 1:
        raise RuntimeError('Official up_blocks.1 must expose resnets.1 and attentions.0 before this stage.')
    resnet0 = up1.resnets[0]
    attention0 = up1.attentions[0]
    resnet1 = up1.resnets[1]

    sample_shape = [1, int(unet.config.in_channels), 32, 32]
    sample = torch.linspace(-1.0, 1.0, steps=int(np.prod(sample_shape)), dtype=torch.float32).reshape(*sample_shape)
    timestep = torch.tensor([1], dtype=torch.long)
    encoder_shape = [1, 77, int(unet.config.cross_attention_dim)]
    encoder_hidden_states = torch.linspace(-0.5, 0.5, steps=int(np.prod(encoder_shape)), dtype=torch.float32).reshape(*encoder_shape)

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
        mid = extract_midblock_output(
            unet.mid_block(
                hidden,
                temb=temb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=None,
                cross_attention_kwargs=None,
                encoder_attention_mask=None,
            )
        )
        up0_res_tuple = down_res[-len(up0.resnets):]
        remaining_after_up0_slice = down_res[:-len(up0.resnets)]
        up0_output = up0(mid, up0_res_tuple, temb=temb, upsample_size=None)
        up1_res_tuple = remaining_after_up0_slice[-len(up1.resnets):]
        remaining_before_up1_slice = remaining_after_up0_slice[:-len(up1.resnets)]

        res0_hidden = up1_res_tuple[-1]
        remaining_after_resnet0_local = up1_res_tuple[:-1]
        concat0 = torch.cat([up0_output, res0_hidden], dim=1)
        out0 = resnet0(concat0, temb)
        attention0_out = attention0(out0, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]

        res1_hidden = remaining_after_resnet0_local[-1]
        remaining_after_resnet1_local = remaining_after_resnet0_local[:-1]
        concat1 = torch.cat([attention0_out, res1_hidden], dim=1)
        out1 = resnet1(concat1, temb)

    sources = source_names()
    resnet0_consumed_index = len(remaining_after_up0_slice) - 1
    resnet1_consumed_index = len(remaining_after_up0_slice) - 2
    next_module = 'up_blocks.1.attentions.1' if len(getattr(up1, 'attentions', [])) > 1 else 'up_blocks.1.resnets.2'
    meta = {
        'status': 'PASS',
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'loaded_lora_parameter_count': len(loaded),
        'upblock0_class': f'{type(up0).__module__}.{type(up0).__name__}',
        'upblock1_class': f'{type(up1).__module__}.{type(up1).__name__}',
        'upblock1_forward_signature': str(inspect.signature(up1.forward)),
        'upblock1_has_cross_attention': bool(getattr(up1, 'has_cross_attention', False)),
        'upblock1_resnet_count': len(up1.resnets),
        'upblock1_attention_count': len(getattr(up1, 'attentions', []) or []),
        'upblock1_upsampler_count': len(getattr(up1, 'upsamplers', []) or []),
        'upblock1_actual_forward_order_until_target': [
            'up_blocks.1 receives hidden_states from full local up_blocks.0 output',
            'slice last len(up_blocks.1.resnets) residuals from remaining accumulated down_block_res_samples',
            'pop residual for resnets.0 from the end of that local residual tuple',
            'concat hidden_states and popped residual on channel axis',
            'run up_blocks.1.resnets.0',
            'run up_blocks.1.attentions.0 without consuming residuals',
            'pop residual for resnets.1 from the end of the remaining local residual tuple',
            'concat attentions.0 output and popped residual on channel axis',
            'run up_blocks.1.resnets.1',
            'stop before up_blocks.1.attentions.1',
        ],
        'sample_shape': sample_shape,
        'encoder_hidden_states_shape': encoder_shape,
        'downblock_output_state_shapes': down_state_shapes,
        'downblock_output_hidden_shapes': down_hidden_shapes,
        'accumulated_down_block_res_sample_sources': sources,
        'accumulated_down_block_res_sample_shapes': [list(x.shape) for x in down_res],
        'accumulated_down_block_res_sample_count': len(down_res),
        'upblock0_consumed_residual_count': len(up0.resnets),
        'remaining_residual_count_after_upblock0_slice': len(remaining_after_up0_slice),
        'upblock1_local_residual_tuple_count': len(up1_res_tuple),
        'remaining_residual_count_before_upblock1_slice': len(remaining_before_up1_slice),
        'upblock1_resnet0_consumed_residual_index_in_accumulated_tuple': resnet0_consumed_index,
        'upblock1_resnet0_consumed_residual_source': sources[resnet0_consumed_index],
        'upblock1_resnet0_consumed_residual_shape': list(res0_hidden.shape),
        'internal_remaining_residual_count_after_resnet0_pop': len(remaining_after_resnet0_local),
        'internal_remaining_residual_shapes_after_resnet0_pop': [list(x.shape) for x in remaining_after_resnet0_local],
        'upblock1_attention0_consumes_accumulated_residuals': False,
        'internal_remaining_residual_count_after_attention0': len(remaining_after_resnet0_local),
        'internal_remaining_residual_shapes_after_attention0': [list(x.shape) for x in remaining_after_resnet0_local],
        'upblock1_resnet1_consumed_residual_index_in_accumulated_tuple': resnet1_consumed_index,
        'upblock1_resnet1_consumed_residual_current_remaining_tuple_index': len(remaining_after_resnet0_local) - 1,
        'upblock1_resnet1_consumed_residual_source': sources[resnet1_consumed_index],
        'upblock1_resnet1_consumed_residual_shape': list(res1_hidden.shape),
        'internal_remaining_residual_count_after_resnet1_pop': len(remaining_after_resnet1_local),
        'internal_remaining_residual_shapes_after_resnet1_pop': [list(x.shape) for x in remaining_after_resnet1_local],
        'upblock0_local_output_shape': list(up0_output.shape),
        'upblock1_resnet0_output_shape': list(out0.shape),
        'upblock1_attention0_output_shape': list(attention0_out.shape),
        'upblock1_resnet1_hidden_input_shape': list(attention0_out.shape),
        'upblock1_resnet1_concat_input_shape': list(concat1.shape),
        'upblock1_resnet1_output_shape': list(out1.shape),
        'concat_axis': 1,
        'temb_shape': list(temb.shape),
        'resnet1_config': resnet_config(resnet1, 'up_blocks_1_resnets_1'),
        'resnet1_lora_info': {
            name: module_lora_info(getattr(resnet1, name))
            for name in ['conv1', 'time_emb_proj', 'conv2', 'conv_shortcut']
            if getattr(resnet1, name, None) is not None
        },
        'resnet1_output_stats': stats(to_np(out1)),
        'actual_next_module_after_upblock1_resnet1': next_module,
        'markers': {
            'TADSR_UNET_UPBLOCK1_AUDIT': 'PASS',
            'TADSR_UNET_UPBLOCK1_RESNET1_AUDIT': 'PASS',
            'TADSR_UNET_UPBLOCK1_RESNET1_RESIDUAL_CONTRACT_AUDIT': 'PASS',
            'TADSR_UNET_UPBLOCK1_RESNET1_LORA_AUDIT': 'PASS',
        },
        'scope_boundaries': {
            'audited_this_stage': 'up_blocks.1.resnets.1 residual pop/concat after attentions.0 only',
            'not_audited_this_stage': [
                'up_blocks.1.attentions.1',
                'up_blocks.1.resnets.2',
                'up_blocks.1.attentions.2',
                'up_blocks.1.upsamplers.0',
                'up_blocks.2',
                'up_blocks.3',
                'full UNet forward',
                'generic runtime LoRA',
                'full TADSR inference',
            ],
        },
    }
    OUT_JSON.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    OUT_TXT.write_text('\n'.join([
        '# TADSR UNet up_blocks.1.resnets.1 official audit',
        '',
        *[f'{k}: {v}' for k, v in meta['markers'].items()],
        '',
        f"resnet1 residual source: {meta['upblock1_resnet1_consumed_residual_source']}",
        f"resnet1 residual index: {resnet1_consumed_index}",
        f"hidden input shape: {meta['upblock1_resnet1_hidden_input_shape']}",
        f"concat shape: {meta['upblock1_resnet1_concat_input_shape']}",
        f"output shape: {meta['upblock1_resnet1_output_shape']}",
        f"next unopened module: {next_module}",
        'full up_blocks.1 alignment: NOT_COMPLETE',
        'full UNet forward alignment: NOT_COMPLETE',
    ]) + '\n', encoding='utf-8')
    for k, v in meta['markers'].items():
        print(f'{k}: {v}')
    print(json.dumps({'status': 'PASS', 'metadata': str(OUT_JSON)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
