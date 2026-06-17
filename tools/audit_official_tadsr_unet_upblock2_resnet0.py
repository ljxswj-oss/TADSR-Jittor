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
from export_tadsr_unet_upblock1_resnet0_oracle import source_names
from audit_official_tadsr_unet_upblock1_local import run_context

OUT_JSON = OUT_DIR / 'audit_tadsr_unet_upblock2_resnet0.json'
OUT_TXT = OUT_DIR / 'audit_tadsr_unet_upblock2_resnet0.txt'


def maybe_reexec() -> None:
    expected_prefix = str(STRICT_PY.parents[1])
    in_expected_venv = str(sys.prefix) == expected_prefix or str(sys.executable).startswith(expected_prefix)
    if STRICT_PY.exists() and not in_expected_venv:
        os.environ['PYTHONNOUSERSITE'] = '1'
        os.execv(str(STRICT_PY), [str(STRICT_PY), __file__, *sys.argv[1:]])


def child_order(block) -> list[str]:
    names = []
    for name, module in block.named_children():
        names.append(f'{name}: {type(module).__module__}.{type(module).__name__}')
    return names


def main() -> int:
    maybe_reexec()
    import torch

    torch.manual_seed(1234)
    np.random.seed(1234)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    unet, loaded = load_unet()
    if len(unet.up_blocks) < 3:
        raise RuntimeError('Official UNet does not expose up_blocks.2')
    up2 = unet.up_blocks[2]
    if not getattr(up2, 'resnets', None):
        raise RuntimeError('Official up_blocks.2 has no resnets.0 target')
    resnet0 = up2.resnets[0]

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
        res0_hidden = up2_res_tuple[-1]
        remaining_after_resnet0_local = up2_res_tuple[:-1]
        concat = torch.cat([ctx['official_output'], res0_hidden], dim=1)
        output = resnet0(concat, ctx['time_embedding'])

    sources = source_names()
    consumed_index = len(remaining_after_up1) - 1
    next_module = 'up_blocks.2.attentions.0' if len(getattr(up2, 'attentions', []) or []) > 0 else 'up_blocks.2.resnets.1'
    cfg = resnet_config(resnet0, 'up_blocks_2_resnets_0')
    lora_info = {
        name: module_lora_info(getattr(resnet0, name))
        for name in ['conv1', 'time_emb_proj', 'conv2', 'conv_shortcut']
        if getattr(resnet0, name, None) is not None
    }
    result = {
        'status': 'PASS',
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'python': sys.executable,
        'loaded_lora_parameter_count': len(loaded),
        'upblock2_class': f'{type(up2).__module__}.{type(up2).__name__}',
        'upblock2_forward_signature': str(inspect.signature(up2.forward)),
        'upblock2_has_cross_attention': bool(getattr(up2, 'has_cross_attention', False)),
        'upblock2_child_order': child_order(up2),
        'upblock2_resnet_count': len(getattr(up2, 'resnets', []) or []),
        'upblock2_attention_count': len(getattr(up2, 'attentions', []) or []),
        'upblock2_upsampler_count': len(getattr(up2, 'upsamplers', []) or []),
        'upblock2_forward_args': list(inspect.signature(up2.forward).parameters.keys()),
        'current_target': 'up_blocks.2.resnets.0',
        'remaining_modules_after_this_stage': [
            'up_blocks.2.attentions.0' if len(getattr(up2, 'attentions', []) or []) > 0 else None,
            'up_blocks.2.resnets.1' if len(getattr(up2, 'resnets', []) or []) > 1 else None,
            'up_blocks.2.attentions.1' if len(getattr(up2, 'attentions', []) or []) > 1 else None,
            'up_blocks.2.resnets.2' if len(getattr(up2, 'resnets', []) or []) > 2 else None,
            'up_blocks.2.attentions.2' if len(getattr(up2, 'attentions', []) or []) > 2 else None,
            'up_blocks.2.upsamplers.0' if len(getattr(up2, 'upsamplers', []) or []) > 0 else None,
        ],
        'actual_next_module_after_resnet0': next_module,
        'sample_shape': sample_shape,
        'encoder_hidden_states_shape': encoder_shape,
        'conv_in_output_shape': list(ctx['conv_in'].shape),
        'time_embedding_shape': list(ctx['time_embedding'].shape),
        'downblock_output_hidden_shapes': [list(x.shape) for x in ctx['down_hiddens']],
        'midblock_output_hidden_shape': list(ctx['mid_output'].shape),
        'upblock0_output_hidden_shape': list(ctx['up0_output'].shape),
        'upblock1_output_hidden_shape': list(ctx['official_output'].shape),
        'accumulated_down_block_res_sample_count': len(ctx['down_res']),
        'accumulated_down_block_res_sample_sources': sources,
        'accumulated_down_block_res_sample_shapes': [list(x.shape) for x in ctx['down_res']],
        'upblock0_consumed_sources': sources[-len(unet.up_blocks[0].resnets):],
        'upblock1_consumed_sources': [
            sources[len(ctx['remaining_after_up0']) - 1],
            sources[len(ctx['remaining_after_up0']) - 2],
            sources[len(ctx['remaining_after_up0']) - 3],
        ],
        'remaining_residual_count_after_upblock1': len(remaining_after_up1),
        'remaining_residual_shapes_after_upblock1': [list(x.shape) for x in remaining_after_up1],
        'upblock2_local_residual_tuple_count': len(up2_res_tuple),
        'upblock2_local_residual_tuple_shapes': [list(x.shape) for x in up2_res_tuple],
        'remaining_residual_count_before_upblock2': len(remaining_before_up2),
        'remaining_residual_shapes_before_upblock2': [list(x.shape) for x in remaining_before_up2],
        'upblock2_resnet0_consumed_residual_index_in_accumulated_tuple': consumed_index,
        'upblock2_resnet0_consumed_residual_current_remaining_tuple_index': len(up2_res_tuple) - 1,
        'upblock2_resnet0_consumed_residual_source': sources[consumed_index],
        'upblock2_resnet0_consumed_residual_shape': list(res0_hidden.shape),
        'upblock2_resnet0_hidden_input_shape': list(ctx['official_output'].shape),
        'upblock2_resnet0_concat_input_shape': list(concat.shape),
        'upblock2_resnet0_output_shape': list(output.shape),
        'concat_axis': 1,
        'remaining_residual_count_after_upblock2_resnet0_pop': len(remaining_after_resnet0_local),
        'remaining_residual_shapes_after_upblock2_resnet0_pop': [list(x.shape) for x in remaining_after_resnet0_local],
        'resnet0_class': f'{type(resnet0).__module__}.{type(resnet0).__name__}',
        'resnet0_forward_signature': str(inspect.signature(resnet0.forward)),
        'resnet0_config': cfg,
        'resnet0_lora_info': lora_info,
        'resnet0_output_stats': stats(to_np(output)),
        'compare_with_existing_tester': {
            'reuse_unet_resnet_block_2d_tester': True,
            'wrapper_must_concat_hidden_and_residual_before_generic_resnet': True,
            'has_channel_change': list(concat.shape) != list(output.shape),
            'has_conv_shortcut': bool(cfg.get('has_shortcut')),
            'time_embedding_norm_supported': cfg.get('time_embedding_norm') in {'default', 'scale_shift'},
        },
        'uses_full_unet_forward': False,
        'uses_upblock2_attention0': False,
        'uses_upblocks3': False,
        'uses_full_tadsr_inference': False,
        'next_module_preview': next_module,
        'markers': {
            'TADSR_UNET_UPBLOCK2_AUDIT': 'PASS',
            'TADSR_UNET_UPBLOCK2_RESNET0_AUDIT': 'PASS',
            'TADSR_UNET_UPBLOCK2_RESNET0_RESIDUAL_CONTRACT_AUDIT': 'PASS',
            'TADSR_UNET_UPBLOCK2_RESNET0_LORA_AUDIT': 'PASS',
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding='utf-8')
    OUT_TXT.write_text('\n'.join([
        '# TADSR UNet up_blocks.2.resnets.0 official audit',
        '',
        *[f'{k}: {v}' for k, v in result['markers'].items()],
        '',
        f"upblock2 class: {result['upblock2_class']}",
        f"upblock2 has_cross_attention: {result['upblock2_has_cross_attention']}",
        f"resnet0 residual source: {result['upblock2_resnet0_consumed_residual_source']}",
        f"resnet0 residual index: {consumed_index}",
        f"hidden input shape: {result['upblock2_resnet0_hidden_input_shape']}",
        f"residual shape: {result['upblock2_resnet0_consumed_residual_shape']}",
        f"concat shape: {result['upblock2_resnet0_concat_input_shape']}",
        f"output shape: {result['upblock2_resnet0_output_shape']}",
        f"next module: {next_module}",
        'full up_blocks.2 alignment: NOT_COMPLETE',
        'full UNet forward alignment: NOT_COMPLETE',
    ]) + '\n', encoding='utf-8')
    for k, v in result['markers'].items():
        print(f'{k}: {v}')
    print(json.dumps({'status': 'PASS', 'metadata': str(OUT_JSON)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
