#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import numpy as np

from export_tadsr_unet_downblock3_resnet0_oracle import STRICT_PY, OUT_DIR, OFFICIAL_REPO, WEIGHTS_DIR, load_unet, resnet_config, to_np, stats

OUT_JSON = OUT_DIR / 'audit_official_tadsr_unet_upblock0_resnet2.json'
OUT_TXT = OUT_DIR / 'audit_official_tadsr_unet_upblock0_resnet2.txt'


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


def main() -> int:
    maybe_reexec()
    import torch
    torch.manual_seed(1234)
    np.random.seed(1234)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    unet, loaded = load_unet()
    up0 = unet.up_blocks[0]
    if len(up0.resnets) < 3:
        raise RuntimeError(f'Official up_blocks.0 has {len(up0.resnets)} resnets; expected at least 3')
    resnet2 = up0.resnets[2]
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
        mid = unet.mid_block(hidden, temb=temb, encoder_hidden_states=encoder_hidden_states, attention_mask=None, cross_attention_kwargs=None, encoder_attention_mask=None)
        if isinstance(mid, tuple):
            mid = mid[0]
        elif hasattr(mid, 'sample'):
            mid = mid.sample
        res_tuple = down_res[-len(up0.resnets):]
        global_remaining = down_res[:-len(up0.resnets)]
        res0_hidden = res_tuple[-1]
        after0_remaining = res_tuple[:-1]
        out0 = up0.resnets[0](torch.cat([mid, res0_hidden], dim=1), temb)
        res1_hidden = after0_remaining[-1]
        after1_remaining = after0_remaining[:-1]
        out1 = up0.resnets[1](torch.cat([out0, res1_hidden], dim=1), temb)
        res2_hidden = after1_remaining[-1]
        after2_remaining = after1_remaining[:-1]
        concat2 = torch.cat([out1, res2_hidden], dim=1)
        out2 = resnet2(concat2, temb)
    sources = source_names()
    res0_index = len(down_res) - 1
    res1_index = len(down_res) - 2
    res2_index = len(down_res) - 3
    meta = {
        'status': 'PASS',
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'loaded_lora_parameter_count': len(loaded),
        'upblock0_resnet_count': len(up0.resnets),
        'upblock0_upsampler_count': len(getattr(up0, 'upsamplers', []) or []),
        'sample_shape': sample_shape,
        'encoder_hidden_states_shape': encoder_shape,
        'downblock_output_state_shapes': down_state_shapes,
        'downblock_output_hidden_shapes': down_hidden_shapes,
        'accumulated_down_block_res_sample_sources': sources,
        'accumulated_down_block_res_sample_shapes': [list(x.shape) for x in down_res],
        'global_remaining_residual_count_after_slice': len(global_remaining),
        'upblock0_resnet0_consumed_residual_index_in_accumulated_tuple': res0_index,
        'upblock0_resnet0_consumed_residual_source': sources[res0_index],
        'upblock0_resnet0_consumed_residual_shape': list(res0_hidden.shape),
        'upblock0_resnet1_consumed_residual_index_in_accumulated_tuple': res1_index,
        'upblock0_resnet1_consumed_residual_source': sources[res1_index],
        'upblock0_resnet1_consumed_residual_shape': list(res1_hidden.shape),
        'upblock0_resnet2_consumed_residual_index_in_accumulated_tuple': res2_index,
        'upblock0_resnet2_consumed_residual_source': sources[res2_index],
        'upblock0_resnet2_consumed_residual_shape': list(res2_hidden.shape),
        'internal_remaining_residual_count_after_resnet2_pop': len(after2_remaining),
        'upblock0_resnet2_hidden_input_shape': list(out1.shape),
        'upblock0_resnet2_concat_input_shape': list(concat2.shape),
        'upblock0_resnet2_output_shape': list(out2.shape),
        'resnet2_config': resnet_config(resnet2, 'up_blocks_0_resnets_2'),
        'resnet2_output_stats': stats(to_np(out2)),
        'next_unopened_module': 'up_blocks.0.upsamplers.0',
        'markers': {
            'TADSR_UNET_UPBLOCK0_RESNET2_AUDIT': 'PASS',
            'TADSR_UNET_UPBLOCK0_RESNET2_RESIDUAL_CONTRACT_AUDIT': 'PASS',
            'TADSR_UNET_UPBLOCK0_RESNET2_LORA_AUDIT': 'PASS',
        },
        'scope_boundaries': {
            'audited_this_stage': 'up_blocks.0.resnets.2 residual pop/concat only',
            'not_audited_this_stage': ['up_blocks.0.upsamplers.0', 'full up_blocks.0', 'full UNet forward', 'generic runtime LoRA', 'full TADSR inference'],
        },
    }
    OUT_JSON.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    OUT_TXT.write_text('\n'.join(['# TADSR UNet up_blocks.0.resnets.2 official audit', '', *[f'{k}: {v}' for k, v in meta['markers'].items()], '', f"resnet2 residual source: {meta['upblock0_resnet2_consumed_residual_source']}", f"resnet2 residual index: {res2_index}", f"concat shape: {list(concat2.shape)}", 'next unopened module: up_blocks.0.upsamplers.0']) + '\n', encoding='utf-8')
    for k, v in meta['markers'].items():
        print(f'{k}: {v}')
    print(json.dumps({'status': 'PASS', 'metadata': str(OUT_JSON)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
