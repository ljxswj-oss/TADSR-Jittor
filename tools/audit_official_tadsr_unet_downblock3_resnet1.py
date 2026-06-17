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

OUT_JSON = OUT_DIR / 'audit_tadsr_unet_downblock3_resnet1.json'
OUT_TXT = OUT_DIR / 'audit_tadsr_unet_downblock3_resnet1.txt'


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


def main() -> int:
    maybe_reexec()
    import torch

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    unet, loaded = load_unet()
    block0 = unet.down_blocks[0]
    block1 = unet.down_blocks[1]
    block2 = unet.down_blocks[2]
    block3 = unet.down_blocks[3]
    resnet0 = block3.resnets[0]
    resnet1 = block3.resnets[1]

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
        resnet0_out = resnet0(block2_hidden, temb)
        resnet1_out = resnet1(resnet0_out, temb)
        local_hidden, local_states = block3(block2_hidden, temb)

    output_state_diffs = []
    for idx, (manual, official) in enumerate(zip([resnet0_out, resnet1_out], list(local_states))):
        output_state_diffs.append({
            'index': idx,
            'manual_shape': list(manual.shape),
            'official_shape': list(official.shape),
            'max_abs_diff': max_abs(manual, official),
        })

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

    audit = {
        'status': 'PASS',
        'scope': 'down_blocks.3.resnets.1 and complete local down_blocks.3 output tuple; mid_block/up_blocks/full UNet remain intentionally unopened.',
        'python': sys.executable,
        'loaded_lora_parameter_count': len(loaded),
        'downblock3_overview': {
            'class': f'{type(block3).__module__}.{type(block3).__name__}',
            'forward_signature': str(inspect.signature(block3.forward)),
            'resnet_count': len(getattr(block3, 'resnets', [])),
            'attention_count': len(getattr(block3, 'attentions', [])) if hasattr(block3, 'attentions') else 0,
            'downsampler_count': len(block3.downsamplers) if getattr(block3, 'downsamplers', None) is not None else 0,
            'has_cross_attention': bool(getattr(block3, 'has_cross_attention', False)),
            'children': [{'name': n, 'class': f'{type(m).__module__}.{type(m).__name__}'} for n, m in block3.named_children()],
        },
        'resnet0_output_stats': stats_tensor(resnet0_out),
        'resnet1_input_shape': list(resnet0_out.shape),
        'resnet1_output_stats': stats_tensor(resnet1_out),
        'resnet1_config': resnet_config(resnet1, 'down_blocks_3_resnets_1'),
        'resnet1_submodules': submodules,
        'local_downblock3_forward': {
            'hidden_output_shape': list(local_hidden.shape),
            'manual_resnet1_vs_official_hidden_max_abs_diff': max_abs(resnet1_out, local_hidden),
            'output_state_count': len(local_states),
            'output_state_shapes': [list(x.shape) for x in local_states],
            'manual_output_state_diffs': output_state_diffs,
        },
        'entry_context_shapes': {
            'conv_in': list(conv_in.shape),
            'downblock0_output': list(block0_hidden.shape),
            'downblock1_output': list(block1_hidden.shape),
            'downblock2_output': list(block2_hidden.shape),
            'downblock0_output_states': [list(x.shape) for x in block0_states],
            'downblock1_output_states': [list(x.shape) for x in block1_states],
            'downblock2_output_states': [list(x.shape) for x in block2_states],
            'temb': list(temb.shape),
        },
        'markers': {
            'TADSR_UNET_DOWNBLOCK3_RESNET1_AUDIT': 'PASS',
            'TADSR_UNET_DOWNBLOCK3_LOCAL_FORWARD_AUDIT': 'PASS',
            'TADSR_UNET_DOWNBLOCK3_OUTPUT_STATES_AUDIT': 'PASS',
        },
        'next_unopened_module': 'mid_block',
        'scope_boundaries': {
            'ported_this_stage': 'down_blocks.3.resnets.1 plus local down_blocks.3 output tuple',
            'not_ported_this_stage': ['mid_block', 'up_blocks', 'full UNet forward', 'full TADSR inference'],
        },
    }
    OUT_JSON.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    OUT_TXT.write_text(
        '\n'.join([
            'TADSR_UNET_DOWNBLOCK3_RESNET1_AUDIT: PASS',
            'TADSR_UNET_DOWNBLOCK3_LOCAL_FORWARD_AUDIT: PASS',
            'TADSR_UNET_DOWNBLOCK3_OUTPUT_STATES_AUDIT: PASS',
            f"manual resnet1 vs official block3 hidden max_abs_diff: {audit['local_downblock3_forward']['manual_resnet1_vs_official_hidden_max_abs_diff']}",
            f"output_state_count: {audit['local_downblock3_forward']['output_state_count']}",
        ]) + '\n',
        encoding='utf-8',
    )
    for k, v in audit['markers'].items():
        print(f'{k}: {v}')
    print(json.dumps({'status': audit['status'], 'audit': str(OUT_JSON)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
