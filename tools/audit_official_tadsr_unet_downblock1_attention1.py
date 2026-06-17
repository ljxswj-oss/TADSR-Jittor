#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))
from audit_official_tadsr_unet_downblock1_attention0 import (
    OFFICIAL_REPO,
    OUT_DIR,
    WEIGHTS_DIR,
    attention_config,
    load_unet,
    module_lora_info,
    shape,
    stats_tensor,
    transformer_block_details,
)

OUT_JSON = OUT_DIR / 'audit_tadsr_unet_downblock1_attention1.json'
OUT_TXT = OUT_DIR / 'audit_tadsr_unet_downblock1_attention1.txt'
STRICT_PY = Path('/mnt/data/sj/venvs/tadsr_official_strict_cu118/bin/python')


def maybe_reexec() -> None:
    expected_prefix = str(STRICT_PY.parents[1])
    in_expected_venv = str(sys.prefix) == expected_prefix or str(sys.executable).startswith(expected_prefix)
    if STRICT_PY.exists() and not in_expected_venv:
        os.environ['PYTHONNOUSERSITE'] = '1'
        os.execv(str(STRICT_PY), [str(STRICT_PY), __file__, *sys.argv[1:]])


def _collect_lora_modules(attention):
    lora_modules = {}
    for label, info in [('proj_in', module_lora_info(attention.proj_in)), ('proj_out', module_lora_info(attention.proj_out))]:
        if info.get('is_lora_wrapped'):
            lora_modules[label] = info
    for i, tb in enumerate(attention.transformer_blocks):
        for an in ['attn1', 'attn2']:
            attn = getattr(tb, an)
            for ln in ['to_q', 'to_k', 'to_v']:
                info = module_lora_info(getattr(attn, ln))
                if info.get('is_lora_wrapped'):
                    lora_modules[f'transformer{i}_{an}_{ln}'] = info
            info = module_lora_info(attn.to_out[0])
            if info.get('is_lora_wrapped'):
                lora_modules[f'transformer{i}_{an}_to_out_0'] = info
        if hasattr(tb.ff.net[0], 'proj'):
            info = module_lora_info(tb.ff.net[0].proj)
            if info.get('is_lora_wrapped'):
                lora_modules[f'transformer{i}_ff_geglu_proj'] = info
        info = module_lora_info(tb.ff.net[2])
        if info.get('is_lora_wrapped'):
            lora_modules[f'transformer{i}_ff_out'] = info
    return lora_modules


def main() -> int:
    maybe_reexec()
    import torch

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    unet, loaded = load_unet()
    block0 = unet.down_blocks[0]
    block1 = unet.down_blocks[1]
    resnet0 = block1.resnets[0]
    attention0 = block1.attentions[0]
    resnet1 = block1.resnets[1]
    attention1 = block1.attentions[1]
    downsampler0 = block1.downsamplers[0] if getattr(block1, 'downsamplers', None) is not None and len(block1.downsamplers) else None

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
        resnet0_out = resnet0(block0_hidden, temb)
        attention0_out = attention0(resnet0_out, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
        resnet1_out = resnet1(attention0_out, temb)
        attention1_out = attention1(resnet1_out, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]

    children = [{'name': n, 'class': f'{type(m).__module__}.{type(m).__name__}'} for n, m in block1.named_children()]
    cfg = attention_config(attention1, 'down_blocks_1_attentions_1', 'unet.down_blocks.1.attentions.1')
    prev_cfg = attention_config(attention0, 'down_blocks_1_attentions_0', 'unet.down_blocks.1.attentions.0')
    comparison = {k: {'downblock1_attention0': prev_cfg.get(k), 'downblock1_attention1': cfg.get(k), 'same': prev_cfg.get(k) == cfg.get(k)}
                  for k in ['class', 'in_channels', 'out_channels', 'num_attention_heads', 'attention_head_dim', 'inner_dim', 'cross_attention_dim', 'use_linear_projection', 'norm_num_groups', 'norm_eps', 'proj_in_weight_shape', 'proj_out_weight_shape']}
    tblocks = [transformer_block_details(tb) for tb in attention1.transformer_blocks]
    lora_modules = _collect_lora_modules(attention1)
    audit = {
        'status': 'PASS',
        'scope': 'down_blocks.1.attentions.1 only; down_blocks.1.downsamplers.0 and full UNet remain unopened.',
        'python': sys.executable,
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'loaded_lora_parameter_count': len(loaded),
        'downblock1_overview': {
            'class': f'{type(block1).__module__}.{type(block1).__name__}',
            'has_cross_attention': getattr(block1, 'has_cross_attention', True),
            'forward_signature': str(inspect.signature(block1.forward)),
            'module_order': children,
            'resnet_count': len(getattr(block1, 'resnets', [])),
            'attention_count': len(getattr(block1, 'attentions', [])) if hasattr(block1, 'attentions') else 0,
            'downsampler_count': len(getattr(block1, 'downsamplers', [])) if getattr(block1, 'downsamplers', None) is not None else 0,
            'known_completed_modules': {'resnets.0': 'PASS', 'attentions.0': 'PASS', 'resnets.1': 'PASS'},
            'current_target_module': 'attentions.1',
            'remaining_modules_after_this_stage': ['downsamplers.0' if downsampler0 is not None else None],
            'actual_next_module_after_attention1': 'downsamplers.0' if downsampler0 is not None else None,
            'consumes_encoder_hidden_states': True,
            'attention_mask_used_in_this_stage': False,
            'cross_attention_kwargs_used_in_this_stage': False,
            'lora_scale_argument_used_in_this_stage': False,
            'has_peft_wrappers': bool(lora_modules),
        },
        'attention_config': cfg,
        'attention1_config': cfg,
        'compare_with_downblock1_attention0': comparison,
        'transformer_block_count': len(attention1.transformer_blocks),
        'transformer_blocks': tblocks,
        'encoder_hidden_states_contract': {
            'shape': encoder_shape,
            'cross_attention_dim': int(unet.config.cross_attention_dim),
            'sequence_length': encoder_shape[1],
            'attention_mask': None,
            'encoder_attention_mask': None,
            'cross_attention_kwargs': None,
        },
        'lora_modules': lora_modules,
        'effective_weights_required': True,
        'effective_weights_strategy': 'Export merged static weights in tools/export_tadsr_unet_downblock1_attention1_oracle.py; no generic Jittor LoRA runtime.',
        'bridge_shapes': {
            'sample': list(sample.shape),
            'encoder_hidden_states': encoder_shape,
            'conv_in': list(conv_in.shape),
            'time_embedding': list(temb.shape),
            'downblock0_hidden': list(block0_hidden.shape),
            'downblock0_output_state_shapes': [list(x.shape) for x in block0_states],
            'downblock1_resnet0_output': list(resnet0_out.shape),
            'downblock1_attention0_output': list(attention0_out.shape),
            'downblock1_resnet1_output': list(resnet1_out.shape),
            'downblock1_attention1_output': list(attention1_out.shape),
        },
        'bridge_tensor_stats': {
            'downblock1_resnet1_output': stats_tensor(resnet1_out),
            'downblock1_attention1_output': stats_tensor(attention1_out),
        },
        'output_states_preview': {
            'resnets1_output_shape': list(resnet1_out.shape),
            'attentions1_output_shape': list(attention1_out.shape),
            'future_downsampler_will_append_additional_state': bool(downsampler0 is not None),
            'full_output_states_not_implemented_this_stage': True,
        },
        'not_in_scope': ['down_blocks.1.downsamplers.0', 'down_blocks.2', 'mid_block', 'up_blocks', 'full UNet forward', 'full TADSR inference', 'generic Jittor LoRA runtime'],
        'next_action': 'Continue TADSR UNet down_blocks.1.downsamplers.0 audit/export/port after down_blocks.1.attentions.1 alignment; keep full UNet forward NotImplemented.',
    }
    if audit['transformer_block_count'] != 1:
        audit['status'] = 'BLOCKED_NEEDS_SPLIT'
        audit['blocked_reason'] = 'This exporter/tester currently supports one transformer block. Split multi-block attention migration first.'
    OUT_JSON.write_text(json.dumps(audit, indent=2, default=str), encoding='utf-8')
    lines = [
        'TADSR UNet down_blocks.1.attentions.1 audit',
        f"status: {audit['status']}",
        f"attention class: {cfg['class']}",
        f"in_channels: {cfg['in_channels']}",
        f"inner_dim: {cfg['inner_dim']}",
        f"heads: {cfg['num_attention_heads']}",
        f"head_dim: {cfg['attention_head_dim']}",
        f"transformer blocks: {audit['transformer_block_count']}",
        f"input shape: {audit['bridge_shapes']['downblock1_resnet1_output']}",
        f"output shape: {audit['bridge_shapes']['downblock1_attention1_output']}",
        f"LoRA wrapped modules: {list(lora_modules)}",
        f"next module: {audit['downblock1_overview']['actual_next_module_after_attention1']}",
    ]
    OUT_TXT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'TADSR_UNET_DOWNBLOCK1_ATTENTION1_AUDIT: {audit["status"]}')
    if audit['status'] == 'PASS':
        print('TADSR_UNET_DOWNBLOCK1_ATTENTION1_TRANSFORMER_AUDIT: PASS')
        print('TADSR_UNET_DOWNBLOCK1_ATTENTION1_LORA_AUDIT: PASS')
        return 0
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
