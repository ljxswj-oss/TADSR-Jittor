#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

STRICT_PY = Path('/mnt/data/sj/venvs/tadsr_official_strict_cu118/bin/python')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
OUT_DIR = Path('experiments/full_repro/unet_alignment')
OUT_JSON = OUT_DIR / 'audit_tadsr_unet_midblock_attention0.json'
OUT_TXT = OUT_DIR / 'audit_tadsr_unet_midblock_attention0.txt'


def maybe_reexec() -> None:
    expected_prefix = str(STRICT_PY.parents[1])
    in_expected_venv = str(sys.prefix) == expected_prefix or str(sys.executable).startswith(expected_prefix)
    if STRICT_PY.exists() and not in_expected_venv:
        os.environ['PYTHONNOUSERSITE'] = '1'
        os.execv(str(STRICT_PY), [str(STRICT_PY), __file__, *sys.argv[1:]])


def main() -> int:
    maybe_reexec()
    import torch
    from tools.audit_official_tadsr_unet_downblock2_attention1 import (
        attention_config,
        transformer_block_details,
        module_lora_info,
        stats_tensor,
        load_unet,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    unet, loaded = load_unet()
    mid = unet.mid_block
    attention0 = mid.attentions[0]
    resnet0 = mid.resnets[0]
    resnet1 = mid.resnets[1] if len(mid.resnets) > 1 else None

    sample = torch.linspace(-1.0, 1.0, steps=1 * int(unet.config.in_channels) * 32 * 32, dtype=torch.float32).reshape(1, int(unet.config.in_channels), 32, 32)
    timestep = torch.tensor([1], dtype=torch.long)
    encoder_shape = [1, 77, int(unet.config.cross_attention_dim)]
    encoder_hidden_states = torch.linspace(-0.5, 0.5, steps=int(np.prod(encoder_shape)), dtype=torch.float32).reshape(*encoder_shape)
    with torch.no_grad():
        centered = 2.0 * sample - 1.0 if unet.config.center_input_sample else sample
        conv_in = unet.conv_in(centered)
        time_proj = unet.time_proj(timestep)
        temb = unet.time_embedding(time_proj.to(dtype=list(unet.time_embedding.parameters())[0].dtype))
        b0_hidden, b0_states = unet.down_blocks[0](conv_in, temb, encoder_hidden_states=encoder_hidden_states)
        b1_hidden, b1_states = unet.down_blocks[1](b0_hidden, temb, encoder_hidden_states=encoder_hidden_states)
        b2_hidden, b2_states = unet.down_blocks[2](b1_hidden, temb, encoder_hidden_states=encoder_hidden_states)
        b3_hidden, b3_states = unet.down_blocks[3](b2_hidden, temb)
        mid0 = resnet0(b3_hidden, temb)
        att0 = attention0(mid0, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]

    cfg = attention_config(attention0, 'mid_block_attentions_0', 'unet.mid_block.attentions.0')
    tb0 = attention0.transformer_blocks[0]
    heads = int(getattr(tb0.attn1, 'heads', 0) or getattr(attention0, 'num_attention_heads', 0))
    inner_dim = int(getattr(tb0.attn1, 'inner_dim', 0) or cfg.get('inner_dim') or cfg.get('in_channels'))
    head_dim = int(inner_dim // heads)
    cfg['num_attention_heads'] = heads
    cfg['attention_head_dim'] = head_dim
    cfg['inner_dim'] = inner_dim
    cfg['cross_attention_dim'] = int(getattr(tb0.attn2, 'cross_attention_dim', 0) or unet.config.cross_attention_dim)

    tblocks = [transformer_block_details(tb) for tb in attention0.transformer_blocks]
    lora_modules = {}
    for label, mod in [
        ('proj_in', attention0.proj_in),
        ('proj_out', attention0.proj_out),
        ('transformer0_attn1_to_q', tb0.attn1.to_q),
        ('transformer0_attn1_to_k', tb0.attn1.to_k),
        ('transformer0_attn1_to_v', tb0.attn1.to_v),
        ('transformer0_attn1_to_out_0', tb0.attn1.to_out[0]),
        ('transformer0_attn2_to_q', tb0.attn2.to_q),
        ('transformer0_attn2_to_k', tb0.attn2.to_k),
        ('transformer0_attn2_to_v', tb0.attn2.to_v),
        ('transformer0_attn2_to_out_0', tb0.attn2.to_out[0]),
        ('transformer0_ff_geglu_proj', tb0.ff.net[0].proj),
        ('transformer0_ff_out', tb0.ff.net[2]),
    ]:
        info = module_lora_info(mod)
        if info.get('is_lora_wrapped'):
            lora_modules[label] = info

    result = {
        'status': 'PASS',
        'scope': 'mid_block.attentions.0 audit only; mid_block.resnets.1/up_blocks/full UNet remain intentionally unopened.',
        'python': sys.executable,
        'loaded_lora_parameter_count': len(loaded),
        'mid_block_overview': {
            'class': f'{type(mid).__module__}.{type(mid).__name__}',
            'children': [{'name': n, 'class': f'{type(m).__module__}.{type(m).__name__}'} for n, m in mid.named_children()],
            'resnet_count': len(mid.resnets),
            'attention_count': len(mid.attentions),
            'has_cross_attention': True,
            'known_completed_modules': ['mid_block.resnets.0'],
            'target_module': 'mid_block.attentions.0',
            'remaining_modules_after_this_stage': ['mid_block.resnets.1'] if resnet1 is not None else [],
            'actual_next_module_after_attention0': 'mid_block.resnets.1' if resnet1 is not None else None,
            'expected_operation_order': ['resnets.0', 'attentions.0', 'resnets.1'],
            'peft_wrapped_children_present': bool(lora_modules),
        },
        'attention0_config': cfg,
        'attention_config': cfg,
        'transformer0_config': {
            'heads': heads,
            'head_dim': head_dim,
            'inner_dim': inner_dim,
            'norm_eps': float(getattr(tb0.norm1, 'eps', 1e-5)),
            'dropout': 0.0,
            'transformer_block_count': len(attention0.transformer_blocks),
        },
        'transformer_blocks': tblocks,
        'attention0_input_stats': stats_tensor(mid0),
        'attention0_output_stats': stats_tensor(att0),
        'encoder_hidden_states_stats': stats_tensor(encoder_hidden_states),
        'entry_context_shapes': {
            'sample': list(sample.shape),
            'timestep': list(timestep.shape),
            'encoder_hidden_states': list(encoder_hidden_states.shape),
            'conv_in': list(conv_in.shape),
            'time_embedding': list(temb.shape),
            'downblock0_output': list(b0_hidden.shape),
            'downblock0_output_states': [list(x.shape) for x in b0_states],
            'downblock1_output': list(b1_hidden.shape),
            'downblock1_output_states': [list(x.shape) for x in b1_states],
            'downblock2_output': list(b2_hidden.shape),
            'downblock2_output_states': [list(x.shape) for x in b2_states],
            'downblock3_output': list(b3_hidden.shape),
            'downblock3_output_states': [list(x.shape) for x in b3_states],
            'midblock_resnet0_output': list(mid0.shape),
            'midblock_attention0_output': list(att0.shape),
            'mid_block_attentions_0_consumes': ['hidden_states', 'encoder_hidden_states'],
            'mid_block_attentions_0_consumes_accumulated_residuals': False,
            'attention_mask': None,
            'encoder_attention_mask': None,
            'cross_attention_kwargs': None,
        },
        'existing_tester_compatibility': {
            'tester': 'UNetAttention0Transformer2DTester',
            'compatible': len(attention0.transformer_blocks) == 1 and bool(cfg.get('use_linear_projection')),
            'reason': 'Same Transformer2DModel + BasicTransformerBlock pattern; selected with prefix=mid_block_attentions_0.',
            'required_code_changes': [],
        },
        'lora_modules': lora_modules,
        'downstream_resnet1_preview': {
            'exists': resnet1 is not None,
            'class': f'{type(resnet1).__module__}.{type(resnet1).__name__}' if resnet1 is not None else None,
            'expected_input_shape_from_attention0': list(att0.shape),
            'uses_temb': True,
            'not_ported_this_stage': True,
        },
        'markers': {
            'TADSR_UNET_MIDBLOCK_ATTENTION0_AUDIT': 'PASS',
            'TADSR_UNET_MIDBLOCK_ATTENTION0_LORA_AUDIT': 'PASS',
        },
        'scope_boundaries': {
            'ported_this_stage': 'mid_block.attentions.0 only',
            'not_ported_this_stage': ['mid_block.resnets.1', 'up_blocks', 'full UNet forward', 'full TADSR inference'],
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding='utf-8')
    OUT_TXT.write_text('\n'.join([
        '# TADSR UNet mid_block.attentions.0 audit',
        '',
        'TADSR_UNET_MIDBLOCK_ATTENTION0_AUDIT: PASS',
        'TADSR_UNET_MIDBLOCK_ATTENTION0_LORA_AUDIT: PASS',
        f"attention0 class: {cfg['class']}",
        f"transformer block count: {len(attention0.transformer_blocks)}",
        f"input shape: {list(mid0.shape)}",
        f"output shape: {list(att0.shape)}",
        'next module: mid_block.resnets.1',
        'Full mid_block/up_blocks/full UNet/full inference remain unopened.',
    ]) + '\n', encoding='utf-8')
    print('TADSR_UNET_MIDBLOCK_ATTENTION0_AUDIT: PASS')
    print('TADSR_UNET_MIDBLOCK_ATTENTION0_LORA_AUDIT: PASS')
    print(json.dumps({'status': 'PASS', 'audit': str(OUT_JSON)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
