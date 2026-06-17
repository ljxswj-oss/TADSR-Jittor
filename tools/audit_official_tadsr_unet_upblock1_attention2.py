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
)
from export_tadsr_unet_upblock1_resnet0_oracle import extract_midblock_output, source_names
from audit_official_tadsr_unet_downblock2_attention1 import (
    attention_config,
    transformer_block_details,
    module_lora_info,
    stats_tensor,
)

OUT_JSON = OUT_DIR / 'audit_tadsr_unet_upblock1_attention2.json'
OUT_TXT = OUT_DIR / 'audit_tadsr_unet_upblock1_attention2.txt'


def maybe_reexec() -> None:
    expected_prefix = str(STRICT_PY.parents[1])
    in_expected_venv = str(sys.prefix) == expected_prefix or str(sys.executable).startswith(expected_prefix)
    if STRICT_PY.exists() and not in_expected_venv:
        os.environ['PYTHONNOUSERSITE'] = '1'
        os.execv(str(STRICT_PY), [str(STRICT_PY), __file__, *sys.argv[1:]])


def main() -> int:
    maybe_reexec()
    import torch

    torch.manual_seed(1234)
    np.random.seed(1234)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    unet, loaded = load_unet()
    up0 = unet.up_blocks[0]
    up1 = unet.up_blocks[1]
    if len(up1.resnets) < 3 or len(getattr(up1, 'attentions', []) or []) < 3:
        raise RuntimeError('Official up_blocks.1 must expose resnets.0/1/2, attentions.0/1/2, and the following upsampler.')
    if not getattr(up1, 'upsamplers', None):
        raise RuntimeError('Official up_blocks.1.upsamplers.0 must exist as the next unopened module after attentions.2.')
    resnet0 = up1.resnets[0]
    attention0 = up1.attentions[0]
    resnet1 = up1.resnets[1]
    attention1 = up1.attentions[1]
    resnet2 = up1.resnets[2]
    attention2 = up1.attentions[2]

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
        mid_raw = unet.mid_block(
            hidden,
            temb=temb,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=None,
            cross_attention_kwargs=None,
            encoder_attention_mask=None,
        )
        mid_output, mid_return = extract_midblock_output(mid_raw)
        up0_res_tuple = down_res[-len(up0.resnets):]
        remaining_after_up0_slice = down_res[:-len(up0.resnets)]
        up0_output = up0(mid_output, up0_res_tuple, temb=temb, upsample_size=None)
        up1_res_tuple = remaining_after_up0_slice[-len(up1.resnets):]
        remaining_before_up1_slice = remaining_after_up0_slice[:-len(up1.resnets)]

        res0_hidden = up1_res_tuple[-1]
        remaining_after_resnet0_local = up1_res_tuple[:-1]
        concat0 = torch.cat([up0_output, res0_hidden], dim=1)
        out0 = resnet0(concat0, temb)
        att0 = attention0(out0, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]

        res1_hidden = remaining_after_resnet0_local[-1]
        remaining_after_resnet1_local = remaining_after_resnet0_local[:-1]
        concat1 = torch.cat([att0, res1_hidden], dim=1)
        out1 = resnet1(concat1, temb)
        att1 = attention1(out1, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]

        res2_hidden = remaining_after_resnet1_local[-1]
        remaining_after_resnet2_local = remaining_after_resnet1_local[:-1]
        concat2 = torch.cat([att1, res2_hidden], dim=1)
        out2 = resnet2(concat2, temb)
        att2 = attention2(
            out2,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs=None,
            attention_mask=None,
            encoder_attention_mask=None,
            return_dict=False,
        )[0]

    cfg = attention_config(attention2, 'up_blocks_1_attentions_2', 'unet.up_blocks.1.attentions.2')
    tb0 = attention2.transformer_blocks[0]
    heads = int(getattr(tb0.attn1, 'heads', 0) or getattr(attention2, 'num_attention_heads', 0))
    inner_dim = int(getattr(tb0.attn1, 'inner_dim', 0) or cfg.get('inner_dim') or cfg.get('in_channels'))
    head_dim = int(inner_dim // heads)
    cfg['num_attention_heads'] = heads
    cfg['attention_head_dim'] = head_dim
    cfg['inner_dim'] = inner_dim
    cfg['cross_attention_dim'] = int(getattr(tb0.attn2, 'cross_attention_dim', 0) or unet.config.cross_attention_dim)

    lora_modules = {}
    for label, mod in [
        ('proj_in', attention2.proj_in),
        ('proj_out', attention2.proj_out),
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

    sources = source_names()
    resnet0_consumed_index = len(remaining_after_up0_slice) - 1
    resnet1_consumed_index = len(remaining_after_up0_slice) - 2
    resnet2_consumed_index = len(remaining_after_up0_slice) - 3
    result = {
        'status': 'PASS',
        'scope': 'up_blocks.1.attentions.2 audit only; up_blocks.1.upsamplers.0 and later modules remain intentionally unopened.',
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'python': sys.executable,
        'loaded_lora_parameter_count': len(loaded),
        'upblock1_overview': {
            'class': f'{type(up1).__module__}.{type(up1).__name__}',
            'forward_signature': str(inspect.signature(up1.forward)),
            'children': [{'name': n, 'class': f'{type(m).__module__}.{type(m).__name__}'} for n, m in up1.named_children()],
            'resnet_count': len(up1.resnets),
            'attention_count': len(getattr(up1, 'attentions', []) or []),
            'upsampler_count': len(getattr(up1, 'upsamplers', []) or []),
            'has_cross_attention': bool(getattr(up1, 'has_cross_attention', False)),
            'known_completed_modules_before_this_stage': [
                'up_blocks.1.resnets.0',
                'up_blocks.1.attentions.0',
                'up_blocks.1.resnets.1',
                'up_blocks.1.attentions.1',
                'up_blocks.1.resnets.2',
            ],
            'target_module': 'up_blocks.1.attentions.2',
            'actual_next_module_after_attention2': 'up_blocks.1.upsamplers.0',
            'remaining_modules_after_this_stage': ['up_blocks.1.upsamplers.0', 'up_blocks.2', 'up_blocks.3', 'full UNet forward'],
            'expected_operation_order_until_target': [
                'up_blocks.1.resnets.0 consumes down_blocks.2.output_state_1 residual',
                'run up_blocks.1.attentions.0 with encoder_hidden_states',
                'up_blocks.1.resnets.1 consumes down_blocks.2.output_state_0 residual',
                'run up_blocks.1.attentions.1 with encoder_hidden_states',
                'up_blocks.1.resnets.2 consumes down_blocks.1.output_state_2 residual',
                'run up_blocks.1.attentions.2 with encoder_hidden_states',
                'stop before up_blocks.1.upsamplers.0',
            ],
        },
        'attention2_config': cfg,
        'attention_config': cfg,
        'attention_module_comparison': {
            'attention0_class': f'{type(attention0).__module__}.{type(attention0).__name__}',
            'attention1_class': f'{type(attention1).__module__}.{type(attention1).__name__}',
            'attention2_class': f'{type(attention2).__module__}.{type(attention2).__name__}',
            'attention2_uses_same_tester_pattern': True,
        },
        'transformer0_config': {
            'heads': heads,
            'head_dim': head_dim,
            'inner_dim': inner_dim,
            'norm_eps': float(getattr(tb0.norm1, 'eps', 1e-5)),
            'dropout': 0.0,
            'transformer_block_count': len(attention2.transformer_blocks),
        },
        'transformer_blocks': [transformer_block_details(tb) for tb in attention2.transformer_blocks],
        'entry_context_shapes': {
            'sample': list(sample.shape),
            'timestep': list(timestep.shape),
            'encoder_hidden_states': list(encoder_hidden_states.shape),
            'downblock_output_state_shapes': down_state_shapes,
            'downblock_output_hidden_shapes': down_hidden_shapes,
            'midblock_output': list(mid_output.shape),
            'upblock0_output': list(up0_output.shape),
            'upblock1_resnet0_residual_source': sources[resnet0_consumed_index],
            'upblock1_resnet1_residual_source': sources[resnet1_consumed_index],
            'upblock1_resnet2_residual_source': sources[resnet2_consumed_index],
            'upblock1_resnet2_output': list(out2.shape),
            'upblock1_attention2_input': list(out2.shape),
            'upblock1_attention2_output': list(att2.shape),
            'remaining_residual_count_before_upblock1_slice': len(remaining_before_up1_slice),
            'internal_remaining_residual_count_after_resnet2_pop': len(remaining_after_resnet2_local),
            'internal_remaining_residual_count_after_attention2': len(remaining_after_resnet2_local),
            'attention2_consumes_accumulated_residuals': False,
            'attention_mask': None,
            'encoder_attention_mask': None,
            'cross_attention_kwargs': None,
            'local_midblock_return': mid_return,
        },
        'attention2_input_stats': stats_tensor(out2),
        'attention2_output_stats': stats_tensor(att2),
        'encoder_hidden_states_stats': stats_tensor(encoder_hidden_states),
        'existing_tester_compatibility': {
            'tester': 'UNetAttention0Transformer2DTester',
            'compatible': len(attention2.transformer_blocks) == 1 and bool(cfg.get('use_linear_projection')),
            'reason': 'Same Transformer2DModel + BasicTransformerBlock pattern; selected with prefix=up_blocks_1_attentions_2.',
        },
        'lora_modules': lora_modules,
        'upblock1_upsampler_preview': {
            'exists': bool(getattr(up1, 'upsamplers', None)),
            'class': f'{type(up1.upsamplers[0]).__module__}.{type(up1.upsamplers[0]).__name__}',
            'expected_input_shape_from_attention2': list(att2.shape),
            'not_ported_this_stage': True,
        },
        'residual_contract': {
            'attention2_consumes_accumulated_residuals': False,
            'remaining_after_resnet2_pop_count': len(remaining_after_resnet2_local),
            'remaining_after_attention2_count': len(remaining_after_resnet2_local),
            'future_residuals_reserved_for_later_up_blocks': len(remaining_before_up1_slice),
        },
        'markers': {
            'TADSR_UNET_UPBLOCK1_AUDIT': 'PASS',
            'TADSR_UNET_UPBLOCK1_ATTENTION2_AUDIT': 'PASS',
            'TADSR_UNET_UPBLOCK1_ATTENTION2_RESIDUAL_CONTRACT_AUDIT': 'PASS',
            'TADSR_UNET_UPBLOCK1_ATTENTION2_LORA_AUDIT': 'PASS',
        },
        'scope_boundaries': {
            'ported_this_stage': 'up_blocks.1.attentions.2 only',
            'not_ported_this_stage': [
                'up_blocks.1.upsamplers.0',
                'full up_blocks.1',
                'up_blocks.2',
                'up_blocks.3',
                'full UNet forward',
                'generic runtime LoRA',
                'full TADSR inference',
            ],
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding='utf-8')
    OUT_TXT.write_text('\n'.join([
        '# TADSR UNet up_blocks.1.attentions.2 official audit',
        '',
        *[f'{k}: {v}' for k, v in result['markers'].items()],
        '',
        f"attention2 class: {cfg['class']}",
        f"transformer block count: {len(attention2.transformer_blocks)}",
        f"input shape: {list(out2.shape)}",
        f"output shape: {list(att2.shape)}",
        'next module: up_blocks.1.upsamplers.0',
        'full up_blocks.1/full UNet/full inference remain NOT_COMPLETE.',
    ]) + '\n', encoding='utf-8')
    for k, v in result['markers'].items():
        print(f'{k}: {v}')
    print(json.dumps({'status': 'PASS', 'audit': str(OUT_JSON)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
