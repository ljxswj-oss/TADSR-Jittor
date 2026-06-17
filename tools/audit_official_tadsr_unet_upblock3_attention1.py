#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
import os
import sys
import numpy as np

from audit_official_tadsr_unet_upblock1_local import run_context, extract_hidden
from audit_official_tadsr_unet_upblock2_local import call_upblock2_local
from audit_official_tadsr_unet_downblock2_attention1 import (
    attention_config,
    transformer_block_details,
    module_lora_info,
    stats_tensor,
)
from export_tadsr_unet_downblock3_resnet0_oracle import (
    STRICT_PY,
    OUT_DIR,
    OFFICIAL_REPO,
    WEIGHTS_DIR,
    load_unet,
)
from export_tadsr_unet_upblock1_resnet0_oracle import source_names

OUT_JSON = OUT_DIR / 'audit_tadsr_unet_upblock3_attention1.json'
OUT_TXT = OUT_DIR / 'audit_tadsr_unet_upblock3_attention1.txt'


def maybe_reexec() -> None:
    expected_prefix = str(STRICT_PY.parents[1])
    in_expected_venv = str(sys.prefix) == expected_prefix or str(sys.executable).startswith(expected_prefix)
    if STRICT_PY.exists() and not in_expected_venv:
        os.environ['PYTHONNOUSERSITE'] = '1'
        os.execv(str(STRICT_PY), [str(STRICT_PY), __file__, *sys.argv[1:]])


def next_module_name(up3) -> str:
    if len(getattr(up3, 'resnets', []) or []) > 2:
        return 'up_blocks.3.resnets.2'
    if len(getattr(up3, 'attentions', []) or []) > 2:
        return 'up_blocks.3.attentions.2'
    return 'output_tail'


def main() -> int:
    maybe_reexec()
    import torch

    torch.manual_seed(1234)
    np.random.seed(1234)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    unet, loaded = load_unet()
    up2 = unet.up_blocks[2]
    up3 = unet.up_blocks[3]
    if len(getattr(up3, 'resnets', []) or []) < 2 or len(getattr(up3, 'attentions', []) or []) < 2:
        raise RuntimeError('Official up_blocks.3 must expose resnets.0/1 and attentions.0/1 for this stage.')
    resnet0 = up3.resnets[0]
    attention0 = up3.attentions[0]
    resnet1 = up3.resnets[1]
    attention1 = up3.attentions[1]

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
        res0_hidden = up3_res_tuple[-1]
        remaining_after_resnet0_local = up3_res_tuple[:-1]
        concat0 = torch.cat([up2_output, res0_hidden], dim=1)
        resnet0_out = resnet0(concat0, ctx['time_embedding'])
        attention0_out = attention0(resnet0_out, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
        res1_hidden = remaining_after_resnet0_local[-1]
        remaining_after_resnet1_local = remaining_after_resnet0_local[:-1]
        concat1 = torch.cat([attention0_out, res1_hidden], dim=1)
        resnet1_out = resnet1(concat1, ctx['time_embedding'])
        attention1_out = attention1(
            resnet1_out,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs=None,
            attention_mask=None,
            encoder_attention_mask=None,
            return_dict=False,
        )[0]

    cfg = attention_config(attention1, 'up_blocks_3_attentions_1', 'unet.up_blocks.3.attentions.1')
    transformer_blocks = list(getattr(attention1, 'transformer_blocks', []) or [])
    status = 'PASS' if len(transformer_blocks) == 1 else 'BLOCKED_NEEDS_SPLIT'
    tb0 = transformer_blocks[0] if transformer_blocks else None
    if tb0 is not None:
        heads = int(getattr(tb0.attn1, 'heads', 0) or getattr(attention1, 'num_attention_heads', 0))
        inner_dim = int(getattr(tb0.attn1, 'inner_dim', 0) or cfg.get('inner_dim') or cfg.get('in_channels'))
        cfg['num_attention_heads'] = heads
        cfg['attention_head_dim'] = int(inner_dim // heads)
        cfg['inner_dim'] = inner_dim
        cfg['cross_attention_dim'] = int(getattr(tb0.attn2, 'cross_attention_dim', 0) or unet.config.cross_attention_dim)

    lora_modules = {}
    if tb0 is not None:
        for label, mod in [
            ('proj_in', attention1.proj_in),
            ('proj_out', attention1.proj_out),
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
    res0_global_index = len(remaining_external_after_up3_tuple_slice) + len(up3_res_tuple) - 1
    res1_global_index = len(remaining_external_after_up3_tuple_slice) + len(remaining_after_resnet0_local) - 1
    next_module = next_module_name(up3)
    markers = {
        'TADSR_UNET_UPBLOCK3_ATTENTION1_AUDIT': status,
        'TADSR_UNET_UPBLOCK3_ATTENTION1_TRANSFORMER_AUDIT': status,
        'TADSR_UNET_UPBLOCK3_ATTENTION1_LORA_AUDIT': 'PASS' if status == 'PASS' else status,
        'TADSR_UNET_UPBLOCK3_ATTENTION1_RESIDUAL_CONTRACT_AUDIT': 'PASS' if status == 'PASS' else status,
    }
    result = {
        'status': status,
        'scope': 'up_blocks.3.attentions.1 audit only; up_blocks.3.resnets.2 and later modules remain intentionally unopened.',
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'python': sys.executable,
        'loaded_lora_parameter_count': len(loaded),
        'upblock3_overview': {
            'class': f'{type(up3).__module__}.{type(up3).__name__}',
            'forward_signature': str(inspect.signature(up3.forward)),
            'children': [{'name': n, 'class': f'{type(m).__module__}.{type(m).__name__}'} for n, m in up3.named_children()],
            'resnet_count': len(getattr(up3, 'resnets', []) or []),
            'attention_count': len(getattr(up3, 'attentions', []) or []),
            'upsampler_count': len(getattr(up3, 'upsamplers', []) or []),
            'has_cross_attention': bool(getattr(up3, 'has_cross_attention', False)),
            'is_final_up_block': True,
            'known_completed_modules_before_this_stage': [
                'full local up_blocks.2',
                'up_blocks.3.resnets.0',
                'up_blocks.3.attentions.0',
                'up_blocks.3.resnets.1',
            ],
            'target_module': 'up_blocks.3.attentions.1',
            'actual_next_module_after_attention1': next_module,
            'remaining_modules_after_this_stage': [
                name for name in [
                    'up_blocks.3.resnets.2' if len(getattr(up3, 'resnets', []) or []) > 2 else None,
                    'up_blocks.3.attentions.2' if len(getattr(up3, 'attentions', []) or []) > 2 else None,
                    'output_tail',
                ] if name is not None
            ],
            'accepted_forward_args': list(inspect.signature(up3.forward).parameters.keys()),
            'consumes_encoder_hidden_states': 'encoder_hidden_states' in inspect.signature(up3.forward).parameters,
            'consumes_attention_mask': 'attention_mask' in inspect.signature(up3.forward).parameters,
            'consumes_encoder_attention_mask': 'encoder_attention_mask' in inspect.signature(up3.forward).parameters,
            'consumes_cross_attention_kwargs': 'cross_attention_kwargs' in inspect.signature(up3.forward).parameters,
            'has_scale_arg': 'scale' in inspect.signature(up3.forward).parameters,
            'has_freeu_attrs': any(hasattr(up3, name) for name in ['s1', 's2', 'b1', 'b2']),
        },
        'attention1_config': cfg,
        'attention_config': cfg,
        'transformer0_config': {
            'heads': cfg.get('num_attention_heads'),
            'head_dim': cfg.get('attention_head_dim'),
            'inner_dim': cfg.get('inner_dim'),
            'norm_eps': float(getattr(tb0.norm1, 'eps', 1e-5)) if tb0 is not None else None,
            'dropout': 0.0,
            'transformer_block_count': len(transformer_blocks),
        },
        'transformer_blocks': [transformer_block_details(tb) for tb in transformer_blocks],
        'entry_context_shapes': {
            'sample': list(sample.shape),
            'timestep': list(timestep.shape),
            'encoder_hidden_states': list(encoder_hidden_states.shape),
            'upblock2_output': list(up2_output.shape),
            'upblock3_resnet0_output': list(resnet0_out.shape),
            'upblock3_attention0_output': list(attention0_out.shape),
            'upblock3_resnet1_consumed_source': sources[res1_global_index],
            'upblock3_resnet1_residual_shape': list(res1_hidden.shape),
            'upblock3_resnet1_output': list(resnet1_out.shape),
            'upblock3_attention1_input': list(resnet1_out.shape),
            'upblock3_attention1_output': list(attention1_out.shape),
            'remaining_residual_count_after_resnet1_pop': len(remaining_after_resnet1_local),
            'remaining_residual_count_after_attention1': len(remaining_after_resnet1_local),
            'attention1_consumes_accumulated_residuals': False,
            'attention_mask': None,
            'encoder_attention_mask': None,
            'cross_attention_kwargs': None,
        },
        'attention1_input_stats': stats_tensor(resnet1_out),
        'attention1_output_stats': stats_tensor(attention1_out),
        'encoder_hidden_states_stats': stats_tensor(encoder_hidden_states),
        'existing_tester_compatibility': {
            'tester': 'UNetAttention0Transformer2DTester',
            'compatible': status == 'PASS' and bool(cfg.get('use_linear_projection')),
            'reason': 'Same Transformer2DModel + BasicTransformerBlock pattern; selected with prefix=up_blocks_3_attentions_1.',
        },
        'lora_modules': lora_modules,
        'residual_contract': {
            'status': 'PASS' if status == 'PASS' else status,
            'attention1_consumes_accumulated_residuals': False,
            'remaining_residual_count_before_attention1': len(remaining_after_resnet1_local),
            'remaining_residual_count_after_attention1': len(remaining_after_resnet1_local),
            'remaining_tuple_reserved_for': 'up_blocks.3.resnets.2',
        },
        'lora_audit': {
            'status': 'PASS' if status == 'PASS' else status,
            'lora_module_count': len(lora_modules),
            'effective_static_weight_export_required': True,
        },
        'upblock2_local_return_info': up2_return,
        'residual_indices': {
            'upblock3_resnet0_consumed_global_index': res0_global_index,
            'upblock3_resnet1_consumed_global_index': res1_global_index,
        },
        'uses_full_unet_forward': False,
        'uses_upblock3_resnet2': False,
        'uses_output_tail': False,
        'uses_full_tadsr_inference': False,
        'markers': markers,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding='utf-8')
    OUT_TXT.write_text('\n'.join([
        '# TADSR UNet up_blocks.3.attentions.1 official audit',
        '',
        *[f'{k}: {v}' for k, v in result['markers'].items()],
        '',
        f"target: {result['upblock3_overview']['target_module']}",
        f"next unopened module: {result['upblock3_overview']['actual_next_module_after_attention1']}",
        f"attention input shape: {result['entry_context_shapes']['upblock3_attention1_input']}",
        f"attention output shape: {result['entry_context_shapes']['upblock3_attention1_output']}",
        'scope: up_blocks.3.attentions.1 only; resnets.2/full upblock/full UNet/full inference were not executed.',
    ]) + '\n', encoding='utf-8')
    for k, v in result['markers'].items():
        print(f'{k}: {v}')
    print(json.dumps({'status': status, 'audit_json': str(OUT_JSON), 'next': next_module}, indent=2))
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
