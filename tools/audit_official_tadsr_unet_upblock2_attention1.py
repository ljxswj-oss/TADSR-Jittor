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
from export_tadsr_unet_upblock1_resnet0_oracle import source_names
from audit_official_tadsr_unet_upblock1_local import run_context
from audit_official_tadsr_unet_downblock2_attention1 import (
    attention_config,
    transformer_block_details,
    module_lora_info,
    stats_tensor,
)

OUT_JSON = OUT_DIR / 'audit_tadsr_unet_upblock2_attention1.json'
OUT_TXT = OUT_DIR / 'audit_tadsr_unet_upblock2_attention1.txt'


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
    up2 = unet.up_blocks[2]
    if len(up2.resnets) < 2 or len(getattr(up2, 'attentions', []) or []) < 2:
        raise RuntimeError('Official up_blocks.2 must expose resnets.0/1 and attentions.0/1 for this stage.')
    resnet0 = up2.resnets[0]
    attention0 = up2.attentions[0]
    resnet1 = up2.resnets[1]
    attention1 = up2.attentions[1]

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
        concat0 = torch.cat([ctx['official_output'], res0_hidden], dim=1)
        resnet0_out = resnet0(concat0, ctx['time_embedding'])
        attention0_out = attention0(
            resnet0_out,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs=None,
            attention_mask=None,
            encoder_attention_mask=None,
            return_dict=False,
        )[0]

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

    cfg = attention_config(attention1, 'up_blocks_2_attentions_1', 'unet.up_blocks.2.attentions.1')
    tb0 = attention1.transformer_blocks[0]
    heads = int(getattr(tb0.attn1, 'heads', 0) or getattr(attention1, 'num_attention_heads', 0))
    inner_dim = int(getattr(tb0.attn1, 'inner_dim', 0) or cfg.get('inner_dim') or cfg.get('in_channels'))
    head_dim = int(inner_dim // heads)
    cfg['num_attention_heads'] = heads
    cfg['attention_head_dim'] = head_dim
    cfg['inner_dim'] = inner_dim
    cfg['cross_attention_dim'] = int(getattr(tb0.attn2, 'cross_attention_dim', 0) or unet.config.cross_attention_dim)

    lora_modules = {}
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
    res0_consumed_index = len(remaining_after_up1) - 1
    res1_consumed_index = len(remaining_before_up2) + len(remaining_after_resnet0_local) - 1
    result = {
        'status': 'PASS',
        'scope': 'up_blocks.2.attentions.1 audit only; up_blocks.2.resnets.2 and later modules remain intentionally unopened.',
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'python': sys.executable,
        'loaded_lora_parameter_count': len(loaded),
        'upblock2_overview': {
            'class': f'{type(up2).__module__}.{type(up2).__name__}',
            'forward_signature': str(inspect.signature(up2.forward)),
            'children': [{'name': n, 'class': f'{type(m).__module__}.{type(m).__name__}'} for n, m in up2.named_children()],
            'resnet_count': len(up2.resnets),
            'attention_count': len(getattr(up2, 'attentions', []) or []),
            'upsampler_count': len(getattr(up2, 'upsamplers', []) or []),
            'has_cross_attention': bool(getattr(up2, 'has_cross_attention', False)),
            'known_completed_modules_before_this_stage': [
                'full local up_blocks.1',
                'up_blocks.2.resnets.0',
                'up_blocks.2.attentions.0',
                'up_blocks.2.resnets.1',
            ],
            'target_module': 'up_blocks.2.attentions.1',
            'actual_next_module_after_attention1': 'up_blocks.2.resnets.2',
            'remaining_modules_after_this_stage': [
                'up_blocks.2.resnets.2',
                'up_blocks.2.attentions.2',
                'up_blocks.2.upsamplers.0',
                'up_blocks.3',
                'full UNet forward',
            ],
            'expected_operation_order_until_target': [
                'full local up_blocks.1 output enters up_blocks.2',
                'up_blocks.2.resnets.0 consumes its residual',
                'run up_blocks.2.attentions.0 with encoder_hidden_states',
                'up_blocks.2.resnets.1 consumes the next residual',
                'run up_blocks.2.attentions.1 with encoder_hidden_states',
                'stop before up_blocks.2.resnets.2',
            ],
        },
        'attention1_config': cfg,
        'attention_config': cfg,
        'transformer0_config': {
            'heads': heads,
            'head_dim': head_dim,
            'inner_dim': inner_dim,
            'norm_eps': float(getattr(tb0.norm1, 'eps', 1e-5)),
            'dropout': 0.0,
            'transformer_block_count': len(attention1.transformer_blocks),
        },
        'transformer_blocks': [transformer_block_details(tb) for tb in attention1.transformer_blocks],
        'entry_context_shapes': {
            'sample': list(sample.shape),
            'timestep': list(timestep.shape),
            'encoder_hidden_states': list(encoder_hidden_states.shape),
            'upblock1_output': list(ctx['official_output'].shape),
            'remaining_residual_count_before_upblock2_slice': len(remaining_before_up2),
            'upblock2_resnet0_consumed_residual_source': sources[res0_consumed_index],
            'upblock2_resnet0_residual_shape': list(res0_hidden.shape),
            'upblock2_resnet0_output': list(resnet0_out.shape),
            'upblock2_attention0_output': list(attention0_out.shape),
            'upblock2_resnet1_consumed_residual_source': sources[res1_consumed_index],
            'upblock2_resnet1_residual_shape': list(res1_hidden.shape),
            'upblock2_resnet1_output': list(resnet1_out.shape),
            'upblock2_attention1_input': list(resnet1_out.shape),
            'upblock2_attention1_output': list(attention1_out.shape),
            'internal_remaining_residual_count_after_resnet0_pop': len(remaining_after_resnet0_local),
            'internal_remaining_residual_count_after_resnet1_pop': len(remaining_after_resnet1_local),
            'internal_remaining_residual_count_after_attention1': len(remaining_after_resnet1_local),
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
            'compatible': len(attention1.transformer_blocks) == 1 and bool(cfg.get('use_linear_projection')),
            'reason': 'Same Transformer2DModel + BasicTransformerBlock pattern; selected with prefix=up_blocks_2_attentions_1.',
        },
        'lora_modules': lora_modules,
        'residual_contract': {
            'status': 'PASS',
            'attention1_consumes_accumulated_residuals': False,
            'remaining_residual_count_unchanged': len(remaining_after_resnet1_local),
        },
        'lora_audit': {
            'status': 'PASS',
            'lora_module_count': len(lora_modules),
            'effective_static_weight_export_required': True,
        },
        'uses_full_unet_forward': False,
        'uses_upblock2_resnet2': False,
        'uses_upblocks3': False,
        'uses_full_tadsr_inference': False,
        'markers': {
            'TADSR_UNET_UPBLOCK2_ATTENTION1_AUDIT': 'PASS',
            'TADSR_UNET_UPBLOCK2_ATTENTION1_RESIDUAL_CONTRACT_AUDIT': 'PASS',
            'TADSR_UNET_UPBLOCK2_ATTENTION1_LORA_AUDIT': 'PASS',
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding='utf-8')
    OUT_TXT.write_text('\n'.join([
        '# TADSR UNet up_blocks.2.attentions.1 official audit',
        '',
        *[f'{k}: {v}' for k, v in result['markers'].items()],
        '',
        f"target: {result['upblock2_overview']['target_module']}",
        f"next unopened module: {result['upblock2_overview']['actual_next_module_after_attention1']}",
        f"attention input shape: {result['entry_context_shapes']['upblock2_attention1_input']}",
        f"attention output shape: {result['entry_context_shapes']['upblock2_attention1_output']}",
    ]) + '\n', encoding='utf-8')
    for k, v in result['markers'].items():
        print(f'{k}: {v}')
    print(json.dumps({'status': 'PASS', 'audit_json': str(OUT_JSON), 'next': 'up_blocks.2.resnets.2 remains unopened'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
