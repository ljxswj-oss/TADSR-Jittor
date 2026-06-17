#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
import os
import sys
import numpy as np

from audit_official_tadsr_unet_manual_wrapper import np_metrics, run_tail
from audit_official_tadsr_unet_upblock1_local import run_context, extract_hidden
from audit_official_tadsr_unet_upblock2_local import call_upblock2_local
from audit_official_tadsr_unet_upblock3_local import call_upblock3_local
from export_tadsr_unet_downblock3_resnet0_oracle import (
    STRICT_PY,
    OUT_DIR,
    OFFICIAL_REPO,
    WEIGHTS_DIR,
    load_unet,
    to_np,
    stats,
)

OUT_JSON = OUT_DIR / 'audit_tadsr_unet_full_forward.json'
OUT_TXT = OUT_DIR / 'audit_tadsr_unet_full_forward.txt'


def maybe_reexec() -> None:
    expected_prefix = str(STRICT_PY.parents[1])
    in_expected_venv = str(sys.prefix) == expected_prefix or str(sys.executable).startswith(expected_prefix)
    if STRICT_PY.exists() and not in_expected_venv:
        os.environ['PYTHONNOUSERSITE'] = '1'
        os.execv(str(STRICT_PY), [str(STRICT_PY), __file__, *sys.argv[1:]])


def full_forward_output(obj):
    if hasattr(obj, 'sample'):
        return obj.sample, {'return_type': type(obj).__name__, 'extraction': 'sample'}
    if isinstance(obj, tuple):
        return obj[0], {'return_type': 'tuple', 'extraction': 'tuple[0]', 'tuple_len': len(obj)}
    return obj, {'return_type': type(obj).__name__, 'extraction': 'identity'}


def manual_chain_output(unet, sample, timestep, encoder_hidden_states):
    up2 = unet.up_blocks[2]
    up3 = unet.up_blocks[3]
    ctx = run_context(unet, sample, timestep, encoder_hidden_states)
    remaining_after_up1 = ctx['remaining_before_up1']
    up2_res_tuple = remaining_after_up1[-len(up2.resnets):]
    remaining_before_up2 = remaining_after_up1[:-len(up2.resnets)]
    up2_raw = call_upblock2_local(up2, ctx['official_output'], up2_res_tuple, ctx['time_embedding'], encoder_hidden_states)
    up2_output, up2_return = extract_hidden(up2_raw)
    up3_res_tuple = remaining_before_up2[-len(up3.resnets):]
    up3_raw = call_upblock3_local(up3, up2_output, up3_res_tuple, ctx['time_embedding'], encoder_hidden_states)
    up3_output, up3_return = extract_hidden(up3_raw)
    tail = run_tail(unet, up3_output)
    return tail['conv_out'], {
        'context': ctx,
        'up2_output': up2_output,
        'up3_output': up3_output,
        'up2_return_info': up2_return,
        'up3_return_info': up3_return,
    }


def main() -> int:
    maybe_reexec()
    import torch

    torch.manual_seed(1234)
    np.random.seed(1234)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    unet, loaded = load_unet()
    unet.eval()

    sample_shape = [1, int(unet.config.in_channels), 32, 32]
    sample = torch.linspace(-1.0, 1.0, steps=int(np.prod(sample_shape)), dtype=torch.float32).reshape(*sample_shape)
    timestep = torch.tensor([1], dtype=torch.long)
    encoder_shape = [1, 77, int(unet.config.cross_attention_dim)]
    encoder_hidden_states = torch.linspace(-0.5, 0.5, steps=int(np.prod(encoder_shape)), dtype=torch.float32).reshape(*encoder_shape)

    with torch.no_grad():
        official_obj = unet(sample, timestep, encoder_hidden_states=encoder_hidden_states, return_dict=True)
        official, official_return = full_forward_output(official_obj)
        tuple_obj = unet(sample, timestep, encoder_hidden_states=encoder_hidden_states, return_dict=False)
        tuple_output, tuple_return = full_forward_output(tuple_obj)
        manual, manual_meta = manual_chain_output(unet, sample, timestep, encoder_hidden_states)

    manual_vs_official = np_metrics(to_np(manual), to_np(official))
    tuple_vs_official = np_metrics(to_np(tuple_output), to_np(official))
    status = 'PASS' if manual_vs_official['shape_match'] and manual_vs_official['max_abs_error'] <= 1e-5 and tuple_vs_official['max_abs_error'] <= 1e-5 else 'FAIL'
    result = {
        'status': status,
        'python': sys.executable,
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'loaded_lora_parameter_count': len(loaded),
        'official_forward_signature': str(inspect.signature(unet.forward)),
        'input_contract': {
            'sample_shape': list(sample.shape),
            'sample_dtype': str(sample.dtype),
            'timestep_shape': list(timestep.shape),
            'timestep_dtype': str(timestep.dtype),
            'timestep_value': to_np(timestep).astype(int).tolist(),
            'timestep_supported_types_checked': ['batch tensor'],
            'encoder_hidden_states_shape': list(encoder_hidden_states.shape),
            'encoder_hidden_states_dtype': str(encoder_hidden_states.dtype),
            'class_labels_required': False,
            'timestep_cond_required': False,
            'attention_mask': None,
            'encoder_attention_mask': None,
            'cross_attention_kwargs': None,
            'added_cond_kwargs_required': False,
            'down_block_additional_residuals': None,
            'mid_block_additional_residual': None,
            'down_intrablock_additional_residuals': None,
            'center_input_sample': bool(unet.config.center_input_sample),
            'in_channels': int(unet.config.in_channels),
            'out_channels': int(unet.config.out_channels),
            'sample_size': unet.config.sample_size,
            'cross_attention_dim': int(unet.config.cross_attention_dim),
        },
        'return_dict_behavior': {
            'return_dict_true': official_return,
            'return_dict_false': tuple_return,
            'tuple_matches_return_dict': tuple_vs_official,
        },
        'official_output_stats': stats(to_np(official)),
        'manual_output_stats': stats(to_np(manual)),
        'manual_vs_official_full_forward': manual_vs_official,
        'official_output_shape': list(official.shape),
        'manual_wrapper_output_shape': list(manual.shape),
        'upblock2_return_info': manual_meta['up2_return_info'],
        'upblock3_return_info': manual_meta['up3_return_info'],
        'official_full_forward_executed': True,
        'scheduler_executed': False,
        'vae_executed': False,
        'full_tadsr_inference_executed': False,
        'runtime_lora_implemented': False,
        'markers': {
            'TADSR_UNET_OFFICIAL_FULL_FORWARD_AUDIT': status,
            'TADSR_UNET_FULL_FORWARD_CONTRACT_AUDIT': 'PASS',
            'TADSR_UNET_MANUAL_VS_OFFICIAL_FULL_FORWARD_AUDIT': status,
        },
        'scope_boundaries': {
            'ported_this_stage': 'official UNet full forward oracle/contract audit only',
            'not_ported_this_stage': ['full TADSR inference', 'scheduler denoising loop', 'VAE pipeline', 'image generation', 'generic runtime LoRA'],
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, default=str), encoding='utf-8')
    OUT_TXT.write_text('\n'.join([
        '# TADSR UNet official full forward audit',
        '',
        *[f'{k}: {v}' for k, v in result['markers'].items()],
        '',
        f"official output shape: {result['official_output_shape']}",
        f"manual vs official: {manual_vs_official}",
        'scope: official UNet.forward was called only in tools; scheduler/VAE/full TADSR inference were not executed.',
    ]) + '\n', encoding='utf-8')
    for k, v in result['markers'].items():
        print(f'{k}: {v}')
    print(json.dumps({'status': status, 'metadata': str(OUT_JSON)}, indent=2))
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
