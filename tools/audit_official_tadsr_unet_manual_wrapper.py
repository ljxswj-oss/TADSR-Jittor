#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
import os
import sys
import numpy as np

from audit_official_tadsr_unet_upblock1_local import run_context, tensor_metrics, extract_hidden
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

OUT_JSON = OUT_DIR / 'audit_tadsr_unet_manual_wrapper.json'
OUT_TXT = OUT_DIR / 'audit_tadsr_unet_manual_wrapper.txt'
PREV_OUTPUT_TAIL = OUT_DIR / 'oracle_tensors_unet_output_tail' / 'entry_output_tail_conv_out_output.npy'


def maybe_reexec() -> None:
    expected_prefix = str(STRICT_PY.parents[1])
    in_expected_venv = str(sys.prefix) == expected_prefix or str(sys.executable).startswith(expected_prefix)
    if STRICT_PY.exists() and not in_expected_venv:
        os.environ['PYTHONNOUSERSITE'] = '1'
        os.execv(str(STRICT_PY), [str(STRICT_PY), __file__, *sys.argv[1:]])


def np_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    diff = np.abs(aa - bb)
    aa64 = aa.astype(np.float64).reshape(-1)
    bb64 = bb.astype(np.float64).reshape(-1)
    denom = float(np.linalg.norm(aa64) * np.linalg.norm(bb64))
    cosine = float(np.dot(aa64, bb64) / denom) if denom else 1.0
    return {
        'shape_match': list(aa.shape) == list(bb.shape),
        'shape': list(aa.shape),
        'max_abs_error': float(diff.max()) if diff.size else 0.0,
        'mean_abs_error': float(diff.mean()) if diff.size else 0.0,
        'cosine_similarity': cosine,
    }


def run_tail(unet, hidden):
    norm = unet.conv_norm_out(hidden)
    act = unet.conv_act(norm)
    conv = unet.conv_out(act)
    return {'norm': norm, 'act': act, 'conv_out': conv}


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
    up2 = unet.up_blocks[2]
    up3 = unet.up_blocks[3]

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
        up3_res_tuple = remaining_before_up2[-len(up3.resnets):]
        up3_raw = call_upblock3_local(up3, up2_output, up3_res_tuple, ctx['time_embedding'], encoder_hidden_states)
        up3_output, up3_return = extract_hidden(up3_raw)
        tail = run_tail(unet, up3_output)

    previous_compare = {'previous_output_tail_oracle_available': False}
    if PREV_OUTPUT_TAIL.exists():
        previous_compare = {
            'previous_output_tail_oracle_available': True,
            **np_metrics(to_np(tail['conv_out']), np.load(PREV_OUTPUT_TAIL)),
        }

    final_shape = list(tail['conv_out'].shape)
    input_shape_status = 'PASS' if list(sample.shape) == sample_shape and list(encoder_hidden_states.shape) == encoder_shape else 'FAIL'
    output_contract_status = 'PASS' if int(tail['conv_out'].shape[1]) == int(unet.config.out_channels) and final_shape[-2:] == sample_shape[-2:] else 'FAIL'
    previous_status = 'PASS' if (not previous_compare.get('previous_output_tail_oracle_available') or previous_compare.get('max_abs_error', 0.0) == 0.0) else 'FAIL'
    status = 'PASS' if input_shape_status == output_contract_status == previous_status == 'PASS' else 'FAIL'

    down_res_counts = [1]
    total = 1
    for states in ctx['down_states']:
        total += len(states)
        down_res_counts.append(total)
    residual_counts = {
        'after_down_blocks': len(ctx['down_res']),
        'before_upblock0': len(ctx['down_res']),
        'after_upblock0': len(ctx['remaining_after_up0']),
        'before_upblock1': len(ctx['up1_res_tuple']),
        'after_upblock1': len(ctx['remaining_before_up1']),
        'before_upblock2': len(up2_res_tuple),
        'after_upblock2': len(remaining_before_up2),
        'before_upblock3': len(up3_res_tuple),
        'after_upblock3': 0,
    }

    result = {
        'status': status,
        'python': sys.executable,
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'loaded_lora_parameter_count': len(loaded),
        'input_contract': {
            'sample_shape': list(sample.shape),
            'sample_dtype': str(sample.dtype),
            'timestep_shape': list(timestep.shape),
            'timestep_dtype': str(timestep.dtype),
            'timestep_value': to_np(timestep).astype(int).tolist(),
            'encoder_hidden_states_shape': list(encoder_hidden_states.shape),
            'encoder_hidden_states_dtype': str(encoder_hidden_states.dtype),
            'center_input_sample': bool(unet.config.center_input_sample),
            'attention_mask': None,
            'encoder_attention_mask': None,
            'cross_attention_kwargs': None,
            'class_labels_unused': True,
            'added_cond_kwargs_unused': True,
            'return_dict_irrelevant_for_manual_wrapper': True,
            'lora_scale_strategy': 'static effective weights in Jittor tests; no runtime LoRA',
        },
        'manual_chain_order': [
            'center_input_sample_if_enabled',
            'conv_in',
            'time_proj',
            'time_embedding',
            'down_blocks.0',
            'down_blocks.1',
            'down_blocks.2',
            'down_blocks.3',
            'mid_block',
            'up_blocks.0',
            'up_blocks.1',
            'up_blocks.2',
            'up_blocks.3',
            'conv_norm_out',
            'conv_act',
            'conv_out',
        ],
        'stage_shapes': {
            'centered': list(ctx['centered'].shape),
            'conv_in': list(ctx['conv_in'].shape),
            'time_proj': list(ctx['time_proj'].shape),
            'time_embedding': list(ctx['time_embedding'].shape),
            'down_blocks': [list(x.shape) for x in ctx['down_hiddens']],
            'mid_block': list(ctx['mid_output'].shape),
            'upblock0': list(ctx['up0_output'].shape),
            'upblock1': list(ctx['official_output'].shape),
            'upblock2': list(up2_output.shape),
            'upblock3': list(up3_output.shape),
            'output_tail_norm': list(tail['norm'].shape),
            'output_tail_act': list(tail['act'].shape),
            'output': final_shape,
        },
        'residual_tuple_counts': residual_counts,
        'down_residual_cumulative_counts': down_res_counts,
        'upblock2_return_info': up2_return,
        'upblock3_return_info': up3_return,
        'previous_output_tail_compare': previous_compare,
        'wrapper_output_contract': {
            'status': output_contract_status,
            'output_shape': final_shape,
            'expected_out_channels': int(unet.config.out_channels),
            'spatial_matches_sample': final_shape[-2:] == sample_shape[-2:],
            'return_format': 'tensor for final output; optional intermediates for audit/debug',
        },
        'official_forward_signature': str(inspect.signature(unet.forward)),
        'official_full_forward_executed': False,
        'full_tadsr_inference_executed': False,
        'manual_wrapper_only': True,
        'manual_wrapper_alignment_is_not_official_full_forward': True,
        'reason_official_full_forward_deferred': 'deferred to next stage after manual wrapper alignment',
        'markers': {
            'TADSR_UNET_MANUAL_FULL_WRAPPER_AUDIT': status,
            'TADSR_UNET_MANUAL_FULL_WRAPPER_CONTRACT_AUDIT': 'PASS' if input_shape_status == output_contract_status == 'PASS' else 'FAIL',
            'TADSR_UNET_MANUAL_FULL_CHAIN_TO_TAIL_AUDIT': status,
        },
        'scope_boundaries': {
            'ported_this_stage': 'manual full-chain wrapper contract/audit only',
            'not_ported_this_stage': ['official unet.forward', 'production Jittor UNet forward', 'scheduler denoising', 'VAE integration', 'image generation', 'generic runtime LoRA'],
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, default=str), encoding='utf-8')
    OUT_TXT.write_text('\n'.join([
        '# TADSR UNet manual full-chain wrapper audit',
        '',
        *[f'{k}: {v}' for k, v in result['markers'].items()],
        '',
        f"manual chain: {' -> '.join(result['manual_chain_order'])}",
        f"output shape: {final_shape}",
        f"previous output-tail compare: {previous_compare}",
        'scope: manual wrapper only; official unet.forward/full inference were not executed.',
    ]) + '\n', encoding='utf-8')
    for k, v in result['markers'].items():
        print(f'{k}: {v}')
    print(json.dumps({'status': status, 'metadata': str(OUT_JSON)}, indent=2))
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
