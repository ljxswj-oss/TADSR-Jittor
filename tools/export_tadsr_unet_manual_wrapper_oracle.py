#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import numpy as np

from audit_official_tadsr_unet_upblock1_local import run_context, extract_hidden
from audit_official_tadsr_unet_upblock2_local import call_upblock2_local
from audit_official_tadsr_unet_upblock3_local import call_upblock3_local
from audit_official_tadsr_unet_manual_wrapper import np_metrics, run_tail
from export_tadsr_unet_downblock3_resnet0_oracle import (
    STRICT_PY,
    OUT_DIR,
    OFFICIAL_REPO,
    WEIGHTS_DIR,
    load_unet,
    to_np,
    stats,
)

ORACLE_DIR = OUT_DIR / 'oracle_tensors_unet_manual_wrapper'
META_JSON = ORACLE_DIR / 'unet_manual_wrapper_oracle_metadata.json'
SUMMARY_TXT = ORACLE_DIR / 'oracle_summary.txt'
PREV_OUTPUT_TAIL = OUT_DIR / 'oracle_tensors_unet_output_tail' / 'entry_output_tail_conv_out_output.npy'


def maybe_reexec() -> None:
    expected_prefix = str(STRICT_PY.parents[1])
    in_expected_venv = str(sys.prefix) == expected_prefix or str(sys.executable).startswith(expected_prefix)
    if STRICT_PY.exists() and not in_expected_venv:
        os.environ['PYTHONNOUSERSITE'] = '1'
        os.execv(str(STRICT_PY), [str(STRICT_PY), __file__, *sys.argv[1:]])


def save_tensor(saved: dict, name: str, tensor):
    arr = to_np(tensor)
    np.save(ORACLE_DIR / f'{name}.npy', arr)
    saved[name] = stats(arr)
    return arr


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
    ORACLE_DIR.mkdir(parents=True, exist_ok=True)
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

    saved = {}
    for name, tensor in [
        ('manual_wrapper_sample', sample),
        ('manual_wrapper_timestep', timestep),
        ('manual_wrapper_encoder_hidden_states', encoder_hidden_states),
        ('manual_wrapper_sample_after_center', ctx['centered']),
        ('manual_wrapper_conv_in_output', ctx['conv_in']),
        ('manual_wrapper_time_proj_output', ctx['time_proj']),
        ('manual_wrapper_time_embedding_output', ctx['time_embedding']),
    ]:
        save_tensor(saved, name, tensor)
    for i, (hidden, states) in enumerate(zip(ctx['down_hiddens'], ctx['down_states'])):
        save_tensor(saved, f'manual_wrapper_downblock{i}_output', hidden)
        for j, state in enumerate(states):
            save_tensor(saved, f'manual_wrapper_downblock{i}_state_{j}', state)
    for name, tensor in [
        ('manual_wrapper_midblock_output', ctx['mid_output']),
        ('manual_wrapper_upblock0_output', ctx['up0_output']),
        ('manual_wrapper_upblock1_output', ctx['official_output']),
        ('manual_wrapper_upblock2_output', up2_output),
        ('manual_wrapper_upblock3_output', up3_output),
        ('manual_wrapper_output_tail_norm_output', tail['norm']),
        ('manual_wrapper_output_tail_act_output', tail['act']),
        ('manual_wrapper_output', tail['conv_out']),
    ]:
        save_tensor(saved, name, tensor)

    previous_compare = {'previous_output_tail_oracle_available': False}
    if PREV_OUTPUT_TAIL.exists():
        previous_compare = {
            'previous_output_tail_oracle_available': True,
            **np_metrics(to_np(tail['conv_out']), np.load(PREV_OUTPUT_TAIL)),
        }

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
    previous_status = 'PASS' if (not previous_compare.get('previous_output_tail_oracle_available') or previous_compare.get('max_abs_error', 0.0) == 0.0) else 'FAIL'
    status = 'PASS' if previous_status == 'PASS' else 'FAIL'
    metadata = {
        'status': status,
        'python': sys.executable,
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'oracle_dir': str(ORACLE_DIR),
        'loaded_lora_parameter_count': len(loaded),
        'sample_shape': list(sample.shape),
        'sample_dtype': str(sample.dtype),
        'timestep_shape': list(timestep.shape),
        'timestep_dtype': str(timestep.dtype),
        'timestep_value': to_np(timestep).astype(int).tolist(),
        'encoder_hidden_states_shape': list(encoder_hidden_states.shape),
        'encoder_hidden_states_dtype': str(encoder_hidden_states.dtype),
        'stage_shapes': {
            'conv_in': list(ctx['conv_in'].shape),
            'time_embedding': list(ctx['time_embedding'].shape),
            'down_blocks': [list(x.shape) for x in ctx['down_hiddens']],
            'mid_block': list(ctx['mid_output'].shape),
            'upblock0': list(ctx['up0_output'].shape),
            'upblock1': list(ctx['official_output'].shape),
            'upblock2': list(up2_output.shape),
            'upblock3': list(up3_output.shape),
            'output_tail_norm': list(tail['norm'].shape),
            'output_tail_act': list(tail['act'].shape),
            'output': list(tail['conv_out'].shape),
        },
        'residual_tuple_counts': residual_counts,
        'upblock2_return_info': up2_return,
        'upblock3_return_info': up3_return,
        'output_tail_config_summary': {
            'conv_norm_out_num_groups': int(unet.conv_norm_out.num_groups),
            'conv_norm_out_eps': float(unet.conv_norm_out.eps),
            'conv_out_padding': list((unet.conv_out.base_layer if hasattr(unet.conv_out, 'base_layer') else unet.conv_out).padding),
            'conv_out_stride': list((unet.conv_out.base_layer if hasattr(unet.conv_out, 'base_layer') else unet.conv_out).stride),
        },
        'existing_effective_weight_paths': {
            'entry_and_down_mid_up': 'experiments/full_repro/unet_alignment/converted_unet_*_effective_weights.npz files from previous stages',
            'output_tail': 'experiments/full_repro/unet_alignment/converted_unet_output_tail_effective_weights.npz',
        },
        'saved_tensors': saved,
        'previous_output_tail_oracle_available': previous_compare.get('previous_output_tail_oracle_available', False),
        'previous_output_tail_max_abs_diff': previous_compare.get('max_abs_error'),
        'previous_output_tail_compare': previous_compare,
        'official_full_forward_executed': False,
        'full_tadsr_inference_executed': False,
        'manual_wrapper_only': True,
        'recommended_next_stage': 'official full unet.forward alignment after manual wrapper alignment',
        'markers': {
            'TADSR_UNET_MANUAL_FULL_WRAPPER_ORACLE_TENSORS': status,
        },
    }
    META_JSON.write_text(json.dumps(metadata, indent=2, default=str), encoding='utf-8')
    SUMMARY_TXT.write_text('\n'.join([
        '# TADSR UNet manual full-chain wrapper oracle export',
        '',
        *[f'{k}: {v}' for k, v in metadata['markers'].items()],
        '',
        f"output shape: {metadata['stage_shapes']['output']}",
        f"previous output-tail compare: {previous_compare}",
        'scope: manual wrapper oracle only; official unet.forward/full inference were not executed.',
    ]) + '\n', encoding='utf-8')
    for k, v in metadata['markers'].items():
        print(f'{k}: {v}')
    print(json.dumps({'status': status, 'oracle_dir': str(ORACLE_DIR), 'metadata': str(META_JSON)}, indent=2))
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
