#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import numpy as np

from audit_official_tadsr_unet_upblock1_local import run_context, tensor_metrics, extract_hidden
from audit_official_tadsr_unet_upblock2_local import call_upblock2_local
from audit_official_tadsr_unet_upblock3_local import call_upblock3_local, manual_upblock3_chain, tail_boundary_preview
from export_tadsr_unet_downblock3_resnet0_oracle import (
    STRICT_PY,
    OUT_DIR,
    OFFICIAL_REPO,
    WEIGHTS_DIR,
    load_unet,
    to_np,
    stats,
)
from export_tadsr_unet_upblock1_resnet0_oracle import source_names

ORACLE_DIR = OUT_DIR / 'oracle_tensors_unet_upblock3_local'
META_JSON = ORACLE_DIR / 'unet_upblock3_local_oracle_metadata.json'
SUMMARY_TXT = ORACLE_DIR / 'oracle_summary.txt'


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
        remaining_before_up3 = remaining_before_up2
        up3_res_tuple = remaining_before_up3[-len(up3.resnets):]
        remaining_external_after_up3_tuple_slice = remaining_before_up3[:-len(up3.resnets)]
        official_raw = call_upblock3_local(up3, up2_output, up3_res_tuple, ctx['time_embedding'], encoder_hidden_states)
        official_output, official_return = extract_hidden(official_raw)
        manual = manual_upblock3_chain(up3, up2_output, up3_res_tuple, ctx['time_embedding'], encoder_hidden_states)

    saved = {}
    for name, tensor in [
        ('entry_synthetic_unet_sample', sample),
        ('entry_synthetic_unet_timestep', timestep),
        ('entry_encoder_hidden_states', encoder_hidden_states),
        ('entry_synthetic_unet_sample_after_center', ctx['centered']),
        ('entry_synthetic_unet_conv_in_output', ctx['conv_in']),
        ('entry_synthetic_unet_time_proj_output', ctx['time_proj']),
        ('entry_synthetic_unet_time_embedding_output', ctx['time_embedding']),
    ]:
        save_tensor(saved, name, tensor)
    for i, (hidden, states) in enumerate(zip(ctx['down_hiddens'], ctx['down_states'])):
        save_tensor(saved, f'entry_downblock{i}_output_hidden', hidden)
        for j, state in enumerate(states):
            save_tensor(saved, f'entry_downblock{i}_output_state_{j}', state)
    for name, tensor in [
        ('entry_midblock_output_hidden', ctx['mid_output']),
        ('entry_upblock0_output_hidden', ctx['up0_output']),
        ('entry_upblock1_output_hidden', ctx['official_output']),
        ('entry_upblock2_output_hidden', up2_output),
        ('manual_upblock3_resnet0_output', manual['resnet0']),
        ('manual_upblock3_attention0_output', manual['attention0']),
        ('manual_upblock3_resnet1_output', manual['resnet1']),
        ('manual_upblock3_attention1_output', manual['attention1']),
        ('manual_upblock3_resnet2_output', manual['resnet2']),
        ('manual_upblock3_attention2_output', manual['attention2']),
        ('local_upblock3_hidden_states_output', official_output),
    ]:
        save_tensor(saved, name, tensor)

    local_diff = tensor_metrics(manual['attention2'], official_output)
    status = 'PASS' if local_diff['shape_match'] and local_diff['max_abs_error'] <= 1e-5 else 'FAIL'
    output_states_status = official_return.get('output_states_status', 'NOT_APPLICABLE')
    residual_status = 'PASS' if len(manual['remaining_after_manual_upblock3']) == 0 and len(remaining_external_after_up3_tuple_slice) == 0 else 'FAIL'
    sources = source_names()
    res0_index = len(remaining_external_after_up3_tuple_slice) + len(up3_res_tuple) - 1
    res1_index = len(remaining_external_after_up3_tuple_slice) + len(up3_res_tuple[:-1]) - 1
    res2_index = len(remaining_external_after_up3_tuple_slice) + len(up3_res_tuple[:-1][:-1]) - 1
    output_tail_boundary = tail_boundary_preview(unet, official_output)
    existing_effective_weights = {
        'upblock3_resnet0': str(OUT_DIR / 'converted_unet_upblock3_resnet0_effective_weights.npz'),
        'upblock3_attention0': str(OUT_DIR / 'converted_unet_upblock3_attention0_effective_weights.npz'),
        'upblock3_resnet1': str(OUT_DIR / 'converted_unet_upblock3_resnet1_effective_weights.npz'),
        'upblock3_attention1': str(OUT_DIR / 'converted_unet_upblock3_attention1_effective_weights.npz'),
        'upblock3_resnet2': str(OUT_DIR / 'converted_unet_upblock3_resnet2_effective_weights.npz'),
        'upblock3_attention2': str(OUT_DIR / 'converted_unet_upblock3_attention2_effective_weights.npz'),
    }
    metadata = {
        'status': status,
        'python': sys.executable,
        'strict_python': str(STRICT_PY),
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'loaded_lora_parameter_count': len(loaded),
        'oracle_dir': str(ORACLE_DIR),
        'entry_sample_shape': sample_shape,
        'selected_timestep': [1],
        'encoder_hidden_states_shape': encoder_shape,
        'saved_tensors': saved,
        'existing_effective_weight_paths': existing_effective_weights,
        'residual_metadata': {
            'accumulated_down_block_res_sample_count': len(ctx['down_res']),
            'accumulated_down_block_res_sample_sources': sources,
            'accumulated_down_block_res_sample_shapes': [list(x.shape) for x in ctx['down_res']],
            'remaining_residual_count_before_upblock3': len(remaining_before_up3),
            'remaining_residual_shapes_before_upblock3': [list(x.shape) for x in remaining_before_up3],
            'local_upblock3_residual_tuple_count': len(up3_res_tuple),
            'local_upblock3_residual_tuple_shapes': [list(x.shape) for x in up3_res_tuple],
            'upblock3_resnet0_consumed_source': sources[res0_index],
            'upblock3_resnet0_consumed_shape': list(manual['resnet0_residual'].shape),
            'upblock3_resnet1_consumed_source': sources[res1_index],
            'upblock3_resnet1_consumed_shape': list(manual['resnet1_residual'].shape),
            'upblock3_resnet2_consumed_source': sources[res2_index],
            'upblock3_resnet2_consumed_shape': list(manual['resnet2_residual'].shape),
            'attention0_consumes_residual': False,
            'attention1_consumes_residual': False,
            'attention2_consumes_residual': False,
            'upsampler_count': len(getattr(up3, 'upsamplers', []) or []),
            'remaining_internal_residual_count_after_upblock3': len(manual['remaining_after_manual_upblock3']),
            'remaining_external_residual_count_after_upblock3': len(remaining_external_after_up3_tuple_slice),
            'remaining_external_residual_shapes_after_upblock3': [list(x.shape) for x in remaining_external_after_up3_tuple_slice],
        },
        'output_states_status': output_states_status,
        'local_upblock3_call_status': status,
        'manual_chain_vs_official_local': local_diff,
        'upblock2_local_return_info': up2_return,
        'output_tail_boundary_status': output_tail_boundary['status'],
        'output_tail_boundary_preview': output_tail_boundary,
        'uses_full_unet_forward': False,
        'uses_output_tail': False,
        'uses_full_tadsr_inference': False,
        'recommended_next_stage': output_tail_boundary['recommended_next_stage'],
        'markers': {
            'TADSR_UNET_UPBLOCK3_LOCAL_ORACLE_TENSORS': status,
            'TADSR_UNET_UPBLOCK3_RESIDUAL_CONTRACT_ORACLE': residual_status,
            'TADSR_UNET_OUTPUT_TAIL_BOUNDARY_AUDIT': output_tail_boundary['status'],
        },
    }
    META_JSON.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    SUMMARY_TXT.write_text('\n'.join([
        '# TADSR UNet up_blocks.3 local aggregate oracle',
        '',
        *[f'{k}: {v}' for k, v in metadata['markers'].items()],
        '',
        f"manual vs official max_abs_error: {local_diff['max_abs_error']}",
        f"manual vs official mean_abs_error: {local_diff['mean_abs_error']}",
        f"manual vs official cosine: {local_diff['cosine_similarity']}",
        f"output_states_status: {output_states_status}",
        f"output-tail boundary: {output_tail_boundary['actual_next_logical_stage_after_upblock3']}",
        'scope: local up_blocks.3 only; output tail/full UNet/full inference were not executed.',
    ]) + '\n', encoding='utf-8')
    for k, v in metadata['markers'].items():
        print(f'{k}: {v}')
    print(json.dumps({'status': status, 'oracle_dir': str(ORACLE_DIR), 'metadata': str(META_JSON)}, indent=2))
    return 0 if status == 'PASS' and residual_status == 'PASS' and output_tail_boundary['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
