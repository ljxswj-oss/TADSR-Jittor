#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import numpy as np

from audit_official_tadsr_unet_upblock1_local import run_context, tensor_metrics
from audit_official_tadsr_unet_upblock2_local import call_upblock2_local, manual_upblock2_chain
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
from audit_official_tadsr_unet_upblock1_local import extract_hidden

ORACLE_DIR = OUT_DIR / 'oracle_tensors_unet_upblock2_local'
META_JSON = ORACLE_DIR / 'unet_upblock2_local_oracle_metadata.json'
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
        official_raw = call_upblock2_local(up2, ctx['official_output'], up2_res_tuple, ctx['time_embedding'], encoder_hidden_states)
        official_output, official_return = extract_hidden(official_raw)
        manual = manual_upblock2_chain(up2, ctx, up2_res_tuple, encoder_hidden_states)

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
        ('manual_upblock2_resnet0_output', manual['resnet0']),
        ('manual_upblock2_attention0_output', manual['attention0']),
        ('manual_upblock2_resnet1_output', manual['resnet1']),
        ('manual_upblock2_attention1_output', manual['attention1']),
        ('manual_upblock2_resnet2_output', manual['resnet2']),
        ('manual_upblock2_attention2_output', manual['attention2']),
        ('manual_upblock2_upsampler_output', manual['upsampler']),
        ('local_upblock2_hidden_states_output', official_output),
    ]:
        save_tensor(saved, name, tensor)

    sources = source_names()
    res0_index = len(remaining_after_up1) - 1
    res1_index = len(remaining_before_up2) + len(up2_res_tuple[:-1]) - 1
    res2_index = len(remaining_before_up2) + len(up2_res_tuple[:-1][:-1]) - 1
    local_diff = tensor_metrics(manual['upsampler'], official_output)
    status = 'PASS' if local_diff['shape_match'] and local_diff['max_abs_error'] <= 1e-5 else 'FAIL'
    existing_effective_weights = {
        'upblock2_resnet0': str(OUT_DIR / 'converted_unet_upblock2_resnet0_effective_weights.npz'),
        'upblock2_attention0': str(OUT_DIR / 'converted_unet_upblock2_attention0_effective_weights.npz'),
        'upblock2_resnet1': str(OUT_DIR / 'converted_unet_upblock2_resnet1_effective_weights.npz'),
        'upblock2_attention1': str(OUT_DIR / 'converted_unet_upblock2_attention1_effective_weights.npz'),
        'upblock2_resnet2': str(OUT_DIR / 'converted_unet_upblock2_resnet2_effective_weights.npz'),
        'upblock2_attention2': str(OUT_DIR / 'converted_unet_upblock2_attention2_effective_weights.npz'),
        'upblock2_upsampler': str(OUT_DIR / 'converted_unet_upblock2_upsampler_effective_weights.npz'),
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
            'remaining_residual_count_before_upblock2': len(remaining_after_up1),
            'remaining_residual_shapes_before_upblock2': [list(x.shape) for x in remaining_after_up1],
            'local_upblock2_residual_tuple_count': len(up2_res_tuple),
            'local_upblock2_residual_tuple_shapes': [list(x.shape) for x in up2_res_tuple],
            'upblock2_resnet0_consumed_source': sources[res0_index],
            'upblock2_resnet1_consumed_source': sources[res1_index],
            'upblock2_resnet2_consumed_source': sources[res2_index],
            'attention0_consumes_residual': False,
            'attention1_consumes_residual': False,
            'attention2_consumes_residual': False,
            'upsampler_consumes_residual': False,
            'remaining_internal_residual_count_after_upblock2': len(manual['remaining_after_manual_upblock2']),
            'remaining_external_residual_count_after_upblock2': len(remaining_before_up2),
            'remaining_external_residual_shapes_after_upblock2': [list(x.shape) for x in remaining_before_up2],
            'remaining_residuals_reserved_for': ['up_blocks.3'],
        },
        'output_states_status': official_return.get('output_states_status', 'NOT_APPLICABLE'),
        'local_upblock2_call_status': status,
        'manual_chain_vs_official_local': local_diff,
        'uses_full_unet_forward': False,
        'uses_upblocks3': False,
        'uses_full_tadsr_inference': False,
        'next_module_preview': 'up_blocks.3.resnets.0',
        'markers': {
            'TADSR_UNET_UPBLOCK2_LOCAL_ORACLE_TENSORS': status,
        },
    }
    META_JSON.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    SUMMARY_TXT.write_text('\n'.join([
        '# TADSR UNet up_blocks.2 local aggregate oracle',
        '',
        f'TADSR_UNET_UPBLOCK2_LOCAL_ORACLE_TENSORS: {status}',
        '',
        f"manual vs official max_abs_error: {local_diff['max_abs_error']}",
        f"manual vs official mean_abs_error: {local_diff['mean_abs_error']}",
        f"manual vs official cosine: {local_diff['cosine_similarity']}",
        f"output_states_status: {metadata['output_states_status']}",
        'scope: local up_blocks.2 only; up_blocks.3/full UNet/full inference were not executed.',
    ]) + '\n', encoding='utf-8')
    print(f'TADSR_UNET_UPBLOCK2_LOCAL_ORACLE_TENSORS: {status}')
    print(json.dumps({'status': status, 'oracle_dir': str(ORACLE_DIR), 'metadata': str(META_JSON)}, indent=2))
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
