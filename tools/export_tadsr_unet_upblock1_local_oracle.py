#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import numpy as np

from audit_official_tadsr_unet_upblock1_local import run_context, tensor_metrics
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

ORACLE_DIR = OUT_DIR / 'oracle_tensors_unet_upblock1_local'
META_JSON = ORACLE_DIR / 'unet_upblock1_local_oracle_metadata.json'
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
    sample_shape = [1, int(unet.config.in_channels), 32, 32]
    sample = torch.linspace(-1.0, 1.0, steps=int(np.prod(sample_shape)), dtype=torch.float32).reshape(*sample_shape)
    timestep = torch.tensor([1], dtype=torch.long)
    encoder_shape = [1, 77, int(unet.config.cross_attention_dim)]
    encoder_hidden_states = torch.linspace(-0.5, 0.5, steps=int(np.prod(encoder_shape)), dtype=torch.float32).reshape(*encoder_shape)

    with torch.no_grad():
        ctx = run_context(unet, sample, timestep, encoder_hidden_states)

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
        ('manual_upblock1_resnet0_output', ctx['manual_outputs']['resnet0']),
        ('manual_upblock1_attention0_output', ctx['manual_outputs']['attention0']),
        ('manual_upblock1_resnet1_output', ctx['manual_outputs']['resnet1']),
        ('manual_upblock1_attention1_output', ctx['manual_outputs']['attention1']),
        ('manual_upblock1_resnet2_output', ctx['manual_outputs']['resnet2']),
        ('manual_upblock1_attention2_output', ctx['manual_outputs']['attention2']),
        ('manual_upblock1_upsampler_output', ctx['manual_outputs']['upsampler']),
        ('local_upblock1_hidden_states_output', ctx['official_output']),
    ]:
        save_tensor(saved, name, tensor)

    sources = source_names()
    resnet_sources = [
        sources[len(ctx['remaining_after_up0']) - 1],
        sources[len(ctx['remaining_after_up0']) - 2],
        sources[len(ctx['remaining_after_up0']) - 3],
    ]
    local_diff = tensor_metrics(ctx['manual_outputs']['upsampler'], ctx['official_output'])
    status = 'PASS' if local_diff['shape_match'] and local_diff['max_abs_error'] <= 1e-5 else 'FAIL'
    existing_effective_weights = {
        'upblock1_resnet0': str(OUT_DIR / 'converted_unet_upblock1_resnet0_effective_weights.npz'),
        'upblock1_attention0': str(OUT_DIR / 'converted_unet_upblock1_attention0_effective_weights.npz'),
        'upblock1_resnet1': str(OUT_DIR / 'converted_unet_upblock1_resnet1_effective_weights.npz'),
        'upblock1_attention1': str(OUT_DIR / 'converted_unet_upblock1_attention1_effective_weights.npz'),
        'upblock1_resnet2': str(OUT_DIR / 'converted_unet_upblock1_resnet2_effective_weights.npz'),
        'upblock1_attention2': str(OUT_DIR / 'converted_unet_upblock1_attention2_effective_weights.npz'),
        'upblock1_upsampler': str(OUT_DIR / 'converted_unet_upblock1_upsampler_effective_weights.npz'),
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
            'upblock0_consumed_sources': sources[-len(unet.up_blocks[0].resnets):],
            'remaining_residual_count_before_upblock1': len(ctx['remaining_after_up0']),
            'remaining_residual_shapes_before_upblock1': [list(x.shape) for x in ctx['remaining_after_up0']],
            'local_upblock1_residual_tuple_count': len(ctx['up1_res_tuple']),
            'local_upblock1_residual_tuple_shapes': [list(x.shape) for x in ctx['up1_res_tuple']],
            'upblock1_resnet0_consumed_source': resnet_sources[0],
            'upblock1_resnet1_consumed_source': resnet_sources[1],
            'upblock1_resnet2_consumed_source': resnet_sources[2],
            'attention0_consumes_residual': False,
            'attention1_consumes_residual': False,
            'attention2_consumes_residual': False,
            'upsampler_consumes_residual': False,
            'remaining_residual_count_after_upblock1': len(ctx['remaining_before_up1']),
            'remaining_residual_shapes_after_upblock1': [list(x.shape) for x in ctx['remaining_before_up1']],
            'remaining_residuals_reserved_for': ['up_blocks.2', 'up_blocks.3'],
        },
        'output_states_status': ctx['official_return'].get('output_states_status', 'NOT_APPLICABLE'),
        'local_upblock1_call_status': status,
        'manual_chain_vs_official_local': local_diff,
        'uses_full_unet_forward': False,
        'uses_upblocks2_3': False,
        'uses_full_tadsr_inference': False,
        'next_module_preview': 'up_blocks.2',
        'markers': {
            'TADSR_UNET_UPBLOCK1_LOCAL_ORACLE_TENSORS': status,
        },
    }
    META_JSON.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    SUMMARY_TXT.write_text('\n'.join([
        '# TADSR UNet up_blocks.1 local aggregate oracle',
        '',
        f'TADSR_UNET_UPBLOCK1_LOCAL_ORACLE_TENSORS: {status}',
        '',
        f"manual vs official max_abs_error: {local_diff['max_abs_error']}",
        f"manual vs official mean_abs_error: {local_diff['mean_abs_error']}",
        f"manual vs official cosine: {local_diff['cosine_similarity']}",
        f"output_states_status: {metadata['output_states_status']}",
        'scope: local up_blocks.1 only; up_blocks.2/full UNet/full inference were not executed.',
    ]) + '\n', encoding='utf-8')
    print(f'TADSR_UNET_UPBLOCK1_LOCAL_ORACLE_TENSORS: {status}')
    print(json.dumps({'status': status, 'oracle_dir': str(ORACLE_DIR), 'metadata': str(META_JSON)}, indent=2))
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
