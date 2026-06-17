#!/usr/bin/env python3
from __future__ import annotations

import json
import inspect
import os
import sys
import numpy as np

from audit_official_tadsr_unet_full_forward import full_forward_output, manual_chain_output
from audit_official_tadsr_unet_manual_wrapper import np_metrics
from export_tadsr_unet_downblock3_resnet0_oracle import (
    STRICT_PY,
    OUT_DIR,
    OFFICIAL_REPO,
    WEIGHTS_DIR,
    load_unet,
    to_np,
    stats,
)

ORACLE_DIR = OUT_DIR / 'oracle_tensors_unet_full_forward'
META_JSON = ORACLE_DIR / 'unet_full_forward_oracle_metadata.json'
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

    saved = {}
    save_tensor(saved, 'full_forward_sample', sample)
    save_tensor(saved, 'full_forward_timestep', timestep)
    save_tensor(saved, 'full_forward_encoder_hidden_states', encoder_hidden_states)
    save_tensor(saved, 'official_full_forward_output', official)
    save_tensor(saved, 'manual_wrapper_output', manual)
    save_tensor(saved, 'official_full_forward_tuple_output', tuple_output)

    manual_vs_official = np_metrics(to_np(manual), to_np(official))
    tuple_vs_official = np_metrics(to_np(tuple_output), to_np(official))
    manual_status = 'PASS' if manual_vs_official['shape_match'] and manual_vs_official['max_abs_error'] <= 1e-5 else 'FAIL'
    tuple_status = 'PASS' if tuple_vs_official['shape_match'] and tuple_vs_official['max_abs_error'] <= 1e-5 else 'FAIL'
    status = 'PASS' if manual_status == tuple_status == 'PASS' else 'FAIL'
    metadata = {
        'status': status,
        'python': sys.executable,
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'oracle_dir': str(ORACLE_DIR),
        'loaded_lora_parameter_count': len(loaded),
        'official_forward_signature': str(inspect.signature(unet.forward)),
        'sample_shape': list(sample.shape),
        'sample_dtype': str(sample.dtype),
        'timestep_shape': list(timestep.shape),
        'timestep_dtype': str(timestep.dtype),
        'timestep_value': to_np(timestep).astype(int).tolist(),
        'encoder_hidden_states_shape': list(encoder_hidden_states.shape),
        'encoder_hidden_states_dtype': str(encoder_hidden_states.dtype),
        'return_dict_behavior': {
            'return_dict_true': official_return,
            'return_dict_false': tuple_return,
            'tuple_matches_return_dict': tuple_vs_official,
        },
        'official_output_shape': list(official.shape),
        'manual_wrapper_output_shape': list(manual.shape),
        'manual_vs_official_full_forward': manual_vs_official,
        'saved_tensors': saved,
        'upblock2_return_info': manual_meta['up2_return_info'],
        'upblock3_return_info': manual_meta['up3_return_info'],
        'official_full_forward_executed': True,
        'scheduler_executed': False,
        'vae_executed': False,
        'full_tadsr_inference_executed': False,
        'runtime_lora_implemented': False,
        'recommended_next_stage': 'Jittor alignment-only full forward wrapper comparison, then pipeline boundary planning; do not enable full inference yet.',
        'markers': {
            'TADSR_UNET_OFFICIAL_FULL_FORWARD_ORACLE_TENSORS': status,
            'TADSR_UNET_MANUAL_VS_OFFICIAL_FULL_FORWARD_ALIGNMENT': manual_status,
        },
    }
    META_JSON.write_text(json.dumps(metadata, indent=2, default=str), encoding='utf-8')
    SUMMARY_TXT.write_text('\n'.join([
        '# TADSR UNet official full forward oracle export',
        '',
        *[f'{k}: {v}' for k, v in metadata['markers'].items()],
        '',
        f"official output shape: {metadata['official_output_shape']}",
        f"manual vs official: {manual_vs_official}",
        'scope: official UNet.forward oracle only; scheduler/VAE/full inference were not executed.',
    ]) + '\n', encoding='utf-8')
    for k, v in metadata['markers'].items():
        print(f'{k}: {v}')
    print(json.dumps({'status': status, 'oracle_dir': str(ORACLE_DIR), 'metadata': str(META_JSON)}, indent=2))
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
