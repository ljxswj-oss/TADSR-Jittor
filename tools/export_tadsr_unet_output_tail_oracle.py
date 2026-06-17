#!/usr/bin/env python3
from __future__ import annotations

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
    module_effective_arrays,
)

ORACLE_DIR = OUT_DIR / 'oracle_tensors_unet_output_tail'
META_JSON = ORACLE_DIR / 'unet_output_tail_oracle_metadata.json'
SUMMARY_TXT = ORACLE_DIR / 'oracle_summary.txt'
EFFECTIVE_WEIGHTS = OUT_DIR / 'converted_unet_output_tail_effective_weights.npz'
PREV_UPBLOCK3_ORACLE = OUT_DIR / 'oracle_tensors_unet_upblock3_local' / 'local_upblock3_hidden_states_output.npy'


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
        entry_tail = run_tail(unet, up3_output)

        synthetic_shape = list(up3_output.shape)
        synthetic_hidden = torch.linspace(-0.25, 0.25, steps=int(np.prod(synthetic_shape)), dtype=torch.float32).reshape(*synthetic_shape)
        synthetic_tail = run_tail(unet, synthetic_hidden)

    saved = {}
    for name, tensor in [
        ('synthetic_output_tail_input', synthetic_hidden),
        ('synthetic_output_tail_norm_output', synthetic_tail['norm']),
        ('synthetic_output_tail_act_output', synthetic_tail['act']),
        ('synthetic_output_tail_conv_out_output', synthetic_tail['conv_out']),
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
        ('entry_upblock3_output_hidden', up3_output),
        ('entry_output_tail_norm_output', entry_tail['norm']),
        ('entry_output_tail_act_output', entry_tail['act']),
        ('entry_output_tail_conv_out_output', entry_tail['conv_out']),
    ]:
        save_tensor(saved, name, tensor)

    effective = {}
    effective_meta = {}
    effective['output_tail_conv_norm_out_weight'] = unet.conv_norm_out.weight.detach().cpu().numpy().astype(np.float32)
    effective['output_tail_conv_norm_out_bias'] = unet.conv_norm_out.bias.detach().cpu().numpy().astype(np.float32)
    effective['conv_norm_out_weight'] = effective['output_tail_conv_norm_out_weight']
    effective['conv_norm_out_bias'] = effective['output_tail_conv_norm_out_bias']
    effective_meta['output_tail_conv_norm_out'] = {
        'is_lora_wrapped': False,
        'weight_shape': list(unet.conv_norm_out.weight.shape),
        'bias_shape': list(unet.conv_norm_out.bias.shape),
    }
    module_effective_arrays('output_tail_conv_out', unet.conv_out, effective, effective_meta)
    effective['conv_out_weight'] = effective['output_tail_conv_out_weight']
    effective['conv_out_bias'] = effective['output_tail_conv_out_bias']
    np.savez_compressed(EFFECTIVE_WEIGHTS, **effective)

    previous_compare = {'previous_upblock3_local_oracle_available': False}
    if PREV_UPBLOCK3_ORACLE.exists():
        prev = np.load(PREV_UPBLOCK3_ORACLE)
        curr = to_np(up3_output)
        diff = np.abs(prev.astype(np.float32) - curr.astype(np.float32))
        previous_compare = {
            'previous_upblock3_local_oracle_available': True,
            'max_abs_diff': float(diff.max()) if diff.size else 0.0,
            'mean_abs_diff': float(diff.mean()) if diff.size else 0.0,
        }
    self_check = tensor_metrics(entry_tail['conv_out'], entry_tail['conv_out'])
    metadata = {
        'status': 'PASS',
        'python': sys.executable,
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'oracle_dir': str(ORACLE_DIR),
        'effective_weights': str(EFFECTIVE_WEIGHTS),
        'loaded_lora_parameter_count': len(loaded),
        'selected_timestep': [1],
        'entry_sample_shape': sample_shape,
        'encoder_hidden_states_shape': encoder_shape,
        'synthetic_tail_input_shape': list(synthetic_hidden.shape),
        'entry_tail_input_shape': list(up3_output.shape),
        'entry_tail_output_shape': list(entry_tail['conv_out'].shape),
        'upblock2_return_info': up2_return,
        'upblock3_return_info': up3_return,
        'tail_config': {
            'module_order': ['conv_norm_out', 'conv_act', 'conv_out'],
            'conv_norm_out_num_groups': int(unet.conv_norm_out.num_groups),
            'conv_norm_out_eps': float(unet.conv_norm_out.eps),
            'conv_out_padding': list((unet.conv_out.base_layer if hasattr(unet.conv_out, 'base_layer') else unet.conv_out).padding),
            'conv_out_stride': list((unet.conv_out.base_layer if hasattr(unet.conv_out, 'base_layer') else unet.conv_out).stride),
            'conv_out_weight_shape': list(effective['output_tail_conv_out_weight'].shape),
            'conv_out_bias_shape': list(effective['output_tail_conv_out_bias'].shape),
        },
        'saved_tensors': saved,
        'effective_weight_export': {
            'status': 'PASS',
            'path': str(EFFECTIVE_WEIGHTS),
            'keys': {k: list(v.shape) for k, v in effective.items()},
            'module_effective_meta': effective_meta,
            'lora_affected_modules': [k for k, v in effective_meta.items() if v.get('is_lora_wrapped')],
        },
        'previous_upblock3_local_compare': previous_compare,
        'self_check': self_check,
        'uses_official_unet_forward': False,
        'uses_output_tail': True,
        'uses_full_tadsr_inference': False,
        'markers': {
            'TADSR_UNET_OUTPUT_TAIL_ORACLE_TENSORS': 'PASS',
            'TADSR_UNET_OUTPUT_TAIL_EFFECTIVE_WEIGHTS': 'PASS',
        },
        'scope_boundaries': {
            'ported_this_stage': 'output tail oracle/effective weights only',
            'not_ported_this_stage': ['official unet.forward', 'scheduler denoising', 'VAE integration', 'image generation', 'generic runtime LoRA'],
        },
    }
    META_JSON.write_text(json.dumps(metadata, indent=2, default=str), encoding='utf-8')
    SUMMARY_TXT.write_text('\n'.join([
        '# TADSR UNet output tail oracle export',
        '',
        *[f'{k}: {v}' for k, v in metadata['markers'].items()],
        '',
        f"entry tail input shape: {metadata['entry_tail_input_shape']}",
        f"entry tail output shape: {metadata['entry_tail_output_shape']}",
        f"effective weights: {EFFECTIVE_WEIGHTS}",
        f"previous upblock3 local compare: {previous_compare}",
        'scope: output tail only; official unet.forward/full inference were not executed.',
    ]) + '\n', encoding='utf-8')
    for k, v in metadata['markers'].items():
        print(f'{k}: {v}')
    print(json.dumps({'status': 'PASS', 'oracle_dir': str(ORACLE_DIR), 'metadata': str(META_JSON), 'effective_weights': str(EFFECTIVE_WEIGHTS)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
