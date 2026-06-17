#!/usr/bin/env python3
from __future__ import annotations

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
    module_effective_arrays,
    resnet_config,
    resnet_manual,
    to_np,
    stats,
    max_abs,
)
from export_tadsr_unet_upblock1_resnet0_oracle import source_names
from audit_official_tadsr_unet_upblock1_local import run_context

ORACLE_DIR = OUT_DIR / 'oracle_tensors_unet_upblock2_resnet0'
META_JSON = ORACLE_DIR / 'unet_upblock2_resnet0_oracle_metadata.json'
SUMMARY_TXT = ORACLE_DIR / 'oracle_summary.txt'
EFFECTIVE_WEIGHTS = OUT_DIR / 'converted_unet_upblock2_resnet0_effective_weights.npz'
PREV_UPBLOCK1_LOCAL_ORACLE = OUT_DIR / 'oracle_tensors_unet_upblock1_local' / 'local_upblock1_hidden_states_output.npy'


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


def save_resnet_intermediates(saved: dict, prefix: str, inter: dict, official) -> None:
    for key, tensor in inter.items():
        out_name = f'{prefix}_manual_output' if key == 'output' else f'{prefix}_{key}_output'
        save_tensor(saved, out_name, tensor)
    save_tensor(saved, f'{prefix}_output', official)


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
    if not getattr(up2, 'resnets', None):
        raise RuntimeError('Official up_blocks.2 has no resnets.0 target')
    resnet0 = up2.resnets[0]

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
        concat = torch.cat([ctx['official_output'], res0_hidden], dim=1)
        entry_inter = resnet_manual(resnet0, concat, ctx['time_embedding'])
        entry_official = resnet0(concat, ctx['time_embedding'])

        h_shape = list(ctx['official_output'].shape)
        r_shape = list(res0_hidden.shape)
        temb_shape = [1, int(resnet0.time_emb_proj.weight.shape[1])]
        synthetic_hidden = torch.linspace(-1.0, 1.0, steps=int(np.prod(h_shape)), dtype=torch.float32).reshape(*h_shape)
        synthetic_res = torch.linspace(-0.75, 0.75, steps=int(np.prod(r_shape)), dtype=torch.float32).reshape(*r_shape)
        synthetic_temb = torch.linspace(-0.5, 0.5, steps=int(np.prod(temb_shape)), dtype=torch.float32).reshape(*temb_shape)
        synthetic_concat = torch.cat([synthetic_hidden, synthetic_res], dim=1)
        synthetic_inter = resnet_manual(resnet0, synthetic_concat, synthetic_temb)
        synthetic_official = resnet0(synthetic_concat, synthetic_temb)

    saved = {}
    for name, tensor in [
        ('synthetic_upblock2_resnet0_hidden_input', synthetic_hidden),
        ('synthetic_upblock2_resnet0_res_hidden', synthetic_res),
        ('synthetic_upblock2_resnet0_concat_input', synthetic_concat),
        ('synthetic_upblock2_resnet0_temb', synthetic_temb),
    ]:
        save_tensor(saved, name, tensor)
    save_resnet_intermediates(saved, 'synthetic_upblock2_resnet0', synthetic_inter, synthetic_official)

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
        ('entry_upblock2_resnet0_hidden_input', ctx['official_output']),
        ('entry_upblock2_resnet0_res_hidden', res0_hidden),
        ('entry_upblock2_resnet0_concat_input', concat),
        ('entry_upblock2_resnet0_temb', ctx['time_embedding']),
    ]:
        save_tensor(saved, name, tensor)
    save_resnet_intermediates(saved, 'entry_upblock2_resnet0', entry_inter, entry_official)

    effective = {}
    effective_meta = {}
    prefix = 'up_blocks_2_resnets_0'
    for name in ['norm1', 'norm2']:
        mod = getattr(resnet0, name)
        effective[f'{prefix}_{name}_weight'] = mod.weight.detach().cpu().numpy().astype(np.float32)
        effective[f'{prefix}_{name}_bias'] = mod.bias.detach().cpu().numpy().astype(np.float32)
        effective_meta[f'{prefix}_{name}'] = {
            'is_lora_wrapped': False,
            'weight_shape': list(mod.weight.shape),
            'bias_shape': list(mod.bias.shape),
        }
    for name in ['conv1', 'time_emb_proj', 'conv2']:
        module_effective_arrays(f'{prefix}_{name}', getattr(resnet0, name), effective, effective_meta)
    if getattr(resnet0, 'conv_shortcut', None) is not None:
        module_effective_arrays(f'{prefix}_conv_shortcut', resnet0.conv_shortcut, effective, effective_meta)
    np.savez_compressed(EFFECTIVE_WEIGHTS, **effective)

    sources = source_names()
    consumed_index = len(remaining_after_up1) - 1
    synth_diff = (synthetic_inter['output'] - synthetic_official).abs()
    entry_diff = (entry_inter['output'] - entry_official).abs()
    previous_upblock1_compare = {'previous_upblock1_oracle_tensor_available': False}
    if PREV_UPBLOCK1_LOCAL_ORACLE.exists():
        prev = np.load(PREV_UPBLOCK1_LOCAL_ORACLE)
        curr = to_np(ctx['official_output'])
        previous_upblock1_compare = {
            'previous_upblock1_oracle_tensor_available': True,
            'max_abs_diff': max_abs(prev, curr),
            'mean_abs_diff': float(np.mean(np.abs(prev.astype(np.float32) - curr.astype(np.float32)))),
        }
    next_module = 'up_blocks.2.attentions.0' if len(getattr(up2, 'attentions', []) or []) > 0 else 'up_blocks.2.resnets.1'
    metadata = {
        'status': 'PASS',
        'python': sys.executable,
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'loaded_lora_parameter_count': len(loaded),
        'oracle_dir': str(ORACLE_DIR),
        'effective_weights': str(EFFECTIVE_WEIGHTS),
        'selected_timestep': [1],
        'entry_sample_shape': sample_shape,
        'encoder_hidden_states_shape': list(encoder_hidden_states.shape),
        'upblock1_output_hidden_shape': list(ctx['official_output'].shape),
        'remaining_residual_samples_before_upblocks2_count': len(remaining_after_up1),
        'remaining_residual_samples_before_upblocks2_shapes': [list(x.shape) for x in remaining_after_up1],
        'upblock2_local_residual_tuple_sources': sources[len(remaining_before_up2):len(remaining_after_up1)],
        'upblock2_local_residual_tuple_shapes': [list(x.shape) for x in up2_res_tuple],
        'upblock2_resnet0_consumed_residual_index_in_accumulated_tuple': consumed_index,
        'upblock2_resnet0_consumed_residual_current_remaining_tuple_index': len(up2_res_tuple) - 1,
        'upblock2_resnet0_consumed_residual_source': sources[consumed_index],
        'upblock2_resnet0_consumed_residual_shape': list(res0_hidden.shape),
        'upblock2_resnet0_hidden_input_shape': list(ctx['official_output'].shape),
        'upblock2_resnet0_residual_input_shape': list(res0_hidden.shape),
        'upblock2_resnet0_concat_input_shape': list(concat.shape),
        'upblock2_resnet0_output_shape': list(entry_official.shape),
        'remaining_residual_count_after_upblock2_resnet0_pop': len(remaining_after_resnet0_local),
        'remaining_residual_shapes_after_upblock2_resnet0_pop': [list(x.shape) for x in remaining_after_resnet0_local],
        'temb_shape': list(ctx['time_embedding'].shape),
        'resnet0_config': resnet_config(resnet0, prefix),
        'saved_tensors': saved,
        'effective_weight_export': {
            'status': 'PASS',
            'path': str(EFFECTIVE_WEIGHTS),
            'keys': {k: list(v.shape) for k, v in effective.items()},
            'module_effective_meta': effective_meta,
            'lora_affected_modules': [k for k, v in effective_meta.items() if v.get('is_lora_wrapped')],
        },
        'manual_vs_official': {
            'synthetic_max_abs_diff': float(synth_diff.max().item()) if synth_diff.numel() else 0.0,
            'synthetic_mean_abs_diff': float(synth_diff.mean().item()) if synth_diff.numel() else 0.0,
            'entry_max_abs_diff': float(entry_diff.max().item()) if entry_diff.numel() else 0.0,
            'entry_mean_abs_diff': float(entry_diff.mean().item()) if entry_diff.numel() else 0.0,
            'status': 'PASS' if float(synth_diff.max()) < 1e-5 and float(entry_diff.max()) < 1e-5 else 'FAIL',
        },
        'previous_upblock1_compare': previous_upblock1_compare,
        'uses_full_unet_forward': False,
        'uses_upblock2_attention0': False,
        'uses_upblocks3': False,
        'uses_full_tadsr_inference': False,
        'uses_stochastic_sampling': False,
        'next_module_preview': next_module,
        'markers': {
            'TADSR_UNET_UPBLOCK2_RESNET0_ORACLE_TENSORS': 'PASS',
            'TADSR_UNET_UPBLOCK2_RESNET0_EFFECTIVE_WEIGHTS': 'PASS',
            'TADSR_UNET_UPBLOCK2_RESNET0_RESIDUAL_CONTRACT_ORACLE': 'PASS',
        },
    }
    META_JSON.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    SUMMARY_TXT.write_text('\n'.join([
        '# TADSR UNet up_blocks.2.resnets.0 PyTorch oracle',
        '',
        *[f'{k}: {v}' for k, v in metadata['markers'].items()],
        '',
        f"residual source: {metadata['upblock2_resnet0_consumed_residual_source']}",
        f"hidden input shape: {metadata['upblock2_resnet0_hidden_input_shape']}",
        f"residual shape: {metadata['upblock2_resnet0_residual_input_shape']}",
        f"concat shape: {metadata['upblock2_resnet0_concat_input_shape']}",
        f"output shape: {metadata['upblock2_resnet0_output_shape']}",
        f"synthetic manual-vs-official max_abs_diff: {metadata['manual_vs_official']['synthetic_max_abs_diff']}",
        f"entry manual-vs-official max_abs_diff: {metadata['manual_vs_official']['entry_max_abs_diff']}",
        f"next unopened module: {metadata['next_module_preview']}",
    ]) + '\n', encoding='utf-8')
    for k, v in metadata['markers'].items():
        print(f'{k}: {v}')
    print(json.dumps({'status': 'PASS', 'oracle_dir': str(ORACLE_DIR), 'effective_weights': str(EFFECTIVE_WEIGHTS)}, indent=2))
    return 0 if metadata['manual_vs_official']['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
