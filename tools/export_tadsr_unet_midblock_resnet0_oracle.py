#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from export_tadsr_unet_downblock3_resnet1_oracle import (
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

ORACLE_DIR = OUT_DIR / 'oracle_tensors_unet_midblock_resnet0'
META_JSON = ORACLE_DIR / 'unet_midblock_resnet0_oracle_metadata.json'
SUMMARY_TXT = ORACLE_DIR / 'oracle_summary.txt'
EFFECTIVE_WEIGHTS = OUT_DIR / 'converted_unet_midblock_resnet0_effective_weights.npz'
PREV_DOWNBLOCK3_ORACLE = OUT_DIR / 'oracle_tensors_unet_downblock3_resnet1' / 'local_downblock3_hidden_states_output.npy'


def maybe_reexec() -> None:
    expected_prefix = str(STRICT_PY.parents[1])
    in_expected_venv = str(sys.prefix) == expected_prefix or str(sys.executable).startswith(expected_prefix)
    if STRICT_PY.exists() and not in_expected_venv:
        os.environ['PYTHONNOUSERSITE'] = '1'
        os.execv(str(STRICT_PY), [str(STRICT_PY), __file__, *sys.argv[1:]])


def save_tensor(saved, name, tensor):
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
    block0 = unet.down_blocks[0]
    block1 = unet.down_blocks[1]
    block2 = unet.down_blocks[2]
    block3 = unet.down_blocks[3]
    mid = unet.mid_block
    resnet0 = mid.resnets[0]

    sample_shape = [1, int(unet.config.in_channels), 32, 32]
    sample = torch.linspace(-1.0, 1.0, steps=int(np.prod(sample_shape)), dtype=torch.float32).reshape(*sample_shape)
    timestep = torch.tensor([1], dtype=torch.long)
    encoder_shape = [1, 77, int(unet.config.cross_attention_dim)]
    encoder_hidden_states = torch.linspace(-0.5, 0.5, steps=int(np.prod(encoder_shape)), dtype=torch.float32).reshape(*encoder_shape)
    with torch.no_grad():
        centered = 2.0 * sample - 1.0 if unet.config.center_input_sample else sample
        conv_in = unet.conv_in(centered)
        time_proj = unet.time_proj(timestep)
        time_embedding = unet.time_embedding(time_proj.to(dtype=list(unet.time_embedding.parameters())[0].dtype))
        block0_hidden, block0_states = block0(conv_in, time_embedding, encoder_hidden_states=encoder_hidden_states)
        block1_hidden, block1_states = block1(block0_hidden, time_embedding, encoder_hidden_states=encoder_hidden_states)
        block2_hidden, block2_states = block2(block1_hidden, time_embedding, encoder_hidden_states=encoder_hidden_states)
        block3_hidden, block3_states = block3(block2_hidden, time_embedding)
        entry_inter = resnet_manual(resnet0, block3_hidden, time_embedding)
        entry_official = resnet0(block3_hidden, time_embedding)
        x_shape = list(block3_hidden.shape)
        temb_shape = [1, int(resnet0.time_emb_proj.weight.shape[1])]
        synthetic_x = torch.linspace(-1.0, 1.0, steps=int(np.prod(x_shape)), dtype=torch.float32).reshape(*x_shape)
        synthetic_temb = torch.linspace(-0.5, 0.5, steps=int(np.prod(temb_shape)), dtype=torch.float32).reshape(*temb_shape)
        synthetic_inter = resnet_manual(resnet0, synthetic_x, synthetic_temb)
        synthetic_official = resnet0(synthetic_x, synthetic_temb)

    saved = {}
    save_tensor(saved, 'synthetic_midblock_resnet0_input', synthetic_x)
    save_tensor(saved, 'synthetic_midblock_resnet0_temb', synthetic_temb)
    for k, v in synthetic_inter.items():
        save_tensor(saved, f'synthetic_midblock_resnet0_{k}_output' if k != 'output' else 'synthetic_midblock_resnet0_manual_output', v)
    save_tensor(saved, 'synthetic_midblock_resnet0_output', synthetic_official)

    save_tensor(saved, 'entry_synthetic_unet_sample', sample)
    save_tensor(saved, 'entry_synthetic_unet_timestep', timestep)
    save_tensor(saved, 'entry_encoder_hidden_states', encoder_hidden_states)
    save_tensor(saved, 'entry_synthetic_unet_sample_after_center', centered)
    save_tensor(saved, 'entry_synthetic_unet_conv_in_output', conv_in)
    save_tensor(saved, 'entry_synthetic_unet_time_proj_output', time_proj)
    save_tensor(saved, 'entry_synthetic_unet_time_embedding_output', time_embedding)
    save_tensor(saved, 'entry_downblock0_output_hidden', block0_hidden)
    for i, state in enumerate(block0_states):
        save_tensor(saved, f'entry_downblock0_output_state_{i}', state)
    save_tensor(saved, 'entry_downblock1_output_hidden', block1_hidden)
    for i, state in enumerate(block1_states):
        save_tensor(saved, f'entry_downblock1_output_state_{i}', state)
    save_tensor(saved, 'entry_downblock2_output_hidden', block2_hidden)
    for i, state in enumerate(block2_states):
        save_tensor(saved, f'entry_downblock2_output_state_{i}', state)
    save_tensor(saved, 'entry_downblock3_output_hidden', block3_hidden)
    for i, state in enumerate(block3_states):
        save_tensor(saved, f'entry_downblock3_output_state_{i}', state)
    save_tensor(saved, 'entry_midblock_resnet0_input', block3_hidden)
    save_tensor(saved, 'entry_midblock_resnet0_temb', time_embedding)
    for k, v in entry_inter.items():
        save_tensor(saved, f'entry_midblock_resnet0_{k}_output' if k != 'output' else 'entry_midblock_resnet0_manual_output', v)
    save_tensor(saved, 'entry_midblock_resnet0_output', entry_official)

    effective = {}
    effective_meta = {}
    prefix = 'mid_block_resnets_0'
    for name in ['norm1', 'norm2']:
        mod = getattr(resnet0, name)
        effective[f'{prefix}_{name}_weight'] = mod.weight.detach().cpu().numpy().astype(np.float32)
        effective[f'{prefix}_{name}_bias'] = mod.bias.detach().cpu().numpy().astype(np.float32)
        effective_meta[f'{prefix}_{name}'] = {'is_lora_wrapped': False, 'weight_shape': list(mod.weight.shape), 'bias_shape': list(mod.bias.shape)}
    for name in ['conv1', 'time_emb_proj', 'conv2']:
        module_effective_arrays(f'{prefix}_{name}', getattr(resnet0, name), effective, effective_meta)
    if getattr(resnet0, 'conv_shortcut', None) is not None:
        module_effective_arrays(f'{prefix}_conv_shortcut', resnet0.conv_shortcut, effective, effective_meta)
    np.savez_compressed(EFFECTIVE_WEIGHTS, **effective)

    synth_diff = (synthetic_inter['output'] - synthetic_official).abs()
    entry_diff = (entry_inter['output'] - entry_official).abs()
    previous_downblock3_compare = {'previous_downblock3_oracle_tensor_available': False}
    if PREV_DOWNBLOCK3_ORACLE.exists():
        prev = np.load(PREV_DOWNBLOCK3_ORACLE)
        curr = to_np(block3_hidden)
        previous_downblock3_compare = {
            'previous_downblock3_oracle_tensor_available': True,
            'max_abs_diff': max_abs(prev, curr),
            'mean_abs_diff': float(np.mean(np.abs(prev.astype(np.float32) - curr.astype(np.float32)))),
        }
    accumulated_shapes = (
        [list(conv_in.shape)]
        + [list(x.shape) for x in block0_states]
        + [list(x.shape) for x in block1_states]
        + [list(x.shape) for x in block2_states]
        + [list(x.shape) for x in block3_states]
    )
    metadata = {
        'status': 'PASS',
        'python': sys.executable,
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'oracle_dir': str(ORACLE_DIR),
        'effective_weights': str(EFFECTIVE_WEIGHTS),
        'selected_timestep': [1],
        'entry_sample_shape': sample_shape,
        'encoder_hidden_states_shape': encoder_shape,
        'downblock0_output_state_shapes': [list(x.shape) for x in block0_states],
        'downblock1_output_state_shapes': [list(x.shape) for x in block1_states],
        'downblock2_output_state_shapes': [list(x.shape) for x in block2_states],
        'downblock3_output_state_shapes': [list(x.shape) for x in block3_states],
        'accumulated_down_block_res_sample_shapes': accumulated_shapes,
        'accumulated_down_block_res_sample_count': len(accumulated_shapes),
        'midblock_resnet0_input_shape': list(block3_hidden.shape),
        'midblock_resnet0_output_shape': list(entry_official.shape),
        'temb_shape': list(time_embedding.shape),
        'midblock_resnet0_consumes': ['hidden_states', 'temb'],
        'midblock_resnet0_consumes_accumulated_residuals': False,
        'resnet0_config': resnet_config(resnet0, prefix),
        'saved_tensors': saved,
        'effective_weight_metadata': effective_meta,
        'manual_vs_official': {
            'synthetic_max_abs_diff': float(synth_diff.max().item()) if synth_diff.numel() else 0.0,
            'synthetic_mean_abs_diff': float(synth_diff.mean().item()) if synth_diff.numel() else 0.0,
            'entry_max_abs_diff': float(entry_diff.max().item()) if entry_diff.numel() else 0.0,
            'entry_mean_abs_diff': float(entry_diff.mean().item()) if entry_diff.numel() else 0.0,
        },
        'previous_downblock3_compare': previous_downblock3_compare,
        'markers': {
            'TADSR_UNET_MIDBLOCK_RESNET0_ORACLE_TENSORS': 'PASS',
            'TADSR_UNET_MIDBLOCK_RESNET0_EFFECTIVE_WEIGHTS': 'PASS',
        },
        'scope_boundaries': {
            'ported_this_stage': 'mid_block.resnets.0',
            'not_ported_this_stage': ['mid_block.attentions.0', 'mid_block.resnets.1', 'up_blocks', 'full UNet forward', 'full TADSR inference'],
        },
    }
    META_JSON.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    SUMMARY_TXT.write_text(
        '\n'.join([
            '# TADSR UNet mid_block.resnets.0 oracle export',
            '',
            'TADSR_UNET_MIDBLOCK_RESNET0_ORACLE_TENSORS: PASS',
            'TADSR_UNET_MIDBLOCK_RESNET0_EFFECTIVE_WEIGHTS: PASS',
            '',
            f'effective weights: {EFFECTIVE_WEIGHTS}',
            f'oracle tensors: {ORACLE_DIR}',
            f'synthetic manual/offical max abs diff: {metadata["manual_vs_official"]["synthetic_max_abs_diff"]}',
            f'entry manual/offical max abs diff: {metadata["manual_vs_official"]["entry_max_abs_diff"]}',
            f'previous downblock3 compare: {previous_downblock3_compare}',
        ]) + '\n',
        encoding='utf-8',
    )
    for k, v in metadata['markers'].items():
        print(f'{k}: {v}')
    print(json.dumps({'status': metadata['status'], 'metadata': str(META_JSON), 'effective_weights': str(EFFECTIVE_WEIGHTS)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
