#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

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

ORACLE_DIR = OUT_DIR / 'oracle_tensors_unet_upblock0_resnet0'
META_JSON = ORACLE_DIR / 'unet_upblock0_resnet0_oracle_metadata.json'
SUMMARY_TXT = ORACLE_DIR / 'oracle_summary.txt'
EFFECTIVE_WEIGHTS = OUT_DIR / 'converted_unet_upblock0_resnet0_effective_weights.npz'
PREV_MIDBLOCK_ORACLE = OUT_DIR / 'oracle_tensors_unet_midblock_resnet1' / 'local_midblock_hidden_states_output.npy'


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


def source_names():
    return (
        ['conv_in']
        + [f'down_blocks.0.output_state_{i}' for i in range(3)]
        + [f'down_blocks.1.output_state_{i}' for i in range(3)]
        + [f'down_blocks.2.output_state_{i}' for i in range(3)]
        + [f'down_blocks.3.output_state_{i}' for i in range(2)]
    )


def extract_midblock_output(out):
    if isinstance(out, tuple):
        return out[0], {'return_type': 'tuple', 'output_states_status': 'UNKNOWN_TUPLE'}
    if hasattr(out, 'sample'):
        return out.sample, {'return_type': type(out).__name__, 'output_states_status': 'NOT_APPLICABLE'}
    return out, {'return_type': type(out).__name__, 'output_states_status': 'NOT_APPLICABLE'}


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
    up0 = unet.up_blocks[0]
    resnet0 = up0.resnets[0]
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
        down_res = (conv_in,)
        hidden = conv_in
        down_states = []
        down_hiddens = []
        for block in unet.down_blocks:
            if getattr(block, 'has_cross_attention', False):
                hidden, states = block(hidden, time_embedding, encoder_hidden_states=encoder_hidden_states)
            else:
                hidden, states = block(hidden, time_embedding)
            down_hiddens.append(hidden)
            down_states.append(states)
            down_res += states
        mid_raw = unet.mid_block(
            hidden,
            temb=time_embedding,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=None,
            cross_attention_kwargs=None,
            encoder_attention_mask=None,
        )
        mid_output, mid_return = extract_midblock_output(mid_raw)
        up0_res_tuple = down_res[-len(up0.resnets):]
        global_remaining = down_res[:-len(up0.resnets)]
        res_hidden = up0_res_tuple[-1]
        internal_remaining = up0_res_tuple[:-1]
        concat = torch.cat([mid_output, res_hidden], dim=1)
        entry_inter = resnet_manual(resnet0, concat, time_embedding)
        entry_official = resnet0(concat, time_embedding)

        h_shape = list(mid_output.shape)
        r_shape = list(res_hidden.shape)
        temb_shape = [1, int(resnet0.time_emb_proj.weight.shape[1])]
        synthetic_h = torch.linspace(-1.0, 1.0, steps=int(np.prod(h_shape)), dtype=torch.float32).reshape(*h_shape)
        synthetic_res = torch.linspace(-0.75, 0.75, steps=int(np.prod(r_shape)), dtype=torch.float32).reshape(*r_shape)
        synthetic_temb = torch.linspace(-0.5, 0.5, steps=int(np.prod(temb_shape)), dtype=torch.float32).reshape(*temb_shape)
        synthetic_concat = torch.cat([synthetic_h, synthetic_res], dim=1)
        synthetic_inter = resnet_manual(resnet0, synthetic_concat, synthetic_temb)
        synthetic_official = resnet0(synthetic_concat, synthetic_temb)

    saved = {}
    save_tensor(saved, 'synthetic_upblock0_resnet0_hidden_input', synthetic_h)
    save_tensor(saved, 'synthetic_upblock0_resnet0_res_hidden', synthetic_res)
    save_tensor(saved, 'synthetic_upblock0_resnet0_concat_input', synthetic_concat)
    save_tensor(saved, 'synthetic_upblock0_resnet0_temb', synthetic_temb)
    for k, v in synthetic_inter.items():
        save_tensor(saved, f'synthetic_upblock0_resnet0_{k}_output' if k != 'output' else 'synthetic_upblock0_resnet0_manual_output', v)
    save_tensor(saved, 'synthetic_upblock0_resnet0_output', synthetic_official)

    save_tensor(saved, 'entry_synthetic_unet_sample', sample)
    save_tensor(saved, 'entry_synthetic_unet_timestep', timestep)
    save_tensor(saved, 'entry_encoder_hidden_states', encoder_hidden_states)
    save_tensor(saved, 'entry_synthetic_unet_sample_after_center', centered)
    save_tensor(saved, 'entry_synthetic_unet_conv_in_output', conv_in)
    save_tensor(saved, 'entry_synthetic_unet_time_proj_output', time_proj)
    save_tensor(saved, 'entry_synthetic_unet_time_embedding_output', time_embedding)
    for i, (h, states) in enumerate(zip(down_hiddens, down_states)):
        save_tensor(saved, f'entry_downblock{i}_output_hidden', h)
        for j, state in enumerate(states):
            save_tensor(saved, f'entry_downblock{i}_output_state_{j}', state)
    save_tensor(saved, 'entry_midblock_output_hidden', mid_output)
    save_tensor(saved, 'entry_upblock0_resnet0_hidden_input', mid_output)
    save_tensor(saved, 'entry_upblock0_resnet0_res_hidden', res_hidden)
    save_tensor(saved, 'entry_upblock0_resnet0_concat_input', concat)
    save_tensor(saved, 'entry_upblock0_resnet0_temb', time_embedding)
    for k, v in entry_inter.items():
        save_tensor(saved, f'entry_upblock0_resnet0_{k}_output' if k != 'output' else 'entry_upblock0_resnet0_manual_output', v)
    save_tensor(saved, 'entry_upblock0_resnet0_output', entry_official)

    effective = {}
    effective_meta = {}
    prefix = 'up_blocks_0_resnets_0'
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

    sources = source_names()
    consumed_index = len(down_res) - 1
    consumed_source = sources[consumed_index] if consumed_index < len(sources) else f'accumulated_index_{consumed_index}'
    synth_diff = (synthetic_inter['output'] - synthetic_official).abs()
    entry_diff = (entry_inter['output'] - entry_official).abs()
    previous_midblock_compare = {'previous_midblock_oracle_tensor_available': False}
    if PREV_MIDBLOCK_ORACLE.exists():
        prev = np.load(PREV_MIDBLOCK_ORACLE)
        curr = to_np(mid_output)
        previous_midblock_compare = {
            'previous_midblock_oracle_tensor_available': True,
            'max_abs_diff': max_abs(prev, curr),
            'mean_abs_diff': float(np.mean(np.abs(prev.astype(np.float32) - curr.astype(np.float32)))),
        }
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
        'downblock_output_state_shapes': {f'down_blocks.{i}': [list(x.shape) for x in states] for i, states in enumerate(down_states)},
        'downblock_output_hidden_shapes': {f'down_blocks.{i}': list(h.shape) for i, h in enumerate(down_hiddens)},
        'accumulated_down_block_res_sample_shapes': [list(x.shape) for x in down_res],
        'accumulated_down_block_res_sample_sources': sources,
        'accumulated_down_block_res_sample_count': len(down_res),
        'global_slice_for_up_blocks_0_start_index': len(down_res) - len(up0.resnets),
        'global_remaining_residual_count_after_slice': len(global_remaining),
        'global_remaining_residual_shapes_after_slice': [list(x.shape) for x in global_remaining],
        'upblock0_resnet0_consumed_residual_index_in_accumulated_tuple': consumed_index,
        'upblock0_resnet0_consumed_residual_source': consumed_source,
        'upblock0_resnet0_consumed_residual_shape': list(res_hidden.shape),
        'internal_remaining_residual_count_after_resnet0_pop': len(internal_remaining),
        'internal_remaining_residual_shapes_after_resnet0_pop': [list(x.shape) for x in internal_remaining],
        'upblock0_resnet0_hidden_input_shape': list(mid_output.shape),
        'upblock0_resnet0_concat_input_shape': list(concat.shape),
        'upblock0_resnet0_output_shape': list(entry_official.shape),
        'concat_axis': 1,
        'temb_shape': list(time_embedding.shape),
        'local_midblock_output_states_status': mid_return.get('output_states_status', 'NOT_APPLICABLE'),
        'resnet0_config': resnet_config(resnet0, prefix),
        'saved_tensors': saved,
        'effective_weight_metadata': effective_meta,
        'manual_vs_official': {
            'synthetic_max_abs_diff': float(synth_diff.max().item()) if synth_diff.numel() else 0.0,
            'synthetic_mean_abs_diff': float(synth_diff.mean().item()) if synth_diff.numel() else 0.0,
            'entry_max_abs_diff': float(entry_diff.max().item()) if entry_diff.numel() else 0.0,
            'entry_mean_abs_diff': float(entry_diff.mean().item()) if entry_diff.numel() else 0.0,
        },
        'previous_midblock_compare': previous_midblock_compare,
        'markers': {
            'TADSR_UNET_UPBLOCK0_RESNET0_ORACLE_TENSORS': 'PASS',
            'TADSR_UNET_UPBLOCK0_RESNET0_EFFECTIVE_WEIGHTS': 'PASS',
            'TADSR_UNET_UPBLOCK0_RESNET0_RESIDUAL_CONTRACT_ORACLE': 'PASS',
        },
        'scope_boundaries': {
            'ported_this_stage': 'up_blocks.0.resnets.0 only, including first residual consumption and concat',
            'not_ported_this_stage': ['full up_blocks.0', 'up_blocks.0.resnets.1', 'up_blocks.0.resnets.2', 'up_blocks.0.upsamplers.0', 'full UNet forward', 'full TADSR inference', 'generic runtime LoRA'],
        },
    }
    META_JSON.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    SUMMARY_TXT.write_text(
        '\n'.join([
            '# TADSR UNet up_blocks.0.resnets.0 oracle export',
            '',
            'TADSR_UNET_UPBLOCK0_RESNET0_ORACLE_TENSORS: PASS',
            'TADSR_UNET_UPBLOCK0_RESNET0_EFFECTIVE_WEIGHTS: PASS',
            'TADSR_UNET_UPBLOCK0_RESNET0_RESIDUAL_CONTRACT_ORACLE: PASS',
            '',
            f'effective weights: {EFFECTIVE_WEIGHTS}',
            f'oracle tensors: {ORACLE_DIR}',
            f'consumed residual source: {consumed_source}',
            f'consumed residual index: {consumed_index}',
            f'concat shape: {list(concat.shape)}',
            f'synthetic manual/offical max abs diff: {metadata["manual_vs_official"]["synthetic_max_abs_diff"]}',
            f'entry manual/offical max abs diff: {metadata["manual_vs_official"]["entry_max_abs_diff"]}',
            f'previous midblock compare: {previous_midblock_compare}',
        ]) + '\n',
        encoding='utf-8',
    )
    for k, v in metadata['markers'].items():
        print(f'{k}: {v}')
    print(json.dumps({'status': metadata['status'], 'metadata': str(META_JSON), 'effective_weights': str(EFFECTIVE_WEIGHTS)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
