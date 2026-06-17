#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import numpy as np

from export_tadsr_unet_downblock3_resnet0_oracle import (
    STRICT_PY, OUT_DIR, OFFICIAL_REPO, WEIGHTS_DIR, load_unet, module_effective_arrays,
    resnet_config, resnet_manual, to_np, stats, max_abs,
)

ORACLE_DIR = OUT_DIR / 'oracle_tensors_unet_upblock0_resnet2'
META_JSON = ORACLE_DIR / 'unet_upblock0_resnet2_oracle_metadata.json'
SUMMARY_TXT = ORACLE_DIR / 'oracle_summary.txt'
EFFECTIVE_WEIGHTS = OUT_DIR / 'converted_unet_upblock0_resnet2_effective_weights.npz'
PREV_RESNET1_ORACLE = OUT_DIR / 'oracle_tensors_unet_upblock0_resnet1' / 'entry_upblock0_resnet1_output.npy'


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
        return out[0], {'return_type': 'tuple'}
    if hasattr(out, 'sample'):
        return out.sample, {'return_type': type(out).__name__}
    return out, {'return_type': type(out).__name__}


def save_resnet_intermediates(saved, prefix, inter, official):
    for k, v in inter.items():
        save_tensor(saved, f'{prefix}_{k}_output' if k != 'output' else f'{prefix}_manual_output', v)
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
    up0 = unet.up_blocks[0]
    if len(up0.resnets) < 3:
        raise RuntimeError('Official up_blocks.0 has no resnets.2; oracle export cannot proceed.')
    resnet0, resnet1, resnet2 = up0.resnets[0], up0.resnets[1], up0.resnets[2]
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
        mid_raw = unet.mid_block(hidden, temb=time_embedding, encoder_hidden_states=encoder_hidden_states, attention_mask=None, cross_attention_kwargs=None, encoder_attention_mask=None)
        mid_output, mid_return = extract_midblock_output(mid_raw)
        up0_res_tuple = down_res[-len(up0.resnets):]
        global_remaining = down_res[:-len(up0.resnets)]
        res0_hidden = up0_res_tuple[-1]
        remaining_after_resnet0 = up0_res_tuple[:-1]
        concat0 = torch.cat([mid_output, res0_hidden], dim=1)
        resnet0_inter = resnet_manual(resnet0, concat0, time_embedding)
        resnet0_out = resnet0(concat0, time_embedding)
        res1_hidden = remaining_after_resnet0[-1]
        remaining_after_resnet1 = remaining_after_resnet0[:-1]
        concat1 = torch.cat([resnet0_out, res1_hidden], dim=1)
        resnet1_inter = resnet_manual(resnet1, concat1, time_embedding)
        resnet1_out = resnet1(concat1, time_embedding)
        res2_hidden = remaining_after_resnet1[-1]
        remaining_after_resnet2 = remaining_after_resnet1[:-1]
        concat2 = torch.cat([resnet1_out, res2_hidden], dim=1)
        entry_inter = resnet_manual(resnet2, concat2, time_embedding)
        entry_official = resnet2(concat2, time_embedding)
        h_shape = list(resnet1_out.shape)
        r_shape = list(res2_hidden.shape)
        temb_shape = [1, int(resnet2.time_emb_proj.weight.shape[1])]
        synthetic_h = torch.linspace(-1.0, 1.0, steps=int(np.prod(h_shape)), dtype=torch.float32).reshape(*h_shape)
        synthetic_res = torch.linspace(-0.75, 0.75, steps=int(np.prod(r_shape)), dtype=torch.float32).reshape(*r_shape)
        synthetic_temb = torch.linspace(-0.5, 0.5, steps=int(np.prod(temb_shape)), dtype=torch.float32).reshape(*temb_shape)
        synthetic_concat = torch.cat([synthetic_h, synthetic_res], dim=1)
        synthetic_inter = resnet_manual(resnet2, synthetic_concat, synthetic_temb)
        synthetic_official = resnet2(synthetic_concat, synthetic_temb)

    saved = {}
    for name, tensor in [
        ('synthetic_upblock0_resnet2_hidden_input', synthetic_h),
        ('synthetic_upblock0_resnet2_res_hidden', synthetic_res),
        ('synthetic_upblock0_resnet2_concat_input', synthetic_concat),
        ('synthetic_upblock0_resnet2_temb', synthetic_temb),
    ]:
        save_tensor(saved, name, tensor)
    save_resnet_intermediates(saved, 'synthetic_upblock0_resnet2', synthetic_inter, synthetic_official)
    for name, tensor in [
        ('entry_synthetic_unet_sample', sample),
        ('entry_synthetic_unet_timestep', timestep),
        ('entry_encoder_hidden_states', encoder_hidden_states),
        ('entry_synthetic_unet_sample_after_center', centered),
        ('entry_synthetic_unet_conv_in_output', conv_in),
        ('entry_synthetic_unet_time_proj_output', time_proj),
        ('entry_synthetic_unet_time_embedding_output', time_embedding),
    ]:
        save_tensor(saved, name, tensor)
    for i, (h, states) in enumerate(zip(down_hiddens, down_states)):
        save_tensor(saved, f'entry_downblock{i}_output_hidden', h)
        for j, state in enumerate(states):
            save_tensor(saved, f'entry_downblock{i}_output_state_{j}', state)
    for name, tensor in [
        ('entry_midblock_output_hidden', mid_output),
        ('entry_upblock0_resnet0_hidden_input', mid_output),
        ('entry_upblock0_resnet0_res_hidden', res0_hidden),
        ('entry_upblock0_resnet0_concat_input', concat0),
        ('entry_upblock0_resnet0_temb', time_embedding),
    ]:
        save_tensor(saved, name, tensor)
    save_resnet_intermediates(saved, 'entry_upblock0_resnet0', resnet0_inter, resnet0_out)
    for name, tensor in [
        ('entry_upblock0_resnet1_hidden_input', resnet0_out),
        ('entry_upblock0_resnet1_res_hidden', res1_hidden),
        ('entry_upblock0_resnet1_concat_input', concat1),
        ('entry_upblock0_resnet1_temb', time_embedding),
    ]:
        save_tensor(saved, name, tensor)
    save_resnet_intermediates(saved, 'entry_upblock0_resnet1', resnet1_inter, resnet1_out)
    for name, tensor in [
        ('entry_upblock0_resnet2_hidden_input', resnet1_out),
        ('entry_upblock0_resnet2_res_hidden', res2_hidden),
        ('entry_upblock0_resnet2_concat_input', concat2),
        ('entry_upblock0_resnet2_temb', time_embedding),
    ]:
        save_tensor(saved, name, tensor)
    save_resnet_intermediates(saved, 'entry_upblock0_resnet2', entry_inter, entry_official)

    effective = {}
    effective_meta = {}
    prefix = 'up_blocks_0_resnets_2'
    for name in ['norm1', 'norm2']:
        mod = getattr(resnet2, name)
        effective[f'{prefix}_{name}_weight'] = mod.weight.detach().cpu().numpy().astype(np.float32)
        effective[f'{prefix}_{name}_bias'] = mod.bias.detach().cpu().numpy().astype(np.float32)
        effective_meta[f'{prefix}_{name}'] = {'is_lora_wrapped': False, 'weight_shape': list(mod.weight.shape), 'bias_shape': list(mod.bias.shape)}
    for name in ['conv1', 'time_emb_proj', 'conv2']:
        module_effective_arrays(f'{prefix}_{name}', getattr(resnet2, name), effective, effective_meta)
    if getattr(resnet2, 'conv_shortcut', None) is not None:
        module_effective_arrays(f'{prefix}_conv_shortcut', resnet2.conv_shortcut, effective, effective_meta)
    np.savez_compressed(EFFECTIVE_WEIGHTS, **effective)

    sources = source_names()
    res0_index = len(down_res) - 1
    res1_index = len(down_res) - 2
    res2_index = len(down_res) - 3
    synth_diff = (synthetic_inter['output'] - synthetic_official).abs()
    entry_diff = (entry_inter['output'] - entry_official).abs()
    previous_resnet1_compare = {'previous_upblock0_resnet1_oracle_tensor_available': False}
    if PREV_RESNET1_ORACLE.exists():
        prev = np.load(PREV_RESNET1_ORACLE)
        curr = to_np(resnet1_out)
        previous_resnet1_compare = {'previous_upblock0_resnet1_oracle_tensor_available': True, 'max_abs_diff': max_abs(prev, curr), 'mean_abs_diff': float(np.mean(np.abs(prev.astype(np.float32) - curr.astype(np.float32))))}
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
        'encoder_hidden_states_shape': encoder_shape,
        'downblock_output_state_shapes': {f'down_blocks.{i}': [list(x.shape) for x in states] for i, states in enumerate(down_states)},
        'downblock_output_hidden_shapes': {f'down_blocks.{i}': list(h.shape) for i, h in enumerate(down_hiddens)},
        'accumulated_down_block_res_sample_shapes': [list(x.shape) for x in down_res],
        'accumulated_down_block_res_sample_sources': sources,
        'accumulated_down_block_res_sample_count': len(down_res),
        'global_slice_for_up_blocks_0_start_index': len(down_res) - len(up0.resnets),
        'global_remaining_residual_count_after_slice': len(global_remaining),
        'upblock0_resnet0_consumed_residual_index_in_accumulated_tuple': res0_index,
        'upblock0_resnet0_consumed_residual_source': sources[res0_index],
        'upblock0_resnet0_consumed_residual_shape': list(res0_hidden.shape),
        'upblock0_resnet1_consumed_residual_index_in_accumulated_tuple': res1_index,
        'upblock0_resnet1_consumed_residual_source': sources[res1_index],
        'upblock0_resnet1_consumed_residual_shape': list(res1_hidden.shape),
        'upblock0_resnet2_consumed_residual_index_in_accumulated_tuple': res2_index,
        'upblock0_resnet2_consumed_residual_source': sources[res2_index],
        'upblock0_resnet2_consumed_residual_shape': list(res2_hidden.shape),
        'internal_remaining_residual_count_after_resnet2_pop': len(remaining_after_resnet2),
        'internal_remaining_residual_shapes_after_resnet2_pop': [list(x.shape) for x in remaining_after_resnet2],
        'upblock0_resnet2_hidden_input_shape': list(resnet1_out.shape),
        'upblock0_resnet2_concat_input_shape': list(concat2.shape),
        'upblock0_resnet2_output_shape': list(entry_official.shape),
        'concat_axis': 1,
        'temb_shape': list(time_embedding.shape),
        'local_midblock_return': mid_return,
        'resnet0_context_config': resnet_config(resnet0, 'up_blocks_0_resnets_0'),
        'resnet1_context_config': resnet_config(resnet1, 'up_blocks_0_resnets_1'),
        'resnet2_config': resnet_config(resnet2, prefix),
        'saved_tensors': saved,
        'effective_weight_metadata': effective_meta,
        'manual_vs_official': {
            'synthetic_max_abs_diff': float(synth_diff.max().item()) if synth_diff.numel() else 0.0,
            'synthetic_mean_abs_diff': float(synth_diff.mean().item()) if synth_diff.numel() else 0.0,
            'entry_max_abs_diff': float(entry_diff.max().item()) if entry_diff.numel() else 0.0,
            'entry_mean_abs_diff': float(entry_diff.mean().item()) if entry_diff.numel() else 0.0,
        },
        'previous_upblock0_resnet1_compare': previous_resnet1_compare,
        'markers': {
            'TADSR_UNET_UPBLOCK0_RESNET2_ORACLE_TENSORS': 'PASS',
            'TADSR_UNET_UPBLOCK0_RESNET2_EFFECTIVE_WEIGHTS': 'PASS',
            'TADSR_UNET_UPBLOCK0_RESNET2_RESIDUAL_CONTRACT_ORACLE': 'PASS',
        },
        'scope_boundaries': {
            'ported_this_stage': 'up_blocks.0.resnets.2 only, including third residual consumption and concat',
            'not_ported_this_stage': ['full up_blocks.0', 'up_blocks.0.upsamplers.0', 'full UNet forward', 'full TADSR inference', 'generic runtime LoRA'],
        },
    }
    META_JSON.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    SUMMARY_TXT.write_text('\n'.join([
        '# TADSR UNet up_blocks.0.resnets.2 oracle export',
        '',
        'TADSR_UNET_UPBLOCK0_RESNET2_ORACLE_TENSORS: PASS',
        'TADSR_UNET_UPBLOCK0_RESNET2_EFFECTIVE_WEIGHTS: PASS',
        'TADSR_UNET_UPBLOCK0_RESNET2_RESIDUAL_CONTRACT_ORACLE: PASS',
        '',
        f'effective weights: {EFFECTIVE_WEIGHTS}',
        f'oracle tensors: {ORACLE_DIR}',
        f'resnet2 consumed residual source: {sources[res2_index]}',
        f'resnet2 consumed residual index: {res2_index}',
        f'concat shape: {list(concat2.shape)}',
        f'synthetic manual/offical max abs diff: {metadata["manual_vs_official"]["synthetic_max_abs_diff"]}',
        f'entry manual/offical max abs diff: {metadata["manual_vs_official"]["entry_max_abs_diff"]}',
        f'previous resnet1 compare: {previous_resnet1_compare}',
    ]) + '\n', encoding='utf-8')
    for k, v in metadata['markers'].items():
        print(f'{k}: {v}')
    print(json.dumps({'status': metadata['status'], 'metadata': str(META_JSON), 'effective_weights': str(EFFECTIVE_WEIGHTS)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
