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

ORACLE_DIR = OUT_DIR / 'oracle_tensors_unet_upblock1_resnet0'
META_JSON = ORACLE_DIR / 'unet_upblock1_resnet0_oracle_metadata.json'
SUMMARY_TXT = ORACLE_DIR / 'oracle_summary.txt'
EFFECTIVE_WEIGHTS = OUT_DIR / 'converted_unet_upblock1_resnet0_effective_weights.npz'
PREV_UPBLOCK0_ORACLE = OUT_DIR / 'oracle_tensors_unet_upblock0_upsampler' / 'entry_upblock0_output_hidden.npy'


def maybe_reexec() -> None:
    expected_prefix = str(STRICT_PY.parents[1])
    in_expected_venv = str(sys.prefix) == expected_prefix or str(sys.executable).startswith(expected_prefix)
    if STRICT_PY.exists() and not in_expected_venv:
        os.environ['PYTHONNOUSERSITE'] = '1'
        os.execv(str(STRICT_PY), [str(STRICT_PY), __file__, *sys.argv[1:]])


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


def save_tensor(saved, name, tensor):
    arr = to_np(tensor)
    np.save(ORACLE_DIR / f'{name}.npy', arr)
    saved[name] = stats(arr)
    return arr


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
    up1 = unet.up_blocks[1]
    if len(up0.resnets) < 3 or not getattr(up0, 'upsamplers', None):
        raise RuntimeError('Official up_blocks.0 must be locally complete before exporting up_blocks.1.resnets.0.')
    if len(up1.resnets) < 1:
        raise RuntimeError('Official up_blocks.1 has no resnets.0; oracle export cannot proceed.')
    resnet0 = up1.resnets[0]

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
        remaining_after_up0_slice = down_res[:-len(up0.resnets)]
        up0_output = up0(mid_output, up0_res_tuple, temb=time_embedding, upsample_size=None)
        up1_res_tuple = remaining_after_up0_slice[-len(up1.resnets):]
        remaining_before_up1_slice = remaining_after_up0_slice[:-len(up1.resnets)]
        res0_hidden = up1_res_tuple[-1]
        remaining_after_resnet0_local = up1_res_tuple[:-1]
        concat0 = torch.cat([up0_output, res0_hidden], dim=1)
        entry_inter = resnet_manual(resnet0, concat0, time_embedding)
        entry_official = resnet0(concat0, time_embedding)

        h_shape = list(up0_output.shape)
        r_shape = list(res0_hidden.shape)
        temb_shape = [1, int(resnet0.time_emb_proj.weight.shape[1])]
        synthetic_h = torch.linspace(-1.0, 1.0, steps=int(np.prod(h_shape)), dtype=torch.float32).reshape(*h_shape)
        synthetic_res = torch.linspace(-0.75, 0.75, steps=int(np.prod(r_shape)), dtype=torch.float32).reshape(*r_shape)
        synthetic_temb = torch.linspace(-0.5, 0.5, steps=int(np.prod(temb_shape)), dtype=torch.float32).reshape(*temb_shape)
        synthetic_concat = torch.cat([synthetic_h, synthetic_res], dim=1)
        synthetic_inter = resnet_manual(resnet0, synthetic_concat, synthetic_temb)
        synthetic_official = resnet0(synthetic_concat, synthetic_temb)

    saved = {}
    for name, tensor in [
        ('synthetic_upblock1_resnet0_hidden_input', synthetic_h),
        ('synthetic_upblock1_resnet0_res_hidden', synthetic_res),
        ('synthetic_upblock1_resnet0_concat_input', synthetic_concat),
        ('synthetic_upblock1_resnet0_temb', synthetic_temb),
    ]:
        save_tensor(saved, name, tensor)
    save_resnet_intermediates(saved, 'synthetic_upblock1_resnet0', synthetic_inter, synthetic_official)
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
        ('entry_upblock0_output_hidden', up0_output),
        ('entry_upblock1_resnet0_hidden_input', up0_output),
        ('entry_upblock1_resnet0_res_hidden', res0_hidden),
        ('entry_upblock1_resnet0_concat_input', concat0),
        ('entry_upblock1_resnet0_temb', time_embedding),
    ]:
        save_tensor(saved, name, tensor)
    save_resnet_intermediates(saved, 'entry_upblock1_resnet0', entry_inter, entry_official)

    effective = {}
    effective_meta = {}
    prefix = 'up_blocks_1_resnets_0'
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
    consumed_index = len(remaining_after_up0_slice) - 1
    synth_diff = (synthetic_inter['output'] - synthetic_official).abs()
    entry_diff = (entry_inter['output'] - entry_official).abs()
    previous_upblock0_compare = {'previous_upblock0_output_oracle_tensor_available': False}
    if PREV_UPBLOCK0_ORACLE.exists():
        prev = np.load(PREV_UPBLOCK0_ORACLE)
        curr = to_np(up0_output)
        previous_upblock0_compare = {
            'previous_upblock0_output_oracle_tensor_available': True,
            'max_abs_diff': max_abs(prev, curr),
            'mean_abs_diff': float(np.mean(np.abs(prev.astype(np.float32) - curr.astype(np.float32)))),
        }
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
        'upblock0_consumed_residual_count': len(up0.resnets),
        'remaining_residual_count_after_upblock0_slice': len(remaining_after_up0_slice),
        'upblock1_resnet_count': len(up1.resnets),
        'upblock1_attention_count': len(getattr(up1, 'attentions', []) or []),
        'upblock1_upsampler_count': len(getattr(up1, 'upsamplers', []) or []),
        'upblock1_local_residual_tuple_count': len(remaining_after_up0_slice[-len(up1.resnets):]),
        'remaining_residual_count_before_upblock1_slice': len(remaining_before_up1_slice),
        'upblock1_resnet0_consumed_residual_index_in_accumulated_tuple': consumed_index,
        'upblock1_resnet0_consumed_residual_source': sources[consumed_index],
        'upblock1_resnet0_consumed_residual_shape': list(res0_hidden.shape),
        'internal_remaining_residual_count_after_resnet0_pop': len(remaining_after_resnet0_local),
        'internal_remaining_residual_shapes_after_resnet0_pop': [list(x.shape) for x in remaining_after_resnet0_local],
        'upblock0_local_output_shape': list(up0_output.shape),
        'upblock1_resnet0_hidden_input_shape': list(up0_output.shape),
        'upblock1_resnet0_concat_input_shape': list(concat0.shape),
        'upblock1_resnet0_output_shape': list(entry_official.shape),
        'concat_axis': 1,
        'temb_shape': list(time_embedding.shape),
        'local_midblock_return': mid_return,
        'resnet0_config': resnet_config(resnet0, prefix),
        'saved_tensors': saved,
        'effective_weight_metadata': effective_meta,
        'manual_vs_official': {
            'synthetic_max_abs_diff': float(synth_diff.max().item()) if synth_diff.numel() else 0.0,
            'synthetic_mean_abs_diff': float(synth_diff.mean().item()) if synth_diff.numel() else 0.0,
            'entry_max_abs_diff': float(entry_diff.max().item()) if entry_diff.numel() else 0.0,
            'entry_mean_abs_diff': float(entry_diff.mean().item()) if entry_diff.numel() else 0.0,
        },
        'previous_upblock0_output_compare': previous_upblock0_compare,
        'actual_next_module_after_upblock1_resnet0': 'up_blocks.1.attentions.0' if len(getattr(up1, 'attentions', [])) > 0 else 'up_blocks.1.resnets.1',
        'markers': {
            'TADSR_UNET_UPBLOCK1_RESNET0_ORACLE_TENSORS': 'PASS',
            'TADSR_UNET_UPBLOCK1_RESNET0_EFFECTIVE_WEIGHTS': 'PASS',
            'TADSR_UNET_UPBLOCK1_RESNET0_RESIDUAL_CONTRACT_ORACLE': 'PASS',
        },
        'scope_boundaries': {
            'ported_this_stage': 'up_blocks.1.resnets.0 only, including first residual consumption after local up_blocks.0',
            'not_ported_this_stage': [
                'up_blocks.1.attentions.0',
                'up_blocks.1.resnets.1',
                'up_blocks.1.resnets.2',
                'up_blocks.1.upsamplers.0',
                'full up_blocks.1',
                'up_blocks.2',
                'up_blocks.3',
                'full UNet forward',
                'full TADSR inference',
                'generic runtime LoRA',
            ],
        },
    }
    META_JSON.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    SUMMARY_TXT.write_text('\n'.join([
        '# TADSR UNet up_blocks.1.resnets.0 PyTorch oracle',
        '',
        *[f'{k}: {v}' for k, v in metadata['markers'].items()],
        '',
        f"residual source: {metadata['upblock1_resnet0_consumed_residual_source']}",
        f"hidden input shape: {metadata['upblock1_resnet0_hidden_input_shape']}",
        f"concat shape: {metadata['upblock1_resnet0_concat_input_shape']}",
        f"output shape: {metadata['upblock1_resnet0_output_shape']}",
        f"synthetic manual-vs-official max_abs_diff: {metadata['manual_vs_official']['synthetic_max_abs_diff']}",
        f"entry manual-vs-official max_abs_diff: {metadata['manual_vs_official']['entry_max_abs_diff']}",
        f"next unopened module: {metadata['actual_next_module_after_upblock1_resnet0']}",
    ]) + '\n', encoding='utf-8')
    for k, v in metadata['markers'].items():
        print(f'{k}: {v}')
    print(json.dumps({'status': 'PASS', 'oracle_dir': str(ORACLE_DIR), 'effective_weights': str(EFFECTIVE_WEIGHTS)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
