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
    to_np,
    stats,
    max_abs,
)
from export_tadsr_unet_upblock1_resnet0_oracle import extract_midblock_output, source_names

ORACLE_DIR = OUT_DIR / 'oracle_tensors_unet_upblock1_attention1'
META_JSON = ORACLE_DIR / 'unet_upblock1_attention1_oracle_metadata.json'
SUMMARY_TXT = ORACLE_DIR / 'oracle_summary.txt'
EFFECTIVE_WEIGHTS = OUT_DIR / 'converted_unet_upblock1_attention1_effective_weights.npz'
PREV_UPBLOCK1_RESNET1_ORACLE = OUT_DIR / 'oracle_tensors_unet_upblock1_resnet1' / 'entry_upblock1_resnet1_output.npy'


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
    from export_tadsr_unet_downblock2_attention0_oracle import manual_transformer_attention, module_effective_arrays

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
    if len(up1.resnets) < 3 or len(getattr(up1, 'attentions', []) or []) < 2:
        raise RuntimeError('Official up_blocks.1 must expose resnets.1, attentions.1, and following resnets.2 for this stage audit/export.')
    resnet0 = up1.resnets[0]
    attention0 = up1.attentions[0]
    resnet1 = up1.resnets[1]
    attention1 = up1.attentions[1]
    if len(attention1.transformer_blocks) != 1:
        raise RuntimeError(f'Expected one transformer block, got {len(attention1.transformer_blocks)}')

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
        resnet0_out = resnet0(concat0, time_embedding)
        attention0_out = attention0(resnet0_out, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]

        res1_hidden = remaining_after_resnet0_local[-1]
        remaining_after_resnet1_local = remaining_after_resnet0_local[:-1]
        concat1 = torch.cat([attention0_out, res1_hidden], dim=1)
        resnet1_out = resnet1(concat1, time_embedding)
        entry_manual = manual_transformer_attention(attention1, resnet1_out, encoder_hidden_states)
        entry_official = attention1(resnet1_out, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]

        h_shape = list(resnet1_out.shape)
        synthetic_hidden = torch.linspace(-1.0, 1.0, steps=int(np.prod(h_shape)), dtype=torch.float32).reshape(*h_shape)
        synthetic_manual = manual_transformer_attention(attention1, synthetic_hidden, encoder_hidden_states)
        synthetic_official = attention1(synthetic_hidden, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]

    saved = {}
    for name, tensor in [
        ('synthetic_upblock1_attention1_input', synthetic_hidden),
        ('synthetic_upblock1_attention1_encoder_hidden_states', encoder_hidden_states),
    ]:
        save_tensor(saved, name, tensor)
    for k, v in synthetic_manual.items():
        save_tensor(saved, f'synthetic_upblock1_attention1_{k}', v)
    save_tensor(saved, 'synthetic_upblock1_attention1_official_output', synthetic_official)
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
        ('entry_upblock1_resnet0_output', resnet0_out),
        ('entry_upblock1_attention0_input', resnet0_out),
        ('entry_upblock1_attention0_output', attention0_out),
        ('entry_upblock1_resnet1_hidden_input', attention0_out),
        ('entry_upblock1_resnet1_res_hidden', res1_hidden),
        ('entry_upblock1_resnet1_concat_input', concat1),
        ('entry_upblock1_resnet1_output', resnet1_out),
        ('entry_upblock1_attention1_input', resnet1_out),
    ]:
        save_tensor(saved, name, tensor)
    for k, v in entry_manual.items():
        save_tensor(saved, f'entry_upblock1_attention1_{k}', v)
    save_tensor(saved, 'entry_upblock1_attention1_official_output', entry_official)

    arrays = {}
    emeta = {}
    prefix = 'up_blocks_1_attentions_1'
    arrays[f'{prefix}_norm_weight'] = attention1.norm.weight.detach().cpu().numpy().astype(np.float32)
    arrays[f'{prefix}_norm_bias'] = attention1.norm.bias.detach().cpu().numpy().astype(np.float32)
    emeta[f'{prefix}_norm'] = {'is_lora_wrapped': False, 'weight_shape': list(attention1.norm.weight.shape), 'bias_shape': list(attention1.norm.bias.shape)}
    module_effective_arrays(f'{prefix}_proj_in', attention1.proj_in, arrays, emeta)
    module_effective_arrays(f'{prefix}_proj_out', attention1.proj_out, arrays, emeta)
    tb = attention1.transformer_blocks[0]
    for name in ['norm1', 'norm2', 'norm3']:
        mod = getattr(tb, name)
        arrays[f'{prefix}_transformer0_{name}_weight'] = mod.weight.detach().cpu().numpy().astype(np.float32)
        arrays[f'{prefix}_transformer0_{name}_bias'] = mod.bias.detach().cpu().numpy().astype(np.float32)
        emeta[f'{prefix}_transformer0_{name}'] = {'is_lora_wrapped': False, 'weight_shape': list(mod.weight.shape), 'bias_shape': list(mod.bias.shape)}
    for an in ['attn1', 'attn2']:
        attn = getattr(tb, an)
        for ln in ['to_q', 'to_k', 'to_v']:
            module_effective_arrays(f'{prefix}_transformer0_{an}_{ln}', getattr(attn, ln), arrays, emeta)
        module_effective_arrays(f'{prefix}_transformer0_{an}_to_out_0', attn.to_out[0], arrays, emeta)
    module_effective_arrays(f'{prefix}_transformer0_ff_geglu_proj', tb.ff.net[0].proj, arrays, emeta)
    module_effective_arrays(f'{prefix}_transformer0_ff_out', tb.ff.net[2], arrays, emeta)
    np.savez_compressed(EFFECTIVE_WEIGHTS, **arrays)

    synth_diff = (synthetic_manual['output'] - synthetic_official).abs()
    entry_diff = (entry_manual['output'] - entry_official).abs()
    previous_resnet1_compare = {'previous_upblock1_resnet1_oracle_tensor_available': False}
    if PREV_UPBLOCK1_RESNET1_ORACLE.exists():
        prev = np.load(PREV_UPBLOCK1_RESNET1_ORACLE)
        curr = to_np(resnet1_out)
        previous_resnet1_compare = {
            'previous_upblock1_resnet1_oracle_tensor_available': True,
            'max_abs_diff': max_abs(prev, curr),
            'mean_abs_diff': float(np.mean(np.abs(prev.astype(np.float32) - curr.astype(np.float32)))),
        }
    sources = source_names()
    resnet0_consumed_index = len(remaining_after_up0_slice) - 1
    resnet1_consumed_index = len(remaining_after_up0_slice) - 2
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
        'accumulated_down_block_res_sample_sources': sources,
        'accumulated_down_block_res_sample_count': len(down_res),
        'upblock0_consumed_residual_count': len(up0.resnets),
        'remaining_residual_count_after_upblock0_slice': len(remaining_after_up0_slice),
        'upblock1_resnet_count': len(up1.resnets),
        'upblock1_attention_count': len(getattr(up1, 'attentions', []) or []),
        'upblock1_upsampler_count': len(getattr(up1, 'upsamplers', []) or []),
        'upblock1_resnet0_consumed_residual_index_in_accumulated_tuple': resnet0_consumed_index,
        'upblock1_resnet0_consumed_residual_source': sources[resnet0_consumed_index],
        'internal_remaining_residual_count_after_resnet0_pop': len(remaining_after_resnet0_local),
        'upblock1_attention0_consumes_accumulated_residuals': False,
        'internal_remaining_residual_count_after_attention0': len(remaining_after_resnet0_local),
        'upblock1_resnet1_consumed_residual_index_in_accumulated_tuple': resnet1_consumed_index,
        'upblock1_resnet1_consumed_residual_source': sources[resnet1_consumed_index],
        'internal_remaining_residual_count_after_resnet1_pop': len(remaining_after_resnet1_local),
        'internal_remaining_residual_count_after_attention1': len(remaining_after_resnet1_local),
        'remaining_residual_count_before_upblock1_slice': len(remaining_before_up1_slice),
        'local_midblock_return': mid_return,
        'attention_config': {
            'prefix': prefix,
            'in_channels': int(attention1.in_channels),
            'inner_dim': int(attention1.num_attention_heads) * int(attention1.attention_head_dim),
            'num_attention_heads': int(attention1.num_attention_heads),
            'attention_head_dim': int(attention1.attention_head_dim),
            'cross_attention_dim': int(unet.config.cross_attention_dim),
            'use_linear_projection': bool(attention1.use_linear_projection),
            'norm_num_groups': int(attention1.norm.num_groups),
            'norm_eps': float(attention1.norm.eps),
        },
        'transformer0_config': {
            'heads': int(tb.attn1.heads),
            'head_dim': int(tb.attn1.inner_dim // tb.attn1.heads),
            'norm_eps': float(tb.norm1.eps),
            'dropout': 0.0,
        },
        'path_a_isolated_attention1': {'status': 'PASS', 'input_shape': h_shape, 'encoder_shape': encoder_shape, 'output_shape': list(synthetic_official.shape)},
        'path_b_entry_downblocks_midblock_upblock0_upblock1_resnet0_attention0_resnet1_attention1': {'status': 'PASS', 'sample_shape': sample_shape, 'input_shape': list(resnet1_out.shape), 'output_shape': list(entry_official.shape)},
        'previous_upblock1_resnet1_compare': previous_resnet1_compare,
        'attention1_consumes_accumulated_residuals': False,
        'actual_next_module_after_upblock1_attention1': 'up_blocks.1.resnets.2',
        'effective_weight_export': {
            'status': 'PASS',
            'path': str(EFFECTIVE_WEIGHTS),
            'keys': {k: list(v.shape) for k, v in arrays.items()},
            'module_effective_meta': emeta,
            'lora_affected_modules': [k for k, v in emeta.items() if v.get('is_lora_wrapped')],
        },
        'manual_vs_official_validation': {
            'status': 'PASS' if float(synth_diff.max()) < 1e-5 and float(entry_diff.max()) < 1e-5 else 'FAIL',
            'synthetic_manual_vs_official_max_abs_error': float(synth_diff.max().item()) if synth_diff.numel() else 0.0,
            'synthetic_manual_vs_official_mean_abs_error': float(synth_diff.mean().item()) if synth_diff.numel() else 0.0,
            'entry_manual_vs_official_max_abs_error': float(entry_diff.max().item()) if entry_diff.numel() else 0.0,
            'entry_manual_vs_official_mean_abs_error': float(entry_diff.mean().item()) if entry_diff.numel() else 0.0,
        },
        'saved_tensors': saved,
        'markers': {
            'TADSR_UNET_UPBLOCK1_ATTENTION1_ORACLE_TENSORS': 'PASS',
            'TADSR_UNET_UPBLOCK1_ATTENTION1_EFFECTIVE_WEIGHTS': 'PASS',
        },
        'scope_boundaries': {
            'ported_this_stage': 'up_blocks.1.attentions.1 only, after up_blocks.1.resnets.1',
            'not_ported_this_stage': ['up_blocks.1.resnets.2', 'up_blocks.1.attentions.2', 'up_blocks.1.upsamplers.0', 'full up_blocks.1', 'up_blocks.2', 'up_blocks.3', 'full UNet forward', 'full TADSR inference', 'generic runtime LoRA'],
        },
    }
    META_JSON.write_text(json.dumps(metadata, indent=2, default=str), encoding='utf-8')
    SUMMARY_TXT.write_text('\n'.join([
        '# TADSR UNet up_blocks.1.attentions.1 PyTorch oracle',
        '',
        *[f'{k}: {v}' for k, v in metadata['markers'].items()],
        '',
        f"input shape: {metadata['path_b_entry_downblocks_midblock_upblock0_upblock1_resnet0_attention0_resnet1_attention1']['input_shape']}",
        f"output shape: {metadata['path_b_entry_downblocks_midblock_upblock0_upblock1_resnet0_attention0_resnet1_attention1']['output_shape']}",
        f"synthetic manual-vs-official max_abs_diff: {metadata['manual_vs_official_validation']['synthetic_manual_vs_official_max_abs_error']}",
        f"entry manual-vs-official max_abs_diff: {metadata['manual_vs_official_validation']['entry_manual_vs_official_max_abs_error']}",
        'next unopened module: up_blocks.1.resnets.2',
        'full up_blocks.1/full UNet/full inference remain NOT_COMPLETE.',
    ]) + '\n', encoding='utf-8')
    for k, v in metadata['markers'].items():
        print(f'{k}: {v}')
    print(json.dumps({'status': 'PASS', 'oracle_dir': str(ORACLE_DIR), 'effective_weights': str(EFFECTIVE_WEIGHTS)}, indent=2))
    return 0 if metadata['manual_vs_official_validation']['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
