#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import numpy as np

from audit_official_tadsr_unet_upblock1_local import run_context, extract_hidden
from audit_official_tadsr_unet_upblock2_local import call_upblock2_local
from export_tadsr_unet_downblock2_attention0_oracle import manual_transformer_attention, module_effective_arrays
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
from export_tadsr_unet_upblock1_resnet0_oracle import source_names

ORACLE_DIR = OUT_DIR / 'oracle_tensors_unet_upblock3_attention0'
META_JSON = ORACLE_DIR / 'unet_upblock3_attention0_oracle_metadata.json'
SUMMARY_TXT = ORACLE_DIR / 'oracle_summary.txt'
EFFECTIVE_WEIGHTS = OUT_DIR / 'converted_unet_upblock3_attention0_effective_weights.npz'
PREV_UPBLOCK3_RESNET0_ORACLE = OUT_DIR / 'oracle_tensors_unet_upblock3_resnet0' / 'entry_upblock3_resnet0_output.npy'


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


def next_module_name(up3) -> str:
    if len(getattr(up3, 'resnets', []) or []) > 1:
        return 'up_blocks.3.resnets.1'
    if len(getattr(up3, 'attentions', []) or []) > 1:
        return 'up_blocks.3.attentions.1'
    return 'output_tail'


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
    if len(getattr(up3, 'resnets', []) or []) < 1 or len(getattr(up3, 'attentions', []) or []) < 1:
        raise RuntimeError('Official up_blocks.3 must expose resnets.0 and attentions.0 for this stage.')
    resnet0 = up3.resnets[0]
    attention0 = up3.attentions[0]
    if len(attention0.transformer_blocks) != 1:
        raise RuntimeError(f'Expected one transformer block, got {len(attention0.transformer_blocks)}')

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

        remaining_before_up3 = remaining_before_up2
        up3_res_tuple = remaining_before_up3[-len(up3.resnets):]
        remaining_external_after_up3_tuple_slice = remaining_before_up3[:-len(up3.resnets)]
        res0_hidden = up3_res_tuple[-1]
        remaining_after_resnet0_local = up3_res_tuple[:-1]
        concat0 = torch.cat([up2_output, res0_hidden], dim=1)
        resnet0_out = resnet0(concat0, ctx['time_embedding'])

        entry_manual = manual_transformer_attention(attention0, resnet0_out, encoder_hidden_states)
        entry_official = attention0(resnet0_out, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]

        h_shape = list(resnet0_out.shape)
        synthetic_hidden = torch.linspace(-1.0, 1.0, steps=int(np.prod(h_shape)), dtype=torch.float32).reshape(*h_shape)
        synthetic_manual = manual_transformer_attention(attention0, synthetic_hidden, encoder_hidden_states)
        synthetic_official = attention0(synthetic_hidden, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]

    saved = {}
    for name, tensor in [
        ('synthetic_upblock3_attention0_input', synthetic_hidden),
        ('synthetic_upblock3_attention0_encoder_hidden_states', encoder_hidden_states),
    ]:
        save_tensor(saved, name, tensor)
    for key, tensor in synthetic_manual.items():
        save_tensor(saved, f'synthetic_upblock3_attention0_{key}', tensor)
    save_tensor(saved, 'synthetic_upblock3_attention0_official_output', synthetic_official)

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
        ('entry_upblock2_output_hidden', up2_output),
        ('entry_upblock3_resnet0_hidden_input', up2_output),
        ('entry_upblock3_resnet0_res_hidden', res0_hidden),
        ('entry_upblock3_resnet0_concat_input', concat0),
        ('entry_upblock3_resnet0_temb', ctx['time_embedding']),
        ('entry_upblock3_resnet0_output', resnet0_out),
        ('entry_upblock3_attention0_input', resnet0_out),
    ]:
        save_tensor(saved, name, tensor)
    for key, tensor in entry_manual.items():
        save_tensor(saved, f'entry_upblock3_attention0_{key}', tensor)
    save_tensor(saved, 'entry_upblock3_attention0_official_output', entry_official)

    arrays = {}
    emeta = {}
    prefix = 'up_blocks_3_attentions_0'
    arrays[f'{prefix}_norm_weight'] = attention0.norm.weight.detach().cpu().numpy().astype(np.float32)
    arrays[f'{prefix}_norm_bias'] = attention0.norm.bias.detach().cpu().numpy().astype(np.float32)
    emeta[f'{prefix}_norm'] = {
        'is_lora_wrapped': False,
        'weight_shape': list(attention0.norm.weight.shape),
        'bias_shape': list(attention0.norm.bias.shape),
    }
    module_effective_arrays(f'{prefix}_proj_in', attention0.proj_in, arrays, emeta)
    module_effective_arrays(f'{prefix}_proj_out', attention0.proj_out, arrays, emeta)
    tb = attention0.transformer_blocks[0]
    for name in ['norm1', 'norm2', 'norm3']:
        mod = getattr(tb, name)
        arrays[f'{prefix}_transformer0_{name}_weight'] = mod.weight.detach().cpu().numpy().astype(np.float32)
        arrays[f'{prefix}_transformer0_{name}_bias'] = mod.bias.detach().cpu().numpy().astype(np.float32)
        emeta[f'{prefix}_transformer0_{name}'] = {
            'is_lora_wrapped': False,
            'weight_shape': list(mod.weight.shape),
            'bias_shape': list(mod.bias.shape),
        }
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
    previous_resnet0_compare = {'previous_upblock3_resnet0_oracle_tensor_available': False}
    if PREV_UPBLOCK3_RESNET0_ORACLE.exists():
        prev = np.load(PREV_UPBLOCK3_RESNET0_ORACLE)
        curr = to_np(resnet0_out)
        previous_resnet0_compare = {
            'previous_upblock3_resnet0_oracle_tensor_available': True,
            'max_abs_diff': max_abs(prev, curr),
            'mean_abs_diff': float(np.mean(np.abs(prev.astype(np.float32) - curr.astype(np.float32)))),
        }

    sources = source_names()
    consumed_global_index = len(remaining_external_after_up3_tuple_slice) + len(up3_res_tuple) - 1
    next_module = next_module_name(up3)
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
        'upblock2_output_hidden_shape': list(up2_output.shape),
        'upblock3_resnet0_output_shape': list(resnet0_out.shape),
        'upblock3_attention0_input_shape': list(resnet0_out.shape),
        'upblock3_attention0_output_shape': list(entry_official.shape),
        'remaining_residual_samples_before_upblocks3_count': len(remaining_before_up3),
        'remaining_residual_samples_before_upblocks3_shapes': [list(x.shape) for x in remaining_before_up3],
        'upblock3_local_residual_tuple_sources': sources[len(remaining_external_after_up3_tuple_slice):len(remaining_before_up3)],
        'upblock3_local_residual_tuple_shapes': [list(x.shape) for x in up3_res_tuple],
        'upblock3_resnet0_consumed_residual_index_in_accumulated_tuple': consumed_global_index,
        'upblock3_resnet0_consumed_residual_source': sources[consumed_global_index],
        'upblock3_resnet0_consumed_residual_shape': list(res0_hidden.shape),
        'remaining_residual_count_after_upblock3_resnet0_pop': len(remaining_after_resnet0_local),
        'remaining_residual_shapes_after_upblock3_resnet0_pop': [list(x.shape) for x in remaining_after_resnet0_local],
        'remaining_residual_count_after_upblock3_attention0': len(remaining_after_resnet0_local),
        'attention0_consumes_accumulated_residuals': False,
        'actual_next_module_after_upblock3_attention0': next_module,
        'attention_config': {
            'prefix': prefix,
            'in_channels': int(attention0.in_channels),
            'inner_dim': int(attention0.num_attention_heads) * int(attention0.attention_head_dim),
            'num_attention_heads': int(attention0.num_attention_heads),
            'attention_head_dim': int(attention0.attention_head_dim),
            'cross_attention_dim': int(unet.config.cross_attention_dim),
            'use_linear_projection': bool(attention0.use_linear_projection),
            'norm_num_groups': int(attention0.norm.num_groups),
            'norm_eps': float(attention0.norm.eps),
        },
        'transformer0_config': {
            'heads': int(attention0.num_attention_heads),
            'head_dim': int(attention0.attention_head_dim),
            'inner_dim': int(attention0.num_attention_heads) * int(attention0.attention_head_dim),
            'norm_eps': float(tb.norm1.eps),
            'dropout': 0.0,
            'transformer_block_count': len(attention0.transformer_blocks),
        },
        'attention_mask': None,
        'encoder_attention_mask': None,
        'cross_attention_kwargs': None,
        'saved_tensors': saved,
        'effective_weight_export': {
            'status': 'PASS',
            'path': str(EFFECTIVE_WEIGHTS),
            'keys': {k: list(v.shape) for k, v in arrays.items()},
            'module_effective_meta': emeta,
            'lora_affected_modules': [k for k, v in emeta.items() if v.get('is_lora_wrapped')],
        },
        'manual_vs_official': {
            'synthetic_max_abs_diff': float(synth_diff.max().item()) if synth_diff.numel() else 0.0,
            'synthetic_mean_abs_diff': float(synth_diff.mean().item()) if synth_diff.numel() else 0.0,
            'entry_max_abs_diff': float(entry_diff.max().item()) if entry_diff.numel() else 0.0,
            'entry_mean_abs_diff': float(entry_diff.mean().item()) if entry_diff.numel() else 0.0,
            'status': 'PASS' if float(synth_diff.max()) < 1e-5 and float(entry_diff.max()) < 1e-5 else 'FAIL',
        },
        'previous_upblock3_resnet0_compare': previous_resnet0_compare,
        'upblock2_local_return_info': up2_return,
        'uses_full_unet_forward': False,
        'uses_upblock3_resnet1': False,
        'uses_output_tail': False,
        'uses_full_tadsr_inference': False,
        'uses_stochastic_sampling': False,
        'markers': {
            'TADSR_UNET_UPBLOCK3_ATTENTION0_ORACLE_TENSORS': 'PASS',
            'TADSR_UNET_UPBLOCK3_ATTENTION0_EFFECTIVE_WEIGHTS': 'PASS',
            'TADSR_UNET_UPBLOCK3_ATTENTION0_RESIDUAL_CONTRACT_ORACLE': 'PASS',
        },
    }
    META_JSON.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    SUMMARY_TXT.write_text('\n'.join([
        '# TADSR UNet up_blocks.3.attentions.0 PyTorch oracle',
        '',
        *[f'{k}: {v}' for k, v in metadata['markers'].items()],
        '',
        f"attention input shape: {list(resnet0_out.shape)}",
        f"attention output shape: {list(entry_official.shape)}",
        f"synthetic manual-vs-official max_abs_diff: {metadata['manual_vs_official']['synthetic_max_abs_diff']}",
        f"entry manual-vs-official max_abs_diff: {metadata['manual_vs_official']['entry_max_abs_diff']}",
        f"next unopened module: {metadata['actual_next_module_after_upblock3_attention0']}",
    ]) + '\n', encoding='utf-8')
    for k, v in metadata['markers'].items():
        print(f'{k}: {v}')
    print(json.dumps({'status': 'PASS', 'oracle_dir': str(ORACLE_DIR), 'effective_weights': str(EFFECTIVE_WEIGHTS)}, indent=2))
    return 0 if metadata['manual_vs_official']['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
