#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import inspect
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
from export_tadsr_unet_downblock2_attention0_oracle import module_effective_arrays

ORACLE_DIR = OUT_DIR / 'oracle_tensors_unet_upblock1_upsampler'
META_JSON = ORACLE_DIR / 'unet_upblock1_upsampler_oracle_metadata.json'
SUMMARY_TXT = ORACLE_DIR / 'oracle_summary.txt'
EFFECTIVE_WEIGHTS = OUT_DIR / 'converted_unet_upblock1_upsampler_effective_weights.npz'
PREV_ATTENTION2_ORACLE = OUT_DIR / 'oracle_tensors_unet_upblock1_attention2' / 'entry_upblock1_attention2_output.npy'


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


def call_upsampler(upsampler, hidden_states, upsample_size=None):
    sig = inspect.signature(upsampler.forward)
    if 'upsample_size' in sig.parameters:
        return upsampler(hidden_states, upsample_size=upsample_size)
    return upsampler(hidden_states)


def run_upsampler_with_hook(upsampler, hidden, upsample_size=None):
    captures = {}
    conv = getattr(upsampler, 'conv', None)
    handle = None
    if conv is not None:
        def hook(_module, inputs, output):
            captures['interpolation'] = inputs[0].detach()
            captures['conv'] = output.detach()
        handle = conv.register_forward_hook(hook)
    try:
        output = call_upsampler(upsampler, hidden, upsample_size=upsample_size)
    finally:
        if handle is not None:
            handle.remove()
    if conv is not None and ('interpolation' not in captures or 'conv' not in captures):
        raise RuntimeError('Failed to capture upsampler conv input/output.')
    if 'interpolation' not in captures:
        captures['interpolation'] = output.detach()
    if 'conv' not in captures:
        captures['conv'] = output.detach()
    return {
        'input': hidden,
        'interpolation': captures['interpolation'],
        'conv': captures['conv'],
        'output': output,
    }


def conv_base(conv):
    return conv.base_layer if hasattr(conv, 'base_layer') else conv


def upsampler_config(upsampler):
    conv = getattr(upsampler, 'conv', None)
    base = conv_base(conv) if conv is not None else None
    return {
        'prefix': 'up_blocks_1_upsamplers_0',
        'module_name': 'up_blocks.1.upsamplers.0',
        'class': f'{type(upsampler).__module__}.{type(upsampler).__name__}',
        'forward_signature': str(inspect.signature(upsampler.forward)),
        'accepts_upsample_size': 'upsample_size' in inspect.signature(upsampler.forward).parameters,
        'conv_class': f'{type(conv).__module__}.{type(conv).__name__}' if conv is not None else None,
        'conv_base_class': f'{type(base).__module__}.{type(base).__name__}' if base is not None else None,
        'channels': getattr(upsampler, 'channels', None),
        'out_channels': getattr(upsampler, 'out_channels', None),
        'use_conv': getattr(upsampler, 'use_conv', None),
        'use_conv_transpose': getattr(upsampler, 'use_conv_transpose', None),
        'interpolate': getattr(upsampler, 'interpolate', None),
        'scale_factor': 2,
        'padding': int(base.padding[0]) if base is not None else 0,
        'conv_padding': int(base.padding[0]) if base is not None else 0,
        'conv_kernel_size': list(base.kernel_size) if base is not None else None,
        'conv_stride': list(base.stride) if base is not None else None,
        'conv_weight_shape': list(base.weight.shape) if base is not None else None,
        'conv_bias_shape': list(base.bias.shape) if base is not None and getattr(base, 'bias', None) is not None else None,
        'operation_order': [
            'hidden_states -> nearest-neighbor interpolation by scale factor 2',
            'interpolated hidden_states -> effective conv',
        ] if conv is not None else ['hidden_states -> interpolation/output'],
        'upsample_size': None,
    }


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
    if not getattr(up1, 'upsamplers', None):
        raise RuntimeError('Official up_blocks.1 has no upsamplers.0')
    upsampler = up1.upsamplers[0]

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
        up0_output = up0(mid_output, down_res[-len(up0.resnets):], temb=time_embedding, upsample_size=None)
        remaining_after_up0 = down_res[:-len(up0.resnets)]
        up1_res_tuple = remaining_after_up0[-len(up1.resnets):]
        remaining_before_up1 = remaining_after_up0[:-len(up1.resnets)]

        res0_hidden = up1_res_tuple[-1]
        remaining_after_resnet0 = up1_res_tuple[:-1]
        concat0 = torch.cat([up0_output, res0_hidden], dim=1)
        resnet0_out = up1.resnets[0](concat0, time_embedding)
        attention0_out = up1.attentions[0](resnet0_out, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
        res1_hidden = remaining_after_resnet0[-1]
        remaining_after_resnet1 = remaining_after_resnet0[:-1]
        concat1 = torch.cat([attention0_out, res1_hidden], dim=1)
        resnet1_out = up1.resnets[1](concat1, time_embedding)
        attention1_out = up1.attentions[1](resnet1_out, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
        res2_hidden = remaining_after_resnet1[-1]
        remaining_after_resnet2 = remaining_after_resnet1[:-1]
        concat2 = torch.cat([attention1_out, res2_hidden], dim=1)
        resnet2_out = up1.resnets[2](concat2, time_embedding)
        attention2_out = up1.attentions[2](resnet2_out, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
        entry_up = run_upsampler_with_hook(upsampler, attention2_out, upsample_size=None)
        synthetic_shape = list(attention2_out.shape)
        synthetic_hidden = torch.linspace(-1.0, 1.0, steps=int(np.prod(synthetic_shape)), dtype=torch.float32).reshape(*synthetic_shape)
        synthetic_up = run_upsampler_with_hook(upsampler, synthetic_hidden, upsample_size=None)

    saved = {}
    for name, tensor in [
        ('synthetic_upblock1_upsampler_input', synthetic_up['input']),
        ('synthetic_upblock1_upsampler_interpolation_output', synthetic_up['interpolation']),
        ('synthetic_upblock1_upsampler_conv_output', synthetic_up['conv']),
        ('synthetic_upblock1_upsampler_output', synthetic_up['output']),
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
        ('entry_upblock1_attention1_output', attention1_out),
        ('entry_upblock1_resnet2_hidden_input', attention1_out),
        ('entry_upblock1_resnet2_res_hidden', res2_hidden),
        ('entry_upblock1_resnet2_concat_input', concat2),
        ('entry_upblock1_resnet2_output', resnet2_out),
        ('entry_upblock1_attention2_input', resnet2_out),
        ('entry_upblock1_attention2_output', attention2_out),
        ('entry_upblock1_upsampler_input', entry_up['input']),
        ('entry_upblock1_upsampler_interpolation_output', entry_up['interpolation']),
        ('entry_upblock1_upsampler_conv_output', entry_up['conv']),
        ('entry_upblock1_upsampler_output', entry_up['output']),
    ]:
        save_tensor(saved, name, tensor)

    effective = {}
    effective_meta = {}
    conv = getattr(upsampler, 'conv', None)
    if conv is not None:
        module_effective_arrays('up_blocks_1_upsamplers_0_conv', conv, effective, effective_meta)
    np.savez_compressed(EFFECTIVE_WEIGHTS, **effective)

    previous_attention2_compare = {'previous_upblock1_attention2_oracle_tensor_available': False}
    if PREV_ATTENTION2_ORACLE.exists():
        prev = np.load(PREV_ATTENTION2_ORACLE)
        curr = to_np(attention2_out)
        previous_attention2_compare = {
            'previous_upblock1_attention2_oracle_tensor_available': True,
            'max_abs_diff': max_abs(prev, curr),
            'mean_abs_diff': float(np.mean(np.abs(prev.astype(np.float32) - curr.astype(np.float32)))),
        }
    sources = source_names()
    resnet0_idx = len(remaining_after_up0) - 1
    resnet1_idx = len(remaining_after_up0) - 2
    resnet2_idx = len(remaining_after_up0) - 3
    metadata = {
        'status': 'PASS',
        'python': sys.executable,
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'loaded_lora_parameter_count': len(loaded),
        'oracle_dir': str(ORACLE_DIR),
        'effective_weights': str(EFFECTIVE_WEIGHTS),
        'entry_sample_shape': sample_shape,
        'selected_timestep': [1],
        'encoder_hidden_states_shape': encoder_shape,
        'downblock_output_state_shapes': {f'down_blocks.{i}': [list(x.shape) for x in states] for i, states in enumerate(down_states)},
        'downblock_output_hidden_shapes': {f'down_blocks.{i}': list(h.shape) for i, h in enumerate(down_hiddens)},
        'accumulated_down_block_res_sample_sources': sources,
        'accumulated_down_block_res_sample_count': len(down_res),
        'remaining_residual_count_before_upblock1_slice': len(remaining_before_up1),
        'internal_remaining_residual_count_after_attention2': len(remaining_after_resnet2),
        'internal_remaining_residual_count_after_upsampler': len(remaining_after_resnet2),
        'upsampler_consumes_accumulated_residuals': False,
        'upblock1_resnet0_consumed_residual_source': sources[resnet0_idx],
        'upblock1_resnet1_consumed_residual_source': sources[resnet1_idx],
        'upblock1_resnet2_consumed_residual_source': sources[resnet2_idx],
        'upblock1_attention2_output_shape': list(attention2_out.shape),
        'upblock1_upsampler_output_shape': list(entry_up['output'].shape),
        'actual_next_module_after_upsampler': 'up_blocks.2',
        'local_midblock_return': mid_return,
        'upsampler_config': upsampler_config(upsampler),
        'saved_tensors': saved,
        'effective_weight_metadata': effective_meta,
        'previous_upblock1_attention2_compare': previous_attention2_compare,
        'path_a_isolated_upsampler': {
            'status': 'PASS',
            'input_shape': synthetic_shape,
            'output_shape': list(synthetic_up['output'].shape),
            'upsample_size': None,
        },
        'path_b_entry_to_upblock1_upsampler': {
            'status': 'PASS',
            'input_shape': sample_shape,
            'output_shape': list(entry_up['output'].shape),
            'stops_before': 'up_blocks.2',
        },
        'markers': {
            'TADSR_UNET_UPBLOCK1_UPSAMPLER_ORACLE_TENSORS': 'PASS',
            'TADSR_UNET_UPBLOCK1_UPSAMPLER_EFFECTIVE_WEIGHTS': 'PASS' if conv is not None else 'NOT_APPLICABLE',
            'TADSR_UNET_UPBLOCK1_UPSAMPLER_RESIDUAL_CONTRACT_ORACLE': 'PASS',
        },
        'scope_boundaries': {
            'ported_this_stage': 'up_blocks.1.upsamplers.0 only, with selected bridge through prior aligned up_blocks.1 modules',
            'not_ported_this_stage': ['full up_blocks.1 aggregate', 'up_blocks.2', 'up_blocks.3', 'full UNet forward', 'full TADSR inference', 'generic runtime LoRA'],
        },
    }
    META_JSON.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    SUMMARY_TXT.write_text('\n'.join([
        '# TADSR UNet up_blocks.1.upsamplers.0 PyTorch oracle',
        '',
        *[f'{k}: {v}' for k, v in metadata['markers'].items()],
        '',
        f"attention2 output / upsampler input shape: {metadata['upblock1_attention2_output_shape']}",
        f"upsampler output shape: {metadata['upblock1_upsampler_output_shape']}",
        f"effective weights: {EFFECTIVE_WEIGHTS}",
        'scope: stops after up_blocks.1.upsamplers.0; full up_blocks.1 aggregate and up_blocks.2 remain unopened.',
    ]) + '\n', encoding='utf-8')
    for k, v in metadata['markers'].items():
        print(f'{k}: {v}')
    print(json.dumps({'status': 'PASS', 'oracle_dir': str(ORACLE_DIR), 'metadata': str(META_JSON), 'effective_weights': str(EFFECTIVE_WEIGHTS)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
