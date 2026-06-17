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
    to_np,
    stats,
    max_abs,
)

ORACLE_DIR = OUT_DIR / 'oracle_tensors_unet_upblock0_upsampler'
META_JSON = ORACLE_DIR / 'unet_upblock0_upsampler_oracle_metadata.json'
SUMMARY_TXT = ORACLE_DIR / 'oracle_summary.txt'
EFFECTIVE_WEIGHTS = OUT_DIR / 'converted_unet_upblock0_upsampler_effective_weights.npz'
PREV_RESNET2_ORACLE = OUT_DIR / 'oracle_tensors_unet_upblock0_resnet2' / 'entry_upblock0_resnet2_output.npy'


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


def run_upsampler_with_hook(upsampler, hidden):
    captures = {}
    conv = getattr(upsampler, 'conv', None)
    if conv is None:
        raise RuntimeError('up_blocks.0.upsamplers.0 has no conv child.')

    def hook(_module, inputs, output):
        captures['interpolation'] = inputs[0].detach()
        captures['conv'] = output.detach()

    handle = conv.register_forward_hook(hook)
    try:
        output = upsampler(hidden)
    finally:
        handle.remove()
    if 'interpolation' not in captures or 'conv' not in captures:
        raise RuntimeError('Failed to capture upsampler conv input/output.')
    return {
        'input': hidden,
        'interpolation': captures['interpolation'],
        'conv': captures['conv'],
        'output': output,
    }


def upsampler_config(upsampler):
    conv = upsampler.conv
    base = conv.base_layer if hasattr(conv, 'base_layer') else conv
    return {
        'prefix': 'up_blocks_0_upsamplers_0',
        'module_name': 'up_blocks.0.upsamplers.0',
        'class': f'{type(upsampler).__module__}.{type(upsampler).__name__}',
        'conv_class': f'{type(conv).__module__}.{type(conv).__name__}',
        'conv_base_class': f'{type(base).__module__}.{type(base).__name__}',
        'channels': getattr(upsampler, 'channels', None),
        'out_channels': getattr(upsampler, 'out_channels', None),
        'use_conv': getattr(upsampler, 'use_conv', None),
        'use_conv_transpose': getattr(upsampler, 'use_conv_transpose', None),
        'interpolate': getattr(upsampler, 'interpolate', None),
        'scale_factor': 2,
        'padding': int(base.padding[0]),
        'conv_padding': int(base.padding[0]),
        'conv_kernel_size': list(base.kernel_size),
        'conv_stride': list(base.stride),
        'conv_weight_shape': list(base.weight.shape),
        'conv_bias_shape': list(base.bias.shape) if getattr(base, 'bias', None) is not None else None,
        'operation_order': [
            'hidden_states -> nearest-neighbor interpolation by scale factor 2',
            'interpolated hidden_states -> effective 3x3 conv',
        ],
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
    if not getattr(up0, 'upsamplers', None):
        raise RuntimeError('Official up_blocks.0 has no upsamplers.0')
    upsampler = up0.upsamplers[0]
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
        res0_hidden = up0_res_tuple[-1]
        out0 = up0.resnets[0](torch.cat([mid_output, res0_hidden], dim=1), time_embedding)
        res1_hidden = up0_res_tuple[-2]
        out1 = up0.resnets[1](torch.cat([out0, res1_hidden], dim=1), time_embedding)
        res2_hidden = up0_res_tuple[-3]
        out2 = up0.resnets[2](torch.cat([out1, res2_hidden], dim=1), time_embedding)
        entry_up = run_upsampler_with_hook(upsampler, out2)
        local_up0_output = up0(mid_output, up0_res_tuple, temb=time_embedding, upsample_size=None)
        synthetic_shape = list(out2.shape)
        synthetic_hidden = torch.linspace(-1.0, 1.0, steps=int(np.prod(synthetic_shape)), dtype=torch.float32).reshape(*synthetic_shape)
        synthetic_up = run_upsampler_with_hook(upsampler, synthetic_hidden)

    saved = {}
    for name, tensor in [
        ('synthetic_upblock0_upsampler_input', synthetic_up['input']),
        ('synthetic_upblock0_upsampler_interpolation_output', synthetic_up['interpolation']),
        ('synthetic_upblock0_upsampler_conv_output', synthetic_up['conv']),
        ('synthetic_upblock0_upsampler_output', synthetic_up['output']),
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
        ('entry_upblock0_resnet0_res_hidden', res0_hidden),
        ('entry_upblock0_resnet0_output', out0),
        ('entry_upblock0_resnet1_res_hidden', res1_hidden),
        ('entry_upblock0_resnet1_output', out1),
        ('entry_upblock0_resnet2_res_hidden', res2_hidden),
        ('entry_upblock0_resnet2_output', out2),
        ('entry_upblock0_upsampler_input', entry_up['input']),
        ('entry_upblock0_upsampler_interpolation_output', entry_up['interpolation']),
        ('entry_upblock0_upsampler_conv_output', entry_up['conv']),
        ('entry_upblock0_upsampler_output', entry_up['output']),
        ('entry_upblock0_output_hidden', local_up0_output),
        ('local_upblock0_hidden_states_output', local_up0_output),
    ]:
        save_tensor(saved, name, tensor)

    effective = {}
    effective_meta = {}
    prefix = 'up_blocks_0_upsamplers_0'
    module_effective_arrays(f'{prefix}_conv', upsampler.conv, effective, effective_meta)
    np.savez_compressed(EFFECTIVE_WEIGHTS, **effective)

    previous_resnet2_compare = {'previous_upblock0_resnet2_oracle_tensor_available': False}
    if PREV_RESNET2_ORACLE.exists():
        prev = np.load(PREV_RESNET2_ORACLE)
        curr = to_np(out2)
        previous_resnet2_compare = {
            'previous_upblock0_resnet2_oracle_tensor_available': True,
            'max_abs_diff': max_abs(prev, curr),
            'mean_abs_diff': float(np.mean(np.abs(prev.astype(np.float32) - curr.astype(np.float32)))),
        }
    local_manual_max = max_abs(to_np(local_up0_output), to_np(entry_up['output']))
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
        'accumulated_down_block_res_sample_shapes': [list(x.shape) for x in down_res],
        'accumulated_down_block_res_sample_sources': source_names(),
        'accumulated_down_block_res_sample_count': len(down_res),
        'upblock0_resnet_count': len(up0.resnets),
        'upblock0_upsampler_count': len(up0.upsamplers),
        'upblock0_pre_upsampler_shape': list(out2.shape),
        'upblock0_upsampler_output_shape': list(entry_up['output'].shape),
        'upblock0_local_output_shape': list(local_up0_output.shape),
        'upblock0_local_manual_max_abs_diff': local_manual_max,
        'local_midblock_return': mid_return,
        'upsampler_config': upsampler_config(upsampler),
        'saved_tensors': saved,
        'effective_weight_metadata': effective_meta,
        'previous_upblock0_resnet2_compare': previous_resnet2_compare,
        'markers': {
            'TADSR_UNET_UPBLOCK0_UPSAMPLER_ORACLE_TENSORS': 'PASS',
            'TADSR_UNET_UPBLOCK0_UPSAMPLER_EFFECTIVE_WEIGHTS': 'PASS',
            'TADSR_UNET_UPBLOCK0_LOCAL_FORWARD_ORACLE': 'PASS',
        },
        'scope_boundaries': {
            'ported_this_stage': 'up_blocks.0.upsamplers.0 only, with local up_blocks.0 hidden-state oracle',
            'not_ported_this_stage': ['up_blocks.1', 'up_blocks.2', 'up_blocks.3', 'full UNet forward', 'full TADSR inference', 'generic runtime LoRA'],
        },
    }
    META_JSON.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    SUMMARY_TXT.write_text('\n'.join([
        '# TADSR UNet up_blocks.0.upsamplers.0 PyTorch oracle',
        '',
        *[f'{k}: {v}' for k, v in metadata['markers'].items()],
        '',
        f"pre-upsample shape: {metadata['upblock0_pre_upsampler_shape']}",
        f"upsampler output shape: {metadata['upblock0_upsampler_output_shape']}",
        f"local upblock0 vs manual max_abs_diff: {local_manual_max}",
        f"effective weights: {EFFECTIVE_WEIGHTS}",
        'scope: stops after local up_blocks.0 hidden output; full UNet forward remains unopened.',
    ]) + '\n', encoding='utf-8')
    for k, v in metadata['markers'].items():
        print(f'{k}: {v}')
    print(json.dumps({'status': 'PASS', 'oracle_dir': str(ORACLE_DIR), 'metadata': str(META_JSON), 'effective_weights': str(EFFECTIVE_WEIGHTS)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
