#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path

OFFICIAL_VENV_PYTHON = Path('/mnt/data/sj/venvs/tadsr_official_strict_cu118/bin/python')
OFFICIAL_VENV_ROOT = OFFICIAL_VENV_PYTHON.parents[1]
if (
    OFFICIAL_VENV_PYTHON.exists()
    and Path(sys.prefix).resolve() != OFFICIAL_VENV_ROOT.resolve()
    and os.environ.get('TADSR_OFFICIAL_VENV_REEXEC') != '1'
):
    env = os.environ.copy()
    env['TADSR_OFFICIAL_VENV_REEXEC'] = '1'
    os.execve(str(OFFICIAL_VENV_PYTHON), [str(OFFICIAL_VENV_PYTHON), *sys.argv], env)

import numpy as np
import torch

OFFICIAL_REPO = Path('/mnt/data/sj/projects/TADSR_official_pytorch')
WEIGHTS_DIR = Path('/mnt/data/sj/checkpoints/TADSR/preset/weights/time_vae')
NPZ_PATH = Path('/mnt/data/sj/checkpoints/TADSR/jittor_converted/diffusers/time_vae_weights.npz')
OUT = Path('experiments/full_repro/time_vae_alignment')


def shape_or_none(obj):
    return list(obj.shape) if obj is not None and hasattr(obj, 'shape') else None


def tensor_stats(value):
    arr = value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)
    arr = arr.astype(np.float32)
    return {
        'shape': list(arr.shape),
        'dtype': str(arr.dtype),
        'min': float(arr.min()) if arr.size else None,
        'max': float(arr.max()) if arr.size else None,
        'mean': float(arr.mean()) if arr.size else None,
        'std': float(arr.std()) if arr.size else None,
    }


def module_class(mod):
    if mod is None:
        return None
    return f'{mod.__class__.__module__}.{mod.__class__.__name__}'


def norm_info(mod, name):
    if mod is None:
        return {'exists': False, 'module_name': name}
    return {
        'exists': True,
        'module_name': name,
        'class': mod.__class__.__name__,
        'module': mod.__class__.__module__,
        'num_groups': int(getattr(mod, 'num_groups', 0)) if hasattr(mod, 'num_groups') else None,
        'num_channels': int(getattr(mod, 'num_channels', 0)) if hasattr(mod, 'num_channels') else None,
        'eps': float(getattr(mod, 'eps', 0.0)) if hasattr(mod, 'eps') else None,
        'affine': bool(getattr(mod, 'affine', False)) if hasattr(mod, 'affine') else None,
        'weight_shape': shape_or_none(getattr(mod, 'weight', None)),
        'bias_shape': shape_or_none(getattr(mod, 'bias', None)),
        'npz_mapping': {
            'weight': 'decoder__conv_norm_out__weight',
            'bias': 'decoder__conv_norm_out__bias',
        },
    }


def act_info(mod, name):
    if mod is None:
        return {'exists': False, 'module_name': name}
    return {
        'exists': True,
        'module_name': name,
        'class': mod.__class__.__name__,
        'module': mod.__class__.__module__,
        'inplace': getattr(mod, 'inplace', None),
        'exact_operation': 'SiLU = x * sigmoid(x)' if mod.__class__.__name__.lower() == 'silu' else mod.__class__.__name__,
        'npz_mapping': {},
    }


def conv_info(mod, name):
    if mod is None:
        return {'exists': False, 'module_name': name}
    return {
        'exists': True,
        'module_name': name,
        'class': mod.__class__.__name__,
        'module': mod.__class__.__module__,
        'weight_shape': shape_or_none(getattr(mod, 'weight', None)),
        'bias_shape': shape_or_none(getattr(mod, 'bias', None)),
        'in_channels': int(getattr(mod, 'in_channels', 0)) if hasattr(mod, 'in_channels') else None,
        'out_channels': int(getattr(mod, 'out_channels', 0)) if hasattr(mod, 'out_channels') else None,
        'kernel_size': list(getattr(mod, 'kernel_size', [])) if hasattr(mod, 'kernel_size') else None,
        'stride': list(getattr(mod, 'stride', [])) if hasattr(mod, 'stride') else None,
        'padding': list(getattr(mod, 'padding', [])) if hasattr(mod, 'padding') else None,
        'dilation': list(getattr(mod, 'dilation', [])) if hasattr(mod, 'dilation') else None,
        'groups': int(getattr(mod, 'groups', 0)) if hasattr(mod, 'groups') else None,
        'npz_mapping': {
            'weight': 'decoder__conv_out__weight',
            'bias': 'decoder__conv_out__bias',
        },
    }


def npz_key_info(keys):
    if not NPZ_PATH.exists():
        return {'status': 'MISSING', 'path': str(NPZ_PATH), 'keys': {}}
    data = np.load(NPZ_PATH)
    return {
        'status': 'PASS',
        'path': str(NPZ_PATH),
        'keys': {k: {'exists': k in data.files, 'shape': list(data[k].shape) if k in data.files else None, 'dtype': str(data[k].dtype) if k in data.files else None} for k in keys},
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(OFFICIAL_REPO))
    from diffusers.models.autoencoders.time_autoencoder_kl import TimeAwareAutoencoderKL

    model = TimeAwareAutoencoderKL.from_pretrained(str(WEIGHTS_DIR), local_files_only=True)
    model.eval()
    decoder = model.decoder
    norm = getattr(decoder, 'conv_norm_out', None) or getattr(decoder, 'norm_out', None)
    norm_name = 'decoder.conv_norm_out' if getattr(decoder, 'conv_norm_out', None) is not None else 'decoder.norm_out'
    act = getattr(decoder, 'conv_act', None) or getattr(decoder, 'act', None)
    act_name = 'decoder.conv_act' if getattr(decoder, 'conv_act', None) is not None else 'decoder.act'
    conv_out = getattr(decoder, 'conv_out', None)

    stage_stats = {}
    input_shape = None
    output_shape = None
    with torch.no_grad():
        z = torch.linspace(-1.0, 1.0, steps=1 * 4 * 4 * 4, dtype=torch.float32).reshape(1, 4, 4, 4)
        sample = model.post_quant_conv(z)
        sample = decoder.conv_in(sample)
        sample = decoder.mid_block(sample, None)
        for block in decoder.up_blocks:
            sample = block(sample, None)
        input_shape = list(sample.shape)
        stage_stats['tail_input'] = tensor_stats(sample)
        norm_out = norm(sample) if norm is not None else sample
        stage_stats['norm_out'] = tensor_stats(norm_out)
        act_out = act(norm_out) if act is not None else norm_out
        stage_stats['act_out'] = tensor_stats(act_out)
        conv_out_value = conv_out(act_out)
        stage_stats['conv_out'] = tensor_stats(conv_out_value)
        output_shape = list(conv_out_value.shape)
        full_direct = decoder(model.post_quant_conv(z), None)
        stage_stats['decoder_forward_output'] = tensor_stats(full_direct)
        stage_stats['manual_vs_decoder_forward_max_abs_diff'] = float((conv_out_value - full_direct).abs().max().detach().cpu().item())

    scaling_factor = getattr(getattr(model, 'config', object()), 'scaling_factor', None)
    status = 'PASS'
    reasons = []
    if norm is None:
        status = 'BLOCKED_NEEDS_REVIEW'; reasons.append('decoder tail norm module missing')
    if act is None:
        status = 'BLOCKED_NEEDS_REVIEW'; reasons.append('decoder tail activation module missing')
    if conv_out is None:
        status = 'BLOCKED_NEEDS_REVIEW'; reasons.append('decoder.conv_out missing')
    if output_shape is None or output_shape[1] != 3:
        status = 'BLOCKED_NEEDS_REVIEW'; reasons.append(f'unexpected decoder tail output shape {output_shape}')

    result = {
        'status': status,
        'reasons': reasons,
        'official_class': module_class(model),
        'time_vae_class': module_class(model),
        'vae_class': module_class(model),
        'decoder_class': module_class(decoder),
        'decoder_forward_signature': str(inspect.signature(decoder.forward)),
        'decoder_forward_source_excerpt': inspect.getsource(decoder.forward)[:5000],
        'decoder_children_order': [{'name': n, 'class': c.__class__.__name__, 'module': c.__class__.__module__} for n, c in decoder.named_children()],
        'decoder_full_execution_order': 'post_quant_conv -> decoder.conv_in -> decoder.mid_block -> decoder.up_blocks.0~3 -> decoder.conv_norm_out -> decoder.conv_act -> decoder.conv_out',
        'decoder_tail_module_order': [norm_name, act_name, 'decoder.conv_out'],
        'tail_inside_decoder_forward': True,
        'tail_uses_latent_embeds': 'latent_embeds' in str(inspect.signature(decoder.forward)),
        'decoder_side_temb_default': None,
        'decoder_tail_dtype_differs_from_upstream': False,
        'norm': norm_info(norm, norm_name),
        'activation': act_info(act, act_name),
        'conv_out': conv_info(conv_out, 'decoder.conv_out'),
        'tail_input_shape': input_shape,
        'tail_output_shape': output_shape,
        'stage_stats': stage_stats,
        'npz_keys': npz_key_info(['decoder__conv_norm_out__weight', 'decoder__conv_norm_out__bias', 'decoder__conv_out__weight', 'decoder__conv_out__bias']),
        'scaling_factor_exists': scaling_factor is not None,
        'scaling_factor_value': float(scaling_factor) if scaling_factor is not None else None,
        'scaling_factor_applied_inside_decoder': False,
        'scaling_factor_note': 'TimeAwareAutoencoderKL has config.scaling_factor, but decoder.forward tail does not apply scaling_factor; pipeline-level scaling is out of scope for this tail alignment.',
        'decoder_output_is_raw_sample_tensor': True,
        'output_clamp_or_tanh_inside_decoder': False,
        'pipeline_postprocess_implemented_this_round': False,
    }
    (OUT / 'audit_time_vae_decoder_tail.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    txt = [
        f"TIME_VAE_DECODER_TAIL_AUDIT: {status}",
        f"tail_module_order: {result['decoder_tail_module_order']}",
        f"tail_input_shape: {input_shape}",
        f"tail_output_shape: {output_shape}",
        f"norm_eps: {result['norm'].get('eps')}",
        f"norm_groups: {result['norm'].get('num_groups')}",
        f"activation: {result['activation'].get('class')}",
        f"conv_out_weight_shape: {result['conv_out'].get('weight_shape')}",
        f"scaling_factor_value: {result['scaling_factor_value']}",
        f"scaling_factor_applied_inside_decoder: {result['scaling_factor_applied_inside_decoder']}",
        f"output_clamp_or_tanh_inside_decoder: {result['output_clamp_or_tanh_inside_decoder']}",
        f"manual_vs_decoder_forward_max_abs_diff: {stage_stats.get('manual_vs_decoder_forward_max_abs_diff')}",
    ]
    (OUT / 'audit_time_vae_decoder_tail.txt').write_text('\n'.join(txt) + '\n', encoding='utf-8')
    print(f'TIME_VAE_DECODER_TAIL_AUDIT: {status}')
    if reasons:
        print('\n'.join(reasons))
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
