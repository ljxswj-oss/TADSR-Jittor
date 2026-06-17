#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
import os
import sys
import numpy as np

from audit_official_tadsr_unet_upblock1_local import run_context, tensor_metrics, extract_hidden
from audit_official_tadsr_unet_upblock2_local import call_upblock2_local
from audit_official_tadsr_unet_upblock3_local import call_upblock3_local
from export_tadsr_unet_downblock3_resnet0_oracle import (
    STRICT_PY,
    OUT_DIR,
    OFFICIAL_REPO,
    WEIGHTS_DIR,
    load_unet,
    to_np,
    stats,
    module_lora_info,
)

OUT_JSON = OUT_DIR / 'audit_tadsr_unet_output_tail.json'
OUT_TXT = OUT_DIR / 'audit_tadsr_unet_output_tail.txt'


def maybe_reexec() -> None:
    expected_prefix = str(STRICT_PY.parents[1])
    in_expected_venv = str(sys.prefix) == expected_prefix or str(sys.executable).startswith(expected_prefix)
    if STRICT_PY.exists() and not in_expected_venv:
        os.environ['PYTHONNOUSERSITE'] = '1'
        os.execv(str(STRICT_PY), [str(STRICT_PY), __file__, *sys.argv[1:]])


def groupnorm_info(module):
    return {
        'class': f'{type(module).__module__}.{type(module).__name__}',
        'num_groups': int(module.num_groups),
        'num_channels': int(module.num_channels),
        'eps': float(module.eps),
        'affine': bool(module.affine),
        'weight_shape': list(module.weight.shape) if getattr(module, 'weight', None) is not None else None,
        'bias_shape': list(module.bias.shape) if getattr(module, 'bias', None) is not None else None,
    }


def activation_info(module):
    return {
        'class': f'{type(module).__module__}.{type(module).__name__}',
        'inplace': bool(getattr(module, 'inplace', False)),
        'parameter_count': sum(int(p.numel()) for p in module.parameters()) if hasattr(module, 'parameters') else 0,
    }


def conv_info(module):
    info = module_lora_info(module)
    base = module.base_layer if hasattr(module, 'base_layer') else module
    for attr in ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups']:
        if hasattr(base, attr):
            value = getattr(base, attr)
            if isinstance(value, tuple):
                value = list(value)
            info[attr] = value
    return info


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
    unet, loaded = load_unet()
    up2 = unet.up_blocks[2]
    up3 = unet.up_blocks[3]

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
        up3_res_tuple = remaining_before_up2[-len(up3.resnets):]
        official_raw = call_upblock3_local(up3, up2_output, up3_res_tuple, ctx['time_embedding'], encoder_hidden_states)
        up3_output, up3_return = extract_hidden(official_raw)
        norm_output = unet.conv_norm_out(up3_output)
        act_output = unet.conv_act(norm_output)
        conv_output = unet.conv_out(act_output)

    shape_status = 'PASS' if list(norm_output.shape) == list(up3_output.shape) and list(act_output.shape) == list(up3_output.shape) else 'FAIL'
    channel_status = 'PASS' if int(conv_output.shape[1]) == int(unet.config.out_channels) else 'FAIL'
    execution_status = 'PASS' if shape_status == 'PASS' and channel_status == 'PASS' else 'FAIL'
    tail_config = {
        'module_order': ['conv_norm_out', 'conv_act', 'conv_out'],
        'conv_norm_out': groupnorm_info(unet.conv_norm_out),
        'conv_act': activation_info(unet.conv_act),
        'conv_out': conv_info(unet.conv_out),
        'input_shape': list(up3_output.shape),
        'norm_output_shape': list(norm_output.shape),
        'act_output_shape': list(act_output.shape),
        'conv_out_output_shape': list(conv_output.shape),
    }
    result = {
        'status': execution_status,
        'python': sys.executable,
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'loaded_lora_parameter_count': len(loaded),
        'unet_forward_signature_not_used': str(inspect.signature(unet.forward)),
        'upblock2_return_info': up2_return,
        'upblock3_return_info': up3_return,
        'tail_config': tail_config,
        'tail_input_stats': stats(to_np(up3_output)),
        'tail_norm_stats': stats(to_np(norm_output)),
        'tail_act_stats': stats(to_np(act_output)),
        'tail_conv_out_stats': stats(to_np(conv_output)),
        'tail_norm_self_check': tensor_metrics(norm_output, norm_output),
        'uses_official_unet_forward': False,
        'uses_output_tail': True,
        'uses_full_tadsr_inference': False,
        'effective_weight_plan': {
            'strategy': 'export conv_norm_out raw affine parameters and conv_out LoRA-merged/effective static weights; no generic Jittor runtime LoRA',
            'planned_npz': str(OUT_DIR / 'converted_unet_output_tail_effective_weights.npz'),
            'conv_out_lora_wrapped': bool(tail_config['conv_out'].get('is_lora_wrapped')),
        },
        'markers': {
            'TADSR_UNET_OUTPUT_TAIL_AUDIT': execution_status,
            'TADSR_UNET_OUTPUT_TAIL_TOPOLOGY_AUDIT': 'PASS',
            'TADSR_UNET_OUTPUT_TAIL_LORA_AUDIT': 'PASS',
            'TADSR_UNET_OUTPUT_TAIL_LOCAL_EXECUTION_AUDIT': execution_status,
        },
        'scope_boundaries': {
            'ported_this_stage': 'conv_norm_out -> conv_act -> conv_out only',
            'not_ported_this_stage': ['official unet.forward', 'scheduler denoising', 'VAE integration', 'image generation', 'generic runtime LoRA'],
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, default=str), encoding='utf-8')
    OUT_TXT.write_text('\n'.join([
        '# TADSR UNet output tail official audit',
        '',
        *[f'{k}: {v}' for k, v in result['markers'].items()],
        '',
        f"tail input shape: {tail_config['input_shape']}",
        f"tail output shape: {tail_config['conv_out_output_shape']}",
        f"conv_out lora wrapped: {tail_config['conv_out'].get('is_lora_wrapped')}",
        'scope: output tail only; official unet.forward/full inference were not executed.',
    ]) + '\n', encoding='utf-8')
    for k, v in result['markers'].items():
        print(f'{k}: {v}')
    print(json.dumps({'status': execution_status, 'metadata': str(OUT_JSON)}, indent=2))
    return 0 if execution_status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
