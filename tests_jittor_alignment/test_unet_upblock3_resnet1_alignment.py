#!/usr/bin/env python3
from __future__ import annotations

from unet_upblock3_resnet1_common import (
    add_metrics,
    blocked_result,
    bridge_tester,
    load_oracle,
    squeeze_expected_if_needed,
    status_from_diagnostics,
    status_from_metrics,
    write_report,
)


def main() -> int:
    tensors, blocked = load_oracle()
    name = 'jittor_unet_upblock3_resnet1_alignment'
    title = 'TADSR UNet up_blocks.3.resnets.1 Alignment'
    if blocked:
        result = blocked_result(name, title, blocked)
        print('TADSR_UNET_UPBLOCK3_RESNET1_ALIGNMENT:', result['status'])
        return 0 if result['status'] == 'PASS' else 1

    tester = bridge_tester()
    got = tester.run_upblock3_resnet1(
        tensors['entry_upblock3_resnet1_hidden_input'],
        tensors['entry_upblock3_resnet1_res_hidden'],
        tensors['entry_upblock3_resnet1_temb'],
        return_intermediates=True,
    )
    expected_map = {
        'concat_input': 'entry_upblock3_resnet1_concat_input',
        'norm1': 'entry_upblock3_resnet1_norm1_output',
        'conv1': 'entry_upblock3_resnet1_conv1_output',
        'time_emb_proj': 'entry_upblock3_resnet1_time_emb_proj_output',
        'after_temb_add': 'entry_upblock3_resnet1_after_temb_add_output',
        'norm2': 'entry_upblock3_resnet1_norm2_output',
        'conv2': 'entry_upblock3_resnet1_conv2_output',
        'shortcut': 'entry_upblock3_resnet1_shortcut_output',
        'output': 'entry_upblock3_resnet1_output',
    }
    diag = {}
    for stage, tensor_name in expected_map.items():
        if tensor_name not in tensors:
            continue
        tol = 1e-6 if stage == 'concat_input' else 2e-4 if stage in {'conv1', 'after_temb_add'} else 1e-4
        diag[stage] = add_metrics(got[stage], squeeze_expected_if_needed(got[stage], tensors[tensor_name]), tolerance=tol)
    status = status_from_diagnostics(diag)
    result = {
        'status': status,
        'diagnostics': diag,
        'note': (
            'Only up_blocks.3.resnets.1 is executed after official residual concat. '
            'It stops before up_blocks.3.attentions.1, full up_blocks.3, full UNet '
            'forward, runtime LoRA, and full TADSR inference.'
        ),
    }
    write_report(name, title, result)
    labels = {
        'concat_input': 'TADSR_UNET_UPBLOCK3_RESNET1_RESIDUAL_CONCAT_ALIGNMENT',
        'norm1': 'TADSR_UNET_UPBLOCK3_RESNET1_NORM1_ALIGNMENT',
        'conv1': 'TADSR_UNET_UPBLOCK3_RESNET1_CONV1_ALIGNMENT',
        'time_emb_proj': 'TADSR_UNET_UPBLOCK3_RESNET1_TEMB_PROJ_ALIGNMENT',
        'conv2': 'TADSR_UNET_UPBLOCK3_RESNET1_CONV2_ALIGNMENT',
        'shortcut': 'TADSR_UNET_UPBLOCK3_RESNET1_SHORTCUT_ALIGNMENT',
        'output': 'TADSR_UNET_UPBLOCK3_RESNET1_ALIGNMENT',
    }
    for stage, label in labels.items():
        if stage in diag:
            print(f'{label}: {status_from_metrics(diag[stage])}')
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
