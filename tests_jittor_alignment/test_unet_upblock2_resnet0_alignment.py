#!/usr/bin/env python3
from __future__ import annotations

from unet_upblock2_resnet0_common import (
    add_metrics,
    blocked_result,
    bridge_tester,
    load_oracle,
    status_from_diagnostics,
    status_from_metrics,
    write_report,
)
from unet_upblock1_resnet0_common import squeeze_expected_if_needed


def main() -> int:
    tensors, blocked = load_oracle()
    name = 'jittor_unet_upblock2_resnet0_alignment'
    title = 'TADSR UNet up_blocks.2.resnets.0 Alignment'
    if blocked:
        result = blocked_result(name, title, blocked)
        print('TADSR_UNET_UPBLOCK2_RESNET0_ALIGNMENT:', result['status'])
        return 0 if result['status'] == 'PASS' else 1
    tester = bridge_tester()
    got = tester.run_upblock2_resnet0(
        tensors['entry_upblock2_resnet0_hidden_input'],
        tensors['entry_upblock2_resnet0_res_hidden'],
        tensors['entry_upblock2_resnet0_temb'],
        return_intermediates=True,
    )
    expected = {
        'concat_input': 'entry_upblock2_resnet0_concat_input',
        'norm1': 'entry_upblock2_resnet0_norm1_output',
        'conv1': 'entry_upblock2_resnet0_conv1_output',
        'time_emb_proj': 'entry_upblock2_resnet0_time_emb_proj_output',
        'after_temb_add': 'entry_upblock2_resnet0_after_temb_add_output',
        'norm2': 'entry_upblock2_resnet0_norm2_output',
        'conv2': 'entry_upblock2_resnet0_conv2_output',
        'shortcut': 'entry_upblock2_resnet0_shortcut_output',
        'output': 'entry_upblock2_resnet0_output',
    }
    diag = {}
    for stage, tensor_name in expected.items():
        tol = 1e-6 if stage == 'concat_input' else 2e-4 if stage in {'conv1', 'after_temb_add'} else 1e-4
        diag[stage] = add_metrics(got[stage], squeeze_expected_if_needed(got[stage], tensors[tensor_name]), tolerance=tol)
    status = status_from_diagnostics(diag)
    result = {
        'status': status,
        'diagnostics': diag,
        'note': 'Only up_blocks.2.resnets.0 is executed. up_blocks.2.attentions.0/full up_blocks.2/full UNet are not executed.',
    }
    write_report(name, title, result)
    labels = {
        'concat_input': 'TADSR_UNET_UPBLOCK2_RESNET0_RESIDUAL_CONCAT_ALIGNMENT',
        'norm1': 'TADSR_UNET_UPBLOCK2_RESNET0_NORM1_ALIGNMENT',
        'conv1': 'TADSR_UNET_UPBLOCK2_RESNET0_CONV1_ALIGNMENT',
        'time_emb_proj': 'TADSR_UNET_UPBLOCK2_RESNET0_TEMB_PROJ_ALIGNMENT',
        'conv2': 'TADSR_UNET_UPBLOCK2_RESNET0_CONV2_ALIGNMENT',
        'shortcut': 'TADSR_UNET_UPBLOCK2_RESNET0_SHORTCUT_ALIGNMENT',
        'output': 'TADSR_UNET_UPBLOCK2_RESNET0_ALIGNMENT',
    }
    for stage, label in labels.items():
        print(f'{label}: {status_from_metrics(diag[stage])}')
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
