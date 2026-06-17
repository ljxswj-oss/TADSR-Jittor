#!/usr/bin/env python3
from __future__ import annotations

from unet_upblock1_resnet2_common import (
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
    name = 'jittor_unet_upblock1_resnet2_alignment'
    title = 'TADSR UNet up_blocks.1.resnets.2 Alignment'
    if blocked:
        result = blocked_result(name, title, blocked)
        print('TADSR_UNET_UPBLOCK1_RESNET2_ALIGNMENT:', result['status'])
        return 0 if result['status'] == 'PASS' else 1
    tester = bridge_tester()
    got = tester.run_upblock1_resnet2(
        tensors['entry_upblock1_resnet2_hidden_input'],
        tensors['entry_upblock1_resnet2_res_hidden'],
        tensors['entry_upblock1_resnet2_temb'],
        return_intermediates=True,
    )
    expected_map = {
        'concat_input': 'entry_upblock1_resnet2_concat_input',
        'norm1': 'entry_upblock1_resnet2_norm1_output',
        'conv1': 'entry_upblock1_resnet2_conv1_output',
        'time_emb_proj': 'entry_upblock1_resnet2_time_emb_proj_output',
        'after_temb_add': 'entry_upblock1_resnet2_after_temb_add_output',
        'norm2': 'entry_upblock1_resnet2_norm2_output',
        'conv2': 'entry_upblock1_resnet2_conv2_output',
        'shortcut': 'entry_upblock1_resnet2_shortcut_output',
        'output': 'entry_upblock1_resnet2_output',
    }
    diag = {}
    for stage, path in expected_map.items():
        tol = 1e-6 if stage == 'concat_input' else 2e-4 if stage in {'conv1', 'after_temb_add'} else 1e-4
        diag[stage] = add_metrics(got[stage], squeeze_expected_if_needed(got[stage], tensors[path]), tolerance=tol)
    status = status_from_diagnostics(diag)
    result = {
        'status': status,
        'diagnostics': diag,
        'note': 'Only up_blocks.1.resnets.2 is executed after official attention1 output and residual concat; attention2/full upblock1/full UNet are not executed.',
    }
    write_report(name, title, result)
    labels = {
        'concat_input': 'TADSR_UNET_UPBLOCK1_RESNET2_RESIDUAL_CONCAT_ALIGNMENT',
        'norm1': 'TADSR_UNET_UPBLOCK1_RESNET2_NORM1_ALIGNMENT',
        'conv1': 'TADSR_UNET_UPBLOCK1_RESNET2_CONV1_ALIGNMENT',
        'time_emb_proj': 'TADSR_UNET_UPBLOCK1_RESNET2_TEMB_PROJ_ALIGNMENT',
        'conv2': 'TADSR_UNET_UPBLOCK1_RESNET2_CONV2_ALIGNMENT',
        'shortcut': 'TADSR_UNET_UPBLOCK1_RESNET2_SHORTCUT_ALIGNMENT',
        'output': 'TADSR_UNET_UPBLOCK1_RESNET2_ALIGNMENT',
    }
    for stage, label in labels.items():
        print(f'{label}: {status_from_metrics(diag[stage])}')
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
