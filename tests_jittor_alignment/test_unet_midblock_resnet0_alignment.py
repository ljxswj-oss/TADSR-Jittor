#!/usr/bin/env python3
from __future__ import annotations

from unet_midblock_resnet0_common import add_metrics, blocked_result, load_oracle, resnet0_tester, squeeze_expected_if_needed, status_from_diagnostics, status_from_metrics, write_report


def main() -> int:
    tensors, blocked = load_oracle()
    name = 'jittor_unet_midblock_resnet0_alignment'
    title = 'TADSR UNet mid_block.resnets.0 Alignment'
    if blocked:
        result = blocked_result(name, title, blocked)
        print('TADSR_UNET_MIDBLOCK_RESNET0_ALIGNMENT:', result['status'])
        return 0 if result['status'] == 'PASS' else 1
    tester = resnet0_tester()
    got = tester.run(tensors['entry_midblock_resnet0_input'], tensors['entry_midblock_resnet0_temb'], return_intermediates=True)
    expected_map = {
        'norm1': 'entry_midblock_resnet0_norm1_output',
        'conv1': 'entry_midblock_resnet0_conv1_output',
        'time_emb_proj': 'entry_midblock_resnet0_time_emb_proj_output',
        'after_temb_add': 'entry_midblock_resnet0_after_temb_add_output',
        'norm2': 'entry_midblock_resnet0_norm2_output',
        'conv2': 'entry_midblock_resnet0_conv2_output',
        'shortcut': 'entry_midblock_resnet0_shortcut_output',
        'output': 'entry_midblock_resnet0_output',
    }
    diag = {stage: add_metrics(got[stage], squeeze_expected_if_needed(got[stage], tensors[path]), tolerance=1e-4) for stage, path in expected_map.items()}
    status = status_from_diagnostics(diag)
    result = {'status': status, 'diagnostics': diag, 'note': 'Only mid_block.resnets.0 is executed. mid_block.attentions.0/resnets.1/up_blocks/full UNet remain unopened.'}
    write_report(name, title, result)
    labels = {
        'norm1': 'TADSR_UNET_MIDBLOCK_RESNET0_NORM1_ALIGNMENT',
        'conv1': 'TADSR_UNET_MIDBLOCK_RESNET0_CONV1_ALIGNMENT',
        'time_emb_proj': 'TADSR_UNET_MIDBLOCK_RESNET0_TEMB_PROJ_ALIGNMENT',
        'conv2': 'TADSR_UNET_MIDBLOCK_RESNET0_CONV2_ALIGNMENT',
        'shortcut': 'TADSR_UNET_MIDBLOCK_RESNET0_SHORTCUT_ALIGNMENT',
        'output': 'TADSR_UNET_MIDBLOCK_RESNET0_ALIGNMENT',
    }
    for stage, label in labels.items():
        print(f'{label}: {status_from_metrics(diag[stage])}')
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
