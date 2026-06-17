#!/usr/bin/env python3
from __future__ import annotations

from unet_downblock3_resnet1_common import add_metrics, blocked_result, load_oracle, resnet1_tester, status_from_diagnostics, write_report


def main() -> int:
    tensors, blocked = load_oracle()
    name = 'jittor_unet_downblock3_resnet1_synthetic_alignment'
    title = 'TADSR UNet down_blocks.3.resnets.1 Synthetic Alignment'
    if blocked:
        result = blocked_result(name, title, blocked)
        print('TADSR_UNET_DOWNBLOCK3_RESNET1_SYNTHETIC_ALIGNMENT:', result['status'])
        return 0 if result['status'] == 'PASS' else 1
    tester = resnet1_tester()
    out = tester.run(tensors['synthetic_downblock3_resnet1_input'], tensors['synthetic_downblock3_resnet1_temb'], return_intermediates=True)
    diagnostics = {
        'synthetic_output': add_metrics(out['output'], tensors['synthetic_downblock3_resnet1_output'], tolerance=1e-4),
        'synthetic_conv2': add_metrics(out['conv2'], tensors['synthetic_downblock3_resnet1_conv2_output'], tolerance=1e-4),
        'synthetic_shortcut': add_metrics(out['shortcut'], tensors['synthetic_downblock3_resnet1_shortcut_output'], tolerance=1e-4),
    }
    status = status_from_diagnostics(diagnostics)
    result = {'status': status, 'diagnostics': diagnostics, 'note': 'Synthetic isolated hidden/temb -> down_blocks.3.resnets.1 only.'}
    write_report(name, title, result)
    print('TADSR_UNET_DOWNBLOCK3_RESNET1_SYNTHETIC_ALIGNMENT:', status)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
