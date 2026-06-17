#!/usr/bin/env python3
from __future__ import annotations

from unet_downblock3_resnet0_common import add_metrics, blocked_result, load_oracle, resnet0_tester, status_from_diagnostics, write_report


def main() -> int:
    tensors, blocked = load_oracle()
    name = 'jittor_unet_downblock3_resnet0_synthetic_alignment'
    title = 'TADSR UNet down_blocks.3.resnets.0 Synthetic Alignment'
    if blocked:
        result = blocked_result(name, title, blocked)
        print('TADSR_UNET_DOWNBLOCK3_RESNET0_SYNTHETIC_ALIGNMENT:', result['status'])
        return 0 if result['status'] == 'PASS' else 1
    tester = resnet0_tester()
    got = tester.run(tensors['synthetic_downblock3_resnet0_input'], tensors['synthetic_downblock3_resnet0_temb'], return_intermediates=True)
    diag = {
        'synthetic_output': add_metrics(got['output'], tensors['synthetic_downblock3_resnet0_output'], tolerance=2e-4),
        'synthetic_shortcut': add_metrics(got['shortcut'], tensors['synthetic_downblock3_resnet0_shortcut_output'], tolerance=1e-4),
    }
    status = status_from_diagnostics(diag)
    result = {'status': status, 'diagnostics': diag, 'note': 'Synthetic isolated hidden/temb path for down_blocks.3.resnets.0.'}
    write_report(name, title, result)
    print('TADSR_UNET_DOWNBLOCK3_RESNET0_SYNTHETIC_ALIGNMENT:', status)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
