#!/usr/bin/env python3
from __future__ import annotations

from unet_upblock1_resnet0_common import (
    add_metrics,
    blocked_result,
    bridge_tester,
    load_oracle,
    status_from_diagnostics,
    write_report,
)


def main() -> int:
    tensors, blocked = load_oracle()
    name = 'jittor_unet_upblock1_resnet0_synthetic_alignment'
    title = 'TADSR UNet up_blocks.1.resnets.0 Synthetic Alignment'
    if blocked:
        result = blocked_result(name, title, blocked)
        print('TADSR_UNET_UPBLOCK1_RESNET0_SYNTHETIC_ALIGNMENT:', result['status'])
        return 0 if result['status'] == 'PASS' else 1
    tester = bridge_tester()
    got = tester.run_upblock1_resnet0(
        tensors['synthetic_upblock1_resnet0_hidden_input'],
        tensors['synthetic_upblock1_resnet0_res_hidden'],
        tensors['synthetic_upblock1_resnet0_temb'],
        return_intermediates=True,
    )
    diag = {
        'output': add_metrics(got['output'], tensors['synthetic_upblock1_resnet0_output'], tolerance=2e-4),
    }
    status = status_from_diagnostics(diag)
    result = {
        'status': status,
        'diagnostics': diag,
        'note': 'Synthetic hidden/residual concat -> up_blocks.1.resnets.0 only.',
    }
    write_report(name, title, result)
    print('TADSR_UNET_UPBLOCK1_RESNET0_SYNTHETIC_ALIGNMENT:', status)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
