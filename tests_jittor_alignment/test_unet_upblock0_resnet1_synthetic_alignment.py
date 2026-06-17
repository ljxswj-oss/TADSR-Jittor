#!/usr/bin/env python3
from __future__ import annotations
from unet_upblock0_resnet1_common import add_metrics, blocked_result, load_oracle, bridge_tester, status_from_diagnostics, write_report


def main() -> int:
    tensors, blocked = load_oracle()
    name = 'jittor_unet_upblock0_resnet1_synthetic_alignment'
    title = 'TADSR UNet up_blocks.0.resnets.1 Synthetic Alignment'
    if blocked:
        result = blocked_result(name, title, blocked)
        print('TADSR_UNET_UPBLOCK0_RESNET1_SYNTHETIC_ALIGNMENT:', result['status'])
        return 0 if result['status'] == 'PASS' else 1
    tester = bridge_tester()
    got = tester.run_upblock0_resnet1(tensors['synthetic_upblock0_resnet1_hidden_input'], tensors['synthetic_upblock0_resnet1_res_hidden'], tensors['synthetic_upblock0_resnet1_temb'], return_intermediates=True)
    diag = {'output': add_metrics(got['output'], tensors['synthetic_upblock0_resnet1_output'], tolerance=2e-4)}
    status = status_from_diagnostics(diag)
    result = {'status': status, 'diagnostics': diag, 'note': 'Synthetic hidden/residual concat -> up_blocks.0.resnets.1 only.'}
    write_report(name, title, result)
    print('TADSR_UNET_UPBLOCK0_RESNET1_SYNTHETIC_ALIGNMENT:', status)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
