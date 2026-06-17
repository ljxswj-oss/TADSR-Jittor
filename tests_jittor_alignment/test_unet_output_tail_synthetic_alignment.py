#!/usr/bin/env python3
from __future__ import annotations

from unet_output_tail_common import (
    add_metrics,
    blocked_result,
    load_oracle,
    output_tail_tester,
    status_from_diagnostics,
    write_report,
)


def main() -> int:
    tensors, blocked = load_oracle()
    name = 'jittor_unet_output_tail_synthetic_alignment'
    title = 'TADSR UNet Output Tail Synthetic Alignment'
    if blocked:
        result = blocked_result(name, title, blocked)
        print('TADSR_UNET_OUTPUT_TAIL_SYNTHETIC_ALIGNMENT:', result['status'])
        return 0 if result['status'] == 'PASS' else 1

    tester = output_tail_tester()
    got = tester.run_output_tail(tensors['synthetic_output_tail_input'], return_intermediates=True)
    diag = {
        'synthetic_norm': add_metrics(got['output_tail_norm_output'], tensors['synthetic_output_tail_norm_output'], tolerance=1e-4),
        'synthetic_act': add_metrics(got['output_tail_act_output'], tensors['synthetic_output_tail_act_output'], tolerance=1e-4),
        'synthetic_conv_out': add_metrics(got['output_tail_conv_out_output'], tensors['synthetic_output_tail_conv_out_output'], tolerance=2e-4),
    }
    status = status_from_diagnostics(diag)
    result = {
        'status': status,
        'diagnostics': diag,
        'note': 'Synthetic hidden tensor -> output tail alignment; no full UNet forward or full inference.',
    }
    write_report(name, title, result)
    print('TADSR_UNET_OUTPUT_TAIL_SYNTHETIC_ALIGNMENT:', status)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
