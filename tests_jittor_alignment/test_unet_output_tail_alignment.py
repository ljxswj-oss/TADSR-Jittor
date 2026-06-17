#!/usr/bin/env python3
from __future__ import annotations

from unet_output_tail_common import (
    add_metrics,
    blocked_result,
    load_oracle,
    output_tail_tester,
    status_from_diagnostics,
    status_from_metrics,
    write_report,
)


def main() -> int:
    tensors, blocked = load_oracle()
    name = 'jittor_unet_output_tail_alignment'
    title = 'TADSR UNet Output Tail Alignment'
    if blocked:
        result = blocked_result(name, title, blocked)
        print('TADSR_UNET_OUTPUT_TAIL_NORM_ALIGNMENT:', result['status'])
        print('TADSR_UNET_OUTPUT_TAIL_ACT_ALIGNMENT:', result['status'])
        print('TADSR_UNET_OUTPUT_TAIL_CONV_OUT_ALIGNMENT:', result['status'])
        print('TADSR_UNET_OUTPUT_TAIL_ALIGNMENT:', result['status'])
        return 0 if result['status'] == 'PASS' else 1

    tester = output_tail_tester()
    got = tester.run_output_tail(tensors['entry_upblock3_output_hidden'], return_intermediates=True)
    diag = {
        'norm': add_metrics(got['output_tail_norm_output'], tensors['entry_output_tail_norm_output'], tolerance=1e-4),
        'act': add_metrics(got['output_tail_act_output'], tensors['entry_output_tail_act_output'], tolerance=1e-4),
        'conv_out': add_metrics(got['output_tail_conv_out_output'], tensors['entry_output_tail_conv_out_output'], tolerance=2e-4),
    }
    norm_status = status_from_metrics(diag['norm'])
    act_status = status_from_metrics(diag['act'])
    conv_status = status_from_metrics(diag['conv_out'])
    status = status_from_diagnostics(diag)
    result = {
        'status': status,
        'diagnostics': diag,
        'norm_status': norm_status,
        'act_status': act_status,
        'conv_out_status': conv_status,
        'note': (
            'Isolated output-tail validation from the PyTorch up_blocks.3 output tensor. '
            'This executes only conv_norm_out -> conv_act -> conv_out and does not call '
            'official UNet.forward or full TADSR inference.'
        ),
    }
    write_report(name, title, result)
    print('TADSR_UNET_OUTPUT_TAIL_NORM_ALIGNMENT:', norm_status)
    print('TADSR_UNET_OUTPUT_TAIL_ACT_ALIGNMENT:', act_status)
    print('TADSR_UNET_OUTPUT_TAIL_CONV_OUT_ALIGNMENT:', conv_status)
    print('TADSR_UNET_OUTPUT_TAIL_ALIGNMENT:', status)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
