#!/usr/bin/env python3
from __future__ import annotations

from unet_upblock0_upsampler_common import (
    add_metrics,
    blocked_result,
    load_oracle,
    status_from_diagnostics,
    upsampler_tester,
    write_report,
)


def main() -> int:
    tensors, blocked = load_oracle()
    name = 'jittor_unet_upblock0_upsampler_synthetic_alignment'
    title = 'TADSR UNet up_blocks.0.upsamplers.0 Synthetic Alignment'
    if blocked:
        result = blocked_result(name, title, blocked)
        print('TADSR_UNET_UPBLOCK0_UPSAMPLER_SYNTHETIC_ALIGNMENT:', result['status'])
        return 0 if result['status'] == 'PASS' else 1
    tester = upsampler_tester()
    got = tester.run(tensors['synthetic_upblock0_upsampler_input'], return_intermediates=True)
    diag = {
        'interpolation': add_metrics(got['interpolation'], tensors['synthetic_upblock0_upsampler_interpolation_output'], tolerance=2e-4),
        'conv': add_metrics(got['conv'], tensors['synthetic_upblock0_upsampler_conv_output'], tolerance=2e-4),
        'output': add_metrics(got['output'], tensors['synthetic_upblock0_upsampler_output'], tolerance=2e-4),
    }
    status = status_from_diagnostics(diag)
    result = {
        'status': status,
        'diagnostics': diag,
        'note': 'Synthetic isolated upsampler check; no downblocks/midblock/full UNet are executed.',
    }
    write_report(name, title, result)
    print('TADSR_UNET_UPBLOCK0_UPSAMPLER_SYNTHETIC_ALIGNMENT:', status)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
