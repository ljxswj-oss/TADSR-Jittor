#!/usr/bin/env python3
from __future__ import annotations

from unet_upblock0_upsampler_common import (
    add_metrics,
    blocked_result,
    load_oracle,
    squeeze_expected_if_needed,
    status_from_diagnostics,
    status_from_metrics,
    upsampler_tester,
    write_report,
)


def main() -> int:
    tensors, blocked = load_oracle()
    name = 'jittor_unet_upblock0_upsampler_alignment'
    title = 'TADSR UNet up_blocks.0.upsamplers.0 Alignment'
    if blocked:
        result = blocked_result(name, title, blocked)
        print('TADSR_UNET_UPBLOCK0_UPSAMPLER_ALIGNMENT:', result['status'])
        return 0 if result['status'] == 'PASS' else 1
    tester = upsampler_tester()
    got = tester.run(tensors['entry_upblock0_upsampler_input'], return_intermediates=True)
    expected = {
        'interpolation': 'entry_upblock0_upsampler_interpolation_output',
        'conv': 'entry_upblock0_upsampler_conv_output',
        'output': 'entry_upblock0_upsampler_output',
    }
    diag = {}
    for stage, path in expected.items():
        diag[stage] = add_metrics(got[stage], squeeze_expected_if_needed(got[stage], tensors[path]), tolerance=2e-4)
    status = status_from_diagnostics(diag)
    result = {
        'status': status,
        'diagnostics': diag,
        'note': 'Only up_blocks.0.upsamplers.0 is executed: nearest-2x interpolation plus exported effective conv.',
    }
    write_report(name, title, result)
    print('TADSR_UNET_UPBLOCK0_UPSAMPLER_INTERPOLATION_ALIGNMENT:', status_from_metrics(diag['interpolation']))
    print('TADSR_UNET_UPBLOCK0_UPSAMPLER_CONV_ALIGNMENT:', status_from_metrics(diag['conv']))
    print('TADSR_UNET_UPBLOCK0_UPSAMPLER_ALIGNMENT:', status)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
