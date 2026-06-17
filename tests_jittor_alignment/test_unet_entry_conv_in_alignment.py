#!/usr/bin/env python3
from __future__ import annotations
from unet_entry_common import *


def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        print(f"TADSR_UNET_CONV_IN_ALIGNMENT: {blocked['status']}")
        blocked_result('jittor_unet_conv_in_alignment', 'TADSR UNet conv_in Alignment', blocked)
        return 1
    tester = tester_from_metadata()
    centered = tester.run_center_input(oracle['synthetic_sample'])
    conv = tester.run_conv_in(centered, already_centered=True)
    diagnostics = {
        'center_input': add_metrics(centered, oracle['unet_centered_sample'], tolerance=0.0),
        'conv_in': add_metrics(conv, oracle['unet_conv_in_output'], tolerance=1e-4),
    }
    result = {
        'status': status_from_diagnostics(diagnostics),
        'target': 'UNet center_input_sample + conv_in with exported effective conv weights',
        'note': 'Static effective conv_in weights include the official loaded conv_in weights and active conv_in LoRA delta. No downstream UNet blocks are executed.',
        'diagnostics': diagnostics,
        'weights': str(WEIGHTS),
    }
    write_report('jittor_unet_conv_in_alignment', 'TADSR UNet conv_in Alignment', result)
    print(f"TADSR_UNET_CENTER_INPUT_ALIGNMENT: {status_from_metrics(diagnostics['center_input'])}")
    print(f"TADSR_UNET_CONV_IN_ALIGNMENT: {status_from_metrics(diagnostics['conv_in'])}")
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
