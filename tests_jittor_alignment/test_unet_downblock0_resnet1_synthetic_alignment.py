#!/usr/bin/env python3
from __future__ import annotations
from unet_downblock0_resnet1_common import *


def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        print(f"TADSR_UNET_DOWNBLOCK0_RESNET1_SYNTHETIC_ALIGNMENT: {blocked['status']}")
        blocked_result('jittor_unet_downblock0_resnet1_synthetic_alignment', 'TADSR UNet down_blocks.0.resnets.1 Synthetic Alignment', blocked)
        return 1
    t = resnet1_tester()
    out = t.run(oracle['synthetic_downblock0_resnet1_input'], oracle['synthetic_downblock0_resnet1_temb'], return_intermediates=True)
    diagnostics = {
        'synthetic_output': add_metrics(out['output'], oracle['synthetic_downblock0_resnet1_output'], 1e-4),
    }
    result = {
        'status': status_from_diagnostics(diagnostics),
        'target': 'isolated synthetic hidden/temb -> down_blocks.0.resnets.1',
        'note': 'Does not enter entry, resnet0, attention0, attention1, downsampler, full down_blocks.0, or full UNet.',
        'diagnostics': diagnostics,
    }
    write_report('jittor_unet_downblock0_resnet1_synthetic_alignment', 'TADSR UNet down_blocks.0.resnets.1 Synthetic Alignment', result)
    print(f"TADSR_UNET_DOWNBLOCK0_RESNET1_SYNTHETIC_ALIGNMENT: {result['status']}")
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
