#!/usr/bin/env python3
from __future__ import annotations
from unet_downblock0_downsampler_common import *

def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        print(f"TADSR_UNET_DOWNBLOCK0_DOWNSAMPLER_ALIGNMENT: {blocked['status']}")
        blocked_result('jittor_unet_downblock0_downsampler_alignment', 'TADSR UNet down_blocks.0.downsamplers.0 Alignment', blocked)
        return 1
    t = downsampler_tester()
    out = t.run_downsampler(oracle['synthetic_downblock0_downsampler_input'])
    diagnostics = {
        'downsampler_output': add_metrics(out, oracle['synthetic_downblock0_downsampler_output'], 2e-3),
    }
    result = {
        'status': status_from_diagnostics(diagnostics),
        'target': 'isolated down_blocks.0.downsamplers.0 only',
        'note': 'Single effective LoRA-merged stride-2 conv. Full UNet forward remains unopened.',
        'diagnostics': diagnostics,
    }
    write_report('jittor_unet_downblock0_downsampler_alignment', 'TADSR UNet down_blocks.0.downsamplers.0 Alignment', result)
    print(f"TADSR_UNET_DOWNBLOCK0_DOWNSAMPLER_ALIGNMENT: {result['status']}")
    return 0 if result['status'] in {'PASS','PARTIAL'} else 1

if __name__ == '__main__':
    raise SystemExit(main())
