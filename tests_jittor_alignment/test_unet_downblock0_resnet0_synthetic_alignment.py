#!/usr/bin/env python3
from __future__ import annotations
from unet_downblock0_resnet0_common import *


def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        print(f"TADSR_UNET_DOWNBLOCK0_RESNET0_SYNTHETIC_ALIGNMENT: {blocked['status']}")
        blocked_result('jittor_unet_downblock0_resnet0_synthetic_alignment', 'Synthetic TADSR UNet down_blocks.0.resnets.0 Alignment', blocked)
        return 1
    t = tester()
    out = t.run_downblock0_resnet0(oracle['synthetic_downblock0_resnet0_input'], oracle['synthetic_downblock0_resnet0_temb'])
    metrics = add_metrics(out, oracle['synthetic_downblock0_resnet0_output'], tolerance=1e-4)
    result = {'status': status_from_metrics(metrics), 'target': 'isolated synthetic hidden + synthetic temb -> down_blocks.0.resnets.0', 'metrics': metrics, 'note': 'No UNet entry, attention, downsampler, or full UNet forward.'}
    write_report('jittor_unet_downblock0_resnet0_synthetic_alignment', 'Synthetic TADSR UNet down_blocks.0.resnets.0 Alignment', result)
    print(f"TADSR_UNET_DOWNBLOCK0_RESNET0_SYNTHETIC_ALIGNMENT: {result['status']}")
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
