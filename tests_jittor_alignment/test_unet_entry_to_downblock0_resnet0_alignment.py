#!/usr/bin/env python3
from __future__ import annotations
from unet_downblock0_resnet0_common import *


def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        print(f"TADSR_UNET_ENTRY_TO_DOWNBLOCK0_RESNET0_ALIGNMENT: {blocked['status']}")
        blocked_result('jittor_unet_entry_to_downblock0_resnet0_alignment', 'TADSR UNet Entry to down_blocks.0.resnets.0 Alignment', blocked)
        return 1
    t = tester()
    out = t.run_entry_to_downblock0_resnet0(oracle['entry_synthetic_unet_sample'], oracle['entry_synthetic_unet_timestep'], return_intermediates=True)
    diagnostics = {
        'conv_in': add_metrics(out['conv_in'], oracle['entry_synthetic_unet_conv_in_output'], tolerance=1e-4),
        'time_proj': add_metrics(out['time_proj'], oracle['entry_synthetic_unet_time_proj_output'], tolerance=1e-5),
        'time_embedding': add_metrics(out['time_embedding'], oracle['entry_synthetic_unet_time_embedding_output'], tolerance=1e-3),
        'resnet0_output': add_metrics(out['resnet0_output'], oracle['entry_downblock0_resnet0_output'], tolerance=1e-4),
    }
    result = {'status': status_from_diagnostics(diagnostics), 'target': 'UNet entry -> down_blocks.0.resnets.0 bridge', 'diagnostics': diagnostics, 'note': 'Stops before down_blocks.0.attentions.0.'}
    write_report('jittor_unet_entry_to_downblock0_resnet0_alignment', 'TADSR UNet Entry to down_blocks.0.resnets.0 Alignment', result)
    print(f"TADSR_UNET_ENTRY_TO_DOWNBLOCK0_RESNET0_ALIGNMENT: {result['status']}")
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
