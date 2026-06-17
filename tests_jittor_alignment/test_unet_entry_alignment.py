#!/usr/bin/env python3
from __future__ import annotations
from unet_entry_common import *


def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        print(f"TADSR_UNET_ENTRY_ALIGNMENT: {blocked['status']}")
        blocked_result('jittor_unet_entry_alignment', 'TADSR UNet Entry Alignment', blocked)
        return 1
    tester = tester_from_metadata()
    out = tester.run_entry(oracle['synthetic_sample'], oracle['timestep'])
    diagnostics = {
        'center_input': add_metrics(out['centered_sample'], oracle['unet_centered_sample'], tolerance=0.0),
        'conv_in': add_metrics(out['conv_in'], oracle['unet_conv_in_output'], tolerance=1e-4),
        'time_proj': add_metrics(out['time_proj'], oracle['unet_time_proj_output'], tolerance=1e-5),
        'time_embedding_linear1': add_metrics(out['time_embedding_linear1'], oracle['unet_time_embedding_linear1_output'], tolerance=1e-4),
        'time_embedding_act': add_metrics(out['time_embedding_act'], oracle['unet_time_embedding_act_output'], tolerance=1e-4),
        'time_embedding': add_metrics(out['time_embedding'], oracle['unet_time_embedding_output'], tolerance=1e-3),
    }
    result = {
        'status': status_from_diagnostics(diagnostics),
        'target': 'UNet entry aggregate: center input, conv_in, time_proj, time_embedding',
        'note': 'Entry aggregate stops before down_blocks.0. This is not full UNet forward and not full TADSR inference.',
        'diagnostics': diagnostics,
        'weights': str(WEIGHTS),
    }
    write_report('jittor_unet_entry_alignment', 'TADSR UNet Entry Alignment', result)
    print(f"TADSR_UNET_ENTRY_ALIGNMENT: {result['status']}")
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
