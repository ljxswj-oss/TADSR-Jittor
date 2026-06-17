#!/usr/bin/env python3
from __future__ import annotations
from unet_downblock0_downsampler_common import *

def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        print(f"TADSR_UNET_DOWNBLOCK0_ALIGNMENT: {blocked['status']}")
        blocked_result('jittor_unet_downblock0_alignment', 'TADSR UNet Full Local down_blocks.0 Alignment', blocked)
        return 1
    t = downblock0_tester()
    out = t.run_downblock0(
        oracle['entry_synthetic_unet_sample'],
        oracle['entry_synthetic_unet_timestep'],
        oracle['entry_downblock0_encoder_hidden_states'],
    )
    diagnostics = {
        'conv_in': add_metrics(out['conv_in'], oracle['entry_synthetic_unet_conv_in_output'], 1e-4),
        'time_embedding': add_metrics(out['time_embedding'], oracle['entry_synthetic_unet_time_embedding_output'], 1e-4),
        'resnet0_output': add_metrics(out['resnet0_output'], oracle['entry_downblock0_resnet0_output'], 1e-4),
        'attention0_output': add_metrics(out['attention0_output'], oracle['entry_downblock0_attention0_output'], 2e-3),
        'resnet1_output': add_metrics(out['resnet1_output'], oracle['entry_downblock0_resnet1_output'], 2e-3),
        'attention1_output': add_metrics(out['attention1_output'], oracle['entry_downblock0_attention1_output'], 2e-3),
        'downsampler_output': add_metrics(out['downsampler_output'], oracle['entry_downblock0_downsampler_output'], 2e-3),
        'block_forward_output': add_metrics(out['downblock0_output'], oracle['entry_downblock0_block_forward_output'], 2e-3),
    }
    result = {
        'status': status_from_diagnostics(diagnostics),
        'target': 'UNet entry -> resnet0 -> attention0 -> resnet1 -> attention1 -> downsampler',
        'note': 'Completes local down_blocks.0 only. Full UNet forward, later blocks and full TADSR inference remain NotImplemented.',
        'diagnostics': diagnostics,
    }
    write_report('jittor_unet_downblock0_alignment', 'TADSR UNet Full Local down_blocks.0 Alignment', result)
    print(f"TADSR_UNET_DOWNBLOCK0_ALIGNMENT: {result['status']}")
    return 0 if result['status'] in {'PASS','PARTIAL'} else 1

if __name__ == '__main__':
    raise SystemExit(main())
