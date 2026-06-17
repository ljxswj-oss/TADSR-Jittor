#!/usr/bin/env python3
from __future__ import annotations
from unet_downblock0_attention0_common import *

def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        print(f"TADSR_UNET_ENTRY_RESNET0_ATTENTION0_ALIGNMENT: {blocked['status']}")
        blocked_result('jittor_unet_entry_resnet0_attention0_alignment', 'TADSR UNet Entry -> ResNet0 -> Attention0 Alignment', blocked)
        return 1
    t = bridge_tester()
    out = t.run_entry_resnet0_attention0(
        oracle['entry_synthetic_unet_sample'],
        oracle['entry_synthetic_unet_timestep'],
        oracle['entry_attention0_encoder_hidden_states'],
    )
    diagnostics = {
        'conv_in': add_metrics(out['conv_in'], oracle['entry_synthetic_unet_conv_in_output'], 1e-4),
        'time_embedding': add_metrics(out['time_embedding'], oracle['entry_synthetic_unet_time_embedding_output'], 1e-4),
        'resnet0_output': add_metrics(out['resnet0_output'], oracle['entry_downblock0_resnet0_output'], 1e-4),
        'attention0_output': add_metrics(out['attention0_output'], oracle['entry_downblock0_attention0_official_output'], 2e-3),
    }
    result = {'status': status_from_diagnostics(diagnostics), 'target': 'UNet entry -> down_blocks.0.resnets.0 -> down_blocks.0.attentions.0', 'note': 'Stops before resnets.1 / attentions.1 / downsampler.', 'diagnostics': diagnostics}
    write_report('jittor_unet_entry_resnet0_attention0_alignment', 'TADSR UNet Entry -> ResNet0 -> Attention0 Alignment', result)
    print(f"TADSR_UNET_ENTRY_RESNET0_ATTENTION0_ALIGNMENT: {result['status']}")
    return 0 if result['status'] in {'PASS','PARTIAL'} else 1
if __name__ == '__main__': raise SystemExit(main())
