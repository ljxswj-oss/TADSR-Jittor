#!/usr/bin/env python3
from __future__ import annotations

from unet_downblock3_resnet1_common import add_metrics, blocked_result, bridge_tester, load_oracle, status_from_diagnostics, write_report


def main() -> int:
    tensors, blocked = load_oracle()
    name = 'jittor_unet_entry_downblock0_downblock1_downblock2_downblock3_resnet0_resnet1_alignment'
    title = 'TADSR UNet Entry + DownBlock0 + DownBlock1 + DownBlock2 + DownBlock3 ResNet0 + ResNet1 Bridge Alignment'
    if blocked:
        result = blocked_result(name, title, blocked)
        print('TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_DOWNBLOCK2_DOWNBLOCK3_RESNET0_RESNET1_ALIGNMENT:', result['status'])
        return 0 if result['status'] == 'PASS' else 1
    tester = bridge_tester()
    got = tester.run_entry_downblock0_downblock1_downblock2_downblock3_resnet0_resnet1(
        tensors['entry_synthetic_unet_sample'],
        tensors['entry_synthetic_unet_timestep'],
        tensors['entry_encoder_hidden_states'],
        return_intermediates=True,
        return_output_states=True,
    )
    diag = {
        'conv_in': add_metrics(got['conv_in'], tensors['entry_synthetic_unet_conv_in_output'], tolerance=1e-4),
        'time_embedding': add_metrics(got['time_embedding'], tensors['entry_synthetic_unet_time_embedding_output'], tolerance=1e-4),
        'downblock0_output': add_metrics(got['downblock0_output'], tensors['entry_downblock0_output_hidden'], tolerance=2e-3),
        'downblock1_output': add_metrics(got['downblock1_output'], tensors['entry_downblock1_output_hidden'], tolerance=2e-3),
        'downblock2_output': add_metrics(got['downblock2_output'], tensors['entry_downblock2_output_hidden'], tolerance=2e-3),
        'downblock3_resnet0_output': add_metrics(got['downblock3_resnet0_output'], tensors['entry_downblock3_resnet0_output'], tolerance=2e-3),
        'downblock3_resnet1_output': add_metrics(got['downblock3_resnet1_output'], tensors['entry_downblock3_resnet1_output'], tolerance=2e-3),
    }
    status = status_from_diagnostics(diag)
    result = {'status': status, 'diagnostics': diag, 'note': 'Local bridge: entry -> full down_blocks.0 -> full down_blocks.1 -> full down_blocks.2 -> down_blocks.3.resnets.0 -> down_blocks.3.resnets.1. Full UNet is not executed.'}
    write_report(name, title, result)
    print('TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_DOWNBLOCK2_DOWNBLOCK3_RESNET0_RESNET1_ALIGNMENT:', status)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
