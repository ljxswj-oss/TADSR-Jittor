#!/usr/bin/env python3
from __future__ import annotations

from unet_downblock1_attention0_common import add_metrics, blocked_result, bridge_tester, load_oracle, status_from_diagnostics, write_report


def main() -> int:
    tensors, blocked = load_oracle()
    name = 'jittor_unet_entry_downblock0_downblock1_resnet0_attention0_alignment'
    title = 'TADSR UNet Entry + DownBlock0 + DownBlock1 ResNet0 + Attention0 Bridge Alignment'
    if blocked:
        result = blocked_result(name, title, blocked)
        print('TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_RESNET0_ATTENTION0_ALIGNMENT:', result['status'])
        return 0 if result['status'] == 'PASS' else 1
    tester = bridge_tester()
    got = tester.run_entry_downblock0_downblock1_resnet0_attention0(
        tensors['entry_synthetic_unet_sample'],
        tensors['entry_synthetic_unet_timestep'],
        tensors['entry_encoder_hidden_states'],
        return_intermediates=True,
    )
    diag = {
        'conv_in': add_metrics(got['conv_in'], tensors['entry_synthetic_unet_conv_in_output'], tolerance=1e-4),
        'time_embedding': add_metrics(got['time_embedding'], tensors['entry_synthetic_unet_time_embedding_output'], tolerance=1e-4),
        'downblock0_output': add_metrics(got['downblock0_output'], tensors['entry_downblock0_output_hidden'], tolerance=2e-3),
        'downblock1_resnet0_output': add_metrics(got['downblock1_resnet0_output'], tensors['entry_downblock1_resnet0_output'], tolerance=2e-3),
        'downblock1_attention0_output': add_metrics(got['downblock1_attention0_output'], tensors['entry_downblock1_attention0_output'], tolerance=2e-3),
    }
    status = status_from_diagnostics(diag)
    write_report(name, title, {'status': status, 'diagnostics': diag, 'note': 'Local bridge stops after down_blocks.1.attentions.0. Full down_blocks.1 and full UNet are not executed.'})
    print('TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_RESNET0_ATTENTION0_ALIGNMENT:', status)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
