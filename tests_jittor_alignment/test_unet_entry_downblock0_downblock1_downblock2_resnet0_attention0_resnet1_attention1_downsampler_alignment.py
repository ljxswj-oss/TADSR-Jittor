#!/usr/bin/env python3
from __future__ import annotations
from unet_downblock2_downsampler_common import *


def main() -> int:
    tensors, blocked = load_oracle()
    name = 'jittor_unet_entry_downblock0_downblock1_downblock2_resnet0_attention0_resnet1_attention1_downsampler_alignment'
    title = 'TADSR UNet Entry + DownBlock0 + DownBlock1 + DownBlock2 ResNet0 + Attention0 + ResNet1 + Attention1 + Downsampler Bridge Alignment'
    if blocked:
        result = blocked_result(name, title, blocked)
        print('TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_DOWNBLOCK2_RESNET0_ATTENTION0_RESNET1_ATTENTION1_DOWNSAMPLER_ALIGNMENT:', result['status'])
        return 0 if result['status'] == 'PASS' else 1
    tester = downblock2_tester()
    got = tester.run_entry_downblock0_downblock1_downblock2_resnet0_attention0_resnet1_attention1_downsampler(
        tensors['entry_synthetic_unet_sample'],
        tensors['entry_synthetic_unet_timestep'],
        tensors['entry_encoder_hidden_states'],
        return_intermediates=True,
        return_output_states=True,
    )
    diagnostics = {
        'conv_in': add_metrics(got['conv_in'], tensors['entry_synthetic_unet_conv_in_output'], tolerance=1e-4),
        'time_embedding': add_metrics(got['time_embedding'], tensors['entry_synthetic_unet_time_embedding_output'], tolerance=1e-4),
        'downblock0_output': add_metrics(got['downblock0_output'], tensors['entry_downblock0_output_hidden'], tolerance=2e-3),
        'downblock1_output': add_metrics(got['downblock1_output'], tensors['entry_downblock1_output_hidden'], tolerance=2e-3),
        'downblock2_resnet0_output': add_metrics(got['downblock2_resnet0_output'], tensors['entry_downblock2_resnet0_output'], tolerance=2e-3),
        'downblock2_attention0_output': add_metrics(got['downblock2_attention0_output'], tensors['entry_downblock2_attention0_output'], tolerance=2e-3),
        'downblock2_resnet1_output': add_metrics(got['downblock2_resnet1_output'], tensors['entry_downblock2_resnet1_output'], tolerance=2e-3),
        'downblock2_attention1_output': add_metrics(got['downblock2_attention1_output'], tensors['entry_downblock2_attention1_output'], tolerance=2e-3),
        'downblock2_downsampler_output': add_metrics(got['downblock2_downsampler_output'], tensors['entry_downblock2_downsampler_output'], tolerance=2e-3),
    }
    status = status_from_diagnostics(diagnostics)
    result = {'status': status, 'diagnostics': diagnostics, 'note': 'Local bridge stops after down_blocks.2.downsamplers.0. Down_blocks.3 and full UNet are not executed.'}
    write_report(name, title, result)
    print('TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_DOWNBLOCK2_RESNET0_ATTENTION0_RESNET1_ATTENTION1_DOWNSAMPLER_ALIGNMENT:', status)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
