#!/usr/bin/env python3
from __future__ import annotations

from unet_upblock0_upsampler_common import add_metrics, blocked_result, bridge_tester, load_oracle, status_from_diagnostics, write_report


def main() -> int:
    tensors, blocked = load_oracle()
    name = 'jittor_unet_entry_downblocks_midblock_upblock0_resnets012_upsampler_alignment'
    title = 'TADSR UNet Entry -> DownBlocks -> MidBlock -> UpBlock0 ResNets012 -> Upsampler Bridge Alignment'
    if blocked:
        result = blocked_result(name, title, blocked)
        print('TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_RESNETS012_UPSAMPLER_ALIGNMENT:', result['status'])
        return 0 if result['status'] == 'PASS' else 1
    tester = bridge_tester()
    got = tester.run_entry_downblocks_midblock_upblock0_resnets012_upsampler(
        tensors['entry_synthetic_unet_sample'],
        tensors['entry_synthetic_unet_timestep'].astype('int64'),
        tensors['entry_encoder_hidden_states'],
        return_intermediates=True,
    )
    expected = {
        'conv_in': 'entry_synthetic_unet_conv_in_output',
        'time_embedding': 'entry_synthetic_unet_time_embedding_output',
        'downblock0_output': 'entry_downblock0_output_hidden',
        'downblock1_output': 'entry_downblock1_output_hidden',
        'downblock2_output': 'entry_downblock2_output_hidden',
        'downblock3_output': 'entry_downblock3_output_hidden',
        'midblock_hidden_output': 'entry_midblock_output_hidden',
        'upblock0_resnet0_output': 'entry_upblock0_resnet0_output',
        'upblock0_resnet1_output': 'entry_upblock0_resnet1_output',
        'upblock0_resnet2_output': 'entry_upblock0_resnet2_output',
        'upblock0_upsampler_interpolation_output': 'entry_upblock0_upsampler_interpolation_output',
        'upblock0_upsampler_conv_output': 'entry_upblock0_upsampler_conv_output',
        'upblock0_upsampler_output': 'entry_upblock0_upsampler_output',
    }
    diag = {}
    for stage, path in expected.items():
        tol = 1e-4 if stage in {'conv_in', 'time_embedding'} else 2e-3
        diag[stage] = add_metrics(got[stage], tensors[path], tolerance=tol)
    status = status_from_diagnostics(diag)
    result = {
        'status': status,
        'diagnostics': diag,
        'note': 'Bridge validates entry -> all down_blocks -> full local mid_block -> up_blocks.0.resnets.0/1/2 -> upsamplers.0. It still does not execute up_blocks.1 or full UNet.',
    }
    write_report(name, title, result)
    print('TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_RESNETS012_UPSAMPLER_ALIGNMENT:', status)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
