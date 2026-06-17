#!/usr/bin/env python3
from __future__ import annotations
import json
from time_vae_decoder_upblock0_common import *


def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        result = blocked
        write_report('jittor_decoder_upblock0_alignment', 'TimeAware VAE Decoder UpBlock0 Alignment', result)
        print(f"TIME_VAE_DECODER_UPBLOCK0_ALIGNMENT: {result['status']}")
        return 1

    tester = TimeVAEUpDecoderBlockTester(WEIGHTS, block_index=0)
    got = tester.run_upblock(oracle['real_decoder_midblock_output'], None, return_intermediates=True)
    diagnostics = {
        'resnet0': add_metrics(got['resnet0'], oracle['real_decoder_upblock0_resnet0_output']),
        'resnet1': add_metrics(got['resnet1'], oracle['real_decoder_upblock0_resnet1_output']),
        'resnet2': add_metrics(got['resnet2'], oracle['real_decoder_upblock0_resnet2_output']),
        'pre_upsampler': add_metrics(got['pre_upsampler'], oracle['real_decoder_upblock0_pre_upsampler_output']),
        'upsampler0': add_metrics(got['upsampler0'], oracle['real_decoder_upblock0_upsampler0_output']),
        'output': add_metrics(got['output'], oracle['real_decoder_upblock0_output']),
    }
    result = {
        'status': status_from_diagnostics(diagnostics),
        'target': 'real decoder.mid_block output -> decoder.up_blocks.0',
        'note': 'decoder temb is None; upsampler is nearest 2x followed by 3x3 conv',
        'diagnostics': diagnostics,
        'metrics': diagnostics['output'],
        'weights': str(WEIGHTS),
    }
    if result['status'] == 'FAIL':
        result['failure_analysis'] = failure_analysis()

    for idx in range(3):
        stage = f'resnet{idx}'
        write_report(
            f'jittor_decoder_upblock0_{stage}_alignment',
            f'TimeAware VAE Decoder UpBlock0 ResNet{idx} Alignment',
            {'status': status_from_metrics(diagnostics[stage]), 'target': f'decoder.up_blocks.0.{stage}', 'metrics': diagnostics[stage], 'weights': str(WEIGHTS)},
        )
    write_report(
        'jittor_decoder_upblock0_upsampler0_alignment',
        'TimeAware VAE Decoder UpBlock0 Upsampler0 Alignment',
        {'status': status_from_metrics(diagnostics['upsampler0']), 'target': 'decoder.up_blocks.0.upsamplers.0', 'metrics': diagnostics['upsampler0'], 'weights': str(WEIGHTS)},
    )
    write_report('jittor_decoder_upblock0_alignment', 'TimeAware VAE Decoder UpBlock0 Alignment', result)

    for name, key in [
        ('TIME_VAE_DECODER_UPBLOCK0_RESNET0_ALIGNMENT', 'resnet0'),
        ('TIME_VAE_DECODER_UPBLOCK0_RESNET1_ALIGNMENT', 'resnet1'),
        ('TIME_VAE_DECODER_UPBLOCK0_RESNET2_ALIGNMENT', 'resnet2'),
        ('TIME_VAE_DECODER_UPBLOCK0_UPSAMPLER0_ALIGNMENT', 'upsampler0'),
    ]:
        print(f"{name}: {status_from_metrics(diagnostics[key])}")
        print(json.dumps(diagnostics[key], indent=2))
    print(f"TIME_VAE_DECODER_UPBLOCK0_ALIGNMENT: {result['status']}")
    print(json.dumps(result['diagnostics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
