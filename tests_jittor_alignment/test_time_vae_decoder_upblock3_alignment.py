#!/usr/bin/env python3
from __future__ import annotations
import json
from time_vae_decoder_upblock3_common import *


def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        result = blocked
        write_report('jittor_decoder_upblock3_alignment', 'TimeAware VAE Decoder UpBlock3 Alignment', result)
        print(f"TIME_VAE_DECODER_UPBLOCK3_ALIGNMENT: {result['status']}")
        return 1

    tester = TimeVAEUpDecoderBlockTester(WEIGHTS, block_index=3)
    got = tester.run_upblock(oracle['real_decoder_upblock2_output'], None, return_intermediates=True)
    diagnostics = {
        'resnet0': add_metrics(got['resnet0'], oracle['real_decoder_upblock3_resnet0_output']),
        'resnet1': add_metrics(got['resnet1'], oracle['real_decoder_upblock3_resnet1_output']),
        'resnet2': add_metrics(got['resnet2'], oracle['real_decoder_upblock3_resnet2_output']),
        'pre_upsampler': add_metrics(got['pre_upsampler'], oracle['real_decoder_upblock3_pre_upsampler_output']),
        'output': add_metrics(got['output'], oracle['real_decoder_upblock3_output']),
    }
    upsampler_status = 'NOT_APPLICABLE'
    if 'upsampler0' in got and 'real_decoder_upblock3_upsampler0_output' in oracle:
        diagnostics['upsampler0'] = add_metrics(got['upsampler0'], oracle['real_decoder_upblock3_upsampler0_output'])
        upsampler_status = status_from_metrics(diagnostics['upsampler0'])
    result = {
        'status': status_from_diagnostics(diagnostics),
        'target': 'real decoder.up_blocks.2 output -> decoder.up_blocks.3',
        'note': 'decoder temb is None; resnet0 changes 256 channels to 128 through conv_shortcut; official upblock3 has no upsampler',
        'diagnostics': diagnostics,
        'metrics': diagnostics['output'],
        'weights': str(WEIGHTS),
    }
    if result['status'] == 'FAIL':
        result['failure_analysis'] = failure_analysis()

    for idx in range(3):
        stage = f'resnet{idx}'
        write_report(
            f'jittor_decoder_upblock3_{stage}_alignment',
            f'TimeAware VAE Decoder UpBlock3 ResNet{idx} Alignment',
            {'status': status_from_metrics(diagnostics[stage]), 'target': f'decoder.up_blocks.3.{stage}', 'metrics': diagnostics[stage], 'weights': str(WEIGHTS)},
        )
    if 'upsampler0' in diagnostics:
        upsampler_report = {'status': upsampler_status, 'target': 'decoder.up_blocks.3.upsamplers.0', 'metrics': diagnostics['upsampler0'], 'weights': str(WEIGHTS)}
    else:
        upsampler_report = {'status': 'NOT_APPLICABLE', 'target': 'decoder.up_blocks.3.upsamplers.0', 'note': 'official decoder.up_blocks.3 has no upsampler module', 'weights': str(WEIGHTS)}
    write_report('jittor_decoder_upblock3_upsampler0_alignment', 'TimeAware VAE Decoder UpBlock3 Upsampler0 Alignment', upsampler_report)
    write_report('jittor_decoder_upblock3_alignment', 'TimeAware VAE Decoder UpBlock3 Alignment', result)

    for name, key in [
        ('TIME_VAE_DECODER_UPBLOCK3_RESNET0_ALIGNMENT', 'resnet0'),
        ('TIME_VAE_DECODER_UPBLOCK3_RESNET1_ALIGNMENT', 'resnet1'),
        ('TIME_VAE_DECODER_UPBLOCK3_RESNET2_ALIGNMENT', 'resnet2'),
    ]:
        print(f"{name}: {status_from_metrics(diagnostics[key])}")
        print(json.dumps(diagnostics[key], indent=2))
    print(f"TIME_VAE_DECODER_UPBLOCK3_UPSAMPLER0_ALIGNMENT: {upsampler_status}")
    print(f"TIME_VAE_DECODER_UPBLOCK3_ALIGNMENT: {result['status']}")
    print(json.dumps(result['diagnostics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
