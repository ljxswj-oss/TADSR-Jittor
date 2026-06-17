#!/usr/bin/env python3
from __future__ import annotations
import json
from time_vae_decoder_upblock0_common import *


def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        result = blocked
    else:
        tester = TimeVAEUpDecoderBlockTester(WEIGHTS, block_index=0)
        got = tester.run_upblock(oracle['synthetic_decoder_upblock0_input'], None, return_intermediates=True)
        diagnostics = {
            'resnet0': add_metrics(got['resnet0'], oracle['synthetic_decoder_upblock0_resnet0_output']),
            'resnet1': add_metrics(got['resnet1'], oracle['synthetic_decoder_upblock0_resnet1_output']),
            'resnet2': add_metrics(got['resnet2'], oracle['synthetic_decoder_upblock0_resnet2_output']),
            'upsampler0': add_metrics(got['upsampler0'], oracle['synthetic_decoder_upblock0_upsampler0_output']),
            'output': add_metrics(got['output'], oracle['synthetic_decoder_upblock0_output']),
        }
        result = {
            'status': status_from_diagnostics(diagnostics),
            'target': 'synthetic hidden -> decoder.up_blocks.0',
            'note': 'isolated deterministic synthetic hidden; decoder temb is None',
            'diagnostics': diagnostics,
            'metrics': diagnostics['output'],
            'weights': str(WEIGHTS),
        }
        if result['status'] == 'FAIL':
            result['failure_analysis'] = failure_analysis()
    write_report('jittor_decoder_upblock0_synthetic_alignment', 'TimeAware VAE Decoder UpBlock0 Synthetic Alignment', result)
    print(f"TIME_VAE_DECODER_UPBLOCK0_SYNTHETIC_ALIGNMENT: {result['status']}")
    if result.get('diagnostics'):
        print(json.dumps(result['diagnostics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
