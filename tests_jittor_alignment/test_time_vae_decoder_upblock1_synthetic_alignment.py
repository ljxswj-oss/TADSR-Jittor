#!/usr/bin/env python3
from __future__ import annotations
import json
from time_vae_decoder_upblock1_common import *


def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        result = blocked
        write_report('jittor_decoder_upblock1_synthetic_alignment', 'TimeAware VAE Decoder UpBlock1 Synthetic Alignment', result)
        print(f"TIME_VAE_DECODER_UPBLOCK1_SYNTHETIC_ALIGNMENT: {result['status']}")
        return 1

    tester = TimeVAEUpDecoderBlockTester(WEIGHTS, block_index=1)
    got = tester.run_upblock(oracle['synthetic_decoder_upblock1_input'], None, return_intermediates=True)
    tolerance = 1e-3
    diagnostics = {
        'resnet0': add_metrics(got['resnet0'], oracle['synthetic_decoder_upblock1_resnet0_output'], tolerance=tolerance),
        'resnet1': add_metrics(got['resnet1'], oracle['synthetic_decoder_upblock1_resnet1_output'], tolerance=tolerance),
        'resnet2': add_metrics(got['resnet2'], oracle['synthetic_decoder_upblock1_resnet2_output'], tolerance=tolerance),
        'pre_upsampler': add_metrics(got['pre_upsampler'], oracle['synthetic_decoder_upblock1_pre_upsampler_output'], tolerance=tolerance),
        'upsampler0': add_metrics(got['upsampler0'], oracle['synthetic_decoder_upblock1_upsampler0_output'], tolerance=tolerance),
        'output': add_metrics(got['output'], oracle['synthetic_decoder_upblock1_output'], tolerance=tolerance),
    }
    result = {
        'status': status_from_diagnostics(diagnostics),
        'target': 'synthetic hidden -> decoder.up_blocks.1',
        'note': 'isolated deterministic synthetic upblock1 pressure test; uses 1e-3 tolerance for large linspace activations',
        'diagnostics': diagnostics,
        'metrics': diagnostics['output'],
        'weights': str(WEIGHTS),
    }
    if result['status'] == 'FAIL':
        result['failure_analysis'] = failure_analysis()
    write_report('jittor_decoder_upblock1_synthetic_alignment', 'TimeAware VAE Decoder UpBlock1 Synthetic Alignment', result)
    print(f"TIME_VAE_DECODER_UPBLOCK1_SYNTHETIC_ALIGNMENT: {result['status']}")
    print(json.dumps(result['diagnostics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
