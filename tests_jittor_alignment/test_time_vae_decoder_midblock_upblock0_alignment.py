#!/usr/bin/env python3
from __future__ import annotations
import json
from time_vae_decoder_upblock0_common import *


def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        result = blocked
    else:
        tester = TimeVAEBlockTester(WEIGHTS)
        got = tester.run_decoder_midblock_upblock0(oracle['entry_decoder_conv_in_output'], None, return_intermediates=True)
        diagnostics = {
            'decoder_midblock': add_metrics(got['decoder_midblock'], oracle['entry_decoder_midblock_output']),
            'decoder_upblock0': add_metrics(got['output'], oracle['entry_decoder_upblock0_output']),
        }
        result = {
            'status': status_from_diagnostics(diagnostics),
            'target': 'decoder.conv_in output -> decoder.mid_block -> decoder.up_blocks.0',
            'note': 'synthetic latent decoder.conv_in output; decoder temb is None',
            'diagnostics': diagnostics,
            'metrics': diagnostics['decoder_upblock0'],
            'weights': str(WEIGHTS),
        }
        if result['status'] == 'FAIL':
            result['failure_analysis'] = failure_analysis()
    write_report('jittor_decoder_midblock_upblock0_alignment', 'TimeAware VAE Decoder MidBlock To UpBlock0 Alignment', result)
    print(f"TIME_VAE_DECODER_MIDBLOCK_UPBLOCK0_ALIGNMENT: {result['status']}")
    if result.get('diagnostics'):
        print(json.dumps(result['diagnostics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
