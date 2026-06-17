#!/usr/bin/env python3
from __future__ import annotations
import json
from time_vae_decoder_upblock1_common import *


def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        result = blocked
        write_report('jittor_decoder_upblock0_upblock1_alignment', 'TimeAware VAE Decoder UpBlock0+1 Alignment', result)
        print(f"TIME_VAE_DECODER_UPBLOCK0_UPBLOCK1_ALIGNMENT: {result['status']}")
        return 1

    tester = TimeVAEBlockTester(WEIGHTS)
    got = tester.run_decoder_upblock0_upblock1(oracle['entry_decoder_midblock_output'], return_intermediates=True)
    diagnostics = {
        'decoder_upblock0': add_metrics(got['decoder_upblock0'], oracle['entry_decoder_upblock0_output']),
        'decoder_upblock1': add_metrics(got['decoder_upblock1'], oracle['entry_decoder_upblock1_output']),
    }
    result = {
        'status': status_from_diagnostics(diagnostics),
        'target': 'decoder.mid_block output -> decoder.up_blocks.0 -> decoder.up_blocks.1',
        'note': 'bridge test for consecutive decoder upblocks',
        'diagnostics': diagnostics,
        'metrics': diagnostics['decoder_upblock1'],
        'weights': str(WEIGHTS),
    }
    if result['status'] == 'FAIL':
        result['failure_analysis'] = failure_analysis()
    write_report('jittor_decoder_upblock0_upblock1_alignment', 'TimeAware VAE Decoder UpBlock0+1 Alignment', result)
    print(f"TIME_VAE_DECODER_UPBLOCK0_UPBLOCK1_ALIGNMENT: {result['status']}")
    print(json.dumps(result['diagnostics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
