#!/usr/bin/env python3
from __future__ import annotations
import json
from time_vae_decoder_upblock1_common import *


def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        result = blocked
        write_report('jittor_decoder_entry_midblock_upblocks01_alignment', 'TimeAware VAE Decoder Entry+MidBlock+UpBlocks0-1 Alignment', result)
        print(f"TIME_VAE_DECODER_ENTRY_MIDBLOCK_UPBLOCKS01_ALIGNMENT: {result['status']}")
        return 1

    tester = TimeVAEBlockTester(WEIGHTS)
    got = tester.run_decoder_entry_midblock_upblocks01(oracle['entry_synthetic_latent_z'], return_intermediates=True)
    diagnostics = {
        'post_quant_conv': add_metrics(got['post_quant_conv'], oracle['entry_post_quant_conv_output']),
        'decoder_conv_in': add_metrics(got['decoder_conv_in'], oracle['entry_decoder_conv_in_output']),
        'decoder_midblock': add_metrics(got['decoder_midblock'], oracle['entry_decoder_midblock_output']),
        'decoder_upblock0': add_metrics(got['decoder_upblock0'], oracle['entry_decoder_upblock0_output']),
        'decoder_upblock1': add_metrics(got['decoder_upblock1'], oracle['entry_decoder_upblock1_output']),
    }
    result = {
        'status': status_from_diagnostics(diagnostics),
        'target': 'latent z -> decoder entry -> mid_block -> up_blocks.0 -> up_blocks.1',
        'note': 'bridge test from decoder entry through first two decoder upblocks',
        'diagnostics': diagnostics,
        'metrics': diagnostics['decoder_upblock1'],
        'weights': str(WEIGHTS),
    }
    if result['status'] == 'FAIL':
        result['failure_analysis'] = failure_analysis()
    write_report('jittor_decoder_entry_midblock_upblocks01_alignment', 'TimeAware VAE Decoder Entry+MidBlock+UpBlocks0-1 Alignment', result)
    print(f"TIME_VAE_DECODER_ENTRY_MIDBLOCK_UPBLOCKS01_ALIGNMENT: {result['status']}")
    print(json.dumps(result['diagnostics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
