#!/usr/bin/env python3
from __future__ import annotations
import json
from time_vae_decoder_tail_common import *


def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        write_report('jittor_decoder_upblocks0123_to_tail_alignment', 'TimeAware VAE Decoder UpBlocks0-3 To Tail Alignment', blocked)
        print(f"TIME_VAE_DECODER_UPBLOCKS0123_TO_TAIL_ALIGNMENT: {blocked['status']}")
        return 1
    tester = TimeVAEBlockTester(WEIGHTS)
    got = tester.run_decoder_upblocks0123_to_tail(oracle['entry_decoder_midblock_output'], return_intermediates=True)
    tolerance = 2e-3
    diagnostics = {
        'decoder_upblock0': add_metrics(got['decoder_upblock0'], oracle['entry_decoder_upblock0_output'], tolerance=tolerance),
        'decoder_upblock1': add_metrics(got['decoder_upblock1'], oracle['entry_decoder_upblock1_output'], tolerance=tolerance),
        'decoder_upblock2': add_metrics(got['decoder_upblock2'], oracle['entry_decoder_upblock2_output'], tolerance=tolerance),
        'decoder_upblock3': add_metrics(got['decoder_upblock3'], oracle['entry_decoder_upblock3_output'], tolerance=tolerance),
        'decoder_tail_norm_out': add_metrics(got['decoder_tail_norm_out'], oracle['entry_decoder_tail_norm_out_output'], tolerance=tolerance),
        'decoder_tail_act': add_metrics(got['decoder_tail_act'], oracle['entry_decoder_tail_act_output'], tolerance=tolerance),
        'decoder_tail_conv_out': add_metrics(got['decoder_tail_conv_out'], oracle['entry_decoder_tail_conv_out_output'], tolerance=tolerance),
        'decoder_tail': add_metrics(got['decoder_tail'], oracle['entry_decoder_tail_output'], tolerance=tolerance),
    }
    result = {
        'status': status_from_diagnostics(diagnostics),
        'target': 'decoder.up_blocks.0~3 -> decoder tail',
        'note': 'bridge test from decoder midblock output through all upblocks and tail; uses 2e-3 tolerance for accumulated float32 drift',
        'diagnostics': diagnostics,
        'metrics': diagnostics['decoder_tail'],
        'weights': str(WEIGHTS),
    }
    if result['status'] == 'FAIL':
        result['failure_analysis'] = failure_analysis()
    write_report('jittor_decoder_upblocks0123_to_tail_alignment', 'TimeAware VAE Decoder UpBlocks0-3 To Tail Alignment', result)
    print(f"TIME_VAE_DECODER_UPBLOCKS0123_TO_TAIL_ALIGNMENT: {result['status']}")
    print(json.dumps(result['diagnostics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
