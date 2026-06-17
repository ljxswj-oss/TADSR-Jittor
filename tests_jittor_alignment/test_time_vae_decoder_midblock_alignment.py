#!/usr/bin/env python3
from __future__ import annotations
import json
from time_vae_decoder_midblock_common import *

def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        result = blocked
    else:
        tester = TimeVAEDecoderMidBlockTester(WEIGHTS)
        got = tester.run_midblock(oracle['real_decoder_conv_in_output'], None, return_intermediates=True)
        diagnostics = {
            'resnet0': add_metrics(got['resnet0'], oracle['real_decoder_midblock_resnet0_output']),
            'attention': add_metrics(got['attention'], oracle['real_decoder_midblock_attention_output']),
            'resnet1': add_metrics(got['resnet1'], oracle['real_decoder_midblock_resnet1_output']),
            'output': add_metrics(got['output'], oracle['real_decoder_midblock_output']),
        }
        result = {
            'status': status_from_diagnostics(diagnostics),
            'target': 'decoder.conv_in(real path output) -> decoder.mid_block',
            'note': 'real encoder-derived decoder.conv_in output; decoder temb is None',
            'diagnostics': diagnostics,
            'metrics': diagnostics['output'],
            'weights': str(WEIGHTS),
        }
        if result['status'] == 'FAIL': result['failure_analysis'] = failure_analysis()
    write_report('jittor_decoder_midblock_alignment', 'TimeAware VAE Decoder MidBlock Alignment', result)
    print(f"TIME_VAE_DECODER_MIDBLOCK_ALIGNMENT: {result['status']}")
    if result.get('diagnostics'): print(json.dumps(result['diagnostics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1
if __name__ == '__main__': raise SystemExit(main())
