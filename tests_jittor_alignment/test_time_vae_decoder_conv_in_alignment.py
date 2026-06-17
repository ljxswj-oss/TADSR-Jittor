#!/usr/bin/env python3
from __future__ import annotations
import json
from time_vae_decoder_entry_common import *

def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        result = blocked
    else:
        tester = TimeVAEBlockTester(WEIGHTS)
        got = tester.run_decoder_conv_in(oracle['synthetic_post_quant_conv_output'])
        result = result_from_metric('decoder.conv_in synthetic post_quant path', got, oracle['synthetic_decoder_conv_in_output'])
        if result['status'] == 'FAIL': result['failure_analysis'] = failure_analysis()
    write_report('jittor_decoder_conv_in_alignment', 'TimeAware VAE Decoder Conv In Alignment', result)
    print(f"TIME_VAE_DECODER_CONV_IN_ALIGNMENT: {result['status']}")
    if result.get('metrics'): print(json.dumps(result['metrics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1
if __name__ == '__main__': raise SystemExit(main())
