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
        got = tester.run_decoder_entry(oracle['synthetic_latent_z'])
        result = result_from_metric('post_quant_conv + decoder.conv_in synthetic latent path', got, oracle['synthetic_decoder_conv_in_output'])
        if result['status'] == 'FAIL': result['failure_analysis'] = failure_analysis()
    write_report('jittor_decoder_entry_alignment', 'TimeAware VAE Decoder Entry Alignment', result)
    print(f"TIME_VAE_DECODER_ENTRY_ALIGNMENT: {result['status']}")
    if result.get('metrics'): print(json.dumps(result['metrics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1
if __name__ == '__main__': raise SystemExit(main())
