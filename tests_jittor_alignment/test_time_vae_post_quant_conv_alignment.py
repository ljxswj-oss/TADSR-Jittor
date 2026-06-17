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
        got = tester.run_post_quant_conv(oracle['synthetic_latent_z'])
        result = result_from_metric('post_quant_conv synthetic latent path', got, oracle['synthetic_post_quant_conv_output'])
        if result['status'] == 'FAIL': result['failure_analysis'] = failure_analysis()
    write_report('jittor_post_quant_conv_alignment', 'TimeAware VAE Post Quant Conv Alignment', result)
    print(f"TIME_VAE_POST_QUANT_CONV_ALIGNMENT: {result['status']}")
    if result.get('metrics'): print(json.dumps(result['metrics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1
if __name__ == '__main__': raise SystemExit(main())
