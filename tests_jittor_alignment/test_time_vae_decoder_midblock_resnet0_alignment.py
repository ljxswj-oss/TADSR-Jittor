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
        got = tester.run_resnet0(oracle['synthetic_decoder_midblock_input'], None)
        result = result_from_metric('decoder.mid_block.resnets.0 synthetic hidden', got, oracle['synthetic_decoder_midblock_resnet0_output'], 'decoder temb is None')
        if result['status'] == 'FAIL': result['failure_analysis'] = failure_analysis()
    write_report('jittor_decoder_midblock_resnet0_alignment', 'TimeAware VAE Decoder MidBlock ResNet0 Alignment', result)
    print(f"TIME_VAE_DECODER_MIDBLOCK_RESNET0_ALIGNMENT: {result['status']}")
    if result.get('metrics'): print(json.dumps(result['metrics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1
if __name__ == '__main__': raise SystemExit(main())
