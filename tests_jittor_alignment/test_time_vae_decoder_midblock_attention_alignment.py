#!/usr/bin/env python3
from __future__ import annotations
import json
from time_vae_decoder_midblock_common import *

def main() -> int:
    meta = metadata()
    oracle, blocked = load_oracle()
    if blocked:
        result = blocked
    elif not meta.get('attention_exists', False):
        result = {'status': 'NOT_APPLICABLE', 'target': 'decoder.mid_block.attentions.0', 'reason': 'official decoder.mid_block has no attention'}
    else:
        tester = TimeVAEDecoderMidBlockTester(WEIGHTS)
        got = tester.run_attention0(oracle['synthetic_decoder_midblock_resnet0_output'])
        result = result_from_metric('decoder.mid_block.attentions.0 synthetic hidden', got, oracle['synthetic_decoder_midblock_attention_output'])
        if result['status'] == 'FAIL': result['failure_analysis'] = failure_analysis()
    write_report('jittor_decoder_midblock_attention_alignment', 'TimeAware VAE Decoder MidBlock Attention Alignment', result)
    print(f"TIME_VAE_DECODER_MIDBLOCK_ATTENTION_ALIGNMENT: {result['status']}")
    if result.get('metrics'): print(json.dumps(result['metrics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL', 'NOT_APPLICABLE'} else 1
if __name__ == '__main__': raise SystemExit(main())
