#!/usr/bin/env python3
from __future__ import annotations
import json
from time_vae_encoder_tail_common import *


def main() -> int:
    inputs, outputs, blocked = load_oracle()
    if blocked:
        result = blocked
    else:
        tester = TimeVAEBlockTester(WEIGHTS)
        got = tester.run_encoder_stage0123_mid_tail(inputs['stage0123_mid_tail_input'], inputs['tail_temb'])
        expected = outputs['stage0123_mid_tail_output']
        metrics = add_metrics(got, expected, 1e-4)
        result = {'status': status_from_metrics(metrics), 'target': 'encoder.conv_in + down_blocks.0..3 + mid_block + tail', 'metrics': metrics, 'weights': str(WEIGHTS)}
        if result['status'] == 'FAIL': result['failure_analysis'] = failure_analysis()
    write_report('jittor_encoder_stage0123_mid_tail_alignment', 'TimeAware VAE Encoder Stage0123 + MidBlock + Tail Alignment', result)
    print(f"TIME_VAE_ENCODER_STAGE0123_MID_TAIL_ALIGNMENT: {result['status']}")
    if result.get('metrics'): print(json.dumps(result['metrics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1
if __name__ == '__main__': raise SystemExit(main())
