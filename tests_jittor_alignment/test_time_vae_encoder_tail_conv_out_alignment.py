#!/usr/bin/env python3
from __future__ import annotations
import json
from time_vae_encoder_tail_common import *


def main() -> int:
    inputs, outputs, blocked = load_oracle()
    if blocked:
        result = blocked
    else:
        tester = TimeVAEEncoderTailTester(WEIGHTS)
        got = tester.run_conv_out(inputs['conv_out_input'])
        expected = outputs['conv_out_output']
        metrics = add_metrics(got, expected, 1e-4)
        result = {'status': status_from_metrics(metrics), 'target': 'encoder.conv_out', 'metrics': metrics, 'weights': str(WEIGHTS)}
        if result['status'] == 'FAIL': result['failure_analysis'] = failure_analysis()
    write_report('jittor_encoder_tail_conv_out_alignment', 'TimeAware VAE Encoder Tail conv_out Alignment', result)
    print(f"TIME_VAE_ENCODER_TAIL_CONV_OUT_ALIGNMENT: {result['status']}")
    if result.get('metrics'): print(json.dumps(result['metrics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1
if __name__ == '__main__': raise SystemExit(main())
