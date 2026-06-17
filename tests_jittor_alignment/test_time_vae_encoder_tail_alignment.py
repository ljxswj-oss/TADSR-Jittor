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
        after_norm = tester.run_conv_norm_out(inputs['conv_norm_out_input'])
        after_act = tester.run_conv_act(after_norm)
        got = tester.run_conv_out(after_act)
        expected = outputs['encoder_tail_output']
        metrics = add_metrics(got, expected, 1e-4)
        diagnostics = {
            'after_norm': add_metrics(after_norm, outputs['conv_norm_out_output'], 1e-4),
            'after_act': add_metrics(after_act, outputs['conv_act_output'], 1e-4),
            'after_conv_out_final': metrics,
        }
        result = {'status': status_from_metrics(metrics), 'target': 'encoder.conv_norm_out -> conv_act -> conv_out', 'metrics': metrics, 'diagnostics': diagnostics, 'weights': str(WEIGHTS)}
        if result['status'] == 'FAIL': result['failure_analysis'] = failure_analysis()
    write_report('jittor_encoder_tail_alignment', 'TimeAware VAE Full Encoder Tail Alignment', result)
    print(f"TIME_VAE_ENCODER_TAIL_ALIGNMENT: {result['status']}")
    if result.get('metrics'): print(json.dumps(result['metrics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1
if __name__ == '__main__': raise SystemExit(main())
