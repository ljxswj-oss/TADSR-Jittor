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
        got = tester.run_quant_conv(inputs['quant_conv_input'])
        expected = outputs['quant_conv_output']
        metrics = add_metrics(got, expected, 1e-4)
        result = {'status': status_from_metrics(metrics), 'target': 'quant_conv', 'note': 'deterministic moments tensor only; no latent sampling', 'metrics': metrics, 'weights': str(WEIGHTS)}
        if result['status'] == 'FAIL': result['failure_analysis'] = failure_analysis('quant')
    write_report('jittor_quant_conv_alignment', 'TimeAware VAE quant_conv Alignment', result)
    print(f"TIME_VAE_QUANT_CONV_ALIGNMENT: {result['status']}")
    if result.get('metrics'): print(json.dumps(result['metrics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1
if __name__ == '__main__': raise SystemExit(main())
