#!/usr/bin/env python3
from __future__ import annotations
import json
from time_vae_midblock_common import *


def main() -> int:
    inputs, outputs, blocked = load_oracle()
    if blocked:
        result = blocked
    else:
        tester = TimeVAEMidBlockTester(WEIGHTS)
        after_r0 = tester.run_resnet0(inputs['midblock_input'], inputs['midblock_temb'])
        after_attn = tester.run_attention0(after_r0)
        got = tester.run_resnet1(after_attn, inputs['midblock_temb'])
        expected = outputs['midblock_output']
        metrics = add_metrics(got, expected, 1e-4)
        diagnostics = {
            'after_resnet0': add_metrics(after_r0, outputs['resnet0_output'], 1e-4),
            'after_attention0': add_metrics(after_attn, outputs['attention0_output'], 1e-4),
            'after_resnet1_final': metrics,
        }
        result = {'status': status_from_metrics(metrics), 'target': 'encoder.mid_block', 'module_order': ['resnets.0', 'attentions.0', 'resnets.1'], 'metrics': metrics, 'diagnostics': diagnostics, 'weights': str(WEIGHTS)}
        if result['status'] == 'FAIL':
            result['failure_analysis'] = failure_analysis('attention')
    write_report('jittor_midblock_alignment', 'TimeAware VAE Full MidBlock Alignment', result)
    print(f"TIME_VAE_MIDBLOCK_ALIGNMENT: {result['status']}")
    if result.get('metrics'):
        print(json.dumps(result['metrics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
