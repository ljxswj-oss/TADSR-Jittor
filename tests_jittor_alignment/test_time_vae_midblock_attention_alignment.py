#!/usr/bin/env python3
from __future__ import annotations
import json
from time_vae_midblock_common import *


def main() -> int:
    inputs, outputs, blocked = load_oracle()
    if blocked:
        result = blocked
    elif not attention_exists():
        result = {'status': 'NOT_APPLICABLE', 'target': 'encoder.mid_block.attentions.0', 'reason': 'official mid_block has no attention'}
    else:
        tester = TimeVAEMidBlockTester(WEIGHTS)
        got = tester.run_attention0(inputs['attention0_input'])
        expected = outputs['attention0_output']
        metrics = add_metrics(got, expected, 1e-4)
        result = {'status': status_from_metrics(metrics), 'target': 'encoder.mid_block.attentions.0', 'metrics': metrics, 'weights': str(WEIGHTS)}
        if result['status'] == 'FAIL':
            result['failure_analysis'] = failure_analysis('attention')
    write_report('jittor_midblock_attention_alignment', 'TimeAware VAE MidBlock Attention Alignment', result)
    print(f"TIME_VAE_MIDBLOCK_ATTENTION_ALIGNMENT: {result['status']}")
    if result.get('metrics'):
        print(json.dumps(result['metrics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL', 'NOT_APPLICABLE'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
