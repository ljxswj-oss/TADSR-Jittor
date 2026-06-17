#!/usr/bin/env python3
from __future__ import annotations
import json
from time_vae_downblock2_common import *


def main() -> int:
    inputs, outputs, blocked = load_oracle()
    if blocked:
        result = blocked
    else:
        tester = TimeVAEBlockTester(WEIGHTS)
        got = tester.run_encoder_stage012(inputs['stage012_input'], inputs['downblock2_temb'])
        expected = outputs['stage012_output']
        metrics = add_metrics(got, expected, 1e-4)
        result = {'status': status_from_metrics(metrics), 'target': 'encoder.conv_in + encoder.down_blocks.0 + encoder.down_blocks.1 + encoder.down_blocks.2', 'module_order': ['encoder.conv_in', 'encoder.down_blocks.0', 'encoder.down_blocks.1', 'encoder.down_blocks.2'], 'metrics': metrics, 'weights': str(WEIGHTS)}
        if result['status'] == 'FAIL':
            result['failure_analysis'] = failure_analysis()
    write_report('jittor_encoder_stage012_alignment', 'TimeAware VAE Encoder Stage012 Alignment', result)
    print(f"TIME_VAE_ENCODER_STAGE012_ALIGNMENT: {result['status']}")
    if result.get('metrics'):
        print(json.dumps(result['metrics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1

if __name__ == '__main__':
    raise SystemExit(main())