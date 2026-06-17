#!/usr/bin/env python3
from __future__ import annotations
import json
from time_vae_downblock3_common import *


def main() -> int:
    inputs, outputs, blocked = load_oracle()
    if blocked:
        result = blocked
    else:
        tester = TimeVAEBlockTester(WEIGHTS)
        got = tester.run_encoder_stage0123(inputs['stage0123_input'], inputs['downblock3_temb'])
        expected = outputs['stage0123_output']
        metrics = add_metrics(got, expected, 1e-4)
        result = {'status': status_from_metrics(metrics), 'target': 'encoder.conv_in + encoder.down_blocks.0 + encoder.down_blocks.1 + encoder.down_blocks.2 + encoder.down_blocks.3', 'module_order': ['encoder.conv_in', 'encoder.down_blocks.0', 'encoder.down_blocks.1', 'encoder.down_blocks.2', 'encoder.down_blocks.3'], 'metrics': metrics, 'weights': str(WEIGHTS)}
        if result['status'] == 'FAIL':
            result['failure_analysis'] = failure_analysis()
    write_report('jittor_encoder_stage0123_alignment', 'TimeAware VAE Encoder Stage0123 Alignment', result)
    print(f"TIME_VAE_ENCODER_STAGE0123_ALIGNMENT: {result['status']}")
    if result.get('metrics'):
        print(json.dumps(result['metrics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1

if __name__ == '__main__':
    raise SystemExit(main())