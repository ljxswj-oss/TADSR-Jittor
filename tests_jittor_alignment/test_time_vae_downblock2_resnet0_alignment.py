#!/usr/bin/env python3
from __future__ import annotations
import json
from time_vae_downblock2_common import *


def main() -> int:
    inputs, outputs, blocked = load_oracle()
    if blocked:
        result = blocked
    else:
        tester = TimeVAEDownBlockTester(BLOCK_INDEX, WEIGHTS)
        got = tester.run_resnet0(inputs['resnet0_input'], inputs['downblock2_temb'])
        expected = outputs['resnet0_output']
        metrics = add_metrics(got, expected, 1e-4)
        result = {'status': status_from_metrics(metrics), 'target': 'encoder.down_blocks.2.resnets.0', 'channel_change': '256->512', 'uses_conv_shortcut': True, 'metrics': metrics, 'weights': str(WEIGHTS)}
        if result['status'] == 'FAIL':
            result['failure_analysis'] = failure_analysis()
    write_report('jittor_downblock2_resnet0_alignment', 'TimeAware VAE DownBlock2 ResNet0 Alignment', result)
    print(f"TIME_VAE_DOWNBLOCK2_RESNET0_ALIGNMENT: {result['status']}")
    if result.get('metrics'):
        print(json.dumps(result['metrics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1

if __name__ == '__main__':
    raise SystemExit(main())