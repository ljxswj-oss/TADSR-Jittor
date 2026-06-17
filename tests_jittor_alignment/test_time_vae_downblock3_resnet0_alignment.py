#!/usr/bin/env python3
from __future__ import annotations
import json
from time_vae_downblock3_common import *


def main() -> int:
    inputs, outputs, blocked = load_oracle()
    if blocked:
        result = blocked
    else:
        tester = TimeVAEDownBlockTester(BLOCK_INDEX, WEIGHTS)
        got = tester.run_resnet0(inputs['resnet0_input'], inputs['downblock3_temb'])
        expected = outputs['resnet0_output']
        metrics = add_metrics(got, expected, 1e-4)
        audit = load_audit()
        info = audit.get('resnet_info', [{}])[0] if audit else {}
        result = {'status': status_from_metrics(metrics), 'target': 'encoder.down_blocks.3.resnets.0', 'channel_change': f"{info.get('in_channels')}->{info.get('out_channels')}", 'uses_conv_shortcut': info.get('has_conv_shortcut'), 'uses_nin_shortcut': info.get('has_nin_shortcut'), 'metrics': metrics, 'weights': str(WEIGHTS)}
        if result['status'] == 'FAIL':
            result['failure_analysis'] = failure_analysis()
    write_report('jittor_downblock3_resnet0_alignment', 'TimeAware VAE DownBlock3 ResNet0 Alignment', result)
    print(f"TIME_VAE_DOWNBLOCK3_RESNET0_ALIGNMENT: {result['status']}")
    if result.get('metrics'):
        print(json.dumps(result['metrics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1

if __name__ == '__main__':
    raise SystemExit(main())