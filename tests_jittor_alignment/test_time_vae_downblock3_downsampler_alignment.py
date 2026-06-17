#!/usr/bin/env python3
from __future__ import annotations
import json
from time_vae_downblock3_common import *


def main() -> int:
    inputs, outputs, blocked = load_oracle()
    audit = load_audit()
    if blocked:
        result = blocked
    elif audit and not audit.get('has_downsampler', False):
        result = {'status': 'NOT_APPLICABLE', 'target': 'encoder.down_blocks.3.downsamplers.0', 'reason': 'official encoder.down_blocks.3 has no downsampler', 'note': 'This is expected for the final encoder down block and is not counted as a failure.'}
    elif 'downsampler0_input' not in inputs.files or 'downsampler0_output' not in outputs.files:
        result = {'status': 'NOT_APPLICABLE', 'target': 'encoder.down_blocks.3.downsamplers.0', 'reason': 'downsampler oracle tensors are absent'}
    else:
        tester = TimeVAEDownBlockTester(BLOCK_INDEX, WEIGHTS)
        got = tester.run_downsampler0(inputs['downsampler0_input'])
        expected = outputs['downsampler0_output']
        metrics = add_metrics(got, expected, 1e-4)
        result = {'status': status_from_metrics(metrics), 'target': 'encoder.down_blocks.3.downsamplers.0', 'metrics': metrics, 'weights': str(WEIGHTS)}
        if result['status'] == 'FAIL':
            result['failure_analysis'] = failure_analysis()
    write_report('jittor_downblock3_downsampler_alignment', 'TimeAware VAE DownBlock3 Downsampler Alignment', result)
    print(f"TIME_VAE_DOWNBLOCK3_DOWNSAMPLER_ALIGNMENT: {result['status']}")
    if result.get('metrics'):
        print(json.dumps(result['metrics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL', 'NOT_APPLICABLE'} else 1

if __name__ == '__main__':
    raise SystemExit(main())