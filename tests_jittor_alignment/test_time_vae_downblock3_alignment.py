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
        after_resnet0 = tester.run_resnet0(inputs['downblock3_input'], inputs['downblock3_temb'])
        after_resnet1 = tester.run_resnet1(after_resnet0, inputs['downblock3_temb'])
        final = tester.run_downsampler0(after_resnet1)
        diagnostics = {
            'after_resnet0': add_metrics(after_resnet0, outputs['resnet0_output'], 1e-4),
            'after_resnet1': add_metrics(after_resnet1, outputs['resnet1_output'], 1e-4),
            'after_downsampler': add_metrics(final, outputs['downsampler0_output'], 1e-4) if 'downsampler0_output' in outputs.files else None,
            'final_downblock3': add_metrics(final, outputs['downblock3_output'], 1e-4),
            'direct_pytorch_downblock3_reference': add_metrics(outputs['downblock3_direct_output'], outputs['downblock3_output'], 1e-7) if 'downblock3_direct_output' in outputs.files else None,
        }
        diagnostics = {k: v for k, v in diagnostics.items() if v is not None}
        metrics = diagnostics['final_downblock3']
        audit = load_audit()
        result = {'status': status_from_metrics(metrics), 'target': 'encoder.down_blocks.3', 'module_order': ['resnets.0', 'resnets.1'] + (['downsamplers.0'] if audit.get('has_downsampler') else []), 'has_downsampler': audit.get('has_downsampler') if audit else None, 'metrics': metrics, 'diagnostics': diagnostics, 'weights': str(WEIGHTS)}
        if result['status'] == 'FAIL':
            result['failure_analysis'] = failure_analysis()
    write_report('jittor_downblock3_alignment', 'TimeAware VAE Full DownBlock3 Alignment', result)
    print(f"TIME_VAE_DOWNBLOCK3_ALIGNMENT: {result['status']}")
    if result.get('metrics'):
        print(json.dumps(result['metrics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1

if __name__ == '__main__':
    raise SystemExit(main())