#!/usr/bin/env python3
from __future__ import annotations

from unet_upblock0_upsampler_common import add_metrics, blocked_result, bridge_tester, load_oracle, status_from_metrics, write_report


def main() -> int:
    tensors, blocked = load_oracle()
    name = 'jittor_unet_upblock0_local_alignment'
    title = 'TADSR UNet up_blocks.0 Local Hidden Output Alignment'
    if blocked:
        result = blocked_result(name, title, blocked)
        print('TADSR_UNET_UPBLOCK0_ALIGNMENT:', result['status'])
        return 0 if result['status'] == 'PASS' else 1
    tester = bridge_tester()
    got = tester.run_upblock0_local(
        tensors['entry_synthetic_unet_sample'],
        tensors['entry_synthetic_unet_timestep'].astype('int64'),
        tensors['entry_encoder_hidden_states'],
        return_intermediates=True,
    )
    diag = {
        'output_hidden': add_metrics(got['upblock0_output_hidden'], tensors['entry_upblock0_output_hidden'], tolerance=2e-3),
        'output_states': {'status': 'PASS', 'note': 'Official UpBlock2D returns hidden_states only; no output_states are produced for up_blocks.0.'},
    }
    hidden_status = status_from_metrics(diag['output_hidden'])
    states_status = diag['output_states']['status']
    status = 'PASS' if hidden_status == 'PASS' and states_status == 'PASS' else 'FAIL'
    result = {
        'status': status,
        'diagnostics': diag,
        'note': 'Local up_blocks.0 hidden-state output is aligned through resnets.0/1/2 and upsamplers.0 only. Full up_blocks.1+ and full UNet remain unopened.',
    }
    write_report(name, title, result)
    print('TADSR_UNET_UPBLOCK0_OUTPUT_HIDDEN_ALIGNMENT:', hidden_status)
    print('TADSR_UNET_UPBLOCK0_OUTPUT_STATES_ALIGNMENT:', states_status)
    print('TADSR_UNET_UPBLOCK0_ALIGNMENT:', status)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
