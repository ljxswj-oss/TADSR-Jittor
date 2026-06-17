#!/usr/bin/env python3
from __future__ import annotations

from unet_downblock3_resnet1_common import add_metrics, blocked_result, downblock3_tester, load_oracle, status_from_diagnostics, status_from_metrics, write_report


def main() -> int:
    tensors, blocked = load_oracle()
    name = 'jittor_unet_downblock3_alignment'
    title = 'TADSR UNet Full Local down_blocks.3 Alignment'
    if blocked:
        result = blocked_result(name, title, blocked)
        print('TADSR_UNET_DOWNBLOCK3_ALIGNMENT:', result['status'])
        return 0 if result['status'] == 'PASS' else 1
    tester = downblock3_tester()
    got = tester.run_downblock3_local(
        tensors['entry_downblock2_output_hidden'],
        tensors['entry_synthetic_unet_time_embedding_output'],
        return_output_states=True,
        return_intermediates=True,
    )
    diagnostics = {
        'downblock3_resnet0_output': add_metrics(got['downblock3_resnet0_output'], tensors['entry_downblock3_resnet0_output'], tolerance=2e-3),
        'downblock3_resnet1_output': add_metrics(got['downblock3_resnet1_output'], tensors['entry_downblock3_resnet1_output'], tolerance=2e-3),
        'downblock3_hidden_states_output': add_metrics(got['downblock3_output'], tensors['local_downblock3_hidden_states_output'], tolerance=2e-3),
    }
    output_state_diag = {}
    for idx, got_state in enumerate(got.get('output_states', [])):
        key = f'local_downblock3_res_sample_{idx}'
        if key in tensors:
            output_state_diag[f'output_state_{idx}'] = add_metrics(got_state, tensors[key], tolerance=2e-3)
    diagnostics.update(output_state_diag)
    status = status_from_diagnostics(diagnostics)
    output_states_status = status_from_diagnostics(output_state_diag) if output_state_diag else 'FAIL'
    result = {
        'status': status,
        'target': 'down_blocks.3 local: resnet0 -> resnet1',
        'note': 'Completes local down_blocks.3 hidden output and residual output_states tuple only. mid_block, up_blocks and full UNet forward remain NotImplemented.',
        'diagnostics': diagnostics,
    }
    write_report(name, title, result)
    print(f"TADSR_UNET_DOWNBLOCK3_OUTPUT_HIDDEN_ALIGNMENT: {status_from_metrics(diagnostics['downblock3_hidden_states_output'])}")
    print(f"TADSR_UNET_DOWNBLOCK3_OUTPUT_STATES_ALIGNMENT: {output_states_status}")
    print(f"TADSR_UNET_DOWNBLOCK3_ALIGNMENT: {status}")
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
