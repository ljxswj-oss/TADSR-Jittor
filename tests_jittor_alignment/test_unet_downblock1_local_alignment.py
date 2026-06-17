#!/usr/bin/env python3
from __future__ import annotations
from unet_downblock1_downsampler_common import *


def main() -> int:
    tensors, blocked = load_oracle()
    name = 'jittor_unet_downblock1_alignment'
    title = 'TADSR UNet Full Local down_blocks.1 Alignment'
    if blocked:
        result = blocked_result(name, title, blocked)
        print('TADSR_UNET_DOWNBLOCK1_ALIGNMENT:', result['status'])
        return 0 if result['status'] == 'PASS' else 1
    tester = downblock1_tester()
    got = tester.run_downblock1_local(
        tensors['entry_downblock0_output_hidden'],
        tensors['entry_synthetic_unet_time_embedding_output'],
        tensors['entry_encoder_hidden_states'],
        return_output_states=True,
        return_intermediates=True,
    )
    diagnostics = {
        'downblock1_resnet0_output': add_metrics(got['downblock1_resnet0_output'], tensors['entry_downblock1_resnet0_output'], tolerance=2e-3),
        'downblock1_attention0_output': add_metrics(got['downblock1_attention0_output'], tensors['entry_downblock1_attention0_output'], tolerance=2e-3),
        'downblock1_resnet1_output': add_metrics(got['downblock1_resnet1_output'], tensors['entry_downblock1_resnet1_output'], tolerance=2e-3),
        'downblock1_attention1_output': add_metrics(got['downblock1_attention1_output'], tensors['entry_downblock1_attention1_output'], tolerance=2e-3),
        'downblock1_downsampler_output': add_metrics(got['downblock1_downsampler_output'], tensors['entry_downblock1_downsampler_output'], tolerance=2e-3),
        'downblock1_hidden_states_output': add_metrics(got['downblock1_output'], tensors['local_downblock1_hidden_states_output'], tolerance=2e-3),
    }
    output_state_diag = {}
    for idx, got_state in enumerate(got.get('output_states', [])):
        key = f'local_downblock1_res_sample_{idx}'
        if key in tensors:
            output_state_diag[f'output_state_{idx}'] = add_metrics(got_state, tensors[key], tolerance=2e-3)
    diagnostics.update(output_state_diag)
    status = status_from_diagnostics(diagnostics)
    output_states_status = status_from_diagnostics(output_state_diag) if output_state_diag else 'FAIL'
    result = {
        'status': status,
        'target': 'down_blocks.1 local: resnet0 -> attention0 -> resnet1 -> attention1 -> downsampler',
        'note': 'Completes local down_blocks.1 only. Full UNet forward, down_blocks.2, mid_block, up_blocks and full TADSR inference remain NotImplemented.',
        'diagnostics': diagnostics,
    }
    write_report(name, title, result)
    print(f"TADSR_UNET_DOWNBLOCK1_OUTPUT_HIDDEN_ALIGNMENT: {status_from_metrics(diagnostics['downblock1_hidden_states_output'])}")
    print(f"TADSR_UNET_DOWNBLOCK1_OUTPUT_STATES_ALIGNMENT: {output_states_status}")
    print(f"TADSR_UNET_DOWNBLOCK1_ALIGNMENT: {status}")
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
