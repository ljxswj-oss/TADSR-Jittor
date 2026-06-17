#!/usr/bin/env python3
from __future__ import annotations

from unet_upblock3_local_common import (
    META,
    add_metrics,
    blocked_result,
    load_json,
    load_oracle,
    local_tester,
    status_from_diagnostics,
    status_from_metrics,
    write_report,
)


def main() -> int:
    tensors, blocked = load_oracle()
    name = 'jittor_unet_upblock3_local_alignment'
    title = 'TADSR UNet up_blocks.3 Local Aggregate Alignment'
    if blocked:
        result = blocked_result(name, title, blocked)
        print('TADSR_UNET_UPBLOCK3_OUTPUT_HIDDEN_ALIGNMENT:', result['status'])
        print('TADSR_UNET_UPBLOCK3_OUTPUT_STATES_ALIGNMENT:', result['status'])
        print('TADSR_UNET_UPBLOCK3_ALIGNMENT:', result['status'])
        print('TADSR_UNET_OUTPUT_TAIL_BOUNDARY_ALIGNMENT:', result['status'])
        return 0 if result['status'] == 'PASS' else 1

    tester = local_tester()
    got = tester.run_entry_downblocks_midblock_upblock0_upblock1_upblock2_upblock3_local(
        tensors['entry_synthetic_unet_sample'],
        tensors['entry_synthetic_unet_timestep'].astype('int64'),
        tensors['entry_encoder_hidden_states'],
        return_intermediates=True,
    )
    expected = {
        'conv_in': 'entry_synthetic_unet_conv_in_output',
        'time_embedding': 'entry_synthetic_unet_time_embedding_output',
        'downblock0_output': 'entry_downblock0_output_hidden',
        'downblock1_output': 'entry_downblock1_output_hidden',
        'downblock2_output': 'entry_downblock2_output_hidden',
        'downblock3_output': 'entry_downblock3_output_hidden',
        'midblock_hidden_output': 'entry_midblock_output_hidden',
        'upblock0_output_hidden': 'entry_upblock0_output_hidden',
        'upblock1_output_hidden': 'entry_upblock1_output_hidden',
        'upblock2_output_hidden': 'entry_upblock2_output_hidden',
        'upblock3_resnet0_output': 'manual_upblock3_resnet0_output',
        'upblock3_attention0_output': 'manual_upblock3_attention0_output',
        'upblock3_resnet1_output': 'manual_upblock3_resnet1_output',
        'upblock3_attention1_output': 'manual_upblock3_attention1_output',
        'upblock3_resnet2_output': 'manual_upblock3_resnet2_output',
        'upblock3_attention2_output': 'manual_upblock3_attention2_output',
        'upblock3_output_hidden': 'local_upblock3_hidden_states_output',
    }
    diag = {}
    for stage, tensor_name in expected.items():
        tol = 1e-4 if stage in {'conv_in', 'time_embedding'} else 2e-3
        diag[stage] = add_metrics(got[stage], tensors[tensor_name], tolerance=tol)
    hidden_status = status_from_metrics(diag['upblock3_output_hidden'])
    meta = load_json(META)
    output_states_status = meta.get('output_states_status', 'NOT_APPLICABLE')
    output_tail_boundary_status = meta.get('output_tail_boundary_status', 'NOT_APPLICABLE')
    status = status_from_diagnostics(diag)
    if output_states_status not in {'PASS', 'NOT_APPLICABLE'}:
        status = 'FAIL'
    if output_tail_boundary_status != 'PASS':
        status = 'FAIL'
    if len(got.get('remaining_internal_residual_samples_after_upblock3', ())) != 0:
        status = 'FAIL'
    result = {
        'status': status,
        'diagnostics': diag,
        'output_hidden_status': hidden_status,
        'output_states_status': output_states_status,
        'output_tail_boundary_status': output_tail_boundary_status,
        'remaining_internal_residual_count_after_upblock3': len(got.get('remaining_internal_residual_samples_after_upblock3', ())),
        'note': (
            'Bridge validates entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 '
            '-> full local up_blocks.1 -> full local up_blocks.2 -> full local up_blocks.3 aggregate. '
            'It deliberately stops at the output-tail boundary and does not run conv_norm_out, conv_act, '
            'conv_out, full UNet forward, runtime LoRA, or full TADSR inference.'
        ),
    }
    write_report(name, title, result)
    print('TADSR_UNET_UPBLOCK3_OUTPUT_HIDDEN_ALIGNMENT:', hidden_status)
    print('TADSR_UNET_UPBLOCK3_OUTPUT_STATES_ALIGNMENT:', output_states_status)
    print('TADSR_UNET_UPBLOCK3_ALIGNMENT:', status)
    print('TADSR_UNET_OUTPUT_TAIL_BOUNDARY_ALIGNMENT:', output_tail_boundary_status)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
