#!/usr/bin/env python3
from __future__ import annotations

from unet_upblock2_local_common import (
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
    name = 'jittor_unet_upblock2_local_alignment'
    title = 'TADSR UNet up_blocks.2 Local Aggregate Alignment'
    if blocked:
        result = blocked_result(name, title, blocked)
        print('TADSR_UNET_UPBLOCK2_OUTPUT_HIDDEN_ALIGNMENT:', result['status'])
        print('TADSR_UNET_UPBLOCK2_OUTPUT_STATES_ALIGNMENT:', result['status'])
        print('TADSR_UNET_UPBLOCK2_ALIGNMENT:', result['status'])
        return 0 if result['status'] == 'PASS' else 1
    tester = local_tester()
    got = tester.run_entry_downblocks_midblock_upblock0_upblock1_upblock2_local(
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
        'upblock2_resnet0_output': 'manual_upblock2_resnet0_output',
        'upblock2_attention0_output': 'manual_upblock2_attention0_output',
        'upblock2_resnet1_output': 'manual_upblock2_resnet1_output',
        'upblock2_attention1_output': 'manual_upblock2_attention1_output',
        'upblock2_resnet2_output': 'manual_upblock2_resnet2_output',
        'upblock2_attention2_output': 'manual_upblock2_attention2_output',
        'upblock2_upsampler_output': 'manual_upblock2_upsampler_output',
        'upblock2_output_hidden': 'local_upblock2_hidden_states_output',
    }
    diag = {}
    for stage, path in expected.items():
        tol = 1e-4 if stage in {'conv_in', 'time_embedding'} else 2e-3
        diag[stage] = add_metrics(got[stage], tensors[path], tolerance=tol)
    hidden_status = status_from_metrics(diag['upblock2_output_hidden'])
    output_states_status = load_json(META).get('output_states_status', 'NOT_APPLICABLE')
    status = status_from_diagnostics(diag)
    if output_states_status not in {'PASS', 'NOT_APPLICABLE'}:
        status = 'FAIL'
    result = {
        'status': status,
        'diagnostics': diag,
        'output_hidden_status': hidden_status,
        'output_states_status': output_states_status,
        'note': (
            'Bridge validates entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 '
            '-> full local up_blocks.1 -> full local up_blocks.2 aggregate. It deliberately stops before '
            'up_blocks.3, full UNet forward, runtime LoRA, and full TADSR inference.'
        ),
    }
    write_report(name, title, result)
    print('TADSR_UNET_UPBLOCK2_OUTPUT_HIDDEN_ALIGNMENT:', hidden_status)
    print('TADSR_UNET_UPBLOCK2_OUTPUT_STATES_ALIGNMENT:', output_states_status)
    print('TADSR_UNET_UPBLOCK2_ALIGNMENT:', status)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
