#!/usr/bin/env python3
from __future__ import annotations

from unet_output_tail_common import (
    add_metrics,
    blocked_result,
    load_oracle,
    output_tail_tester,
    status_from_diagnostics,
    status_from_metrics,
    write_report,
)


def main() -> int:
    tensors, blocked = load_oracle()
    name = 'jittor_unet_entry_downblocks_midblock_upblocks_output_tail_alignment'
    title = 'TADSR UNet Entry Through Output Tail Alignment'
    if blocked:
        result = blocked_result(name, title, blocked)
        print('TADSR_UNET_ENTRY_TO_OUTPUT_TAIL_ALIGNMENT:', result['status'])
        print('TADSR_UNET_MANUAL_BLOCKS_TO_TAIL_ALIGNMENT:', result['status'])
        return 0 if result['status'] == 'PASS' else 1

    tester = output_tail_tester()
    got = tester.run_entry_downblocks_midblock_upblocks_output_tail(
        tensors['entry_synthetic_unet_sample'],
        tensors['entry_synthetic_unet_timestep'].astype('int64'),
        tensors['entry_encoder_hidden_states'],
        return_intermediates=True,
    )
    expected = {
        'conv_in': ('entry_synthetic_unet_conv_in_output', 1e-4),
        'time_embedding': ('entry_synthetic_unet_time_embedding_output', 1e-4),
        'downblock0_output': ('entry_downblock0_output_hidden', 2e-3),
        'downblock1_output': ('entry_downblock1_output_hidden', 2e-3),
        'downblock2_output': ('entry_downblock2_output_hidden', 2e-3),
        'downblock3_output': ('entry_downblock3_output_hidden', 2e-3),
        'midblock_hidden_output': ('entry_midblock_output_hidden', 2e-3),
        'upblock0_output_hidden': ('entry_upblock0_output_hidden', 2e-3),
        'upblock1_output_hidden': ('entry_upblock1_output_hidden', 2e-3),
        'upblock2_output_hidden': ('entry_upblock2_output_hidden', 2e-3),
        'upblock3_output_hidden': ('entry_upblock3_output_hidden', 2e-3),
        'output_tail_norm_output': ('entry_output_tail_norm_output', 2e-3),
        'output_tail_act_output': ('entry_output_tail_act_output', 2e-3),
        'output_tail_conv_out_output': ('entry_output_tail_conv_out_output', 2e-3),
    }
    diag = {}
    for stage, (tensor_name, tol) in expected.items():
        diag[stage] = add_metrics(got[stage], tensors[tensor_name], tolerance=tol)
    status = status_from_diagnostics(diag)
    output_tail_status = status_from_metrics(diag['output_tail_conv_out_output'])
    result = {
        'status': status,
        'diagnostics': diag,
        'output_tail_status': output_tail_status,
        'note': (
            'Bridge validates entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 '
            '-> full local up_blocks.1 -> full local up_blocks.2 -> full local up_blocks.3 -> output tail. '
            'It still does not call official UNet.forward, scheduler denoising, VAE integration, runtime '
            'LoRA, image postprocessing, or full TADSR inference.'
        ),
    }
    write_report(name, title, result)
    print('TADSR_UNET_ENTRY_TO_OUTPUT_TAIL_ALIGNMENT:', status)
    print('TADSR_UNET_MANUAL_BLOCKS_TO_TAIL_ALIGNMENT:', status)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
