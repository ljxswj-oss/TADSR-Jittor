#!/usr/bin/env python3
from __future__ import annotations

from unet_manual_wrapper_common import (
    add_metrics,
    blocked_result,
    load_oracle,
    manual_wrapper_tester,
    status_from_diagnostics,
    status_from_metrics,
    write_report,
)


def main() -> int:
    tensors, blocked = load_oracle()
    name = 'jittor_unet_manual_wrapper_alignment'
    title = 'TADSR UNet Manual Full-Chain Wrapper Alignment'
    if blocked:
        result = blocked_result(name, title, blocked)
        for marker in [
            'TADSR_UNET_MANUAL_WRAPPER_ENTRY_ALIGNMENT',
            'TADSR_UNET_MANUAL_WRAPPER_DOWNBLOCKS_ALIGNMENT',
            'TADSR_UNET_MANUAL_WRAPPER_MIDBLOCK_ALIGNMENT',
            'TADSR_UNET_MANUAL_WRAPPER_UPBLOCKS_ALIGNMENT',
            'TADSR_UNET_MANUAL_WRAPPER_OUTPUT_TAIL_ALIGNMENT',
            'TADSR_UNET_MANUAL_FULL_WRAPPER_ALIGNMENT',
            'TADSR_UNET_MANUAL_FULL_CHAIN_ALIGNMENT',
        ]:
            print(f'{marker}:', result['status'])
        return 0 if result['status'] == 'PASS' else 1

    tester = manual_wrapper_tester()
    got = tester.run_manual_unet_chain_for_alignment(
        tensors['manual_wrapper_sample'],
        tensors['manual_wrapper_timestep'].astype('int64'),
        tensors['manual_wrapper_encoder_hidden_states'],
        return_intermediates=True,
    )
    expected = {
        'conv_in': ('manual_wrapper_conv_in_output', 1e-4),
        'time_embedding': ('manual_wrapper_time_embedding_output', 1e-4),
        'downblock0_output': ('manual_wrapper_downblock0_output', 2e-3),
        'downblock1_output': ('manual_wrapper_downblock1_output', 2e-3),
        'downblock2_output': ('manual_wrapper_downblock2_output', 2e-3),
        'downblock3_output': ('manual_wrapper_downblock3_output', 2e-3),
        'midblock_hidden_output': ('manual_wrapper_midblock_output', 2e-3),
        'upblock0_output_hidden': ('manual_wrapper_upblock0_output', 2e-3),
        'upblock1_output_hidden': ('manual_wrapper_upblock1_output', 2e-3),
        'upblock2_output_hidden': ('manual_wrapper_upblock2_output', 2e-3),
        'upblock3_output_hidden': ('manual_wrapper_upblock3_output', 2e-3),
        'output_tail_norm_output': ('manual_wrapper_output_tail_norm_output', 2e-3),
        'output_tail_act_output': ('manual_wrapper_output_tail_act_output', 2e-3),
        'manual_wrapper_output': ('manual_wrapper_output', 2e-3),
    }
    diag = {}
    for stage, (tensor_name, tol) in expected.items():
        diag[stage] = add_metrics(got[stage], tensors[tensor_name], tolerance=tol)

    entry_status = status_from_diagnostics({k: diag[k] for k in ['conv_in', 'time_embedding']})
    downblocks_status = status_from_diagnostics({k: diag[k] for k in ['downblock0_output', 'downblock1_output', 'downblock2_output', 'downblock3_output']})
    midblock_status = status_from_metrics(diag['midblock_hidden_output'])
    upblocks_status = status_from_diagnostics({k: diag[k] for k in ['upblock0_output_hidden', 'upblock1_output_hidden', 'upblock2_output_hidden', 'upblock3_output_hidden']})
    tail_status = status_from_diagnostics({k: diag[k] for k in ['output_tail_norm_output', 'output_tail_act_output', 'manual_wrapper_output']})
    status = status_from_diagnostics(diag)
    result = {
        'status': status,
        'diagnostics': diag,
        'entry_status': entry_status,
        'downblocks_status': downblocks_status,
        'midblock_status': midblock_status,
        'upblocks_status': upblocks_status,
        'output_tail_status': tail_status,
        'note': (
            'Alignment-only manual wrapper validates sample -> center -> conv_in -> time embedding '
            '-> down_blocks -> mid_block -> up_blocks.0/1/2/3 -> output tail. It does not call '
            'official UNet.forward, production Jittor forward, scheduler, VAE, runtime LoRA, or full inference.'
        ),
    }
    write_report(name, title, result)
    print('TADSR_UNET_MANUAL_WRAPPER_ENTRY_ALIGNMENT:', entry_status)
    print('TADSR_UNET_MANUAL_WRAPPER_DOWNBLOCKS_ALIGNMENT:', downblocks_status)
    print('TADSR_UNET_MANUAL_WRAPPER_MIDBLOCK_ALIGNMENT:', midblock_status)
    print('TADSR_UNET_MANUAL_WRAPPER_UPBLOCKS_ALIGNMENT:', upblocks_status)
    print('TADSR_UNET_MANUAL_WRAPPER_OUTPUT_TAIL_ALIGNMENT:', tail_status)
    print('TADSR_UNET_MANUAL_FULL_WRAPPER_ALIGNMENT:', status)
    print('TADSR_UNET_MANUAL_FULL_CHAIN_ALIGNMENT:', status)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
