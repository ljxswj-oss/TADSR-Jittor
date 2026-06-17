#!/usr/bin/env python3
from __future__ import annotations

from unet_midblock_resnet1_common import add_metrics, blocked_result, bridge_tester, load_json, load_oracle, status_from_diagnostics, write_report, META


def main() -> int:
    tensors, blocked = load_oracle()
    name = 'jittor_unet_midblock_local_alignment'
    title = 'TADSR UNet Local MidBlock Hidden Output Alignment'
    if blocked:
        result = blocked_result(name, title, blocked)
        print('TADSR_UNET_MIDBLOCK_OUTPUT_HIDDEN_ALIGNMENT:', result['status'])
        print('TADSR_UNET_MIDBLOCK_OUTPUT_STATES_ALIGNMENT: NOT_COMPLETE')
        print('TADSR_UNET_MIDBLOCK_ALIGNMENT:', result['status'])
        return 0 if result['status'] == 'PASS' else 1
    tester = bridge_tester()
    got = tester.run_entry_downblocks_midblock_resnet0_attention0_resnet1(
        tensors['entry_synthetic_unet_sample'],
        tensors['entry_synthetic_unet_timestep'].astype('int64'),
        tensors['entry_encoder_hidden_states'],
        return_intermediates=True,
    )
    diag = {
        'midblock_hidden_output': add_metrics(got['midblock_resnet1_output'], tensors['local_midblock_hidden_states_output'], tolerance=2e-3),
    }
    hidden_status = status_from_diagnostics(diag)
    meta = load_json(META)
    states_status = meta.get('local_midblock_output_states_status', 'NOT_APPLICABLE')
    overall = hidden_status if states_status in {'PASS', 'NOT_APPLICABLE'} else 'PARTIAL'
    result = {
        'status': overall,
        'diagnostics': diag,
        'output_states_status': states_status,
        'note': 'Manual chain resnets.0 -> attentions.0 -> resnets.1 is compared with official local mid_block hidden output. up_blocks/full UNet are not executed.',
    }
    write_report(name, title, result)
    print('TADSR_UNET_MIDBLOCK_OUTPUT_HIDDEN_ALIGNMENT:', hidden_status)
    print(f'TADSR_UNET_MIDBLOCK_OUTPUT_STATES_ALIGNMENT: {states_status}')
    print('TADSR_UNET_MIDBLOCK_ALIGNMENT:', overall)
    return 0 if overall == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
