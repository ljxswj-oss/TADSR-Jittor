#!/usr/bin/env python3
from __future__ import annotations

from unet_downblock2_attention1_common import add_metrics, attention1_tester, blocked_result, load_oracle, status_from_diagnostics, write_report


def main() -> int:
    tensors, blocked = load_oracle()
    name = 'jittor_unet_downblock2_attention1_alignment'
    title = 'TADSR UNet down_blocks.2.attentions.1 Full Local Alignment'
    if blocked:
        result = blocked_result(name, title, blocked)
        print('TADSR_UNET_DOWNBLOCK2_ATTENTION1_ALIGNMENT:', result['status'])
        return 0 if result['status'] == 'PASS' else 1
    tester = attention1_tester()
    got = tester.run_attention0(tensors['synthetic_downblock2_attention1_input'], tensors['synthetic_downblock2_attention1_encoder_hidden_states'], return_intermediates=True)
    diag = {
        'output': add_metrics(got['output'], tensors['synthetic_downblock2_attention1_output'], tolerance=2e-3),
        'official_output': add_metrics(got['output'], tensors['synthetic_downblock2_attention1_official_output'], tolerance=2e-3),
    }
    status = status_from_diagnostics(diag)
    write_report(name, title, {'status': status, 'diagnostics': diag, 'note': 'Only down_blocks.2.attentions.1 is executed; full down_blocks.2 remains unopened.'})
    print('TADSR_UNET_DOWNBLOCK2_ATTENTION1_ALIGNMENT:', status)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
