#!/usr/bin/env python3
from __future__ import annotations

from unet_upblock3_attention0_common import add_metrics, attention0_tester, blocked_result, load_oracle, status_from_diagnostics, write_report


def main() -> int:
    tensors, blocked = load_oracle()
    name = 'jittor_unet_upblock3_attention0_alignment'
    title = 'TADSR UNet up_blocks.3.attentions.0 Full Local Alignment'
    if blocked:
        result = blocked_result(name, title, blocked)
        print('TADSR_UNET_UPBLOCK3_ATTENTION0_ALIGNMENT:', result['status'])
        return 0 if result['status'] == 'PASS' else 1
    tester = attention0_tester()
    got = tester.run_attention0(tensors['synthetic_upblock3_attention0_input'], tensors['synthetic_upblock3_attention0_encoder_hidden_states'], return_intermediates=True)
    diag = {
        'output': add_metrics(got['output'], tensors['synthetic_upblock3_attention0_output'], tolerance=2e-3),
        'official_output': add_metrics(got['output'], tensors['synthetic_upblock3_attention0_official_output'], tolerance=2e-3),
    }
    status = status_from_diagnostics(diag)
    write_report(name, title, {'status': status, 'diagnostics': diag, 'note': 'Only up_blocks.3.attentions.0 is executed; full up_blocks.3 remains unopened.'})
    print('TADSR_UNET_UPBLOCK3_ATTENTION0_ALIGNMENT:', status)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
