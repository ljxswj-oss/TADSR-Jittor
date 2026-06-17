#!/usr/bin/env python3
from __future__ import annotations

from unet_upblock1_attention2_common import add_metrics, attention2_tester, blocked_result, load_oracle, status_from_diagnostics, status_from_metrics, write_report


def main() -> int:
    tensors, blocked = load_oracle()
    name = 'jittor_unet_upblock1_attention2_proj_alignment'
    title = 'TADSR UNet up_blocks.1.attentions.2 Projection Alignment'
    if blocked:
        result = blocked_result(name, title, blocked)
        print('TADSR_UNET_UPBLOCK1_ATTENTION2_PROJ_IN_ALIGNMENT:', result['status'])
        return 0 if result['status'] == 'PASS' else 1
    tester = attention2_tester()
    got = tester.run_input_projection(tensors['synthetic_upblock1_attention2_input'])
    diag = {
        'norm': add_metrics(got['norm'], tensors['synthetic_upblock1_attention2_norm'], tolerance=1e-4),
        'sequence_input': add_metrics(got['sequence_input'], tensors['synthetic_upblock1_attention2_sequence_input'], tolerance=1e-4),
        'proj_in': add_metrics(got['proj_in'], tensors['synthetic_upblock1_attention2_proj_in'], tolerance=1e-4),
    }
    status = status_from_diagnostics(diag)
    write_report(name, title, {'status': status, 'diagnostics': diag})
    print('TADSR_UNET_UPBLOCK1_ATTENTION2_NORM_ALIGNMENT:', status_from_metrics(diag['norm']))
    print('TADSR_UNET_UPBLOCK1_ATTENTION2_SEQUENCE_ALIGNMENT:', status_from_metrics(diag['sequence_input']))
    print('TADSR_UNET_UPBLOCK1_ATTENTION2_PROJ_IN_ALIGNMENT:', status_from_metrics(diag['proj_in']))
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
