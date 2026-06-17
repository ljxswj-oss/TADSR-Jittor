#!/usr/bin/env python3
from __future__ import annotations

from unet_upblock2_attention0_common import add_metrics, attention0_tester, blocked_result, load_oracle, status_from_diagnostics, status_from_metrics, write_report


def main() -> int:
    tensors, blocked = load_oracle()
    name = 'jittor_unet_upblock2_attention0_ff_alignment'
    title = 'TADSR UNet up_blocks.2.attentions.0 Feed-Forward Alignment'
    if blocked:
        result = blocked_result(name, title, blocked)
        print('TADSR_UNET_UPBLOCK2_ATTENTION0_FF_ALIGNMENT:', result['status'])
        return 0 if result['status'] == 'PASS' else 1
    tester = attention0_tester()
    inp = tester.run_input_projection(tensors['synthetic_upblock2_attention0_input'])
    a1 = tester.run_attn1(inp['proj_in'])
    a2 = tester.run_attn2(a1['after_attn1'], tensors['synthetic_upblock2_attention0_encoder_hidden_states'])
    got = tester.run_ff(a2['after_attn2'])
    diag = {
        'norm3': add_metrics(got['norm3'], tensors['synthetic_upblock2_attention0_transformer0_norm3'], tolerance=1e-4),
        'ff_geglu_proj': add_metrics(got['ff_geglu_proj'], tensors['synthetic_upblock2_attention0_transformer0_ff_geglu_proj'], tolerance=1e-4),
        'ff_geglu_output': add_metrics(got['ff_geglu_output'], tensors['synthetic_upblock2_attention0_transformer0_ff_geglu_output'], tolerance=1e-4),
        'ff_output': add_metrics(got['ff_output'], tensors['synthetic_upblock2_attention0_transformer0_ff_output'], tolerance=1e-3),
        'transformer0_output': add_metrics(got['transformer0_output'], tensors['synthetic_upblock2_attention0_transformer0_output'], tolerance=1e-3),
    }
    status = status_from_diagnostics(diag)
    write_report(name, title, {'status': status, 'diagnostics': diag})
    print('TADSR_UNET_UPBLOCK2_ATTENTION0_FF_ALIGNMENT:', status_from_metrics(diag['ff_output']))
    print('TADSR_UNET_UPBLOCK2_ATTENTION0_TRANSFORMER0_ALIGNMENT:', status_from_metrics(diag['transformer0_output']))
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
