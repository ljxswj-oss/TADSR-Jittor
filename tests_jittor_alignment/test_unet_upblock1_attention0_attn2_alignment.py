#!/usr/bin/env python3
from __future__ import annotations

from unet_upblock1_attention0_common import add_metrics, attention0_tester, blocked_result, load_oracle, status_from_diagnostics, status_from_metrics, write_report


def main() -> int:
    tensors, blocked = load_oracle()
    name = 'jittor_unet_upblock1_attention0_attn2_alignment'
    title = 'TADSR UNet up_blocks.1.attentions.0 Attn2 Alignment'
    if blocked:
        result = blocked_result(name, title, blocked)
        print('TADSR_UNET_UPBLOCK1_ATTENTION0_ATTN2_ALIGNMENT:', result['status'])
        return 0 if result['status'] == 'PASS' else 1
    tester = attention0_tester()
    inp = tester.run_input_projection(tensors['synthetic_upblock1_attention0_input'])
    a1 = tester.run_attn1(inp['proj_in'])
    got = tester.run_attn2(a1['after_attn1'], tensors['synthetic_upblock1_attention0_encoder_hidden_states'])
    diag = {
        'norm2': add_metrics(got['norm2'], tensors['synthetic_upblock1_attention0_transformer0_norm2'], tolerance=1e-4),
        'attn2_q': add_metrics(got['attn2_q'], tensors['synthetic_upblock1_attention0_transformer0_attn2_q'], tolerance=1e-4),
        'attn2_k': add_metrics(got['attn2_k'], tensors['synthetic_upblock1_attention0_transformer0_attn2_k'], tolerance=1e-4),
        'attn2_v': add_metrics(got['attn2_v'], tensors['synthetic_upblock1_attention0_transformer0_attn2_v'], tolerance=1e-4),
        'attn2_output': add_metrics(got['attn2_output'], tensors['synthetic_upblock1_attention0_transformer0_attn2_output'], tolerance=1e-3),
        'after_attn2': add_metrics(got['after_attn2'], tensors['synthetic_upblock1_attention0_transformer0_after_attn2'], tolerance=1e-3),
    }
    qkv_status = status_from_diagnostics({k: diag[k] for k in ['attn2_q', 'attn2_k', 'attn2_v']})
    status = status_from_diagnostics(diag)
    write_report(name, title, {'status': status, 'diagnostics': diag})
    print('TADSR_UNET_UPBLOCK1_ATTENTION0_ATTN2_QKV_ALIGNMENT:', qkv_status)
    print('TADSR_UNET_UPBLOCK1_ATTENTION0_ATTN2_ALIGNMENT:', status_from_metrics(diag['attn2_output']))
    print('TADSR_UNET_UPBLOCK1_ATTENTION0_AFTER_ATTN2_ALIGNMENT:', status_from_metrics(diag['after_attn2']))
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
