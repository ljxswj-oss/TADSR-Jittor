#!/usr/bin/env python3
from __future__ import annotations

from unet_upblock2_attention1_common import add_metrics, attention1_tester, blocked_result, load_oracle, status_from_diagnostics, status_from_metrics, write_report


def main() -> int:
    tensors, blocked = load_oracle()
    name = 'jittor_unet_upblock2_attention1_attn1_alignment'
    title = 'TADSR UNet up_blocks.2.attentions.1 Attn1 Alignment'
    if blocked:
        result = blocked_result(name, title, blocked)
        print('TADSR_UNET_UPBLOCK2_ATTENTION1_ATTN1_ALIGNMENT:', result['status'])
        return 0 if result['status'] == 'PASS' else 1
    tester = attention1_tester()
    inp = tester.run_input_projection(tensors['synthetic_upblock2_attention1_input'])
    got = tester.run_attn1(inp['proj_in'])
    diag = {
        'norm1': add_metrics(got['norm1'], tensors['synthetic_upblock2_attention1_transformer0_norm1'], tolerance=1e-4),
        'attn1_q': add_metrics(got['attn1_q'], tensors['synthetic_upblock2_attention1_transformer0_attn1_q'], tolerance=1e-4),
        'attn1_k': add_metrics(got['attn1_k'], tensors['synthetic_upblock2_attention1_transformer0_attn1_k'], tolerance=1e-4),
        'attn1_v': add_metrics(got['attn1_v'], tensors['synthetic_upblock2_attention1_transformer0_attn1_v'], tolerance=1e-4),
        'attn1_output': add_metrics(got['attn1_output'], tensors['synthetic_upblock2_attention1_transformer0_attn1_output'], tolerance=1e-3),
        'after_attn1': add_metrics(got['after_attn1'], tensors['synthetic_upblock2_attention1_transformer0_after_attn1'], tolerance=1e-3),
    }
    qkv_status = status_from_diagnostics({k: diag[k] for k in ['attn1_q', 'attn1_k', 'attn1_v']})
    status = status_from_diagnostics(diag)
    write_report(name, title, {'status': status, 'diagnostics': diag})
    print('TADSR_UNET_UPBLOCK2_ATTENTION1_ATTN1_QKV_ALIGNMENT:', qkv_status)
    print('TADSR_UNET_UPBLOCK2_ATTENTION1_ATTN1_ALIGNMENT:', status_from_metrics(diag['attn1_output']))
    print('TADSR_UNET_UPBLOCK2_ATTENTION1_AFTER_ATTN1_ALIGNMENT:', status_from_metrics(diag['after_attn1']))
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
