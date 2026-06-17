#!/usr/bin/env python3
from __future__ import annotations
from unet_downblock0_attention0_common import *

def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        print(f"TADSR_UNET_DOWNBLOCK0_ATTENTION0_ATTN1_ALIGNMENT: {blocked['status']}")
        blocked_result('jittor_unet_downblock0_attention0_attn1_alignment', 'TADSR UNet attention0 attn1 Alignment', blocked)
        return 1
    t = attention_tester()
    inp = t.run_input_projection(oracle['synthetic_attention0_input'])
    out = t.run_attn1(inp['proj_in'])
    diagnostics = {
        'norm1': add_metrics(out['norm1'], oracle['synthetic_attention0_transformer0_norm1'], 1e-4),
        'attn1_q': add_metrics(out['attn1_q'], oracle['synthetic_attention0_transformer0_attn1_q'], 1e-4),
        'attn1_k': add_metrics(out['attn1_k'], oracle['synthetic_attention0_transformer0_attn1_k'], 1e-4),
        'attn1_v': add_metrics(out['attn1_v'], oracle['synthetic_attention0_transformer0_attn1_v'], 1e-4),
        'attn1_output': add_metrics(out['attn1_output'], oracle['synthetic_attention0_transformer0_attn1_output'], 1e-3),
        'after_attn1': add_metrics(out['after_attn1'], oracle['synthetic_attention0_transformer0_after_attn1'], 1e-3),
    }
    result = {'status': status_from_diagnostics(diagnostics), 'target': 'transformer_blocks.0 norm1 + self-attention attn1', 'diagnostics': diagnostics}
    write_report('jittor_unet_downblock0_attention0_attn1_alignment', 'TADSR UNet attention0 attn1 Alignment', result)
    qkv = status_from_diagnostics({k: diagnostics[k] for k in ['attn1_q','attn1_k','attn1_v']})
    print(f"TADSR_UNET_DOWNBLOCK0_ATTENTION0_ATTN1_QKV_ALIGNMENT: {qkv}")
    print(f"TADSR_UNET_DOWNBLOCK0_ATTENTION0_ATTN1_ALIGNMENT: {status_from_metrics(diagnostics['attn1_output'])}")
    print(f"TADSR_UNET_DOWNBLOCK0_ATTENTION0_AFTER_ATTN1_ALIGNMENT: {status_from_metrics(diagnostics['after_attn1'])}")
    return 0 if result['status'] in {'PASS','PARTIAL'} else 1
if __name__ == '__main__': raise SystemExit(main())
