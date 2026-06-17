#!/usr/bin/env python3
from __future__ import annotations
from unet_downblock0_attention1_common import *

def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        print(f"TADSR_UNET_DOWNBLOCK0_ATTENTION1_ATTN2_ALIGNMENT: {blocked['status']}")
        blocked_result('jittor_unet_downblock0_attention1_attn2_alignment', 'TADSR UNet down_blocks.0.attentions.1 Attn2 Alignment', blocked)
        return 1
    t = attention1_tester()
    inp = t.run_input_projection(oracle['synthetic_attention1_input'])
    a1 = t.run_attn1(inp['proj_in'])
    out = t.run_attn2(a1['after_attn1'], oracle['synthetic_attention1_encoder_hidden_states'])
    diagnostics = {
        'norm2': add_metrics(out['norm2'], oracle['synthetic_attention1_transformer0_norm2'], 1e-4),
        'attn2_q': add_metrics(out['attn2_q'], oracle['synthetic_attention1_transformer0_attn2_q'], 1e-4),
        'attn2_k': add_metrics(out['attn2_k'], oracle['synthetic_attention1_transformer0_attn2_k'], 1e-4),
        'attn2_v': add_metrics(out['attn2_v'], oracle['synthetic_attention1_transformer0_attn2_v'], 1e-4),
        'attn2_output': add_metrics(out['attn2_output'], oracle['synthetic_attention1_transformer0_attn2_output'], 1e-3),
        'after_attn2': add_metrics(out['after_attn2'], oracle['synthetic_attention1_transformer0_after_attn2'], 1e-3),
    }
    result = {'status': status_from_diagnostics(diagnostics), 'target': 'attention1 transformer0 cross-attention only', 'note': 'Stops before FF.', 'diagnostics': diagnostics}
    write_report('jittor_unet_downblock0_attention1_attn2_alignment', 'TADSR UNet down_blocks.0.attentions.1 Attn2 Alignment', result)
    qkv_status = status_from_diagnostics({k: diagnostics[k] for k in ['attn2_q','attn2_k','attn2_v']})
    print(f"TADSR_UNET_DOWNBLOCK0_ATTENTION1_ATTN2_QKV_ALIGNMENT: {qkv_status}")
    print(f"TADSR_UNET_DOWNBLOCK0_ATTENTION1_ATTN2_ALIGNMENT: {status_from_metrics(diagnostics['attn2_output'])}")
    print(f"TADSR_UNET_DOWNBLOCK0_ATTENTION1_AFTER_ATTN2_ALIGNMENT: {status_from_metrics(diagnostics['after_attn2'])}")
    return 0 if result['status'] in {'PASS','PARTIAL'} else 1

if __name__ == '__main__':
    raise SystemExit(main())
