#!/usr/bin/env python3
from __future__ import annotations
from unet_downblock0_attention1_common import *

def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        print(f"TADSR_UNET_DOWNBLOCK0_ATTENTION1_FF_ALIGNMENT: {blocked['status']}")
        blocked_result('jittor_unet_downblock0_attention1_ff_alignment', 'TADSR UNet down_blocks.0.attentions.1 FF Alignment', blocked)
        return 1
    t = attention1_tester()
    inp = t.run_input_projection(oracle['synthetic_attention1_input'])
    a1 = t.run_attn1(inp['proj_in'])
    a2 = t.run_attn2(a1['after_attn1'], oracle['synthetic_attention1_encoder_hidden_states'])
    out = t.run_ff(a2['after_attn2'])
    diagnostics = {
        'norm3': add_metrics(out['norm3'], oracle['synthetic_attention1_transformer0_norm3'], 1e-4),
        'ff_geglu_proj': add_metrics(out['ff_geglu_proj'], oracle['synthetic_attention1_transformer0_ff_geglu_proj'], 1e-4),
        'ff_output': add_metrics(out['ff_output'], oracle['synthetic_attention1_transformer0_ff_output'], 1e-3),
        'transformer0_output': add_metrics(out['transformer0_output'], oracle['synthetic_attention1_transformer0_output'], 1e-3),
    }
    result = {'status': status_from_diagnostics(diagnostics), 'target': 'attention1 transformer0 feed-forward and residual', 'note': 'Stops before proj_out.', 'diagnostics': diagnostics}
    write_report('jittor_unet_downblock0_attention1_ff_alignment', 'TADSR UNet down_blocks.0.attentions.1 FF Alignment', result)
    print(f"TADSR_UNET_DOWNBLOCK0_ATTENTION1_FF_ALIGNMENT: {status_from_metrics(diagnostics['ff_output'])}")
    print(f"TADSR_UNET_DOWNBLOCK0_ATTENTION1_TRANSFORMER0_ALIGNMENT: {status_from_metrics(diagnostics['transformer0_output'])}")
    return 0 if result['status'] in {'PASS','PARTIAL'} else 1

if __name__ == '__main__':
    raise SystemExit(main())
