#!/usr/bin/env python3
from __future__ import annotations
from unet_downblock0_attention1_common import *

def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        print(f"TADSR_UNET_DOWNBLOCK0_ATTENTION1_ALIGNMENT: {blocked['status']}")
        blocked_result('jittor_unet_downblock0_attention1_alignment', 'TADSR UNet down_blocks.0.attentions.1 Alignment', blocked)
        return 1
    t = attention1_tester()
    out = t.run_attention0(oracle['synthetic_attention1_input'], oracle['synthetic_attention1_encoder_hidden_states'], return_intermediates=True)
    diagnostics = {
        'proj_out_sequence': add_metrics(out['proj_out_sequence'], oracle['synthetic_attention1_proj_out_sequence'], 1e-3),
        'proj_out_nchw': add_metrics(out['proj_out_nchw'], oracle['synthetic_attention1_proj_out_nchw'], 1e-3),
        'output': add_metrics(out['output'], oracle['synthetic_attention1_output'], 2e-3),
    }
    result = {'status': status_from_diagnostics(diagnostics), 'target': 'isolated down_blocks.0.attentions.1 only', 'note': 'Stops before downsampler/full down_blocks.0.', 'diagnostics': diagnostics}
    write_report('jittor_unet_downblock0_attention1_alignment', 'TADSR UNet down_blocks.0.attentions.1 Alignment', result)
    print(f"TADSR_UNET_DOWNBLOCK0_ATTENTION1_ALIGNMENT: {result['status']}")
    return 0 if result['status'] in {'PASS','PARTIAL'} else 1

if __name__ == '__main__':
    raise SystemExit(main())
