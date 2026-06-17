#!/usr/bin/env python3
from __future__ import annotations
from unet_downblock0_attention1_common import *

def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        print(f"TADSR_UNET_DOWNBLOCK0_ATTENTION1_PROJ_IN_ALIGNMENT: {blocked['status']}")
        blocked_result('jittor_unet_downblock0_attention1_proj_alignment', 'TADSR UNet down_blocks.0.attentions.1 Projection Alignment', blocked)
        return 1
    t = attention1_tester()
    out = t.run_input_projection(oracle['synthetic_attention1_input'])
    diagnostics = {
        'norm': add_metrics(out['norm'], oracle['synthetic_attention1_norm'], 1e-4),
        'sequence_input': add_metrics(out['sequence_input'], oracle['synthetic_attention1_sequence_input'], 1e-4),
        'proj_in': add_metrics(out['proj_in'], oracle['synthetic_attention1_proj_in'], 1e-4),
    }
    result = {'status': status_from_diagnostics(diagnostics), 'target': 'isolated attention1 norm/sequence/proj_in only', 'note': 'Stops before transformer block.', 'diagnostics': diagnostics}
    write_report('jittor_unet_downblock0_attention1_proj_alignment', 'TADSR UNet down_blocks.0.attentions.1 Projection Alignment', result)
    print(f"TADSR_UNET_DOWNBLOCK0_ATTENTION1_NORM_ALIGNMENT: {status_from_metrics(diagnostics['norm'])}")
    print(f"TADSR_UNET_DOWNBLOCK0_ATTENTION1_PROJ_IN_ALIGNMENT: {status_from_metrics(diagnostics['proj_in'])}")
    print(f"TADSR_UNET_DOWNBLOCK0_ATTENTION1_SEQUENCE_ALIGNMENT: {status_from_metrics(diagnostics['sequence_input'])}")
    return 0 if result['status'] in {'PASS','PARTIAL'} else 1

if __name__ == '__main__':
    raise SystemExit(main())
