#!/usr/bin/env python3
from __future__ import annotations
import json
from time_vae_decoder_entry_common import *

def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        result = blocked
    else:
        tester = TimeVAEBlockTester(WEIGHTS)
        mode = tester.run_posterior_mode(oracle['real_moments'])
        post = tester.run_post_quant_conv(mode)
        dec = tester.run_decoder_conv_in(post)
        diagnostics = {
            'posterior_mode': add_metrics(mode, oracle['real_latent_dist_mode']),
            'post_quant_conv': add_metrics(post, oracle['real_post_quant_conv_output']),
            'decoder_conv_in': add_metrics(dec, oracle['real_decoder_conv_in_output']),
        }
        worst = max(m.get('max_abs_error', 1.0) for m in diagnostics.values())
        result = {'status': 'PASS' if worst <= 1e-4 else ('PARTIAL' if worst <= 1e-3 else 'FAIL'), 'target': 'quant_conv moments -> mode -> post_quant_conv -> decoder.conv_in', 'note': 'real encoder moments oracle; no latent sampling', 'diagnostics': diagnostics, 'metrics': diagnostics['decoder_conv_in'], 'weights': str(WEIGHTS)}
        if result['status'] == 'FAIL': result['failure_analysis'] = failure_analysis()
    write_report('jittor_quant_to_decoder_entry_alignment', 'TimeAware VAE Quant To Decoder Entry Alignment', result)
    print(f"TIME_VAE_QUANT_TO_DECODER_ENTRY_ALIGNMENT: {result['status']}")
    if result.get('diagnostics'): print(json.dumps(result['diagnostics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1
if __name__ == '__main__': raise SystemExit(main())
