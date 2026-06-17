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
        moments = tester.run_encoder_stage0123_mid_tail_quant(oracle['real_synthetic_image_tensor'], oracle['real_temb'])
        mode = tester.run_posterior_mode(moments)
        post = tester.run_post_quant_conv(mode)
        dec = tester.run_decoder_conv_in(post)
        diagnostics = {
            'encoder_to_quant_moments': add_metrics(moments, oracle['real_moments']),
            'posterior_mode': add_metrics(mode, oracle['real_latent_dist_mode']),
            'post_quant_conv': add_metrics(post, oracle['real_post_quant_conv_output']),
            'decoder_conv_in': add_metrics(dec, oracle['real_decoder_conv_in_output']),
        }
        worst = max(m.get('max_abs_error', 1.0) for m in diagnostics.values())
        result = {'status': 'PASS' if worst <= 1e-4 else ('PARTIAL' if worst <= 1e-3 else 'FAIL'), 'target': 'encoder.conv_in + down_blocks.0..3 + mid_block + tail + quant_conv + posterior mode + post_quant_conv + decoder.conv_in', 'note': 'deterministic encoder-to-decoder-entry bridge; no latent sampling', 'diagnostics': diagnostics, 'metrics': diagnostics['decoder_conv_in'], 'weights': str(WEIGHTS)}
        if result['status'] == 'FAIL': result['failure_analysis'] = failure_analysis()
    write_report('jittor_encoder_to_decoder_entry_alignment', 'TimeAware VAE Encoder To Decoder Entry Alignment', result)
    print(f"TIME_VAE_ENCODER_TO_DECODER_ENTRY_ALIGNMENT: {result['status']}")
    if result.get('diagnostics'): print(json.dumps(result['diagnostics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1
if __name__ == '__main__': raise SystemExit(main())
