#!/usr/bin/env python3
from __future__ import annotations
import json
from time_vae_decoder_tail_common import *


def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        write_report('jittor_quant_to_decoder_tail_alignment', 'TimeAware VAE Quant To Decoder Tail Alignment', blocked)
        print(f"TIME_VAE_QUANT_TO_DECODER_TAIL_ALIGNMENT: {blocked['status']}")
        return 1
    tester = TimeVAEBlockTester(WEIGHTS)
    got = tester.run_quant_to_decoder_tail(oracle['real_moments'], return_intermediates=True)
    tolerance = 2e-3
    diagnostics = {
        'posterior_mode': add_metrics(got['posterior_mode'], oracle['real_latent_dist_mode'], tolerance=tolerance),
        'post_quant_conv': add_metrics(got['post_quant_conv'], oracle['real_post_quant_conv_output'], tolerance=tolerance),
        'decoder_conv_in': add_metrics(got['decoder_conv_in'], oracle['real_decoder_conv_in_output'], tolerance=tolerance),
        'decoder_midblock': add_metrics(got['decoder_midblock'], oracle['real_decoder_midblock_output'], tolerance=tolerance),
        'decoder_upblock0': add_metrics(got['decoder_upblock0'], oracle['real_decoder_upblock0_output'], tolerance=tolerance),
        'decoder_upblock1': add_metrics(got['decoder_upblock1'], oracle['real_decoder_upblock1_output'], tolerance=tolerance),
        'decoder_upblock2': add_metrics(got['decoder_upblock2'], oracle['real_decoder_upblock2_output'], tolerance=tolerance),
        'decoder_upblock3': add_metrics(got['decoder_upblock3'], oracle['real_decoder_upblock3_output'], tolerance=tolerance),
        'decoder_tail_norm_out': add_metrics(got['decoder_tail_norm_out'], oracle['real_decoder_tail_norm_out_output'], tolerance=tolerance),
        'decoder_tail_act': add_metrics(got['decoder_tail_act'], oracle['real_decoder_tail_act_output'], tolerance=tolerance),
        'decoder_tail_conv_out': add_metrics(got['decoder_tail_conv_out'], oracle['real_decoder_tail_conv_out_output'], tolerance=tolerance),
        'decoder_tail': add_metrics(got['decoder_tail'], oracle['real_decoder_tail_output'], tolerance=tolerance),
    }
    result = {
        'status': status_from_diagnostics(diagnostics),
        'target': 'moments -> posterior mode -> deterministic decoder stack -> tail',
        'note': 'deterministic quant bridge uses posterior mode, not sampling; no scaling_factor/image postprocess applied',
        'diagnostics': diagnostics,
        'metrics': diagnostics['decoder_tail'],
        'weights': str(WEIGHTS),
    }
    if result['status'] == 'FAIL':
        result['failure_analysis'] = failure_analysis()
    write_report('jittor_quant_to_decoder_tail_alignment', 'TimeAware VAE Quant To Decoder Tail Alignment', result)
    print(f"TIME_VAE_QUANT_TO_DECODER_TAIL_ALIGNMENT: {result['status']}")
    print(json.dumps(result['diagnostics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
