#!/usr/bin/env python3
from __future__ import annotations
import json
from time_vae_decoder_upblock2_common import *


def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        result = blocked
        write_report('jittor_quant_to_decoder_upblock2_alignment', 'TimeAware VAE Quant To Decoder UpBlock2 Alignment', result)
        print(f"TIME_VAE_QUANT_TO_DECODER_UPBLOCK2_ALIGNMENT: {result['status']}")
        return 1

    tester = TimeVAEBlockTester(WEIGHTS)
    got = tester.run_quant_to_decoder_upblock2(oracle['real_moments'], return_intermediates=True)
    diagnostics = {
        'posterior_mode': add_metrics(got['posterior_mode'], oracle['real_latent_dist_mode']),
        'post_quant_conv': add_metrics(got['post_quant_conv'], oracle['real_post_quant_conv_output']),
        'decoder_conv_in': add_metrics(got['decoder_conv_in'], oracle['real_decoder_conv_in_output']),
        'decoder_midblock': add_metrics(got['decoder_midblock'], oracle['real_decoder_midblock_output']),
        'decoder_upblock0': add_metrics(got['decoder_upblock0'], oracle['real_decoder_upblock0_output']),
        'decoder_upblock1': add_metrics(got['decoder_upblock1'], oracle['real_decoder_upblock1_output']),
        'decoder_upblock2': add_metrics(got['decoder_upblock2'], oracle['real_decoder_upblock2_output']),
    }
    result = {
        'status': status_from_diagnostics(diagnostics),
        'target': 'moments -> posterior mode -> decoder up_blocks.2',
        'note': 'deterministic quant bridge uses posterior mode, not sampling',
        'diagnostics': diagnostics,
        'metrics': diagnostics['decoder_upblock2'],
        'weights': str(WEIGHTS),
    }
    if result['status'] == 'FAIL':
        result['failure_analysis'] = failure_analysis()
    write_report('jittor_quant_to_decoder_upblock2_alignment', 'TimeAware VAE Quant To Decoder UpBlock2 Alignment', result)
    print(f"TIME_VAE_QUANT_TO_DECODER_UPBLOCK2_ALIGNMENT: {result['status']}")
    print(json.dumps(result['diagnostics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
