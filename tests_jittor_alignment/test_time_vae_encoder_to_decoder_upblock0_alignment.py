#!/usr/bin/env python3
from __future__ import annotations
import json
from time_vae_decoder_upblock0_common import *


def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        result = blocked
    else:
        tester = TimeVAEBlockTester(WEIGHTS)
        got = tester.run_encoder_stage0123_mid_tail_quant_to_decoder_upblock0(oracle['real_input_image'], oracle['real_encoder_temb'], None, return_intermediates=True)
        diagnostics = {
            'moments': add_metrics(got['moments'], oracle['real_moments']),
            'posterior_mode': add_metrics(got['posterior_mode'], oracle['real_latent_dist_mode']),
            'decoder_conv_in': add_metrics(got['decoder_conv_in'], oracle['real_decoder_conv_in_output']),
            'decoder_midblock': add_metrics(got['decoder_midblock'], oracle['real_decoder_midblock_output']),
            'decoder_upblock0': add_metrics(got['output'], oracle['real_decoder_upblock0_output']),
        }
        result = {
            'status': status_from_diagnostics(diagnostics),
            'target': 'encoder -> quant_conv -> posterior mode -> decoder.mid_block -> decoder.up_blocks.0',
            'note': 'deterministic real bridge; encoder temb used; decoder temb is None; no sampling',
            'diagnostics': diagnostics,
            'metrics': diagnostics['decoder_upblock0'],
            'weights': str(WEIGHTS),
        }
        if result['status'] == 'FAIL':
            result['failure_analysis'] = failure_analysis()
    write_report('jittor_encoder_to_decoder_upblock0_alignment', 'TimeAware VAE Encoder To Decoder UpBlock0 Alignment', result)
    print(f"TIME_VAE_ENCODER_TO_DECODER_UPBLOCK0_ALIGNMENT: {result['status']}")
    if result.get('diagnostics'):
        print(json.dumps(result['diagnostics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
