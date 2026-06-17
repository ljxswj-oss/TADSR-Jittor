#!/usr/bin/env python3
from __future__ import annotations
import json
from time_vae_decoder_upblock1_common import *


def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        result = blocked
        write_report('jittor_encoder_to_decoder_upblock1_alignment', 'TimeAware VAE Encoder To Decoder UpBlock1 Alignment', result)
        print(f"TIME_VAE_ENCODER_TO_DECODER_UPBLOCK1_ALIGNMENT: {result['status']}")
        return 1

    tester = TimeVAEBlockTester(WEIGHTS)
    got = tester.run_encoder_stage0123_mid_tail_quant_to_decoder_upblock1(
        oracle['real_input_image'],
        oracle['real_encoder_temb'],
        return_intermediates=True,
    )
    diagnostics = {
        'moments': add_metrics(got['moments'], oracle['real_moments']),
        'posterior_mode': add_metrics(got['posterior_mode'], oracle['real_latent_dist_mode']),
        'decoder_conv_in': add_metrics(got['decoder_conv_in'], oracle['real_decoder_conv_in_output']),
        'decoder_midblock': add_metrics(got['decoder_midblock'], oracle['real_decoder_midblock_output']),
        'decoder_upblock0': add_metrics(got['decoder_upblock0'], oracle['real_decoder_upblock0_output']),
        'decoder_upblock1': add_metrics(got['decoder_upblock1'], oracle['real_decoder_upblock1_output']),
    }
    result = {
        'status': status_from_diagnostics(diagnostics),
        'target': 'encoder image path -> quant_conv -> decoder up_blocks.1',
        'note': 'deterministic full bridge through encoder and partial decoder, ending at decoder.up_blocks.1',
        'diagnostics': diagnostics,
        'metrics': diagnostics['decoder_upblock1'],
        'weights': str(WEIGHTS),
    }
    if result['status'] == 'FAIL':
        result['failure_analysis'] = failure_analysis()
    write_report('jittor_encoder_to_decoder_upblock1_alignment', 'TimeAware VAE Encoder To Decoder UpBlock1 Alignment', result)
    print(f"TIME_VAE_ENCODER_TO_DECODER_UPBLOCK1_ALIGNMENT: {result['status']}")
    print(json.dumps(result['diagnostics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
