#!/usr/bin/env python3
from __future__ import annotations
import json
from time_vae_decoder_tail_common import *


def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        write_report('jittor_decoder_tail_alignment', 'TimeAware VAE Decoder Tail Alignment', blocked)
        print(f"TIME_VAE_DECODER_TAIL_ALIGNMENT: {blocked['status']}")
        return 1

    tester = TimeVAEBlockTester(WEIGHTS)
    x = oracle['synthetic_decoder_tail_input']
    norm = tester.run_decoder_tail_norm_out(x)
    act = tester.run_decoder_tail_act(norm)
    conv = tester.run_decoder_tail_conv_out(act)
    full = tester.run_decoder_tail(x)
    diagnostics = {
        'norm_out': add_metrics(norm, oracle['synthetic_decoder_tail_norm_out_output'], tolerance=1e-4),
        'act': add_metrics(act, oracle['synthetic_decoder_tail_act_output'], tolerance=1e-4),
        'conv_out': add_metrics(conv, oracle['synthetic_decoder_tail_conv_out_output'], tolerance=1e-4),
        'tail': add_metrics(full, oracle['synthetic_decoder_tail_output'], tolerance=1e-4),
    }
    for stage, key in [('norm_out', 'TIME_VAE_DECODER_TAIL_NORM_ALIGNMENT'), ('act', 'TIME_VAE_DECODER_TAIL_ACT_ALIGNMENT'), ('conv_out', 'TIME_VAE_DECODER_TAIL_CONV_OUT_ALIGNMENT')]:
        result = {
            'status': status_from_metrics(diagnostics[stage]),
            'target': f'isolated decoder tail {stage}',
            'note': 'isolated synthetic hidden -> decoder tail leaf alignment; no scaling_factor or image postprocess is applied',
            'metrics': diagnostics[stage],
            'weights': str(WEIGHTS),
        }
        if result['status'] == 'FAIL':
            result['failure_analysis'] = failure_analysis()
        report_name = {'norm_out': 'jittor_decoder_tail_norm_alignment', 'act': 'jittor_decoder_tail_act_alignment', 'conv_out': 'jittor_decoder_tail_conv_out_alignment'}[stage]
        title = {'norm_out': 'TimeAware VAE Decoder Tail Norm Alignment', 'act': 'TimeAware VAE Decoder Tail Activation Alignment', 'conv_out': 'TimeAware VAE Decoder Tail ConvOut Alignment'}[stage]
        write_report(report_name, title, result)
        print(f"{key}: {result['status']}")
    result = {
        'status': status_from_diagnostics(diagnostics),
        'target': 'isolated decoder tail: conv_norm_out -> conv_act -> conv_out',
        'note': 'isolated synthetic hidden -> full decoder tail alignment; no scaling_factor, tanh, clamp, or pipeline postprocess is applied',
        'diagnostics': diagnostics,
        'metrics': diagnostics['tail'],
        'weights': str(WEIGHTS),
    }
    if result['status'] == 'FAIL':
        result['failure_analysis'] = failure_analysis()
    write_report('jittor_decoder_tail_alignment', 'TimeAware VAE Decoder Tail Alignment', result)
    print(f"TIME_VAE_DECODER_TAIL_ALIGNMENT: {result['status']}")
    print(json.dumps(result['diagnostics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
