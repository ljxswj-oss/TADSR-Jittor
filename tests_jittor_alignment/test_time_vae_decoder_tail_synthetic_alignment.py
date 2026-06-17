#!/usr/bin/env python3
from __future__ import annotations
import json
from time_vae_decoder_tail_common import *


def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        write_report('jittor_decoder_tail_synthetic_alignment', 'TimeAware VAE Decoder Tail Synthetic Alignment', blocked)
        print(f"TIME_VAE_DECODER_TAIL_SYNTHETIC_ALIGNMENT: {blocked['status']}")
        return 1
    tester = TimeVAEBlockTester(WEIGHTS)
    got = tester.run_decoder_tail(oracle['synthetic_decoder_tail_input'])
    result = write_metric_report(
        'jittor_decoder_tail_synthetic_alignment',
        'TimeAware VAE Decoder Tail Synthetic Alignment',
        'synthetic hidden -> decoder tail',
        got,
        oracle['synthetic_decoder_tail_output'],
        note='pressure test for isolated deterministic linspace hidden through decoder tail',
        tolerance=1e-4,
    )
    print(f"TIME_VAE_DECODER_TAIL_SYNTHETIC_ALIGNMENT: {result['status']}")
    print(json.dumps(result.get('metrics', {}), indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
