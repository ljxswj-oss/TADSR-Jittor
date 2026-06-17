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
        got = tester.run_posterior_mode(oracle['real_moments'])
        result = result_from_metric('DiagonalGaussianDistribution.mode() deterministic mean', got, oracle['real_latent_dist_mode'], 'mode returns posterior mean; no sampling')
        if result['status'] == 'FAIL': result['failure_analysis'] = failure_analysis()
    write_report('jittor_posterior_mode_alignment', 'TimeAware VAE Posterior Mode Alignment', result)
    print(f"TIME_VAE_POSTERIOR_MODE_ALIGNMENT: {result['status']}")
    if result.get('metrics'): print(json.dumps(result['metrics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1
if __name__ == '__main__': raise SystemExit(main())
