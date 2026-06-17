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
        mean, raw_logvar, clamped_logvar = tester.run_moments_split(oracle['real_moments'])
        diagnostics = {
            'mean': add_metrics(mean, oracle['real_mean_manual']),
            'raw_logvar': add_metrics(raw_logvar, oracle['real_logvar_raw_manual']),
            'clamped_logvar': add_metrics(clamped_logvar, oracle['real_logvar_clamped_manual']),
        }
        worst = max(m.get('max_abs_error', 1.0) for m in diagnostics.values())
        result = {'status': 'PASS' if worst <= 1e-4 else ('PARTIAL' if worst <= 1e-3 else 'FAIL'), 'target': 'DiagonalGaussianDistribution moments split + logvar clamp', 'note': 'split dim=1; no sampling', 'diagnostics': diagnostics, 'metrics': diagnostics['mean'], 'weights': str(WEIGHTS)}
        if result['status'] == 'FAIL': result['failure_analysis'] = failure_analysis()
    write_report('jittor_moments_split_alignment', 'TimeAware VAE Moments Split Alignment', result)
    print(f"TIME_VAE_MOMENTS_SPLIT_ALIGNMENT: {result['status']}")
    if result.get('diagnostics'): print(json.dumps(result['diagnostics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1
if __name__ == '__main__': raise SystemExit(main())
