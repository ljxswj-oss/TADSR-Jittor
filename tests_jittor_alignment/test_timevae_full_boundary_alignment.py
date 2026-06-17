#!/usr/bin/env python3
from __future__ import annotations

import json

from timevae_full_boundary_common import *


def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        write_report('jittor_timevae_full_boundary_alignment', 'TimeVAE Full Boundary Alignment', blocked)
        print(f"TIME_VAE_FULL_BOUNDARY_ALIGNMENT: {blocked['status']}")
        return 1

    metadata = load_json(ORACLE / 'timevae_full_boundary_oracle_metadata.json')
    tester = TimeVAEFullBoundaryTester(WEIGHTS)
    scaling_factor = float(metadata.get('scaling_factor', 0.18215))
    got = tester.run_encode_sample_scale_decode_clamp_boundary(
        oracle['timevae_full_boundary_input'].astype('float32'),
        oracle['timevae_full_boundary_encoder_temb'].astype('float32'),
        oracle['timevae_full_boundary_sample_epsilon'].astype('float32'),
        scaling_factor=scaling_factor,
        return_intermediates=True,
    )
    tolerance = 2e-3
    diagnostics = {
        'encode_moments': add_metrics(got['moments'], oracle['timevae_full_boundary_moments'], tolerance=tolerance),
        'posterior_mean': add_metrics(got['posterior_mean'], oracle['timevae_full_boundary_posterior_mean'], tolerance=tolerance),
        'posterior_logvar': add_metrics(got['posterior_logvar'], oracle['timevae_full_boundary_posterior_logvar'], tolerance=tolerance),
        'posterior_std': add_metrics(got['posterior_std'], oracle['timevae_full_boundary_posterior_std'], tolerance=tolerance),
        'posterior_sample': add_metrics(got['posterior_sample'], oracle['timevae_full_boundary_posterior_sample'], tolerance=tolerance),
        'scaled_latent': add_metrics(got['scaled_latent'], oracle['timevae_full_boundary_scaled_latent'], tolerance=tolerance),
        'decode_input': add_metrics(got['decode_input'], oracle['timevae_full_boundary_decode_input'], tolerance=tolerance),
        'decoded_output': add_metrics(got['decoded_output'], oracle['timevae_full_boundary_decoded_output'], tolerance=tolerance),
        'final_clamped_output': add_metrics(got['final_clamped_output'], oracle['timevae_full_boundary_final_clamped_output'], tolerance=tolerance),
    }
    status = status_from_diagnostics(diagnostics)
    decode_status = status_from_metrics(diagnostics['decoded_output'])
    encode_status = status_from_metrics(diagnostics['encode_moments'])
    post_status = status_from_metrics(diagnostics['final_clamped_output'])
    result = {
        'status': status,
        'decode_status': decode_status,
        'encode_status': encode_status,
        'postprocess_status': post_status,
        'target': 'TimeAwareAutoencoderKL encode -> fixed-epsilon posterior sample -> scaling -> decode -> clamp',
        'note': 'alignment-only non-tiled deterministic full-boundary check; no scheduler, no image save, no full TADSR inference',
        'diagnostics': diagnostics,
        'metrics': diagnostics['final_clamped_output'],
        'weights': str(WEIGHTS),
        'metadata': metadata,
        'remaining_timevae_gaps': metadata.get('remaining_timevae_gaps', []),
        'timevae_full_alignment_candidate': bool(metadata.get('timevae_full_alignment_candidate', False)),
    }
    write_report('jittor_timevae_full_boundary_alignment', 'TimeVAE Full Boundary Alignment', result)
    print(f"TIME_VAE_FULL_BOUNDARY_ENCODE_ALIGNMENT: {encode_status}")
    print(f"TIME_VAE_FULL_BOUNDARY_DECODE_ALIGNMENT: {decode_status}")
    print(f"TIME_VAE_FULL_BOUNDARY_POSTPROCESS_ALIGNMENT: {post_status}")
    print(f"TIME_VAE_FULL_BOUNDARY_ALIGNMENT: {status}")
    print('TIME_VAE_FULL_ALIGNMENT: NOT_COMPLETE')
    print(json.dumps(result['diagnostics'], indent=2))
    return 0 if status in {'PASS', 'PARTIAL'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
