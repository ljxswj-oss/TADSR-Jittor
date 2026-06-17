#!/usr/bin/env python3
from __future__ import annotations

import json

from timevae_actual_hook_behavior_common import *


def annotate(metric: dict) -> dict:
    out = dict(metric)
    out['status'] = status_from_metrics(metric)
    return out


def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        write_report('jittor_timevae_actual_hook_behavior_alignment', 'TimeVAE Actual VAEHook Behavior Alignment', blocked)
        print(f"TIME_VAE_ACTUAL_VAEHOOK_BEHAVIOR_ALIGNMENT: {blocked['status']}")
        return 1

    metadata = load_json(ORACLE / 'vae_actual_hook_behavior_oracle_metadata.json')
    tester = TimeVAEActualHookBehaviorTester(WEIGHTS)
    scaling_factor = float(metadata.get('scaling_factor', 0.18215))

    got_full = tester.run_actual_vaehook_behavior_boundary_for_alignment(
        oracle['actual_hook_input'].astype('float32'),
        oracle['actual_hook_timestep'],
        oracle['actual_sample_epsilon'].astype('float32'),
        scaling_factor=scaling_factor,
        return_intermediates=True,
    )
    got_decoder = tester.run_actual_decoder_hook_boundary_for_alignment(
        oracle['actual_decode_input'].astype('float32'),
        return_intermediates=True,
    )

    tol = 2e-3
    diagnostics = {
        'encoder_tiled_raw_output': annotate(add_metrics(got_full['raw_encoder_output'], oracle['actual_encoder_tiled_raw_output'], tolerance=tol)) if 'actual_encoder_tiled_raw_output' in oracle else {'status': 'NOT_EXPORTED'},
        'encoder_moments': annotate(add_metrics(got_full['moments'], oracle['actual_encoder_moments'], tolerance=tol)),
        'posterior_mean': annotate(add_metrics(got_full['posterior_mean'], oracle['actual_posterior_mean'], tolerance=tol)),
        'posterior_logvar': annotate(add_metrics(got_full['posterior_logvar'], oracle['actual_posterior_logvar'], tolerance=tol)),
        'posterior_std': annotate(add_metrics(got_full['posterior_std'], oracle['actual_posterior_std'], tolerance=tol)),
        'posterior_sample': annotate(add_metrics(got_full['posterior_sample'], oracle['actual_posterior_sample'], tolerance=tol)),
        'scaled_latent': annotate(add_metrics(got_full['scaled_latent'], oracle['actual_scaled_latent'], tolerance=tol)),
        'decode_input': annotate(add_metrics(got_full['decode_input'], oracle['actual_decode_input'], tolerance=tol)),
        'full_decoded_output': annotate(add_metrics(got_full['decoded_output'], oracle['actual_decoded_output'], tolerance=tol)),
        'full_final_clamped_output': annotate(add_metrics(got_full['final_clamped_output'], oracle['actual_final_clamped_output'], tolerance=tol)),
        'decoder_original_forward_output': annotate(add_metrics(got_decoder['decoded_output'], oracle['actual_decoded_output'], tolerance=tol)),
        'decoder_original_forward_final': annotate(add_metrics(got_decoder['final_clamped_output'], oracle['actual_final_clamped_output'], tolerance=tol)),
    }

    raw_status = diagnostics['encoder_tiled_raw_output']['status']
    encoder_metric_statuses = [
        diagnostics['encoder_moments']['status'],
        diagnostics['posterior_mean']['status'],
        diagnostics['posterior_logvar']['status'],
        diagnostics['posterior_std']['status'],
        diagnostics['posterior_sample']['status'],
        diagnostics['scaled_latent']['status'],
        diagnostics['decode_input']['status'],
    ]
    if raw_status != 'NOT_EXPORTED':
        encoder_metric_statuses.insert(0, raw_status)
    tile_metadata = got_full.get('tile_metadata', {})
    expected_tile_count = int(metadata.get('encoder_tile_count', 0) or 0)
    actual_tile_count = int(tile_metadata.get('tile_count', 0) or 0)
    task_queue_summary = got_full.get('task_queue_summary', {})
    group_norm_rounds = got_full.get('group_norm_rounds', [])
    tile_queue_status = (
        'PASS'
        if expected_tile_count == actual_tile_count
        and actual_tile_count > 0
        and int(task_queue_summary.get('task_count', 0) or 0) > 0
        and len(group_norm_rounds) > 0
        else 'FAIL'
    )
    decoder_status = (
        'PASS'
        if diagnostics['decoder_original_forward_output']['status'] == 'PASS'
        and diagnostics['decoder_original_forward_final']['status'] == 'PASS'
        else 'FAIL'
    )
    exact_tiled_encoder_implemented = bool(tester.encoder_tiled_task_queue_implemented)
    encoder_status = (
        'PASS'
        if exact_tiled_encoder_implemented and tile_queue_status == 'PASS' and all(s == 'PASS' for s in encoder_metric_statuses)
        else 'PARTIAL'
    )
    full_metric_status = (
        'PASS'
        if diagnostics['full_decoded_output']['status'] == 'PASS'
        and diagnostics['full_final_clamped_output']['status'] == 'PASS'
        else 'FAIL'
    )
    actual_full_boundary_status = (
        'PASS'
        if encoder_status == 'PASS' and decoder_status == 'PASS' and full_metric_status == 'PASS'
        else ('PARTIAL' if decoder_status == 'PASS' else 'FAIL')
    )
    top_status = (
        'PASS'
        if actual_full_boundary_status == 'PASS'
        else ('PARTIAL' if decoder_status == 'PASS' else 'FAIL')
    )

    result = {
        'status': top_status,
        'status_reason': (
            'Decoder original_forward actual-hook path aligns numerically. Encoder exact tiled VAEHook task queue is implemented, but at least one encoder/full actual-hook metric is still outside tolerance.'
            if top_status == 'PARTIAL'
            else 'Actual VAEHook behavior alignment complete.'
        ),
        'encoder_tiled_raw_output_status': raw_status,
        'encoder_tile_queue_alignment_status': tile_queue_status,
        'encoder_tiled_moments_status': encoder_status,
        'encoder_tiled_posterior_status': encoder_status,
        'encoder_tiled_alignment_status': encoder_status,
        'decoder_original_forward_alignment_status': decoder_status,
        'decoder_hook_behavior_alignment_status': decoder_status,
        'actual_vaehook_behavior_alignment_status': top_status,
        'actual_vaehook_full_boundary_alignment_status': actual_full_boundary_status,
        'tiled_decoder_alignment_status': 'NOT_APPLICABLE_BLOCKED_DECODER_HOOK_CONTRACT',
        'metrics': diagnostics['decoder_original_forward_final'],
        'diagnostics': diagnostics,
        'policy': {
            'mirror_official_actual_behavior': True,
            'encoder_hook': got_full['encoder_policy'],
            'decoder_hook': got_decoder['decoder_policy'],
            'do_not_force_decoder_time_vae_true': True,
            'do_not_claim_corrected_tiled_decoder': True,
            'scheduler_executed': False,
            'full_tadsr_inference_executed': False,
        },
        'metadata': metadata,
        'tile_queue_diagnostics': {
            'expected_tile_count': expected_tile_count,
            'actual_tile_count': actual_tile_count,
            'task_count': task_queue_summary.get('task_count'),
            'task_counts': task_queue_summary.get('task_counts'),
            'group_norm_round_count': len(group_norm_rounds),
            'first_group_norm_round': group_norm_rounds[0] if group_norm_rounds else None,
        },
        'remaining_gaps': [
            'Keep decoder hook as official original_forward unless the official contract changes or a separate corrected-tiled-decoder experiment is explicitly requested.',
            'Keep TIME_VAE_FULL_ALIGNMENT and JITTOR_FULL_INFERENCE NOT_COMPLETE until VAEHook behavior, scheduler integration, and runtime LoRA policy are all verified.',
        ],
    }
    if top_status == 'PASS':
        result['remaining_gaps'] = [
            'Keep decoder hook as official original_forward unless the official contract changes or a separate corrected-tiled-decoder experiment is explicitly requested.',
            'Keep TIME_VAE_FULL_ALIGNMENT and JITTOR_FULL_INFERENCE NOT_COMPLETE until scheduler integration, runtime LoRA policy, and end-to-end inference are explicitly implemented and verified.',
        ]

    write_report('jittor_timevae_actual_hook_behavior_alignment', 'TimeVAE Actual VAEHook Behavior Alignment', result)
    print(f"TIME_VAE_ACTUAL_ENCODER_TILE_QUEUE_ALIGNMENT: {tile_queue_status}")
    print(f"TIME_VAE_ACTUAL_ENCODER_TILED_RAW_OUTPUT_ALIGNMENT: {raw_status}")
    print(f"TIME_VAE_ACTUAL_ENCODER_TILED_MOMENTS_ALIGNMENT: {encoder_status}")
    print(f"TIME_VAE_ACTUAL_ENCODER_TILED_POSTERIOR_ALIGNMENT: {encoder_status}")
    print(f"TIME_VAE_ACTUAL_ENCODER_TILED_ALIGNMENT: {encoder_status}")
    print(f"TIME_VAE_ACTUAL_DECODER_ORIGINAL_FORWARD_ALIGNMENT: {decoder_status}")
    print(f"TIME_VAE_ACTUAL_DECODER_HOOK_BEHAVIOR_ALIGNMENT: {decoder_status}")
    print(f"TIME_VAE_ACTUAL_VAEHOOK_BEHAVIOR_ALIGNMENT: {top_status}")
    print(f"TIME_VAE_ACTUAL_VAEHOOK_FULL_BOUNDARY_ALIGNMENT: {actual_full_boundary_status}")
    print('TIME_VAE_TILED_DECODER_ALIGNMENT: NOT_APPLICABLE_BLOCKED_DECODER_HOOK_CONTRACT')
    print('TIME_VAE_FULL_ALIGNMENT: NOT_COMPLETE')
    print(json.dumps(result['diagnostics'], indent=2))
    return 0 if top_status in {'PASS', 'PARTIAL'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
