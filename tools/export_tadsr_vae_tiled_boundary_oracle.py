#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

OFFICIAL_VENV_PYTHON = Path('/mnt/data/sj/venvs/tadsr_official_strict_cu118/bin/python')
OFFICIAL_VENV_ROOT = OFFICIAL_VENV_PYTHON.parents[1]
if (
    OFFICIAL_VENV_PYTHON.exists()
    and Path(sys.prefix).resolve() != OFFICIAL_VENV_ROOT.resolve()
    and os.environ.get('TADSR_OFFICIAL_VENV_REEXEC') != '1'
):
    env = os.environ.copy()
    env['TADSR_OFFICIAL_VENV_REEXEC'] = '1'
    os.execve(str(OFFICIAL_VENV_PYTHON), [str(OFFICIAL_VENV_PYTHON), *sys.argv], env)

import numpy as np
import torch

BASE_OUT = Path('experiments/full_repro/vae_alignment')
AUDIT_PATH = BASE_OUT / 'audit_tadsr_vae_tiled_boundary.json'
ORACLE_OUT = BASE_OUT / 'oracle_tensors_vae_tiled_boundary'


def write_blocked(audit: dict, status: str, reason: str) -> int:
    ORACLE_OUT.mkdir(parents=True, exist_ok=True)
    metadata = {
        'status': status,
        'reason': reason,
        'source_audit': str(AUDIT_PATH),
        'VAEHook_class': audit.get('source_file_paths', {}).get('my_utils/vaehook.py'),
        'patch_targets': audit.get('hook_audit', {}).get('patch_targets', {}),
        'tile_params': audit.get('hook_audit', {}).get('hook_init_params', {}),
        'input_shape': None,
        'timestep_shape': None,
        'latent_shape': None,
        'output_shape': None,
        'scaling_factor': 0.18215,
        'deterministic_policy': 'not exported because official tiled encode/decode oracle feasibility is blocked',
        'fixed_epsilon_policy': 'not applicable',
        'tiled_path_triggered': False,
        'encode_tiled_path_triggered': False,
        'decode_tiled_path_triggered': False,
        'nontiled_reference_available': False,
        'scheduler_executed': False,
        'full_tadsr_inference_executed': False,
        'image_saved': False,
        'jittor_tiled_implementation_done': False,
        'timevae_full_alignment_candidate': False,
        'recommended_next_stage': audit.get('oracle_feasibility', {}).get('recommended_next_stage', 'resolve tiled oracle blocker before Jittor tiled implementation'),
    }
    (ORACLE_OUT / 'vae_tiled_boundary_oracle_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    summary = [
        f'TIME_VAE_TILED_ORACLE_TENSORS: {status}',
        '',
        f'Reason: {reason}',
        'No scheduler denoising loop was run.',
        'No full TADSR inference was run.',
        'No image was saved.',
        'No Jittor tiled implementation was added.',
    ]
    (ORACLE_OUT / 'oracle_summary.txt').write_text('\n'.join(summary) + '\n', encoding='utf-8')
    print(summary[0])
    print(json.dumps(metadata, indent=2))
    return 0


def main() -> int:
    if not AUDIT_PATH.exists():
        return write_blocked({}, 'BLOCKED_MISSING_AUDIT', f'Missing {AUDIT_PATH}; run tools/audit_official_tadsr_vae_tiled_boundary.py first')
    audit = json.loads(AUDIT_PATH.read_text(encoding='utf-8'))
    feasibility = audit.get('oracle_feasibility', {})
    feasibility_status = str(feasibility.get('status', 'BLOCKED_UNKNOWN'))
    if feasibility_status != 'PASS':
        return write_blocked(audit, feasibility_status, str(feasibility.get('reason', 'VAE tiled oracle feasibility is not PASS')))

    # This branch is intentionally conservative. The current official audit found
    # that decoder tiled execution is not reachable under official TADSR_test
    # VAEHook parameters. If a future audit changes feasibility to PASS, extend
    # this branch to export tensors; do not silently fake a tiled decode oracle.
    torch.manual_seed(123)
    np.random.seed(123)
    return write_blocked(
        audit,
        'BLOCKED_EXPORT_NOT_IMPLEMENTED_FOR_CHANGED_CONTRACT',
        'Audit reported PASS, but this exporter has not been extended for the changed official tiled encode/decode contract.',
    )


if __name__ == '__main__':
    raise SystemExit(main())
