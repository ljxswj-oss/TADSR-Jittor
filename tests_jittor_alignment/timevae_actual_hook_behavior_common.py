#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jittor_tadsr_full.vae_time_aware import TimeVAEActualHookBehaviorTester
from jittor_tadsr_full.utils import compare_arrays, write_json

OUT = Path('experiments/full_repro/vae_alignment')
ORACLE = OUT / 'oracle_tensors_vae_actual_hook_behavior'
WEIGHTS = Path('/mnt/data/sj/checkpoints/TADSR/jittor_converted/diffusers/time_vae_weights.npz')


def load_json(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 1.0


def add_metrics(got, expected, tolerance=2e-3):
    cmp = compare_arrays(got, expected)
    cmp['cosine_similarity'] = cosine(got, expected) if cmp.get('shape_match') else None
    cmp['tolerance'] = tolerance
    return cmp


def status_from_metrics(metrics):
    tolerance = float(metrics.get('tolerance', 2e-3))
    if metrics.get('shape_match') and metrics.get('max_abs_error', 1.0) < tolerance:
        return 'PASS'
    if metrics.get('shape_match') and metrics.get('max_abs_error', 1.0) < max(1e-3, tolerance * 10.0):
        return 'PARTIAL'
    return 'FAIL'


def load_oracle():
    if not ORACLE.exists():
        return None, {'status': 'BLOCKED', 'reason': 'actual VAEHook behavior oracle tensors missing'}
    arrays = {}
    for path in sorted(ORACLE.glob('*.npy')):
        arrays[path.stem] = np.load(path)
    required = [
        'actual_hook_input',
        'actual_hook_timestep',
        'actual_encoder_moments',
        'actual_posterior_mean',
        'actual_posterior_logvar',
        'actual_posterior_std',
        'actual_sample_epsilon',
        'actual_posterior_sample',
        'actual_scaled_latent',
        'actual_decode_input',
        'actual_decoded_output',
        'actual_final_clamped_output',
    ]
    missing = [name for name in required if name not in arrays]
    if missing:
        return None, {'status': 'BLOCKED', 'reason': f'Missing oracle tensors: {missing}'}
    return arrays, None


def write_report(name: str, title: str, result: dict):
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / f'{name}.json', result)
    md = [f'# {title}', '', f"Status: **{result['status']}**"]
    if result.get('status_reason'):
        md += ['', f"Reason: {result['status_reason']}"]
    if result.get('metrics'):
        md += ['', '## Primary Metrics', '', '| Metric | Value |', '|---|---:|']
        for k, v in result['metrics'].items():
            md.append(f'| {k} | `{v}` |')
    if result.get('diagnostics'):
        md += ['', '## Diagnostics', '', '| Stage | Status | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |', '|---|---|---:|---:|---:|---:|---:|']
        for stage, m in result['diagnostics'].items():
            md.append(f"| {stage} | `{m.get('status')}` | `{m.get('max_abs_error')}` | `{m.get('mean_abs_error')}` | `{m.get('relative_error')}` | `{m.get('cosine_similarity')}` | `{m.get('tolerance')}` |")
    if result.get('policy'):
        md += ['', '## Policy', '', '```json', json.dumps(result['policy'], indent=2, ensure_ascii=False), '```']
    if result.get('remaining_gaps'):
        md += ['', '## Remaining gaps']
        md.extend(f"- {item}" for item in result['remaining_gaps'])
    (OUT / f'{name}.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
