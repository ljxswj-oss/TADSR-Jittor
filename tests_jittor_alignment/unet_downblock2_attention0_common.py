#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jittor_tadsr_full.unet_2d_condition import UNetAttention0Transformer2DTester, TADSRUNetDownBlock2Attention0Tester
from jittor_tadsr_full.utils import compare_arrays, write_json

OUT = Path('experiments/full_repro/unet_alignment')
ORACLE = OUT / 'oracle_tensors_unet_downblock2_attention0'
META = ORACLE / 'unet_downblock2_attention0_oracle_metadata.json'
DOWN0_RESNET0_META = OUT / 'oracle_tensors_unet_downblock0_resnet0' / 'unet_downblock0_resnet0_oracle_metadata.json'
DOWN0_ATTENTION0_META = OUT / 'oracle_tensors_unet_downblock0_attention0' / 'unet_downblock0_attention0_oracle_metadata.json'
DOWN0_RESNET1_META = OUT / 'oracle_tensors_unet_downblock0_resnet1' / 'unet_downblock0_resnet1_oracle_metadata.json'
DOWN0_ATTENTION1_META = OUT / 'oracle_tensors_unet_downblock0_attention1' / 'unet_downblock0_attention1_oracle_metadata.json'
DOWN0_META = OUT / 'oracle_tensors_unet_downblock0_downsampler' / 'unet_downblock0_downsampler_oracle_metadata.json'
DOWN1_RESNET0_META = OUT / 'oracle_tensors_unet_downblock1_resnet0' / 'unet_downblock1_resnet0_oracle_metadata.json'
DOWN1_ATTENTION0_META = OUT / 'oracle_tensors_unet_downblock1_attention0' / 'unet_downblock1_attention0_oracle_metadata.json'
DOWN1_RESNET1_META = OUT / 'oracle_tensors_unet_downblock1_resnet1' / 'unet_downblock1_resnet1_oracle_metadata.json'
DOWN1_ATTENTION1_META = OUT / 'oracle_tensors_unet_downblock1_attention1' / 'unet_downblock1_attention1_oracle_metadata.json'
DOWN1_DOWNSAMPLER_META = OUT / 'oracle_tensors_unet_downblock1_downsampler' / 'unet_downblock1_downsampler_oracle_metadata.json'
DOWN2_RESNET0_META = OUT / 'oracle_tensors_unet_downblock2_resnet0' / 'unet_downblock2_resnet0_oracle_metadata.json'
ENTRY_WEIGHTS = OUT / 'converted_unet_entry_effective_weights.npz'
DOWN0_RESNET0_WEIGHTS = OUT / 'converted_unet_downblock0_resnet0_effective_weights.npz'
DOWN0_ATTENTION0_WEIGHTS = OUT / 'converted_unet_downblock0_attention0_effective_weights.npz'
DOWN0_RESNET1_WEIGHTS = OUT / 'converted_unet_downblock0_resnet1_effective_weights.npz'
DOWN0_ATTENTION1_WEIGHTS = OUT / 'converted_unet_downblock0_attention1_effective_weights.npz'
DOWN0_DOWNSAMPLER_WEIGHTS = OUT / 'converted_unet_downblock0_downsampler_effective_weights.npz'
DOWN1_RESNET0_WEIGHTS = OUT / 'converted_unet_downblock1_resnet0_effective_weights.npz'
DOWN1_ATTENTION0_WEIGHTS = OUT / 'converted_unet_downblock1_attention0_effective_weights.npz'
DOWN1_RESNET1_WEIGHTS = OUT / 'converted_unet_downblock1_resnet1_effective_weights.npz'
DOWN1_ATTENTION1_WEIGHTS = OUT / 'converted_unet_downblock1_attention1_effective_weights.npz'
DOWN1_DOWNSAMPLER_WEIGHTS = OUT / 'converted_unet_downblock1_downsampler_effective_weights.npz'
DOWN2_RESNET0_WEIGHTS = OUT / 'converted_unet_downblock2_resnet0_effective_weights.npz'
DOWN2_ATTENTION0_WEIGHTS = OUT / 'converted_unet_downblock2_attention0_effective_weights.npz'


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def load_oracle():
    required = [ORACLE, META, ENTRY_WEIGHTS, DOWN0_RESNET0_WEIGHTS, DOWN0_ATTENTION0_WEIGHTS, DOWN0_RESNET1_WEIGHTS, DOWN0_ATTENTION1_WEIGHTS, DOWN0_DOWNSAMPLER_WEIGHTS, DOWN1_RESNET0_WEIGHTS, DOWN1_ATTENTION0_WEIGHTS, DOWN1_RESNET1_WEIGHTS, DOWN1_ATTENTION1_WEIGHTS, DOWN1_DOWNSAMPLER_WEIGHTS, DOWN2_RESNET0_WEIGHTS, DOWN2_ATTENTION0_WEIGHTS]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        return None, {'status': 'BLOCKED', 'reason': 'Run python3 tools/export_tadsr_unet_downblock2_attention0_oracle.py first.', 'missing': missing}
    tensors = {p.stem: np.load(p).astype(np.float32) for p in ORACLE.glob('*.npy')}
    if not tensors:
        return None, {'status': 'BLOCKED', 'reason': 'Run python3 tools/export_tadsr_unet_downblock2_attention0_oracle.py first.'}
    return tensors, None


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 1.0


def add_metrics(got, expected, tolerance=1e-4):
    cmp = compare_arrays(got, expected)
    cmp['cosine_similarity'] = cosine(got, expected) if cmp.get('shape_match') else None
    cmp['tolerance'] = tolerance
    return cmp


def status_from_metrics(metrics):
    tol = float(metrics.get('tolerance', 1e-4))
    if metrics.get('shape_match') and metrics.get('max_abs_error', 1.0) <= tol:
        return 'PASS'
    if metrics.get('shape_match') and metrics.get('max_abs_error', 1.0) <= max(tol * 10.0, 2e-3) and (metrics.get('cosine_similarity') is None or metrics.get('cosine_similarity') > 0.99999):
        return 'PARTIAL'
    return 'FAIL'


def status_from_diagnostics(diag):
    statuses = [status_from_metrics(m) for m in diag.values()]
    if statuses and all(s == 'PASS' for s in statuses):
        return 'PASS'
    if statuses and all(s in {'PASS', 'PARTIAL'} for s in statuses):
        return 'PARTIAL'
    return 'FAIL'


def write_report(name, title, result):
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / f'{name}.json', result)
    md = [f'# {title}', '', f"Status: **{result['status']}**"]
    if result.get('diagnostics'):
        md += ['', '## Diagnostics', '', '| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |', '|---|---:|---:|---:|---:|---:|']
        for stage, m in result['diagnostics'].items():
            md.append(f"| {stage} | `{m.get('max_abs_error')}` | `{m.get('mean_abs_error')}` | `{m.get('relative_error')}` | `{m.get('cosine_similarity')}` | `{m.get('tolerance')}` |")
    if result.get('note'):
        md += ['', f"Note: {result['note']}"]
    if result.get('reason'):
        md += ['', f"Reason: {result['reason']}"]
    (OUT / f'{name}.md').write_text('\n'.join(md) + '\n', encoding='utf-8')


def blocked_result(name, title, blocked):
    write_report(name, title, blocked)
    return blocked


def attention0_tester():
    return UNetAttention0Transformer2DTester(DOWN2_ATTENTION0_WEIGHTS, metadata=load_json(META))


def bridge_tester():
    return TADSRUNetDownBlock2Attention0Tester(
        ENTRY_WEIGHTS,
        DOWN0_RESNET0_WEIGHTS,
        DOWN0_ATTENTION0_WEIGHTS,
        DOWN0_RESNET1_WEIGHTS,
        DOWN0_ATTENTION1_WEIGHTS,
        DOWN0_DOWNSAMPLER_WEIGHTS,
        DOWN1_RESNET0_WEIGHTS,
        DOWN1_ATTENTION0_WEIGHTS,
        DOWN1_RESNET1_WEIGHTS,
        DOWN1_ATTENTION1_WEIGHTS,
        DOWN1_DOWNSAMPLER_WEIGHTS,
        DOWN2_RESNET0_WEIGHTS,
        DOWN2_ATTENTION0_WEIGHTS,
        metadata={
            'downblock1_metadata': {
                'downblock0_metadata': {
                    'resnet0_metadata': load_json(DOWN0_RESNET0_META),
                    'attention0_metadata': load_json(DOWN0_ATTENTION0_META),
                    'resnet1_metadata': load_json(DOWN0_RESNET1_META),
                    'attention1_metadata': load_json(DOWN0_ATTENTION1_META),
                    'downsampler_metadata': load_json(DOWN0_META),
                },
                'resnet0_metadata': load_json(DOWN1_RESNET0_META),
                'attention0_metadata': load_json(DOWN1_ATTENTION0_META),
                'resnet1_metadata': load_json(DOWN1_RESNET1_META),
                'attention1_metadata': load_json(DOWN1_ATTENTION1_META),
                'downsampler_metadata': load_json(DOWN1_DOWNSAMPLER_META),
            },
            'resnet0_metadata': load_json(DOWN2_RESNET0_META),
            'attention0_metadata': load_json(META),
        },
    )
