#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jittor_tadsr_full.vae_time_aware import TimeVAEDownBlockTester, TimeVAEBlockTester
from jittor_tadsr_full.utils import compare_arrays, write_json

OUT = Path('experiments/full_repro/time_vae_alignment')
ORACLE = OUT / 'oracle_tensors_downblock3'
WEIGHTS = Path('/mnt/data/sj/checkpoints/TADSR/jittor_converted/diffusers/time_vae_weights.npz')
BLOCK_INDEX = 3


def load_json(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_audit():
    return load_json(OUT / 'downblock3_audit.json')


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 1.0


def add_metrics(got, expected, tolerance):
    cmp = compare_arrays(got, expected)
    cmp['cosine_similarity'] = cosine(got, expected) if cmp.get('shape_match') else None
    cmp['tolerance'] = tolerance
    return cmp


def status_from_metrics(metrics):
    if metrics.get('shape_match') and metrics.get('max_abs_error', 1.0) <= 1e-4:
        return 'PASS'
    if metrics.get('shape_match') and metrics.get('max_abs_error', 1.0) <= 1e-3:
        return 'PARTIAL'
    return 'FAIL'


def write_report(name, title, result):
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / f'{name}.json', result)
    md = [f'# {title}', '', f"Status: **{result['status']}**"]
    if result.get('metrics'):
        md += ['', '## Metrics', '', '| Metric | Value |', '|---|---:|']
        for k, v in result['metrics'].items():
            md.append(f'| {k} | `{v}` |')
    if result.get('diagnostics'):
        md += ['', '## Layer-by-layer diagnostics', '', '| Stage | Max abs error | Mean abs error | Relative error | Cosine |', '|---|---:|---:|---:|---:|']
        for stage, m in result['diagnostics'].items():
            md.append(f"| {stage} | `{m.get('max_abs_error')}` | `{m.get('mean_abs_error')}` | `{m.get('relative_error')}` | `{m.get('cosine_similarity')}` |")
    if result.get('note'):
        md += ['', f"Note: {result['note']}"]
    if result.get('failure_analysis'):
        md += ['', '## Failure analysis']
        for item in result['failure_analysis']:
            md.append(f'- {item}')
    if result.get('reason'):
        md += ['', f"Reason: {result['reason']}"]
    (OUT / f'{name}.md').write_text('\n'.join(md) + '\n', encoding='utf-8')


def load_oracle():
    inp_path = ORACLE / 'time_vae_downblock3_inputs.npz'
    out_path = ORACLE / 'time_vae_downblock3_outputs.npz'
    if not inp_path.exists() or not out_path.exists():
        return None, None, {'status': 'BLOCKED', 'reason': 'DownBlock3 oracle tensors missing'}
    return np.load(inp_path), np.load(out_path), None


def failure_analysis():
    return [
        'Check group_norm channel count for 512-channel final down block.',
        'Check time_emb_proj scale/shift split shape and time_embedding_norm.',
        'Check residual branch order and output_scale_factor.',
        'Check whether audit unexpectedly found conv_shortcut/nin_shortcut.',
        'Check convolution padding and dtype conversions.',
    ]