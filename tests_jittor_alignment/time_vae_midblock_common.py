#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jittor_tadsr_full.vae_time_aware import TimeVAEMidBlockTester, TimeVAEBlockTester
from jittor_tadsr_full.utils import compare_arrays, write_json

OUT = Path('experiments/full_repro/time_vae_alignment')
ORACLE = OUT / 'oracle_tensors_midblock'
WEIGHTS = Path('/mnt/data/sj/checkpoints/TADSR/jittor_converted/diffusers/time_vae_weights.npz')


def load_json(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_audit():
    return load_json(OUT / 'midblock_audit.json')


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
    if result.get('reason'):
        md += ['', f"Reason: {result['reason']}"]
    if result.get('failure_analysis'):
        md += ['', '## Failure analysis']
        for item in result['failure_analysis']:
            md.append(f'- {item}')
    (OUT / f'{name}.md').write_text('\n'.join(md) + '\n', encoding='utf-8')


def load_oracle():
    inp_path = ORACLE / 'time_vae_midblock_inputs.npz'
    out_path = ORACLE / 'time_vae_midblock_outputs.npz'
    if not inp_path.exists() or not out_path.exists():
        return None, None, {'status': 'BLOCKED', 'reason': 'MidBlock oracle tensors missing'}
    return np.load(inp_path), np.load(out_path), None


def attention_exists():
    audit = load_audit()
    return int(audit.get('attention_count', 0)) > 0


def has_resnet1():
    audit = load_audit()
    return bool(audit.get('has_resnet1', False))


def failure_analysis(kind='midblock'):
    base = [
        'Check TimeAwareResnetBlock scale_shift time embedding and group norm epsilon.',
        'Check converted NPZ key mapping for encoder.mid_block.*.',
        'Check dtype conversion and convolution padding.',
    ]
    if kind == 'attention':
        base += [
            'Check attention group_norm is applied on [batch, channels, sequence] after NCHW flatten.',
            'Check Q/K/V projection names: converted NPZ uses query/key/value/proj_attn.',
            'Check heads/head_dim reshape order and scaled dot-product softmax axis.',
            'Check residual connection and rescale_output_factor.',
        ]
    return base
