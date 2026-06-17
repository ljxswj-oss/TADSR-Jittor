#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jittor_tadsr_full.vae_time_aware import TimeVAEBlockTester, TimeVAEUpDecoderBlockTester
from jittor_tadsr_full.utils import compare_arrays, write_json

OUT = Path('experiments/full_repro/time_vae_alignment')
ORACLE = OUT / 'oracle_tensors_decoder_upblock3'
WEIGHTS = Path('/mnt/data/sj/checkpoints/TADSR/jittor_converted/diffusers/time_vae_weights.npz')


def load_json(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def metadata():
    return load_json(ORACLE / 'decoder_upblock3_oracle_metadata.json')


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 1.0


def add_metrics(got, expected, tolerance=1e-3):
    cmp = compare_arrays(got, expected)
    cmp['cosine_similarity'] = cosine(got, expected) if cmp.get('shape_match') else None
    cmp['tolerance'] = tolerance
    return cmp


def status_from_metrics(metrics):
    tolerance = float(metrics.get('tolerance', 1e-4))
    if metrics.get('shape_match') and metrics.get('max_abs_error', 1.0) < tolerance:
        return 'PASS'
    if metrics.get('shape_match') and metrics.get('max_abs_error', 1.0) < max(1e-3, tolerance * 10.0):
        return 'PARTIAL'
    return 'FAIL'


def status_from_diagnostics(diag):
    statuses = [status_from_metrics(m) for m in diag.values()]
    if all(s == 'PASS' for s in statuses):
        return 'PASS'
    if all(s in {'PASS', 'PARTIAL'} for s in statuses):
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
        md += ['', '## Diagnostics', '', '| Stage | Max abs error | Mean abs error | Relative error | Cosine |', '|---|---:|---:|---:|---:|']
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
    if not ORACLE.exists():
        return None, {'status': 'BLOCKED', 'reason': 'Decoder upblock3 oracle tensors missing'}
    tensor_files = sorted(ORACLE.glob('*.npy'))
    if not tensor_files:
        return None, {'status': 'BLOCKED', 'reason': 'Decoder upblock3 .npy oracle tensors missing; run tools/export_time_vae_decoder_upblock3_oracle.py before alignment tests'}
    return {p.stem: np.load(p).astype(np.float32) for p in tensor_files}, None


def result_from_metric(target, got, expected, note='', tolerance=1e-4):
    metrics = add_metrics(got, expected, tolerance)
    return {
        'status': status_from_metrics(metrics),
        'target': target,
        'note': note,
        'metrics': metrics,
        'weights': str(WEIGHTS),
    }


def failure_analysis():
    return [
        'Check audit_time_vae_decoder_upblock3.json for module order, channel change, and upsampler behavior.',
        'Check decoder.up_blocks.3.resnets.0 converted conv_shortcut weights; this block changes channels from 256 to 128.',
        'Check decoder.up_blocks.3 converted NPZ keys and prefix resolution.',
        'decoder.up_blocks.3 has no upsampler in the official topology; upsampler0 should be reported as NOT_APPLICABLE, not as a failing stage.',
        'Check ResnetBlock2D without time_emb_proj skips temb modulation on the decoder side.',
        'Check that decoder.up_blocks.2 output is used as decoder.up_blocks.3 input in bridge tests.',
        'Check NCHW dtype, group norm epsilon, padding rules, and float32/float64 accumulation drift.',
    ]


def write_metric_report(name, title, target, got, expected, note='', tolerance=1e-4):
    result = result_from_metric(target, got, expected, note, tolerance=tolerance)
    if result['status'] == 'FAIL':
        result['failure_analysis'] = failure_analysis()
    write_report(name, title, result)
    return result
