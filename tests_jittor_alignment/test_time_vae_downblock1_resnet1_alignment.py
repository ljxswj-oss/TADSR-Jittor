#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jittor_tadsr_full.vae_time_aware import TimeVAEDownBlock1Tester
from jittor_tadsr_full.utils import compare_arrays, write_json

OUT = Path('experiments/full_repro/time_vae_alignment')
ORACLE = OUT / 'oracle_tensors_downblock1'
WEIGHTS = Path('/mnt/data/sj/checkpoints/TADSR/jittor_converted/diffusers/time_vae_weights.npz')

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
    write_json(OUT / f'{name}.json', result)
    md = [f'# {title}', '', f"Status: **{result['status']}**"]
    if result.get('metrics'):
        md += ['', '| Metric | Value |', '|---|---:|']
        for k, v in result['metrics'].items():
            md.append(f'| {k} | `{v}` |')
    if result.get('reason'):
        md += ['', f"Reason: {result['reason']}"]
    (OUT / f'{name}.md').write_text('\n'.join(md) + '\n', encoding='utf-8')

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    inp_path = ORACLE / 'time_vae_downblock1_inputs.npz'
    out_path = ORACLE / 'time_vae_downblock1_outputs.npz'
    if not inp_path.exists() or not out_path.exists():
        result = {'status': 'BLOCKED', 'reason': 'DownBlock1 oracle tensors missing'}
    else:
        inputs = np.load(inp_path)
        outputs = np.load(out_path)
        tester = TimeVAEDownBlock1Tester(WEIGHTS)
        got = tester.run_resnet1(inputs['resnet1_input'], inputs['downblock1_temb'])
        expected = outputs['resnet1_output']
        metrics = add_metrics(got, expected, 1e-4)
        result = {
            'status': status_from_metrics(metrics),
            'target': 'encoder.down_blocks.1.resnets.1',
            'channel_change': '256->256',
            'uses_conv_shortcut': False,
            'metrics': metrics,
            'weights': str(WEIGHTS),
        }
    write_report('jittor_downblock1_resnet1_alignment', 'TimeAware VAE DownBlock1 ResNet1 Alignment', result)
    print(f"TIME_VAE_DOWNBLOCK1_RESNET1_ALIGNMENT: {result['status']}")
    if result.get('metrics'):
        print(json.dumps(result['metrics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1

if __name__ == '__main__':
    raise SystemExit(main())
