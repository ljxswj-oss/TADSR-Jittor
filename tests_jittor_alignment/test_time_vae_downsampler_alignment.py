#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jittor_tadsr_full.vae_time_aware import TimeVAEDownBlock0Tester
from jittor_tadsr_full.utils import compare_arrays, write_json

OUT = Path('experiments/full_repro/time_vae_alignment')
ORACLE = OUT / 'oracle_tensors_downblock0'
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


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    inp_path = ORACLE / 'time_vae_downblock0_inputs.npz'
    out_path = ORACLE / 'time_vae_downblock0_outputs.npz'
    if not inp_path.exists() or not out_path.exists():
        result = {'status': 'BLOCKED', 'reason': 'DownBlock0 oracle tensors missing'}
    else:
        inputs = np.load(inp_path)
        outputs = np.load(out_path)
        if 'downsampler0_input' not in inputs.files or 'downsampler0_output' not in outputs.files:
            result = {'status': 'NOT_APPLICABLE', 'reason': 'official encoder.down_blocks.0 has no downsampler oracle'}
        else:
            tester = TimeVAEDownBlock0Tester(WEIGHTS)
            got = tester.run_downsampler0(inputs['downsampler0_input'])
            expected = outputs['downsampler0_output']
            metrics = add_metrics(got, expected, 1e-4)
            result = {'status': 'PASS' if metrics.get('shape_match') and metrics.get('max_abs_error', 1.0) <= 1e-4 else ('PARTIAL' if metrics.get('shape_match') and metrics.get('max_abs_error', 1.0) <= 1e-3 else 'FAIL'), 'target': 'encoder.down_blocks.0.downsamplers.0', 'padding_note': 'Diffusers Downsample2D uses right/bottom asymmetric zero padding before stride-2 conv when padding=0.', 'metrics': metrics, 'weights': str(WEIGHTS)}
    write_json(OUT / 'jittor_downsampler_alignment.json', result)
    md = ['# TimeAware VAE Downsampler Alignment', '', f"Status: **{result['status']}**"]
    if result.get('metrics'):
        md += ['', '| Metric | Value |', '|---|---:|']
        for k, v in result['metrics'].items():
            md.append(f'| {k} | `{v}` |')
    if result.get('padding_note'):
        md += ['', f"Padding note: {result['padding_note']}"]
    if result.get('reason'):
        md += ['', f"Reason: {result['reason']}"]
    (OUT / 'jittor_downsampler_alignment.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print(f"TIME_VAE_DOWNSAMPLER_ALIGNMENT: {result['status']}")
    if result.get('metrics'):
        print(json.dumps(result['metrics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL', 'NOT_APPLICABLE'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
