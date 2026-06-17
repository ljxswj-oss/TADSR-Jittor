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
        tester = TimeVAEDownBlock0Tester(WEIGHTS)
        after_resnet0 = tester.run_resnet0(inputs['downblock0_input'], inputs['downblock0_temb'])
        after_resnet1 = tester.run_resnet1(after_resnet0, inputs['downblock0_temb'])
        final = tester.run_downsampler0(after_resnet1)
        diagnostics = {
            'after_resnet0': add_metrics(after_resnet0, outputs['resnet0_output'], 1e-4),
            'after_resnet1': add_metrics(after_resnet1, outputs['resnet1_output'], 1e-4),
            'after_downsampler': add_metrics(final, outputs['downsampler0_output'], 1e-4),
            'final_downblock0': add_metrics(final, outputs['downblock0_output'], 1e-4),
            'direct_pytorch_downblock0_reference': add_metrics(outputs['downblock0_direct_output'], outputs['downblock0_output'], 1e-7),
        }
        final_metrics = diagnostics['final_downblock0']
        status = 'PASS' if final_metrics.get('shape_match') and final_metrics.get('max_abs_error', 1.0) <= 1e-4 else ('PARTIAL' if final_metrics.get('shape_match') and final_metrics.get('max_abs_error', 1.0) <= 1e-3 else 'FAIL')
        result = {'status': status, 'target': 'encoder.down_blocks.0', 'module_order': ['resnets.0', 'resnets.1', 'downsamplers.0'], 'metrics': final_metrics, 'diagnostics': diagnostics, 'weights': str(WEIGHTS)}
    write_json(OUT / 'jittor_downblock0_alignment.json', result)
    md = ['# TimeAware VAE Full DownBlock0 Alignment', '', f"Status: **{result['status']}**"]
    if result.get('metrics'):
        md += ['', '## Final metrics', '', '| Metric | Value |', '|---|---:|']
        for k, v in result['metrics'].items():
            md.append(f'| {k} | `{v}` |')
    if result.get('diagnostics'):
        md += ['', '## Layer-by-layer diagnostics', '', '| Stage | Max abs error | Mean abs error | Relative error | Cosine |', '|---|---:|---:|---:|---:|']
        for name, m in result['diagnostics'].items():
            md.append(f"| {name} | `{m.get('max_abs_error')}` | `{m.get('mean_abs_error')}` | `{m.get('relative_error')}` | `{m.get('cosine_similarity')}` |")
    if result.get('reason'):
        md += ['', f"Reason: {result['reason']}"]
    (OUT / 'jittor_downblock0_alignment.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print(f"TIME_VAE_DOWNBLOCK0_ALIGNMENT: {result['status']}")
    if result.get('metrics'):
        print(json.dumps(result['metrics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
