#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jittor_tadsr_full.vae_time_aware import TimeVAEBlockTester
from jittor_tadsr_full.utils import compare_arrays, write_json

OUT = Path('experiments/full_repro/time_vae_alignment')
ORACLE = OUT / 'oracle_tensors'
WEIGHTS = Path('/mnt/data/sj/checkpoints/TADSR/jittor_converted/diffusers/time_vae_weights.npz')


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 1.0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    inp_path = ORACLE / 'time_vae_oracle_inputs.npz'
    out_path = ORACLE / 'time_vae_oracle_outputs.npz'
    if not inp_path.exists() or not out_path.exists():
        result = {'status': 'BLOCKED', 'reason': 'TimeAware VAE oracle tensors missing'}
    else:
        inputs = np.load(inp_path)
        outputs = np.load(out_path)
        tester = TimeVAEBlockTester(WEIGHTS)
        got = tester.run_first_resnet_block(inputs['first_resnet_input'], inputs['first_resnet_temb'])
        expected = outputs['first_resnet_output']
        cmp = compare_arrays(got, expected)
        cmp['cosine_similarity'] = cosine(got, expected) if cmp.get('shape_match') else None
        cmp['tolerance'] = 2e-4
        result = {
            'status': 'PASS' if cmp.get('shape_match') and cmp.get('max_abs_error', 1.0) <= 2e-4 else 'FAIL',
            'target': 'encoder.down_blocks.0.resnets.0',
            'metrics': cmp,
            'implemented_components': ['GroupNorm', 'SiLU', 'Conv2d', 'time_emb_proj', 'scale_shift', 'residual'],
            'not_claimed': ['full down block', 'mid block', 'decoder', 'full TimeAware VAE forward'],
            'weights': str(WEIGHTS),
        }
    write_json(OUT / 'jittor_first_block_alignment.json', result)
    md = ['# TimeAware VAE First ResNet Block Alignment', '', f"Status: **{result['status']}**"]
    if result.get('metrics'):
        m = result['metrics']
        md += ['', '| Metric | Value |', '|---|---:|']
        for k in ['shape', 'max_abs_error', 'mean_abs_error', 'relative_error', 'cosine_similarity', 'tolerance']:
            md.append(f"| {k} | `{m.get(k)}` |")
        md += ['', 'Implemented components: ' + ', '.join(result.get('implemented_components', [])), 'Still not claimed: ' + ', '.join(result.get('not_claimed', []))]
    if result.get('reason'):
        md += ['', f"Reason: {result['reason']}"]
    (OUT / 'jittor_first_block_alignment.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print(f"TIME_VAE_FIRST_BLOCK_ALIGNMENT: {result['status']}")
    if result.get('metrics'):
        print(json.dumps(result['metrics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
