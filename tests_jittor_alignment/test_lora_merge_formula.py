#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jittor_tadsr_full.utils import compare_arrays, write_json

OUT = Path('experiments/full_repro/lora_alignment')


def numpy_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    if a.ndim == 2 and b.ndim == 2:
        return (b @ a).astype(np.float32)
    if a.ndim == 4 and b.ndim == 4 and b.shape[2:] == (1, 1):
        return np.einsum('orxy,rijk->oijk', b, a, optimize=True).astype(np.float32)
    raise ValueError(f'Unsupported LoRA shapes A={a.shape} B={b.shape}')


def main() -> int:
    oracle_path = OUT / 'lora_merge_oracle.npz'
    if not oracle_path.exists():
        result = {'status': 'PARTIAL', 'reason': 'lora_merge_oracle.npz missing'}
    else:
        oracle = np.load(oracle_path)
        got = numpy_delta(oracle['lora_A'], oracle['lora_B'])
        expected = oracle['merged_delta']
        cmp = compare_arrays(got, expected)
        result = {'status': 'PASS' if cmp.get('shape_match') and cmp.get('max_abs_error', 1.0) <= 1e-6 else 'FAIL', 'metrics': cmp, 'formula': 'linear: B @ A; conv: einsum over rank dimension'}
    write_json(OUT / 'lora_merge_validation.json', result)
    md = ['# LoRA Merge Formula Validation', '', f"Status: **{result['status']}**"]
    if result.get('metrics'):
        md += ['', '| Metric | Value |', '|---|---:|']
        for k, v in result['metrics'].items():
            md.append(f'| {k} | `{v}` |')
    if result.get('reason'):
        md += ['', f"Reason: {result['reason']}"]
    (OUT / 'lora_merge_validation.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print(f"LORA_MERGE_VALIDATION: {result['status']}")
    if result.get('metrics'):
        print(json.dumps(result['metrics'], indent=2))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
