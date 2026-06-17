#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch

WEIGHTS = Path('/mnt/data/sj/checkpoints/TADSR/jittor_converted/tadsr_weights.npz')
PAIRS = Path('experiments/full_repro/lora_alignment/lora_pairs.json')
OUT = Path('experiments/full_repro/lora_alignment')


def torch_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    at = torch.from_numpy(a.astype(np.float32))
    bt = torch.from_numpy(b.astype(np.float32))
    if at.ndim == 2 and bt.ndim == 2:
        return torch.matmul(bt, at).numpy().astype(np.float32)
    if at.ndim == 4 and bt.ndim == 4 and bt.shape[2:] == (1, 1):
        return torch.einsum('orxy,rijk->oijk', bt, at).numpy().astype(np.float32)
    raise ValueError(f'Unsupported LoRA shapes A={tuple(at.shape)} B={tuple(bt.shape)}')


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not PAIRS.exists():
        raise FileNotFoundError('Run tools/analyze_lora_pairs.py first')
    pair_data = json.loads(PAIRS.read_text())
    data = np.load(WEIGHTS)
    selected = None
    for pair in pair_data.get('pairs', []):
        try:
            delta = torch_delta(data[pair['a_key']], data[pair['b_key']])
            selected = pair
            break
        except Exception:
            continue
    if selected is None:
        result = {'status': 'PARTIAL', 'reason': 'no supported linear/conv LoRA pair found'}
        (OUT / 'lora_merge_oracle_report.md').write_text('# LoRA Merge Oracle\n\nStatus: **PARTIAL**\n\nNo supported pair found.\n', encoding='utf-8')
        (OUT / 'lora_merge_oracle.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return 0
    a = data[selected['a_key']].astype(np.float32)
    b = data[selected['b_key']].astype(np.float32)
    delta = torch_delta(a, b)
    np.savez_compressed(OUT / 'lora_merge_oracle.npz', lora_A=a, lora_B=b, merged_delta=delta)
    result = {'status': 'PASS', 'selected_pair': selected, 'delta_shape': list(delta.shape), 'scale': 1.0, 'formula': 'linear: B @ A; conv: einsum over rank dimension'}
    (OUT / 'lora_merge_oracle.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    md = ['# LoRA Merge Oracle', '', 'Status: **PASS**', '', f"A key: `{selected['a_key']}`", f"B key: `{selected['b_key']}`", f"Kind: `{selected['kind']}`", f"Delta shape: `{list(delta.shape)}`", '', 'Scale is 1.0 for this isolated formula validation; integration-specific alpha scaling is left for the full LoRA port.']
    (OUT / 'lora_merge_oracle_report.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
