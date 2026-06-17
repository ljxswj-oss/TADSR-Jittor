#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

WEIGHTS = Path('/mnt/data/sj/checkpoints/TADSR/jittor_converted/tadsr_weights.npz')
OUT = Path('experiments/full_repro/lora_alignment')


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = np.load(WEIGHTS)
    keys = list(data.files)
    pairs = []
    for a_key in keys:
        if 'lora_A' not in a_key:
            continue
        b_key = a_key.replace('lora_A', 'lora_B')
        if b_key in data.files:
            a_shape = list(data[a_key].shape)
            b_shape = list(data[b_key].shape)
            kind = 'conv' if len(a_shape) == 4 or len(b_shape) == 4 else 'linear'
            pairs.append({'a_key': a_key, 'b_key': b_key, 'a_shape': a_shape, 'b_shape': b_shape, 'kind': kind})
    result = {'status': 'PASS' if pairs else 'PARTIAL', 'weight_file': str(WEIGHTS), 'lora_key_count': len([k for k in keys if 'lora' in k.lower()]), 'pair_count': len(pairs), 'pairs': pairs}
    (OUT / 'lora_pairs.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    md = ['# LoRA Pair Analysis', '', f"Status: **{result['status']}**", f"LoRA keys: {result['lora_key_count']}", f"A/B pairs: {result['pair_count']}", '', '| # | Kind | A key | A shape | B key | B shape |', '|---:|---|---|---:|---|---:|']
    for i, pair in enumerate(pairs[:120], 1):
        md.append(f"| {i} | {pair['kind']} | `{pair['a_key']}` | `{pair['a_shape']}` | `{pair['b_key']}` | `{pair['b_shape']}` |")
    if len(pairs) > 120:
        md.append(f"\nOnly the first 120 pairs are shown; full list is in `lora_pairs.json`.")
    (OUT / 'lora_pairs.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['status', 'lora_key_count', 'pair_count']}, indent=2))
    return 0 if pairs else 1


if __name__ == '__main__':
    raise SystemExit(main())
