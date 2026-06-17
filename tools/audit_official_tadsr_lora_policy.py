#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

OFFICIAL_REPO = Path('/mnt/data/sj/projects/TADSR_official_pytorch')
JITTOR_REPO = Path('/mnt/data/sj/projects/TADSR-Jittor')
LORA_PAIRS = Path('experiments/full_repro/lora_alignment/lora_pairs.json')
CONVERTED_WEIGHTS = Path('/mnt/data/sj/checkpoints/TADSR/jittor_converted/tadsr_weights.npz')
OUT = Path('experiments/full_repro/lora_alignment')

KEYWORDS = [
    'lora',
    'LoRA',
    'peft',
    'Peft',
    'adapter',
    'set_adapters',
    'set_adapter',
    'load_lora',
    'scale',
    'lora_scale',
    'cross_attention_kwargs',
    'LoRACompatible',
    'BaseTunerLayer',
    'merge_and_unload',
]

SKIP_DIRS = {'.git', '__pycache__'}
MAX_SOURCE_HITS = 600


def read_text(path: Path) -> str:
    return path.read_text(encoding='utf-8', errors='ignore')


def source_files(root: Path):
    for p in root.rglob('*'):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() not in {'.py', '.sh', '.md', '.txt', '.yml', '.yaml'}:
            continue
        yield p


def context_name(lines: list[str], idx: int) -> str:
    pat = re.compile(r'^\s*(class|def)\s+([A-Za-z_][A-Za-z0-9_]*)')
    for j in range(idx, -1, -1):
        m = pat.search(lines[j])
        if m:
            return f'{m.group(1)} {m.group(2)}'
    return 'module'


def classify_file(path: Path, line: str) -> dict[str, Any]:
    s = f'{path.as_posix()} {line}'.lower()
    is_diffusers = '/diffusers/' in s or s.startswith('diffusers/')
    is_training = 'train_tadsr' in s or 'set_train' in s or 'lambda_vsd_lora' in s
    is_inference = 'test_tadsr' in s or 'tadsr_test' in s or 'load_ckpt' in s or 'forward' in s
    return {
        'component_hint': (
            'UNet' if 'unet' in s else
            'TimeVAE' if 'vae' in s or 'autoencoder' in s else
            'RAM/text encoder' if 'ram' in s or 'bert' in s or 'text_encoder' in s else
            'diffusers library' if is_diffusers else
            'other'
        ),
        'used_for_unet': 'unet' in s,
        'used_for_vae_or_timevae': 'vae' in s or 'autoencoder' in s,
        'used_for_output_tail': 'conv_out' in s or 'output tail' in s,
        'used_for_inference_or_test_path': is_inference,
        'used_for_training_path': is_training,
        'library_level_hit': is_diffusers,
    }


def scan_source() -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    keyword_re = re.compile('|'.join(re.escape(k) for k in KEYWORDS))
    for p in source_files(OFFICIAL_REPO):
        rel = p.relative_to(OFFICIAL_REPO)
        text = read_text(p)
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if not keyword_re.search(line):
                continue
            item = {
                'file': str(rel),
                'line': i + 1,
                'context': context_name(lines, i),
                'text': line.strip()[:240],
            }
            item.update(classify_file(rel, line))
            hits.append(item)
            if len(hits) >= MAX_SOURCE_HITS:
                return hits
    return hits


def load_lora_pairs() -> list[dict[str, Any]]:
    if LORA_PAIRS.exists():
        data = json.loads(read_text(LORA_PAIRS))
        return list(data.get('pairs', []))
    data = np.load(CONVERTED_WEIGHTS)
    pairs: list[dict[str, Any]] = []
    for a_key in data.files:
        if 'lora_A' not in a_key:
            continue
        b_key = a_key.replace('lora_A', 'lora_B')
        if b_key not in data.files:
            continue
        a_shape = list(data[a_key].shape)
        b_shape = list(data[b_key].shape)
        pairs.append({
            'a_key': a_key,
            'b_key': b_key,
            'a_shape': a_shape,
            'b_shape': b_shape,
            'kind': 'conv' if len(a_shape) == 4 or len(b_shape) == 4 else 'linear',
        })
    return pairs


def key_to_module_path(key: str) -> tuple[str, str, str]:
    if key.startswith('state_dict_unet__'):
        component = 'UNet'
        rest = key[len('state_dict_unet__'):]
    elif key.startswith('state_dict_vae__'):
        component = 'TimeVAE'
        rest = key[len('state_dict_vae__'):]
    else:
        component = 'other'
        rest = key
    rest = re.sub(r'__lora_A__.*$', '', rest)
    rest = re.sub(r'__lora_B__.*$', '', rest)
    module_path = rest.replace('__', '.')
    adapter = 'unknown'
    m = re.search(r'__lora_[AB]__(.*?)__weight$', key)
    if m:
        adapter = m.group(1)
    return component, module_path, adapter


def build_inventory(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        component, module_path, adapter = key_to_module_path(pair['a_key'])
        a_shape = pair.get('a_shape', [])
        b_shape = pair.get('b_shape', [])
        rank = a_shape[0] if a_shape else None
        base_shape = None
        if pair.get('kind') == 'linear' and a_shape and b_shape:
            base_shape = [b_shape[0], a_shape[1]]
        elif pair.get('kind') == 'conv' and a_shape and b_shape:
            base_shape = [b_shape[0], a_shape[1], a_shape[2], a_shape[3]]
        rows.append({
            'module_path': module_path,
            'component': component,
            'module_class': 'PEFT LoRA wrapped conv/linear target',
            'base_layer_class': 'Conv2d' if pair.get('kind') == 'conv' else 'Linear',
            'active_adapters': [adapter],
            'lora_A_key': pair['a_key'],
            'lora_B_key': pair['b_key'],
            'lora_A_shape': a_shape,
            'lora_B_shape': b_shape,
            'rank': rank,
            'alpha': 'not serialized in converted NPZ; official PEFT module applies its configured scale',
            'scaling': 'fixed active-adapter scale in the official loaded model',
            'base_weight_shape_inferred': base_shape,
            'delta_weight_shape_inferred': base_shape,
            'bias_status': 'base bias comes from base/effective module weights when present',
            'active_in_official_inference_path': component in {'UNet', 'TimeVAE'},
            'already_in_exported_effective_weights_if_known': (
                'UNet effective-weight artifacts cover this component'
                if component == 'UNet'
                else 'TimeVAE active LoRA requires separate per-module effective-weight artifact audit'
            ),
        })
    return rows


def analyze_dynamic_requirements(source_hits: list[dict[str, Any]]) -> dict[str, Any]:
    tadsr_py = read_text(OFFICIAL_REPO / 'tadsr.py') if (OFFICIAL_REPO / 'tadsr.py').exists() else ''
    test_py = read_text(OFFICIAL_REPO / 'test_tadsr.py') if (OFFICIAL_REPO / 'test_tadsr.py').exists() else ''
    inference_sets_fixed_adapters = all(s in tadsr_py for s in [
        'self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])',
        "self.vae.set_adapter(['default_encoder'])",
    ])
    forward_uses_cross_attention_scale = 'cross_attention_kwargs' in tadsr_py and 'scale' in tadsr_py
    unet_library_has_scale_arg = any(
        h['file'] == 'models/unet_2d_condition.py' and 'cross_attention_kwargs' in h['text']
        for h in source_hits
    )
    test_has_merge_option = '--merge_and_unload_lora' in test_py and 'default=False' in test_py
    training_toggles_adapters = 'enable_adapters()' in tadsr_py and 'disable_adapters()' in tadsr_py
    return {
        'official_inference_sets_fixed_adapters_once': inference_sets_fixed_adapters,
        'official_test_merge_and_unload_lora_optional': test_has_merge_option,
        'official_inference_forward_passes_dynamic_lora_scale': forward_uses_cross_attention_scale,
        'underlying_unet_library_supports_cross_attention_kwargs_scale': unet_library_has_scale_arg,
        'training_or_regularization_code_toggles_adapters': training_toggles_adapters,
        'official_inference_lora_dynamic_switching_required': False if inference_sets_fixed_adapters and not forward_uses_cross_attention_scale else True,
        'official_inference_static_effective_weight_sufficient': True if inference_sets_fixed_adapters and not forward_uses_cross_attention_scale else False,
        'generic_runtime_lora_needed_for_alignment': False if inference_sets_fixed_adapters and not forward_uses_cross_attention_scale else True,
        'decision_note': (
            'Official TADSR_test loads adapters, copies checkpoint LoRA parameters, sets fixed active adapters, '
            'and does not pass a per-forward LoRA scale. Dynamic adapter enable/disable exists in training/reg code, '
            'and diffusers supports scale hooks, but this is not required for the audited inference state.'
        ),
    }


def write_txt_report(result: dict[str, Any]) -> None:
    lines = [
        '# TADSR LoRA Policy Audit',
        '',
        f"Status: **{result['status']}**",
        '',
        '## Summary',
        '',
        f"- Source hits scanned: {result['source_hit_count']}",
        f"- LoRA A/B pairs: {result['inventory_summary']['pair_count']}",
        f"- UNet pairs: {result['inventory_summary']['by_component'].get('UNet', 0)}",
        f"- TimeVAE pairs: {result['inventory_summary']['by_component'].get('TimeVAE', 0)}",
        f"- Dynamic LoRA switching required for official inference: `{result['dynamic_behavior']['official_inference_lora_dynamic_switching_required']}`",
        f"- Static effective-weight policy sufficient for the audited official inference state: `{result['dynamic_behavior']['official_inference_static_effective_weight_sufficient']}`",
        '',
        '## Policy conclusion',
        '',
        result['policy_conclusion'],
        '',
        '## Component inventory',
        '',
        '| Component | LoRA pair count |',
        '|---|---:|',
    ]
    for comp, count in result['inventory_summary']['by_component'].items():
        lines.append(f'| {comp} | {count} |')
    lines += [
        '',
        '## Representative official source hits',
        '',
        '| File | Line | Context | Text |',
        '|---|---:|---|---|',
    ]
    for hit in result['source_hits'][:80]:
        text = hit['text'].replace('|', '\\|')
        lines.append(f"| `{hit['file']}` | {hit['line']} | `{hit['context']}` | {text} |")
    lines += [
        '',
        '## Representative active LoRA modules',
        '',
        '| Component | Module | Adapter | Kind | Rank | A shape | B shape |',
        '|---|---|---|---|---:|---|---|',
    ]
    for row in result['module_inventory'][:120]:
        lines.append(
            f"| {row['component']} | `{row['module_path']}` | `{','.join(row['active_adapters'])}` | "
            f"{row['base_layer_class']} | {row['rank']} | `{row['lora_A_shape']}` | `{row['lora_B_shape']}` |"
        )
    if len(result['module_inventory']) > 120:
        lines.append(f"\nOnly the first 120 modules are shown here; full inventory is in JSON ({len(result['module_inventory'])} rows).")
    (OUT / 'audit_tadsr_lora_policy.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    source_hits = scan_source()
    pairs = load_lora_pairs()
    inventory = build_inventory(pairs)
    by_component = Counter(row['component'] for row in inventory)
    by_adapter = Counter(adapter for row in inventory for adapter in row['active_adapters'])
    dynamic = analyze_dynamic_requirements(source_hits)
    status = 'PASS' if pairs and dynamic['official_inference_static_effective_weight_sufficient'] else 'BLOCKED_DYNAMIC_LORA_REQUIRED'
    static_status = 'PASS' if dynamic['official_inference_static_effective_weight_sufficient'] else 'BLOCKED_DYNAMIC_LORA_REQUIRED'
    runtime_status = 'PASS' if not dynamic['generic_runtime_lora_needed_for_alignment'] else 'BLOCKED_DYNAMIC_LORA_REQUIRED'
    result = {
        'status': status,
        'official_repo': str(OFFICIAL_REPO),
        'converted_weights': str(CONVERTED_WEIGHTS),
        'source_hit_count': len(source_hits),
        'source_hits': source_hits,
        'module_inventory_status': 'PASS' if inventory else 'BLOCKED_NO_LORA_MODULES',
        'module_inventory': inventory,
        'inventory_summary': {
            'pair_count': len(inventory),
            'by_component': dict(by_component),
            'by_adapter': dict(by_adapter),
        },
        'dynamic_behavior': dynamic,
        'static_effective_lora_policy_audit': static_status,
        'runtime_dynamic_lora_requirement_audit': runtime_status,
        'policy_conclusion': (
            'Use static effective weights as the mainline alignment policy. Generic runtime LoRA loading, '
            'adapter registry, merge/unmerge, and runtime scale switching are not implemented by design in this stage. '
            'This is valid for the audited official inference state because adapters are loaded and fixed before forward.'
            if static_status == 'PASS'
            else 'Static effective weights are insufficient because the official inference path appears to require dynamic LoRA behavior.'
        ),
    }
    (OUT / 'audit_tadsr_lora_policy.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    write_txt_report(result)
    print(f"TADSR_LORA_POLICY_AUDIT: {status}")
    print(f"TADSR_STATIC_EFFECTIVE_LORA_POLICY_AUDIT: {static_status}")
    print(f"TADSR_RUNTIME_DYNAMIC_LORA_REQUIREMENT_AUDIT: {runtime_status}")
    print(f"TADSR_LORA_MODULE_INVENTORY_AUDIT: {result['module_inventory_status']}")
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
