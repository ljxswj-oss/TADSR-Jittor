#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

OUT = Path('experiments/full_repro/lora_alignment')
POLICY_JSON = OUT / 'audit_tadsr_lora_policy.json'
TIMEVAE_EFFECTIVE_META = OUT / 'timevae_lora_effective_weights/timevae_lora_effective_weights_metadata.json'

UNET_DIR = Path('experiments/full_repro/unet_alignment')
TIME_VAE_DIR = Path('experiments/full_repro/time_vae_alignment')


def load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception as exc:
        return {'status': 'FAIL', 'error': repr(exc)}


def status_from_json(path: str | Path, default: str = 'BLOCKED') -> str:
    d = load_json(path)
    return str(d.get('status', default)) if d else default


def git_ignored(path: Path) -> bool:
    try:
        proc = subprocess.run(['git', 'check-ignore', '-q', str(path)], cwd='.', check=False)
        return proc.returncode == 0
    except Exception:
        return False


def git_tracked(path: Path) -> bool:
    try:
        proc = subprocess.run(['git', 'ls-files', '--error-unmatch', str(path)], cwd='.', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return proc.returncode == 0
    except Exception:
        return False


def staged_npz_files() -> list[str]:
    try:
        proc = subprocess.run(['git', 'diff', '--cached', '--name-only'], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode != 0:
            return ['GIT_DIFF_CACHED_FAILED']
        return [line for line in proc.stdout.splitlines() if line.endswith(('.npy', '.npz'))]
    except Exception as exc:
        return [f'ERROR:{exc!r}']


def collect_effective_artifacts() -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for p in sorted(Path('experiments/full_repro').rglob('*effective_weights*.npz')):
        posix = p.as_posix()
        component = (
            'UNet' if 'unet_alignment' in posix else
            'TimeVAE' if 'time_vae' in posix or 'timevae_lora_effective_weights' in posix else
            'other'
        )
        artifacts.append({
            'path': str(p),
            'component': component,
            'exists': p.exists(),
            'size_bytes': p.stat().st_size if p.exists() else 0,
            'git_ignored': git_ignored(p),
            'git_tracked': git_tracked(p),
        })
    return artifacts


def timevae_effective_coverage() -> tuple[set[str], dict[str, Any]]:
    meta = load_json(TIMEVAE_EFFECTIVE_META)
    rows = list(meta.get('rows', []))
    good = set()
    if (
        meta.get('status') == 'PASS'
        and meta.get('timevae_lora_effective_weights_audit') == 'PASS'
        and meta.get('timevae_lora_effective_weights_export') == 'PASS'
        and meta.get('timevae_lora_effective_weight_manual_verify') == 'PASS'
        and len(rows) == 32
    ):
        for row in rows:
            if row.get('manual_verify', {}).get('status') == 'PASS':
                good.add(str(row.get('module_path')))
    return good, meta


def coverage_for_module(module: dict[str, Any], unet_effective_exists: bool, unet_full_pass: bool, timevae_actual_pass: bool, timevae_good: set[str]) -> dict[str, Any]:
    comp = module.get('component')
    path = str(module.get('module_path', ''))
    if comp == 'UNet':
        return {
            'module_path': path,
            'component': comp,
            'coverage_status': 'PASS' if unet_effective_exists and unet_full_pass else 'PARTIAL',
            'coverage_evidence': 'UNet static effective-weight NPZ artifacts plus TADSR_UNET_FULL_FORWARD_ALIGNMENT PASS',
        }
    if comp == 'TimeVAE':
        if path in timevae_good:
            return {
                'module_path': path,
                'component': comp,
                'coverage_status': 'PASS',
                'coverage_evidence': (
                    'TimeVAE static effective LoRA artifact metadata includes this module and manual '
                    'official-active-module-vs-effective-weight verification is PASS.'
                ),
            }
        return {
            'module_path': path,
            'component': comp,
            'coverage_status': 'PARTIAL',
            'coverage_evidence': (
                'Official actual VAEHook/full-boundary alignment is PASS, but no per-module VAE LoRA merged effective-weight '
                'artifact is present. This is intentionally reported as partial rather than claimed complete.'
                if timevae_actual_pass else
                'No complete TimeVAE LoRA effective-weight evidence found.'
            ),
        }
    return {
        'module_path': path,
        'component': comp,
        'coverage_status': 'NOT_APPLICABLE_NO_LORA',
        'coverage_evidence': 'No active TADSR inference LoRA component identified.',
    }


def write_md(result: dict[str, Any]) -> None:
    lines = [
        '# Jittor Static Effective LoRA Coverage Audit',
        '',
        f"Overall static effective LoRA coverage: **{result['static_effective_lora_coverage_audit']}**",
        '',
        '## Summary',
        '',
        f"- Active official LoRA modules: {result['active_lora_module_count']}",
        f"- Covered modules: {result['covered_count']}",
        f"- Partial modules: {result['partial_count']}",
        f"- Missing modules: {result['missing_count']}",
        f"- Effective-weight artifacts: {result['effective_weight_artifact_count']}",
        f"- Large NPZ artifacts tracked by git: {result['committed_large_npz_count']}",
        '',
        '## Marker statuses',
        '',
        '| Marker | Status |',
        '|---|---|',
    ]
    for key in [
        'static_effective_lora_coverage_audit',
        'active_lora_module_coverage',
        'effective_weight_artifact_coverage',
        'dynamic_runtime_lora_implementation',
        'timevae_lora_effective_artifact_coverage',
        'timevae_active_lora_module_coverage',
    ]:
        lines.append(f"| `{key}` | `{result[key]}` |")
    lines += [
        '',
        '## Component coverage',
        '',
        '| Component | Total | PASS | PARTIAL | Missing |',
        '|---|---:|---:|---:|---:|',
    ]
    for comp, row in result['component_coverage'].items():
        lines.append(f"| {comp} | {row['total']} | {row.get('PASS', 0)} | {row.get('PARTIAL', 0)} | {row.get('missing', 0)} |")
    lines += [
        '',
        '## Missing or partial active LoRA modules',
        '',
        '| Component | Module | Status | Evidence |',
        '|---|---|---|---|',
    ]
    for row in result['module_coverage']:
        if row['coverage_status'] not in {'PASS', 'NOT_APPLICABLE_NO_LORA'}:
            ev = row['coverage_evidence'].replace('|', '\\|')
            lines.append(f"| {row['component']} | `{row['module_path']}` | `{row['coverage_status']}` | {ev} |")
    if not any(row['coverage_status'] not in {'PASS', 'NOT_APPLICABLE_NO_LORA'} for row in result['module_coverage']):
        lines.append('| - | - | - | No missing or partial active LoRA modules. |')
    lines += [
        '',
        '## Effective-weight artifacts',
        '',
        '| Path | Component | Size bytes | Git ignored | Git tracked |',
        '|---|---|---:|---|---|',
    ]
    for art in result['effective_weight_artifacts'][:120]:
        lines.append(f"| `{art['path']}` | {art['component']} | {art['size_bytes']} | {art['git_ignored']} | {art['git_tracked']} |")
    if len(result['effective_weight_artifacts']) > 120:
        lines.append(f"\nOnly the first 120 artifacts are shown; full list is in JSON ({len(result['effective_weight_artifacts'])} artifacts).")
    (OUT / 'jittor_effective_lora_coverage.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    policy = load_json(POLICY_JSON)
    inventory = list(policy.get('module_inventory', []))
    artifacts = collect_effective_artifacts()
    timevae_good, timevae_meta = timevae_effective_coverage()
    unet_effective_exists = any(a['component'] == 'UNet' for a in artifacts)
    unet_full_pass = status_from_json('experiments/full_repro/unet_alignment/jittor_unet_full_forward_alignment.json') == 'PASS'
    timevae_actual_pass = status_from_json('experiments/full_repro/vae_alignment/jittor_tadsr_vae_actual_hook_behavior_alignment.json') == 'PASS'
    module_coverage = [
        coverage_for_module(m, unet_effective_exists=unet_effective_exists, unet_full_pass=unet_full_pass, timevae_actual_pass=timevae_actual_pass, timevae_good=timevae_good)
        for m in inventory
    ]
    counts = Counter(row['coverage_status'] for row in module_coverage)
    component_coverage: dict[str, dict[str, int]] = {}
    for row in module_coverage:
        comp = row['component']
        component_coverage.setdefault(comp, {'total': 0, 'PASS': 0, 'PARTIAL': 0, 'missing': 0})
        component_coverage[comp]['total'] += 1
        if row['coverage_status'] == 'PASS':
            component_coverage[comp]['PASS'] += 1
        elif row['coverage_status'] == 'PARTIAL':
            component_coverage[comp]['PARTIAL'] += 1
        elif row['coverage_status'] == 'FAIL':
            component_coverage[comp]['missing'] += 1

    committed_large_npz = [
        a for a in artifacts
        if a['git_tracked'] and a['path'].endswith('.npz') and a['size_bytes'] > 1024 * 1024
    ]
    missing = counts.get('FAIL', 0)
    partial = counts.get('PARTIAL', 0)
    coverage_status = 'PASS' if inventory and missing == 0 and partial == 0 else ('PARTIAL' if inventory and missing == 0 else 'FAIL')
    staged_npz = staged_npz_files()
    if not artifacts:
        artifact_status = 'FAIL'
    elif staged_npz:
        artifact_status = 'FAIL'
    else:
        artifact_status = 'PASS'
    result = {
        'status': coverage_status,
        'policy_json': str(POLICY_JSON),
        'active_lora_module_count': len(inventory),
        'covered_count': counts.get('PASS', 0),
        'partial_count': partial,
        'missing_count': missing,
        'module_coverage': module_coverage,
        'component_coverage': component_coverage,
        'timevae_lora_effective_metadata': str(TIMEVAE_EFFECTIVE_META),
        'timevae_lora_effective_metadata_status': timevae_meta.get('status', 'MISSING') if timevae_meta else 'MISSING',
        'timevae_lora_effective_artifact_coverage': 'PASS' if component_coverage.get('TimeVAE', {}).get('PASS', 0) == 32 else 'PARTIAL',
        'timevae_active_lora_module_coverage': 'PASS' if component_coverage.get('TimeVAE', {}).get('PASS', 0) == 32 else 'PARTIAL',
        'effective_weight_artifact_count': len(artifacts),
        'effective_weight_artifacts': artifacts,
        'ignored_large_npz_count': sum(1 for a in artifacts if a['git_ignored'] and a['size_bytes'] > 1024 * 1024),
        'committed_large_npz_count': len(committed_large_npz),
        'committed_large_npz': committed_large_npz,
        'staged_npy_npz_count': len(staged_npz),
        'staged_npy_npz': staged_npz,
        'static_effective_lora_coverage_audit': coverage_status,
        'active_lora_module_coverage': coverage_status,
        'effective_weight_artifact_coverage': artifact_status,
        'dynamic_runtime_lora_implementation': 'NOT_IMPLEMENTED_BY_DESIGN',
        'note': (
            'UNet active LoRA modules are covered by exported static effective weights and full-forward alignment. '
            'TimeVAE active LoRA modules still need explicit per-module static effective-weight artifacts; current VAE evidence is actual-hook/boundary alignment, so coverage is PARTIAL. '
            'Existing tracked large NPZ artifacts are reported separately; this audit only blocks newly staged NPY/NPZ payloads.'
            if coverage_status == 'PARTIAL'
            else 'All active LoRA modules are covered by static effective-weight artifacts. Generic runtime LoRA remains not implemented by design.'
        ),
    }
    (OUT / 'jittor_effective_lora_coverage.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    write_md(result)
    print(f"TADSR_STATIC_EFFECTIVE_LORA_COVERAGE_AUDIT: {coverage_status}")
    print(f"TADSR_ACTIVE_LORA_MODULE_COVERAGE: {coverage_status}")
    print(f"TADSR_EFFECTIVE_WEIGHT_ARTIFACT_COVERAGE: {artifact_status}")
    print("TADSR_DYNAMIC_RUNTIME_LORA_IMPLEMENTATION: NOT_IMPLEMENTED_BY_DESIGN")
    print(f"TIME_VAE_LORA_EFFECTIVE_ARTIFACT_COVERAGE: {result['timevae_lora_effective_artifact_coverage']}")
    print(f"TIME_VAE_ACTIVE_LORA_MODULE_COVERAGE: {result['timevae_active_lora_module_coverage']}")
    return 0 if coverage_status in {'PASS', 'PARTIAL'} and artifact_status in {'PASS', 'PARTIAL'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
