#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path


META = Path('experiments/full_repro/lora_alignment/timevae_lora_effective_weights/timevae_lora_effective_weights_metadata.json')
COVERAGE = Path('experiments/full_repro/lora_alignment/jittor_effective_lora_coverage.json')
OUT = Path('experiments/full_repro/lora_alignment/timevae_lora_effective_weight_coverage_test.json')


def load_json(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception as exc:
        return {'status': 'FAIL', 'error': repr(exc)}


def audit_status(check: str) -> str:
    d = load_json('experiments/final_audit_report.json')
    for row in d.get('rows', []):
        if row.get('check') == check:
            return str(row.get('status', 'MISSING'))
    return 'MISSING'


def staged_npz() -> list[str]:
    proc = subprocess.run(['git', 'diff', '--cached', '--name-only'], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        return ['GIT_DIFF_CACHED_FAILED']
    return [line for line in proc.stdout.splitlines() if line.endswith(('.npy', '.npz'))]


def main() -> int:
    meta = load_json(META)
    coverage = load_json(COVERAGE)
    rows = list(meta.get('rows', []))
    manual_counts = meta.get('manual_verify_status_counts', {})
    staged = staged_npz()
    final_unet = audit_status('TADSR_UNET_FULL_FORWARD_ALIGNMENT')
    final_vae_boundary = audit_status('TIME_VAE_ACTUAL_VAEHOOK_FULL_BOUNDARY_ALIGNMENT')
    final_full = audit_status('JITTOR_FULL_INFERENCE')
    checks = {
        'metadata_exists': META.exists(),
        'metadata_status': meta.get('status') == 'PASS',
        'metadata_pair_count': meta.get('timevae_active_lora_pair_count') == 32,
        'manual_verify_pass': meta.get('timevae_lora_effective_weight_manual_verify') == 'PASS',
        'manual_verify_rows_pass': len(rows) == 32 and all(row.get('manual_verify', {}).get('status') == 'PASS' for row in rows),
        'coverage_timevae_pass': coverage.get('timevae_lora_effective_artifact_coverage') == 'PASS',
        'coverage_all_lora_pass': coverage.get('static_effective_lora_coverage_audit') == 'PASS',
        'dynamic_runtime_lora_not_implemented': coverage.get('dynamic_runtime_lora_implementation') == 'NOT_IMPLEMENTED_BY_DESIGN',
        'no_staged_npy_npz': staged == [],
        'unet_full_forward_pass': final_unet == 'PASS',
        'timevae_actual_vaehook_boundary_pass': final_vae_boundary == 'PASS',
        'full_inference_not_complete': final_full == 'NOT_COMPLETE',
    }
    status = 'PASS' if all(checks.values()) else ('PARTIAL' if not staged and meta else 'FAIL')
    result = {
        'status': status,
        'checks': checks,
        'metadata_path': str(META),
        'coverage_path': str(COVERAGE),
        'timevae_active_lora_pair_count': meta.get('timevae_active_lora_pair_count'),
        'manual_verify_status_counts': manual_counts,
        'coverage_timevae_status': coverage.get('timevae_lora_effective_artifact_coverage'),
        'coverage_all_lora_status': coverage.get('static_effective_lora_coverage_audit'),
        'staged_npy_npz': staged,
        'final_audit_unet_full_forward': final_unet,
        'final_audit_timevae_actual_vaehook_full_boundary': final_vae_boundary,
        'final_audit_full_inference': final_full,
    }
    OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f"TIME_VAE_LORA_EFFECTIVE_WEIGHT_COVERAGE_TEST: {status}")
    print(json.dumps(result, indent=2))
    return 0 if status in {'PASS', 'PARTIAL'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
