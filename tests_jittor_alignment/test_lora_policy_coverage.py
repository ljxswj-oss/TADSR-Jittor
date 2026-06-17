#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path


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
    required = [
        Path('experiments/full_repro/lora_alignment/audit_tadsr_lora_policy.json'),
        Path('experiments/full_repro/lora_alignment/jittor_effective_lora_coverage.json'),
        Path('experiments/full_repro/lora_alignment/timevae_lora_effective_weights/timevae_lora_effective_weights_metadata.json'),
        Path('docs/lora_policy_decision.md'),
    ]
    missing = [str(p) for p in required if not p.exists()]
    policy = load_json(required[0])
    coverage = load_json(required[1])
    staged = staged_npz()
    final_unet = audit_status('TADSR_UNET_FULL_FORWARD_ALIGNMENT')
    final_full = audit_status('JITTOR_FULL_INFERENCE')
    dynamic_status = coverage.get('dynamic_runtime_lora_implementation')
    coverage_status = coverage.get('static_effective_lora_coverage_audit')
    timevae_status = coverage.get('timevae_lora_effective_artifact_coverage')
    if missing or staged:
        status = 'FAIL'
    elif policy.get('status') == 'PASS' and dynamic_status == 'NOT_IMPLEMENTED_BY_DESIGN' and final_unet == 'PASS' and final_full == 'NOT_COMPLETE':
        status = 'PASS' if coverage_status == 'PASS' else 'PARTIAL'
    else:
        status = 'FAIL'
    result = {
        'status': status,
        'missing_required_files': missing,
        'staged_npy_npz': staged,
        'policy_status': policy.get('status'),
        'static_effective_coverage_status': coverage_status,
        'timevae_lora_effective_artifact_coverage': timevae_status,
        'dynamic_runtime_lora_status': dynamic_status,
        'final_audit_unet_full_forward': final_unet,
        'final_audit_full_inference': final_full,
    }
    Path('experiments/full_repro/lora_alignment/lora_policy_contract_test.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f"TADSR_LORA_POLICY_CONTRACT_TEST: {status}")
    print(json.dumps(result, indent=2))
    return 0 if status in {'PASS', 'PARTIAL'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
