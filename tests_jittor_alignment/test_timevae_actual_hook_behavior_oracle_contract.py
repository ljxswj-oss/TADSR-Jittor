#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


META = Path('experiments/full_repro/vae_alignment/oracle_tensors_vae_actual_hook_behavior/vae_actual_hook_behavior_oracle_metadata.json')
FINAL_AUDIT = Path('experiments/final_audit_report.json')
OUT = Path('experiments/full_repro/vae_alignment/jittor_timevae_actual_hook_behavior_oracle_contract.json')


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> int:
    require(META.exists(), f'missing metadata: {META}')
    meta = json.loads(META.read_text(encoding='utf-8'))
    require(meta.get('status') == 'PASS', f"metadata status is not PASS: {meta.get('status')}")
    require(meta.get('oracle_type') == 'official_actual_vaehook_behavior', 'wrong oracle_type')
    require(meta.get('policy_decision') == 'mirror_official_actual_behavior', 'wrong policy decision')
    require(meta.get('decoder_hook_installed') is True, 'decoder hook not recorded as installed')
    require(meta.get('decoder_original_forward_used') is True, 'decoder original_forward was not used')
    require(meta.get('decoder_tiled_path_triggered') is False, 'decoder tiled path should not be triggered')
    require(meta.get('scheduler_executed') is False, 'scheduler must not run')
    require(meta.get('full_tadsr_inference_executed') is False, 'full inference must not run')
    require(meta.get('image_saved') is False, 'image must not be saved')
    if FINAL_AUDIT.exists():
        audit = json.loads(FINAL_AUDIT.read_text(encoding='utf-8'))
        statuses = {
            (row.get('check') or row.get('name')): row.get('status')
            for row in audit.get('rows', [])
            if row.get('check') or row.get('name')
        }
        require(statuses.get('TIME_VAE_FULL_ALIGNMENT') == 'NOT_COMPLETE', 'TIME_VAE_FULL_ALIGNMENT must remain NOT_COMPLETE')
        require(statuses.get('JITTOR_FULL_INFERENCE') == 'NOT_COMPLETE', 'JITTOR_FULL_INFERENCE must remain NOT_COMPLETE')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    result = {
        'status': 'PASS',
        'oracle_type': meta.get('oracle_type'),
        'policy_decision': meta.get('policy_decision'),
        'decoder_original_forward_used': meta.get('decoder_original_forward_used'),
        'decoder_tiled_path_triggered': meta.get('decoder_tiled_path_triggered'),
        'scheduler_executed': meta.get('scheduler_executed'),
        'full_tadsr_inference_executed': meta.get('full_tadsr_inference_executed'),
        'image_saved': meta.get('image_saved'),
    }
    OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('TIME_VAE_ACTUAL_VAEHOOK_ORACLE_CONTRACT_TEST: PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
