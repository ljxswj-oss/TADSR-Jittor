#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

WEIGHTS = Path('/mnt/data/sj/checkpoints/TADSR/preset/weights')
VENV_FALLBACK = Path('/mnt/data/sj/venvs/tadsr_official_pytorch')
WHEELHOUSE = Path('/mnt/data/sj/wheelhouse/tadsr_official_pytorch')
ENV_MATRIX = Path('experiments/full_repro/pytorch_official/env_matrix')
REQ = ['time_vae','unet','vae','text_encoder','tokenizer','scheduler','feature_extractor','bert-base-uncased','DAPE.pth','ram_swin_large_14m.pth','tadsr.pkl']


def load_json(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8', errors='ignore'))
    except Exception as e:
        return {'status': 'FAIL', 'error': repr(e)}


def status_from_json(path: str | Path, default='BLOCKED') -> str:
    d = load_json(path)
    return str(d.get('status', default)) if d else default


def feasibility_marker_status(marker: str, default='BLOCKED') -> str:
    d = load_json('experiments/jittor_migration_feasibility_summary.json')
    markers = d.get('markers', {}) if isinstance(d, dict) else {}
    return str(markers.get(marker, default))


def production_completion_marker_status(marker: str, default='BLOCKED') -> str:
    sources = [
        'experiments/production_completion/phase3_validation.json',
        'experiments/production_completion/phase3d_import_gate_status.json',
        'experiments/production_completion/phase3_import_decision.json',
        'experiments/production_completion/env/official_env_resolution.json',
        'experiments/production_completion/env/remote_connectivity_diagnostic.json',
        'experiments/production_completion/env/official_runtime_dependency_overlay.json',
        'experiments/production_completion/env/official_runtime_dependency_diagnosis.json',
        'experiments/production_completion/timevae_full/official_timevae_full_path_audit.json',
        'experiments/production_completion/timevae_full/timevae_metadata_partial_diagnosis.json',
        'experiments/production_completion/timevae_full/timevae_production_oracle_metadata.json',
        'experiments/production_completion/timevae_full/timevae_live_metadata_completion.json',
        'experiments/production_completion/timevae_full/jittor_timevae_production_alignment_preflight.json',
        'experiments/production_completion/runtime_lora/official_runtime_lora_behavior_audit.json',
        'experiments/production_completion/full_inference/controlled_validation_plan.json',
        'experiments/production_completion/full_inference/metadata_dry_run_contract.json',
        'experiments/production_completion/full_inference/one_step_tensor_alignment_plan.json',
        'experiments/production_completion/full_inference/one_step_tensor_alignment_protocol.json',
        'experiments/production_completion/full_inference/one_step_gate_blocker_summary.json',
        'experiments/production_completion/full_inference/one_step/one_step_official_path_audit.json',
        'experiments/production_completion/full_inference/one_step/official_one_step_oracle_metadata.json',
        'experiments/production_completion/full_inference/one_step/jittor_one_step_alignment.json',
        'experiments/production_completion/full_inference/one_step/one_step_tensor_alignment_validation.json',
        'experiments/production_completion/full_inference/actual_inference_path_audit.json',
        'experiments/production_completion/full_inference/postprocess_contract_audit.json',
        'experiments/production_completion/full_inference/multistep_applicability_decision.json',
        'experiments/production_completion/full_inference/experimental_cli_metadata_only_plan.json',
        'experiments/final_phase5b_submission_freeze_summary.json',
        'experiments/production_completion/full_inference/diagnostic_image_smoke_plan.json',
        'experiments/production_completion/diagnostic_image_smoke/diagnostic_image_smoke_metrics.json',
        'experiments/production_completion/diagnostic_image_smoke/diagnostic_image_smoke_validation.json',
        'experiments/production_completion/diagnostic_cli/one_step_diagnostic_cli_test.json',
        'experiments/timevae_full_alignment_closure_summary.json',
        'experiments/runtime_lora_final_decision_proof.json',
        'experiments/production_completion/phase3b_manual_handoff.json',
        'experiments/production_completion/phase3_import_validation.json',
        'experiments/production_completion/phase3b_live_results_manifest.json',
        'experiments/production_completion/blockers/production_completion_blockers.json',
        'experiments/smoke_training/output_tail/smoke_training_submission_summary.json',
        'experiments/production_completion/phase2_validation.json',
        'experiments/production_completion/phase1_validation.json',
        'experiments/production_completion/readiness.json',
    ]
    for src in sources:
        d = load_json(src)
        if not isinstance(d, dict) or not d:
            continue
        markers = d.get('markers', {})
        if isinstance(markers, dict) and marker in markers:
            return str(markers.get(marker, default))
        if d.get('status_marker') == marker:
            return str(d.get('audit_status', d.get('status', default)))
        phase2 = d.get('phase2_status', {})
        if isinstance(phase2, dict):
            phase2_key_map = {
                'TADSR_PRODUCTION_OFFICIAL_ENV_RESOLUTION': 'official_env_resolution_status',
                'TADSR_TIMEVAE_PRODUCTION_ORACLE_METADATA': 'timevae_oracle_metadata_status',
                'TADSR_TIMEVAE_PRODUCTION_ALIGNMENT_PREFLIGHT': 'timevae_alignment_preflight_status',
                'TADSR_FULL_INFERENCE_METADATA_DRY_RUN_CONTRACT': 'full_inference_metadata_contract_status',
            }
            if marker in phase2_key_map and phase2_key_map[marker] in phase2:
                return str(phase2[phase2_key_map[marker]])
        phase3 = d.get('phase3_status', {})
        if isinstance(phase3, dict):
            phase3_key_map = {
                'TADSR_PRODUCTION_OFFICIAL_ENV_RESOLUTION': 'official_env_resolution_status',
                'TADSR_TIMEVAE_FULL_PRODUCTION_PATH_AUDIT': 'live_timevae_audit_status',
                'TADSR_TIMEVAE_PRODUCTION_ORACLE_METADATA': 'timevae_oracle_metadata_status',
                'TADSR_TIMEVAE_PRODUCTION_ALIGNMENT_PREFLIGHT': 'timevae_preflight_status',
                'TADSR_OFFICIAL_RUNTIME_LORA_BEHAVIOR_AUDIT': 'lora_live_audit_status',
                'TADSR_FULL_INFERENCE_METADATA_DRY_RUN_CONTRACT': 'full_inference_metadata_contract_status',
                'TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT_PLAN': 'one_step_alignment_plan_status',
            }
            if marker in phase3_key_map and phase3_key_map[marker] in phase3:
                return str(phase3[phase3_key_map[marker]])
        if marker == 'TADSR_DYNAMIC_RUNTIME_LORA_NOT_REQUIRED_FOR_OFFICIAL_INFERENCE' and 'dynamic_runtime_lora_not_required_marker' in d:
            return str(d.get('dynamic_runtime_lora_not_required_marker', default))
    return default


def selected_env() -> tuple[str, str]:
    p = ENV_MATRIX / 'selected_env.sh'
    if not p.exists():
        return '', 'NONE'
    env = ''
    kind = 'NONE'
    for line in p.read_text().splitlines():
        if line.startswith('export OFFICIAL_PYTORCH_VENV='):
            env = line.split('=', 1)[1].strip().strip('"')
        if line.startswith('export OFFICIAL_PYTORCH_ENV_KIND='):
            kind = line.split('=', 1)[1].strip().strip('"')
    return env, kind


def assets_status():
    missing = [x for x in REQ if not (WEIGHTS / x).exists()]
    return ('PASS', 'all official assets present') if not missing else ('BLOCKED', 'missing: ' + ', '.join(missing))


def env_matrix_status():
    d = load_json(ENV_MATRIX / 'env_matrix_summary.json')
    if not d:
        return 'BLOCKED', 'env_matrix_summary.json missing; run bash scripts/install_official_env_matrix.sh'
    rows = d.get('rows', [])
    if any(r.get('install_status') == 'PASS' and r.get('import_smoke') == 'PASS' and r.get('check_status') == 'PASS' for r in rows):
        return 'PASS', 'at least one env fully passed'
    if any(r.get('install_status') == 'PASS' or r.get('import_smoke') == 'PASS' for r in rows):
        return 'PARTIAL', 'at least one env installed/imported partially'
    return 'BLOCKED', 'no installable official env found; use offline_pack wheelhouse'


def selected_env_status():
    env, kind = selected_env()
    if env and (Path(env) / 'bin/python').exists():
        return 'PASS', f'{kind}: {env}'
    return 'BLOCKED', 'selected_env.sh missing or points to no usable venv'


def strict_status(name):
    d = load_json(ENV_MATRIX / f'env_check_{name}.json') or load_json(f'experiments/full_repro/pytorch_official/env/env_check_{name}.json')
    if not d:
        return 'BLOCKED', f'env_check_{name}.json missing'
    st = d.get('status')
    if st == 'PASS':
        return 'PASS', f'{name} passed'
    if d.get('critical_missing'):
        return 'BLOCKED', 'critical missing: ' + ', '.join(d.get('critical_missing', []))
    if d.get('version_mismatches'):
        return 'PARTIAL', 'version mismatches: ' + ', '.join(x.get('package', '?') for x in d.get('version_mismatches', []))
    return 'PARTIAL' if 'PARTIAL' in str(st) else 'BLOCKED', str(st)


def repo_import_status():
    d = load_json(ENV_MATRIX / 'official_repo_imports.json')
    if not d:
        return 'BLOCKED', 'official_repo_imports.json missing'
    st = d.get('status')
    if st == 'PASS':
        return 'PASS', 'all official repo imports passed'
    if st == 'PARTIAL':
        return 'PARTIAL', 'some official repo imports passed'
    return 'BLOCKED', st or 'repo imports not run'


def dryrun_status():
    d = load_json('experiments/full_repro/pytorch_official/dryrun/status.json')
    if not d:
        return 'BLOCKED', 'dryrun status missing'
    if d.get('status') == 'PASS':
        return 'PASS', 'test_tadsr.py CLI checked'
    if d.get('status') in {'BLOCKED_NO_SELECTED_ENV', 'BLOCKED_OFFICIAL_REPO_MISSING'}:
        return 'BLOCKED', d.get('status')
    return 'FAIL', d.get('status', 'dryrun failed')


def smoke_status():
    d = load_json('experiments/full_repro/pytorch_official/smoke/smoke_status.json')
    if d:
        if d.get('status') == 'PASS' and d.get('output_count', 0) > 0:
            return 'PASS', f"outputs={d.get('output_count')} inputs={d.get('input_count')}"
        if 'FAIL' in str(d.get('status')):
            return 'FAIL', d.get('reason', d.get('status'))
        return 'BLOCKED', d.get('reason', d.get('status', 'smoke not complete'))
    return 'BLOCKED', 'smoke_status.json missing'


def benchmark_status():
    root = Path('experiments/full_repro/pytorch_official')
    metric_files = list(root.glob('*_subset_*/metrics.json'))
    if metric_files:
        return 'PARTIAL', f'subset benchmark metrics exist: {len(metric_files)} file(s)'
    statuses = []
    for ds in ['RealSR', 'DRealSR', 'RealLR200']:
        files = sorted(root.glob(f'{ds}_subset_*/subset_status.json'))
        if files:
            d = load_json(files[-1])
            statuses.append((ds, d.get('status', 'UNKNOWN'), d.get('reason', '')))
        else:
            statuses.append((ds, 'BLOCKED_DATASET_MISSING', 'subset status not generated'))
    if all(st == 'BLOCKED_DATASET_MISSING' for _, st, _ in statuses):
        missing = ', '.join(ds for ds, _, _ in statuses)
        return 'BLOCKED_DATASET_MISSING', f'missing benchmark datasets: {missing}'
    if any(st == 'PASS' for _, st, _ in statuses):
        return 'PARTIAL', '; '.join(f'{ds}={st}' for ds, st, _ in statuses)
    return 'BLOCKED', '; '.join(f'{ds}={st}' for ds, st, _ in statuses)


def exists_all(paths):
    return all(Path(p).exists() for p in paths)


def row(name, status, note, action=''):
    return (name, status, note, action)


def main():
    rows = []
    tiny_train = exists_all(['experiments/jittor_tiny/logs/train_log.csv', 'experiments/jittor_tiny/curves/loss_curve.png', 'experiments/jittor_tiny/checkpoints/latest.pkl'])
    tiny_test = exists_all(['experiments/jittor_tiny/logs/test_metrics.json', 'experiments/jittor_tiny/results/compare_grid.png'])
    pytorch_tiny = exists_all(['experiments/alignment_report.md', 'experiments/compare_loss_jittor_vs_pytorch.png', 'experiments/compare_psnr_jittor_vs_pytorch.png'])
    rows.append(row('JITTOR_TINY_TRAINING', 'PASS' if tiny_train else 'FAIL', 'tiny train log/loss curve/checkpoint present' if tiny_train else 'missing tiny training evidence', 'run scripts/train_jittor_tiny.sh'))
    rows.append(row('JITTOR_TINY_TESTING', 'PASS' if tiny_test else 'FAIL', 'tiny test metrics and compare grid present' if tiny_test else 'missing tiny testing evidence', 'run scripts/test_jittor_tiny.sh'))
    rows.append(row('PYTORCH_TINY_ALIGNMENT', 'PASS' if pytorch_tiny else 'FAIL', 'alignment report and compare curves present' if pytorch_tiny else 'missing PyTorch tiny alignment evidence', 'run scripts/compare_jittor_pytorch.sh'))
    rows.append(row('PYTORCH_OFFICIAL_VENV', 'PASS' if VENV_FALLBACK.exists() else 'BLOCKED', str(VENV_FALLBACK), 'bash scripts/install_official_env_matrix.sh'))
    rows.append(row('PYTORCH_OFFICIAL_ENV_MATRIX', *env_matrix_status(), 'run bash scripts/install_official_env_matrix.sh or prepare offline_pack'))
    rows.append(row('PYTORCH_OFFICIAL_SELECTED_ENV', *selected_env_status(), 'inspect experiments/full_repro/pytorch_official/env_matrix/selected_env.sh'))
    rows.append(row('PYTORCH_OFFICIAL_ENV_STRICT_CU118', *strict_status('strict-cu118'), 'check install_strict-cu118.log'))
    rows.append(row('PYTORCH_OFFICIAL_ENV_STRICT_PYPI', *strict_status('strict-pypi'), 'check install_strict-pypi.log'))
    rows.append(row('PYTORCH_OFFICIAL_ENV_RELAXED_PYPI', *strict_status('relaxed-pypi'), 'check install_relaxed-pypi.log'))
    rows.append(row('PYTORCH_OFFICIAL_REPO_IMPORTS', *repo_import_status(), 'python3 scripts/check_official_repo_imports.py'))
    rows.append(row('PYTORCH_OFFICIAL_DRYRUN', *dryrun_status(), 'bash scripts/run_official_pytorch_dryrun.sh'))
    rows.append(row('PYTORCH_OFFICIAL_ASSETS', *assets_status(), 'upload weights to /mnt/data/sj/incoming/TADSR_assets/TADSR_weights'))
    rows.append(row('OFFICIAL_PYTORCH_ORACLE', *smoke_status(), 'bash scripts/run_official_pytorch_smoke.sh'))
    rows.append(row('PYTORCH_OFFICIAL_BENCHMARK', *benchmark_status(), 'upload datasets and run scripts/run_official_pytorch_subset.sh'))
    wheels = list(WHEELHOUSE.glob('*')) if WHEELHOUSE.exists() else []
    rows.append(row('WHEELHOUSE', 'PASS' if wheels else 'BLOCKED', f'{len(wheels)} files in {WHEELHOUSE}', 'build offline_pack on networked machine and unpack it'))
    rows.append(row('OFFLINE_PACK_GUIDE', 'PASS' if Path('offline_pack/README_OFFLINE_PACK.md').exists() else 'BLOCKED', 'offline_pack guide exists' if Path('offline_pack/README_OFFLINE_PACK.md').exists() else 'missing guide', 'restore offline_pack docs'))
    glog = Path('experiments/full_repro/gpu_setup/jittor_cuda_user_space.log')
    gtxt = glog.read_text(errors='ignore') if glog.exists() else ''
    gpu = 'Jittor CUDA status: 0' in gtxt
    rows.append(row('JITTOR_GPU', 'PASS' if gpu else 'BLOCKED', 'CUDA check passed' if gpu else 'cuDNN8 required; cuDNN9 not acceptable', 'install cuDNN8 to /mnt/data/sj/local/cudnn8'))
    base_conv = status_from_json('experiments/full_repro/weights/weight_conversion_report.json')
    rows.append(row('JITTOR_BASE_WEIGHT_CONVERSION', base_conv, 'tadsr/DAPE/RAM conversion report', 'python3 tools/verify_converted_weights.py'))
    diff_conv = status_from_json('experiments/full_repro/weights/diffusers_weight_verification.json')
    rows.append(row('JITTOR_DIFFUSERS_WEIGHT_CONVERSION', diff_conv, 'diffusers NPZ conversion verification', 'python3 tools/verify_diffusers_npz_weights.py'))
    oracle_tensors = status_from_json('experiments/full_repro/oracle_tensors/smoke/metadata.json')
    rows.append(row('PYTORCH_ORACLE_TENSORS', oracle_tensors, 'preprocess/scheduler oracle tensors exported', 'python3 tools/export_pytorch_oracle_tensors.py --split smoke'))
    scheduler_audit = load_json('experiments/full_repro/scheduler_alignment/audit_tadsr_scheduler_boundary.json')
    scheduler_oracle = load_json('experiments/full_repro/scheduler_alignment/oracle_tensors_scheduler_boundary/scheduler_boundary_oracle_metadata.json')
    scheduler_boundary = load_json('experiments/full_repro/scheduler_alignment/jittor_scheduler_boundary_alignment.json')
    rows.append(row(
        'TADSR_SCHEDULER_USAGE_AUDIT',
        str(scheduler_audit.get('tadsr_scheduler_usage_audit', 'BLOCKED')) if scheduler_audit else 'BLOCKED',
        'official TADSR scheduler usage path audited without running full inference',
        'python3 tools/audit_official_tadsr_scheduler_boundary.py',
    ))
    rows.append(row(
        'TADSR_SCHEDULER_CONFIG_AUDIT',
        str(scheduler_audit.get('tadsr_scheduler_config_audit', 'BLOCKED')) if scheduler_audit else 'BLOCKED',
        'official scheduler class/config/tensor contract audited',
        'python3 tools/audit_official_tadsr_scheduler_boundary.py',
    ))
    rows.append(row(
        'TADSR_SCHEDULER_TIMESTEP_CONTRACT_AUDIT',
        str(scheduler_audit.get('tadsr_scheduler_timestep_contract_audit', 'BLOCKED')) if scheduler_audit else 'BLOCKED',
        'official set_timesteps(1) contract audited',
        'python3 tools/audit_official_tadsr_scheduler_boundary.py',
    ))
    rows.append(row(
        'TADSR_SCHEDULER_STEP_CONTRACT_AUDIT',
        str(scheduler_audit.get('tadsr_scheduler_step_contract_audit', 'BLOCKED')) if scheduler_audit else 'BLOCKED',
        'official scheduler.step one-step contract audited for boundary testing only',
        'python3 tools/audit_official_tadsr_scheduler_boundary.py',
    ))
    rows.append(row(
        'TADSR_SCHEDULER_ORACLE_TENSORS',
        str(scheduler_oracle.get('tadsr_scheduler_oracle_tensors', 'BLOCKED')) if scheduler_oracle else 'BLOCKED',
        'official scheduler-only one-step oracle tensors exported; no denoising loop or VAE decode',
        'python3 tools/export_tadsr_scheduler_boundary_oracle.py',
    ))
    rows.append(row(
        'TADSR_UNET_SCHEDULER_ONE_STEP_ORACLE',
        str(scheduler_oracle.get('tadsr_unet_scheduler_one_step_oracle', 'NOT_APPLICABLE')) if scheduler_oracle else 'BLOCKED',
        'optional one-step scheduler oracle using existing UNet full-forward tensors',
        'python3 tools/export_tadsr_scheduler_boundary_oracle.py',
    ))
    rows.append(row(
        'TADSR_SCHEDULER_TIMESTEPS_ALIGNMENT',
        str(scheduler_boundary.get('tadsr_scheduler_timesteps_alignment', 'BLOCKED')) if scheduler_boundary else 'BLOCKED',
        'Jittor scheduler boundary tester matches official timesteps',
        'USE_CUDA=0 python3 tests_jittor_alignment/test_scheduler_boundary_alignment.py',
    ))
    rows.append(row(
        'TADSR_SCHEDULER_SCALE_MODEL_INPUT_ALIGNMENT',
        str(scheduler_boundary.get('tadsr_scheduler_scale_model_input_alignment', 'BLOCKED')) if scheduler_boundary else 'BLOCKED',
        'scale_model_input is aligned or explicitly no-op for the audited DDPMScheduler path',
        'USE_CUDA=0 python3 tests_jittor_alignment/test_scheduler_boundary_alignment.py',
    ))
    rows.append(row(
        'TADSR_SCHEDULER_STEP_ALIGNMENT',
        str(scheduler_boundary.get('tadsr_scheduler_step_alignment', 'BLOCKED')) if scheduler_boundary else 'BLOCKED',
        'Jittor one-step scheduler boundary output matches official oracle',
        'USE_CUDA=0 python3 tests_jittor_alignment/test_scheduler_boundary_alignment.py',
    ))
    rows.append(row(
        'TADSR_UNET_SCHEDULER_ONE_STEP_ALIGNMENT',
        str(scheduler_boundary.get('tadsr_unet_scheduler_one_step_alignment', 'NOT_APPLICABLE')) if scheduler_boundary else 'BLOCKED',
        'optional existing-UNet-output plus scheduler one-step boundary matches official oracle',
        'USE_CUDA=0 python3 tests_jittor_alignment/test_scheduler_boundary_alignment.py',
    ))
    rows.append(row(
        'TADSR_SCHEDULER_BOUNDARY_ALIGNMENT',
        str(scheduler_boundary.get('tadsr_scheduler_boundary_alignment', 'BLOCKED')) if scheduler_boundary else 'BLOCKED',
        'scheduler boundary / minimal denoising-step contract is audited and aligned; full loop still unopened',
        'USE_CUDA=0 python3 tests_jittor_alignment/test_scheduler_boundary_alignment.py',
    ))
    minimal_audit = load_json('experiments/full_repro/integration_alignment/audit_tadsr_minimal_integration.json')
    minimal_oracle = load_json('experiments/full_repro/integration_alignment/oracle_tensors_minimal_latent_integration/minimal_latent_integration_oracle_metadata.json')
    minimal_alignment = load_json('experiments/full_repro/integration_alignment/jittor_minimal_latent_integration_alignment.json')
    rows.append(row(
        'TADSR_MINIMAL_INTEGRATION_AUDIT',
        str(minimal_audit.get('tadsr_minimal_integration_audit', 'BLOCKED')) if minimal_audit else 'BLOCKED',
        'official minimal TADSR latent one-step path audited from source code; no production full inference',
        'python3 tools/audit_official_tadsr_minimal_integration.py',
    ))
    rows.append(row(
        'TADSR_GET_X0_FROM_RES_AUDIT',
        str(minimal_audit.get('tadsr_get_x0_from_res_audit', 'BLOCKED')) if minimal_audit else 'BLOCKED',
        'official get_x0_from_res formula audited: latent / sqrt(alpha_prod_t) - model_pred',
        'python3 tools/audit_official_tadsr_minimal_integration.py',
    ))
    rows.append(row(
        'TADSR_MINIMAL_LATENT_ONE_STEP_CONTRACT_AUDIT',
        str(minimal_audit.get('tadsr_minimal_latent_one_step_contract_audit', 'BLOCKED')) if minimal_audit else 'BLOCKED',
        'minimal latent-only encode -> UNet -> x0 contract defined without full loop',
        'python3 tools/audit_official_tadsr_minimal_integration.py',
    ))
    rows.append(row(
        'TADSR_MINIMAL_LATENT_ONE_STEP_ORACLE',
        str(minimal_oracle.get('tadsr_minimal_latent_one_step_oracle', 'BLOCKED')) if minimal_oracle else 'BLOCKED',
        'official minimal latent-only oracle exported; no VAE decode or image output',
        'python3 tools/export_tadsr_minimal_latent_integration_oracle.py',
    ))
    rows.append(row(
        'TADSR_GET_X0_FROM_RES_ORACLE',
        str(minimal_oracle.get('tadsr_get_x0_from_res_oracle', 'BLOCKED')) if minimal_oracle else 'BLOCKED',
        'official get_x0_from_res oracle tensor exported',
        'python3 tools/export_tadsr_minimal_latent_integration_oracle.py',
    ))
    rows.append(row(
        'TADSR_MINIMAL_DECODE_BOUNDARY_ORACLE',
        str(minimal_oracle.get('tadsr_minimal_decode_boundary_oracle', 'NOT_APPLICABLE')) if minimal_oracle else 'BLOCKED',
        'official minimal one-step decode/clamp tensor oracle exported; no image output or full inference',
        'python3 tools/export_tadsr_minimal_latent_integration_oracle.py',
    ))
    rows.append(row(
        'TADSR_MINIMAL_VAE_ENCODE_ALIGNMENT',
        str(minimal_alignment.get('tadsr_minimal_vae_encode_alignment', 'BLOCKED')) if minimal_alignment else 'BLOCKED',
        'Jittor actual TimeVAE encode/sample/scale boundary matches minimal oracle',
        'USE_CUDA=0 python3 tests_jittor_alignment/test_minimal_latent_integration_alignment.py',
    ))
    rows.append(row(
        'TADSR_MINIMAL_UNET_MODEL_PRED_ALIGNMENT',
        str(minimal_alignment.get('tadsr_minimal_unet_model_pred_alignment', 'BLOCKED')) if minimal_alignment else 'BLOCKED',
        'Jittor UNet full-forward model_pred matches minimal oracle',
        'USE_CUDA=0 python3 tests_jittor_alignment/test_minimal_latent_integration_alignment.py',
    ))
    rows.append(row(
        'TADSR_GET_X0_FROM_RES_ALIGNMENT',
        str(minimal_alignment.get('tadsr_get_x0_from_res_alignment', 'BLOCKED')) if minimal_alignment else 'BLOCKED',
        'Jittor alpha/get_x0_from_res output matches official oracle',
        'USE_CUDA=0 python3 tests_jittor_alignment/test_minimal_latent_integration_alignment.py',
    ))
    rows.append(row(
        'TADSR_MINIMAL_LATENT_ONE_STEP_ALIGNMENT',
        str(minimal_alignment.get('tadsr_minimal_latent_one_step_alignment', 'BLOCKED')) if minimal_alignment else 'BLOCKED',
        'minimal latent one-step outputs match official oracle',
        'USE_CUDA=0 python3 tests_jittor_alignment/test_minimal_latent_integration_alignment.py',
    ))
    rows.append(row(
        'TADSR_MINIMAL_DECODE_INPUT_ALIGNMENT',
        str(minimal_alignment.get('tadsr_minimal_decode_input_alignment', 'BLOCKED')) if minimal_alignment else 'BLOCKED',
        'Jittor decode_input tensor matches official x0/scaling_factor oracle',
        'USE_CUDA=0 python3 tests_jittor_alignment/test_minimal_latent_integration_alignment.py',
    ))
    rows.append(row(
        'TADSR_MINIMAL_DECODED_OUTPUT_ALIGNMENT',
        str(minimal_alignment.get('tadsr_minimal_decoded_output_alignment', 'BLOCKED')) if minimal_alignment else 'BLOCKED',
        'Jittor TimeVAE actual decoder original_forward output matches official tensor oracle',
        'USE_CUDA=0 python3 tests_jittor_alignment/test_minimal_latent_integration_alignment.py',
    ))
    rows.append(row(
        'TADSR_MINIMAL_FINAL_CLAMPED_OUTPUT_ALIGNMENT',
        str(minimal_alignment.get('tadsr_minimal_final_clamped_output_alignment', 'BLOCKED')) if minimal_alignment else 'BLOCKED',
        'Jittor decoded output clamped to [-1, 1] matches official tensor oracle',
        'USE_CUDA=0 python3 tests_jittor_alignment/test_minimal_latent_integration_alignment.py',
    ))
    rows.append(row(
        'TADSR_MINIMAL_DECODE_BOUNDARY_ALIGNMENT',
        str(minimal_alignment.get('tadsr_minimal_decode_boundary_alignment', 'NOT_APPLICABLE')) if minimal_alignment else 'BLOCKED',
        'minimal one-step decode boundary is PASS only when decode_input, decoded_output and final_clamped_output all align',
        'USE_CUDA=0 python3 tests_jittor_alignment/test_minimal_latent_integration_alignment.py',
    ))
    rows.append(row(
        'TADSR_MINIMAL_LATENT_INTEGRATION_DRY_RUN',
        str(minimal_alignment.get('tadsr_minimal_latent_integration_dry_run', 'BLOCKED')) if minimal_alignment else 'BLOCKED',
        'minimal latent-only dry-run combines VAE encode, UNet model_pred and get_x0_from_res without opening full inference',
        'USE_CUDA=0 python3 tests_jittor_alignment/test_minimal_latent_integration_alignment.py',
    ))
    rows.append(row(
        'TADSR_MINIMAL_ONE_STEP_DECODE_DRY_RUN',
        str(minimal_alignment.get('tadsr_minimal_one_step_decode_dry_run', 'BLOCKED')) if minimal_alignment else 'BLOCKED',
        'minimal one-step dry-run includes tensor-only decode/clamp boundary while still avoiding full inference',
        'USE_CUDA=0 python3 tests_jittor_alignment/test_minimal_latent_integration_alignment.py',
    ))
    for check, path, note in [
        ('JITTOR_PREPROCESS_ALIGNMENT', 'experiments/full_repro/jittor_alignment/preprocess_alignment.json', 'preprocess tensor alignment'),
        ('JITTOR_SCHEDULER_ALIGNMENT', 'experiments/full_repro/jittor_alignment/scheduler_alignment.json', 'scheduler tensor alignment'),
        ('JITTOR_WEIGHT_LOADING_ALIGNMENT', 'experiments/full_repro/jittor_alignment/weight_loading_alignment.json', 'converted NPZ loading check'),
        ('JITTOR_LORA_MAPPING', 'experiments/full_repro/jittor_alignment/lora_key_mapping.json', 'LoRA key table; forward not claimed'),
        ('JITTOR_TIME_VAE_LOADING', 'experiments/full_repro/jittor_alignment/time_vae_loading.json', 'time-VAE loading checklist; forward not claimed'),
    ]:
        rows.append(row(check, status_from_json(path), note, 'USE_CUDA=0 bash scripts/run_jittor_alignment_tests.sh'))
    rows.append(row('OFFICIAL_TIME_VAE_AUDIT', status_from_json('experiments/full_repro/time_vae_alignment/official_time_vae_audit.json'), 'official TimeAwareAutoencoderKL module tree and state-key audit', 'python3 tools/audit_official_time_vae.py'))
    rows.append(row('TIME_VAE_ORACLE_TENSORS', status_from_json('experiments/full_repro/time_vae_alignment/oracle_tensors/time_vae_hook_metadata.json'), 'PyTorch time-VAE block oracle tensors exported', 'python3 tools/export_time_vae_oracle_tensors.py'))
    rows.append(row('TIME_VAE_CONV_IN_ALIGNMENT', status_from_json('experiments/full_repro/time_vae_alignment/jittor_conv_in_alignment.json'), 'Jittor-side conv_in output aligned with PyTorch oracle', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_conv_in_alignment.py'))
    rows.append(row('TIME_VAE_FIRST_BLOCK_ALIGNMENT', status_from_json('experiments/full_repro/time_vae_alignment/jittor_first_block_alignment.json'), 'first TimeAware ResnetBlock2D numerical alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_first_block_alignment.py'))
    rows.append(row('LORA_PAIR_ANALYSIS', status_from_json('experiments/full_repro/lora_alignment/lora_pairs.json'), 'LoRA A/B key-pair discovery', 'python3 tools/analyze_lora_pairs.py'))
    rows.append(row('LORA_MERGE_VALIDATION', status_from_json('experiments/full_repro/lora_alignment/lora_merge_validation.json'), 'at least one LoRA pair merge formula numerically aligned', 'python3 tools/export_lora_merge_oracle.py && python3 tests_jittor_alignment/test_lora_merge_formula.py'))
    rows.append(row('TIME_VAE_LORA_ALIGNMENT_REPORT', 'PASS' if Path('experiments/full_repro/time_vae_alignment/time_vae_lora_alignment_report.md').exists() else 'BLOCKED', 'TimeAware VAE and LoRA progress report', 'python3 scripts/make_time_vae_lora_alignment_report.py'))
    rows.append(row('TIME_VAE_DOWNBLOCK0_AUDIT', status_from_json('experiments/full_repro/time_vae_alignment/downblock0_audit.json'), 'official encoder.down_blocks.0 structure and key mapping audit', 'python3 tools/audit_official_time_vae_downblock0.py'))
    rows.append(row('TIME_VAE_RESNET1_ALIGNMENT', status_from_json('experiments/full_repro/time_vae_alignment/jittor_resnet1_alignment.json'), 'encoder.down_blocks.0.resnets.1 numerical alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_resnet1_alignment.py'))
    rows.append(row('TIME_VAE_DOWNSAMPLER_ALIGNMENT', status_from_json('experiments/full_repro/time_vae_alignment/jittor_downsampler_alignment.json'), 'encoder.down_blocks.0.downsamplers.0 numerical alignment or NOT_APPLICABLE', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_downsampler_alignment.py'))
    rows.append(row('TIME_VAE_DOWNBLOCK0_ALIGNMENT', status_from_json('experiments/full_repro/time_vae_alignment/jittor_downblock0_alignment.json'), 'full encoder.down_blocks.0 stage numerical alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_downblock0_alignment.py'))
    rows.append(row('TIME_VAE_DOWNBLOCK1_AUDIT', status_from_json('experiments/full_repro/time_vae_alignment/downblock1_audit.json'), 'official encoder.down_blocks.1 structure and key mapping audit', 'python3 tools/audit_official_time_vae_downblock1.py'))
    rows.append(row('TIME_VAE_DOWNBLOCK1_ORACLE_TENSORS', status_from_json('experiments/full_repro/time_vae_alignment/oracle_tensors_downblock1/time_vae_downblock1_hook_metadata.json'), 'PyTorch oracle tensors for encoder.down_blocks.1 exported', 'python3 tools/export_time_vae_downblock1_oracle.py'))
    rows.append(row('TIME_VAE_DOWNBLOCK1_RESNET0_ALIGNMENT', status_from_json('experiments/full_repro/time_vae_alignment/jittor_downblock1_resnet0_alignment.json'), 'encoder.down_blocks.1.resnets.0 128->256 channel-change alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_downblock1_resnet0_alignment.py'))
    rows.append(row('TIME_VAE_DOWNBLOCK1_RESNET1_ALIGNMENT', status_from_json('experiments/full_repro/time_vae_alignment/jittor_downblock1_resnet1_alignment.json'), 'encoder.down_blocks.1.resnets.1 numerical alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_downblock1_resnet1_alignment.py'))
    rows.append(row('TIME_VAE_DOWNBLOCK1_DOWNSAMPLER_ALIGNMENT', status_from_json('experiments/full_repro/time_vae_alignment/jittor_downblock1_downsampler_alignment.json'), 'encoder.down_blocks.1.downsamplers.0 numerical alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_downblock1_downsampler_alignment.py'))
    rows.append(row('TIME_VAE_DOWNBLOCK1_ALIGNMENT', status_from_json('experiments/full_repro/time_vae_alignment/jittor_downblock1_alignment.json'), 'full encoder.down_blocks.1 stage numerical alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_downblock1_alignment.py'))
    rows.append(row('TIME_VAE_ENCODER_STAGE01_ALIGNMENT', status_from_json('experiments/full_repro/time_vae_alignment/jittor_encoder_stage01_alignment.json'), 'encoder.conv_in + down_blocks.0 + down_blocks.1 composition alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_encoder_stage01_alignment.py'))
    rows.append(row('TIME_VAE_DOWNBLOCK2_AUDIT', status_from_json('experiments/full_repro/time_vae_alignment/downblock2_audit.json'), 'official encoder.down_blocks.2 structure and key mapping audit', 'python3 tools/audit_official_time_vae_downblock.py --block-index 2'))
    rows.append(row('TIME_VAE_DOWNBLOCK2_ORACLE_TENSORS', status_from_json('experiments/full_repro/time_vae_alignment/oracle_tensors_downblock2/time_vae_downblock2_hook_metadata.json'), 'PyTorch oracle tensors for encoder.down_blocks.2 exported', 'python3 tools/export_time_vae_downblock_oracle.py --block-index 2'))
    rows.append(row('TIME_VAE_DOWNBLOCK2_RESNET0_ALIGNMENT', status_from_json('experiments/full_repro/time_vae_alignment/jittor_downblock2_resnet0_alignment.json'), 'encoder.down_blocks.2.resnets.0 256->512 channel-change alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_downblock2_resnet0_alignment.py'))
    rows.append(row('TIME_VAE_DOWNBLOCK2_RESNET1_ALIGNMENT', status_from_json('experiments/full_repro/time_vae_alignment/jittor_downblock2_resnet1_alignment.json'), 'encoder.down_blocks.2.resnets.1 numerical alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_downblock2_resnet1_alignment.py'))
    rows.append(row('TIME_VAE_DOWNBLOCK2_DOWNSAMPLER_ALIGNMENT', status_from_json('experiments/full_repro/time_vae_alignment/jittor_downblock2_downsampler_alignment.json'), 'encoder.down_blocks.2.downsamplers.0 numerical alignment or NOT_APPLICABLE', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_downblock2_downsampler_alignment.py'))
    rows.append(row('TIME_VAE_DOWNBLOCK2_ALIGNMENT', status_from_json('experiments/full_repro/time_vae_alignment/jittor_downblock2_alignment.json'), 'full encoder.down_blocks.2 stage numerical alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_downblock2_alignment.py'))
    rows.append(row('TIME_VAE_ENCODER_STAGE012_ALIGNMENT', status_from_json('experiments/full_repro/time_vae_alignment/jittor_encoder_stage012_alignment.json'), 'encoder.conv_in + down_blocks.0 + down_blocks.1 + down_blocks.2 composition alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_encoder_stage012_alignment.py'))
    rows.append(row('TIME_VAE_DOWNBLOCK3_AUDIT', status_from_json('experiments/full_repro/time_vae_alignment/downblock3_audit.json'), 'official encoder.down_blocks.3 structure and key mapping audit', 'python3 tools/audit_official_time_vae_downblock.py --block-index 3'))
    rows.append(row('TIME_VAE_DOWNBLOCK3_ORACLE_TENSORS', status_from_json('experiments/full_repro/time_vae_alignment/oracle_tensors_downblock3/time_vae_downblock3_hook_metadata.json'), 'PyTorch oracle tensors for encoder.down_blocks.3 exported', 'python3 tools/export_time_vae_downblock_oracle.py --block-index 3'))
    rows.append(row('TIME_VAE_DOWNBLOCK3_RESNET0_ALIGNMENT', status_from_json('experiments/full_repro/time_vae_alignment/jittor_downblock3_resnet0_alignment.json'), 'encoder.down_blocks.3.resnets.0 numerical alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_downblock3_resnet0_alignment.py'))
    rows.append(row('TIME_VAE_DOWNBLOCK3_RESNET1_ALIGNMENT', status_from_json('experiments/full_repro/time_vae_alignment/jittor_downblock3_resnet1_alignment.json'), 'encoder.down_blocks.3.resnets.1 numerical alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_downblock3_resnet1_alignment.py'))
    rows.append(row('TIME_VAE_DOWNBLOCK3_DOWNSAMPLER_ALIGNMENT', status_from_json('experiments/full_repro/time_vae_alignment/jittor_downblock3_downsampler_alignment.json'), 'encoder.down_blocks.3 downsampler status; NOT_APPLICABLE if absent', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_downblock3_downsampler_alignment.py'))
    rows.append(row('TIME_VAE_DOWNBLOCK3_ALIGNMENT', status_from_json('experiments/full_repro/time_vae_alignment/jittor_downblock3_alignment.json'), 'full encoder.down_blocks.3 stage numerical alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_downblock3_alignment.py'))
    rows.append(row('TIME_VAE_ENCODER_STAGE0123_ALIGNMENT', status_from_json('experiments/full_repro/time_vae_alignment/jittor_encoder_stage0123_alignment.json'), 'encoder.conv_in + down_blocks.0 + down_blocks.1 + down_blocks.2 + down_blocks.3 composition alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_encoder_stage0123_alignment.py'))
    rows.append(row('TIME_VAE_MIDBLOCK_AUDIT', status_from_json('experiments/full_repro/time_vae_alignment/midblock_audit.json'), 'official encoder.mid_block structure and key mapping audit', 'python3 tools/audit_official_time_vae_midblock.py'))
    rows.append(row('TIME_VAE_MIDBLOCK_ORACLE_TENSORS', status_from_json('experiments/full_repro/time_vae_alignment/oracle_tensors_midblock/time_vae_midblock_hook_metadata.json'), 'PyTorch oracle tensors for encoder.mid_block exported', 'python3 tools/export_time_vae_midblock_oracle.py'))
    rows.append(row('TIME_VAE_MIDBLOCK_RESNET0_ALIGNMENT', status_from_json('experiments/full_repro/time_vae_alignment/jittor_midblock_resnet0_alignment.json'), 'encoder.mid_block.resnets.0 numerical alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_midblock_resnet0_alignment.py'))
    rows.append(row('TIME_VAE_MIDBLOCK_ATTENTION_ALIGNMENT', status_from_json('experiments/full_repro/time_vae_alignment/jittor_midblock_attention_alignment.json'), 'encoder.mid_block.attentions.0 numerical alignment or NOT_APPLICABLE', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_midblock_attention_alignment.py'))
    rows.append(row('TIME_VAE_MIDBLOCK_RESNET1_ALIGNMENT', status_from_json('experiments/full_repro/time_vae_alignment/jittor_midblock_resnet1_alignment.json'), 'encoder.mid_block.resnets.1 numerical alignment or NOT_APPLICABLE', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_midblock_resnet1_alignment.py'))
    rows.append(row('TIME_VAE_MIDBLOCK_ALIGNMENT', status_from_json('experiments/full_repro/time_vae_alignment/jittor_midblock_alignment.json'), 'full encoder.mid_block stage numerical alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_midblock_alignment.py'))
    rows.append(row('TIME_VAE_ENCODER_STAGE0123_MID_ALIGNMENT', status_from_json('experiments/full_repro/time_vae_alignment/jittor_encoder_stage0123_mid_alignment.json'), 'encoder.conv_in + down_blocks.0..3 + mid_block composition alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_encoder_stage0123_mid_alignment.py'))
    rows.append(row('TIME_VAE_ENCODER_TAIL_AUDIT', status_from_json('experiments/full_repro/time_vae_alignment/encoder_tail_audit.json'), 'official encoder tail and quant_conv structure/key mapping audit', 'python3 tools/audit_official_time_vae_encoder_tail.py'))
    rows.append(row('TIME_VAE_ENCODER_TAIL_ORACLE_TENSORS', status_from_json('experiments/full_repro/time_vae_alignment/oracle_tensors_encoder_tail/time_vae_encoder_tail_hook_metadata.json'), 'PyTorch oracle tensors for encoder tail and quant_conv exported', 'python3 tools/export_time_vae_encoder_tail_oracle.py'))
    rows.append(row('TIME_VAE_ENCODER_TAIL_NORM_ALIGNMENT', status_from_json('experiments/full_repro/time_vae_alignment/jittor_encoder_tail_norm_alignment.json'), 'encoder.conv_norm_out numerical alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_encoder_tail_norm_alignment.py'))
    rows.append(row('TIME_VAE_ENCODER_TAIL_ACT_ALIGNMENT', status_from_json('experiments/full_repro/time_vae_alignment/jittor_encoder_tail_act_alignment.json'), 'encoder.conv_act numerical alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_encoder_tail_act_alignment.py'))
    rows.append(row('TIME_VAE_ENCODER_TAIL_CONV_OUT_ALIGNMENT', status_from_json('experiments/full_repro/time_vae_alignment/jittor_encoder_tail_conv_out_alignment.json'), 'encoder.conv_out numerical alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_encoder_tail_conv_out_alignment.py'))
    rows.append(row('TIME_VAE_ENCODER_TAIL_ALIGNMENT', status_from_json('experiments/full_repro/time_vae_alignment/jittor_encoder_tail_alignment.json'), 'full encoder tail numerical alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_encoder_tail_alignment.py'))
    rows.append(row('TIME_VAE_QUANT_CONV_ALIGNMENT', status_from_json('experiments/full_repro/time_vae_alignment/jittor_quant_conv_alignment.json'), 'deterministic quant_conv moments tensor alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_quant_conv_alignment.py'))
    rows.append(row('TIME_VAE_ENCODER_STAGE0123_MID_TAIL_ALIGNMENT', status_from_json('experiments/full_repro/time_vae_alignment/jittor_encoder_stage0123_mid_tail_alignment.json'), 'encoder.conv_in + down_blocks.0..3 + mid_block + tail composition alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_encoder_stage0123_mid_tail_alignment.py'))
    rows.append(row('TIME_VAE_ENCODER_STAGE0123_MID_TAIL_QUANT_ALIGNMENT', status_from_json('experiments/full_repro/time_vae_alignment/jittor_encoder_stage0123_mid_tail_quant_alignment.json'), 'encoder-side deterministic path through quant_conv alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_time_vae_encoder_stage0123_mid_tail_quant_alignment.py'))
    decoder_entry_checks = [
        ('TIME_VAE_DECODER_ENTRY_AUDIT', 'experiments/full_repro/time_vae_alignment/audit_time_vae_decoder_entry.json', 'official decoder entry, DGD split/mode and post_quant_conv audit'),
        ('TIME_VAE_DECODER_ENTRY_ORACLE_TENSORS', 'experiments/full_repro/time_vae_alignment/oracle_tensors_decoder_entry/decoder_entry_oracle_metadata.json', 'PyTorch decoder-entry oracle tensors exported'),
        ('TIME_VAE_MOMENTS_SPLIT_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_moments_split_alignment.json', 'posterior moments mean/logvar split and clamp alignment'),
        ('TIME_VAE_POSTERIOR_MODE_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_posterior_mode_alignment.json', 'posterior mode(mean) alignment; no sampling'),
        ('TIME_VAE_POST_QUANT_CONV_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_post_quant_conv_alignment.json', 'post_quant_conv numerical alignment'),
        ('TIME_VAE_DECODER_CONV_IN_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_conv_in_alignment.json', 'decoder.conv_in numerical alignment'),
        ('TIME_VAE_DECODER_ENTRY_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_entry_alignment.json', 'post_quant_conv + decoder.conv_in synthetic latent path alignment'),
        ('TIME_VAE_QUANT_TO_DECODER_ENTRY_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_quant_to_decoder_entry_alignment.json', 'quant moments -> mode -> decoder entry alignment'),
        ('TIME_VAE_ENCODER_TO_DECODER_ENTRY_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_encoder_to_decoder_entry_alignment.json', 'deterministic encoder-to-decoder-entry bridge alignment'),
    ]
    decoder_statuses = []
    for check, path, note in decoder_entry_checks:
        st = status_from_json(path)
        decoder_statuses.append(st)
        rows.append(row(check, st, note, 'USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    decoder_pass = all(st in {'PASS', 'NOT_APPLICABLE'} for st in decoder_statuses)
    rows.append(row('TIME_VAE_DECODER_ENTRY_BRIDGE_ALIGNMENT', 'PASS' if decoder_pass else 'PARTIAL', 'posterior mode + post_quant_conv + decoder.conv_in deterministic bridge; decoder body is checked in later rows', 'continue with decoder.mid_block / up_blocks'))
    decoder_midblock_checks = [
        ('TIME_VAE_DECODER_MIDBLOCK_AUDIT', 'experiments/full_repro/time_vae_alignment/audit_time_vae_decoder_midblock.json', 'official decoder.mid_block topology and key audit'),
        ('TIME_VAE_DECODER_MIDBLOCK_ORACLE_TENSORS', 'experiments/full_repro/time_vae_alignment/oracle_tensors_decoder_midblock/decoder_midblock_oracle_metadata.json', 'PyTorch decoder.mid_block oracle tensors exported'),
        ('TIME_VAE_DECODER_MIDBLOCK_RESNET0_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_midblock_resnet0_alignment.json', 'decoder.mid_block.resnets.0 alignment'),
        ('TIME_VAE_DECODER_MIDBLOCK_ATTENTION_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_midblock_attention_alignment.json', 'decoder.mid_block.attentions.0 alignment or NOT_APPLICABLE'),
        ('TIME_VAE_DECODER_MIDBLOCK_RESNET1_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_midblock_resnet1_alignment.json', 'decoder.mid_block.resnets.1 alignment'),
        ('TIME_VAE_DECODER_MIDBLOCK_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_midblock_alignment.json', 'full decoder.mid_block deterministic alignment'),
        ('TIME_VAE_DECODER_MIDBLOCK_SYNTHETIC_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_midblock_synthetic_alignment.json', 'isolated synthetic hidden -> decoder.mid_block alignment'),
        ('TIME_VAE_DECODER_ENTRY_MIDBLOCK_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_entry_midblock_alignment.json', 'decoder entry -> decoder.mid_block bridge alignment'),
        ('TIME_VAE_QUANT_TO_DECODER_MIDBLOCK_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_quant_to_decoder_midblock_alignment.json', 'quant moments -> posterior mode -> decoder.mid_block bridge alignment'),
        ('TIME_VAE_ENCODER_TO_DECODER_MIDBLOCK_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_encoder_to_decoder_midblock_alignment.json', 'encoder -> quant -> decoder.mid_block deterministic bridge alignment'),
    ]
    mid_statuses = []
    for check, path, note in decoder_midblock_checks:
        st = status_from_json(path)
        mid_statuses.append(st)
        rows.append(row(check, st, note, 'USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    required_mid = [s for s in mid_statuses if s != 'NOT_APPLICABLE']
    mid_pass = bool(required_mid) and all(st == 'PASS' for st in required_mid)
    rows.append(row('TIME_VAE_DECODER_MIDBLOCK_BRIDGE_ALIGNMENT', 'PASS' if mid_pass else 'PARTIAL', 'deterministic decoder.mid_block aligned through isolated, decoder-entry and encoder-to-decoder paths; later decoder stages are checked separately', 'continue with decoder.up_blocks'))
    decoder_upblock0_checks = [
        ('TIME_VAE_DECODER_UPBLOCKS_AUDIT', 'experiments/full_repro/time_vae_alignment/audit_time_vae_decoder_upblocks.json', 'official decoder.up_blocks audit; up_blocks.0 focus plus later-block topology notes'),
        ('TIME_VAE_DECODER_UPBLOCK0_AUDIT', 'experiments/full_repro/time_vae_alignment/audit_time_vae_decoder_upblock0.json', 'official decoder.up_blocks.0 topology and key audit'),
        ('TIME_VAE_DECODER_UPBLOCK0_ORACLE_TENSORS', 'experiments/full_repro/time_vae_alignment/oracle_tensors_decoder_upblock0/decoder_upblock0_oracle_metadata.json', 'PyTorch decoder.up_blocks.0 oracle tensors exported'),
        ('TIME_VAE_DECODER_UPBLOCK0_RESNET0_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_upblock0_resnet0_alignment.json', 'decoder.up_blocks.0.resnets.0 alignment'),
        ('TIME_VAE_DECODER_UPBLOCK0_RESNET1_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_upblock0_resnet1_alignment.json', 'decoder.up_blocks.0.resnets.1 alignment'),
        ('TIME_VAE_DECODER_UPBLOCK0_RESNET2_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_upblock0_resnet2_alignment.json', 'decoder.up_blocks.0.resnets.2 alignment'),
        ('TIME_VAE_DECODER_UPBLOCK0_UPSAMPLER0_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_upblock0_upsampler0_alignment.json', 'decoder.up_blocks.0.upsamplers.0 nearest-2x + conv alignment'),
        ('TIME_VAE_DECODER_UPBLOCK0_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_upblock0_alignment.json', 'full decoder.up_blocks.0 deterministic alignment'),
        ('TIME_VAE_DECODER_UPBLOCK0_SYNTHETIC_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_upblock0_synthetic_alignment.json', 'isolated synthetic hidden -> decoder.up_blocks.0 alignment'),
        ('TIME_VAE_DECODER_MIDBLOCK_UPBLOCK0_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_midblock_upblock0_alignment.json', 'decoder.mid_block -> decoder.up_blocks.0 bridge alignment'),
        ('TIME_VAE_DECODER_ENTRY_MIDBLOCK_UPBLOCK0_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_entry_midblock_upblock0_alignment.json', 'decoder entry -> decoder.mid_block -> decoder.up_blocks.0 bridge alignment'),
        ('TIME_VAE_QUANT_TO_DECODER_UPBLOCK0_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_quant_to_decoder_upblock0_alignment.json', 'quant moments -> posterior mode -> decoder.up_blocks.0 bridge alignment'),
        ('TIME_VAE_ENCODER_TO_DECODER_UPBLOCK0_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_encoder_to_decoder_upblock0_alignment.json', 'encoder -> quant -> decoder.up_blocks.0 deterministic bridge alignment'),
    ]
    upblock0_statuses = []
    for check, path, note in decoder_upblock0_checks:
        st = status_from_json(path)
        upblock0_statuses.append(st)
        rows.append(row(check, st, note, 'USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    required_upblock0 = [s for s in upblock0_statuses if s != 'NOT_APPLICABLE']
    upblock0_pass = bool(required_upblock0) and all(st == 'PASS' for st in required_upblock0)
    rows.append(row('TIME_VAE_DECODER_UPBLOCK0_BRIDGE_ALIGNMENT', 'PASS' if upblock0_pass else 'PARTIAL', 'deterministic decoder.up_blocks.0 aligned through isolated, decoder-entry and encoder-to-decoder paths; later up blocks and tail are checked separately', 'continue with decoder.up_blocks.1'))
    decoder_upblock1_checks = [
        ('TIME_VAE_DECODER_UPBLOCK1_AUDIT', 'experiments/full_repro/time_vae_alignment/audit_time_vae_decoder_upblock1.json', 'official decoder.up_blocks.1 topology and key audit'),
        ('TIME_VAE_DECODER_UPBLOCK1_ORACLE_TENSORS', 'experiments/full_repro/time_vae_alignment/oracle_tensors_decoder_upblock1/decoder_upblock1_oracle_metadata.json', 'PyTorch decoder.up_blocks.1 oracle tensors exported'),
        ('TIME_VAE_DECODER_UPBLOCK1_RESNET0_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_upblock1_resnet0_alignment.json', 'decoder.up_blocks.1.resnets.0 alignment'),
        ('TIME_VAE_DECODER_UPBLOCK1_RESNET1_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_upblock1_resnet1_alignment.json', 'decoder.up_blocks.1.resnets.1 alignment'),
        ('TIME_VAE_DECODER_UPBLOCK1_RESNET2_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_upblock1_resnet2_alignment.json', 'decoder.up_blocks.1.resnets.2 alignment'),
        ('TIME_VAE_DECODER_UPBLOCK1_UPSAMPLER0_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_upblock1_upsampler0_alignment.json', 'decoder.up_blocks.1.upsamplers.0 nearest-2x + conv alignment'),
        ('TIME_VAE_DECODER_UPBLOCK1_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_upblock1_alignment.json', 'full decoder.up_blocks.1 deterministic alignment'),
        ('TIME_VAE_DECODER_UPBLOCK1_SYNTHETIC_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_upblock1_synthetic_alignment.json', 'isolated synthetic hidden -> decoder.up_blocks.1 pressure alignment'),
        ('TIME_VAE_DECODER_UPBLOCK0_UPBLOCK1_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_upblock0_upblock1_alignment.json', 'decoder.up_blocks.0 -> decoder.up_blocks.1 bridge alignment'),
        ('TIME_VAE_DECODER_ENTRY_MIDBLOCK_UPBLOCKS01_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_entry_midblock_upblocks01_alignment.json', 'decoder entry -> decoder.mid_block -> decoder.up_blocks.0 -> decoder.up_blocks.1 bridge alignment'),
        ('TIME_VAE_QUANT_TO_DECODER_UPBLOCK1_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_quant_to_decoder_upblock1_alignment.json', 'quant moments -> posterior mode -> decoder.up_blocks.1 bridge alignment'),
        ('TIME_VAE_ENCODER_TO_DECODER_UPBLOCK1_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_encoder_to_decoder_upblock1_alignment.json', 'encoder -> quant -> decoder.up_blocks.1 deterministic bridge alignment'),
    ]
    upblock1_statuses = []
    for check, path, note in decoder_upblock1_checks:
        st = status_from_json(path)
        upblock1_statuses.append(st)
        rows.append(row(check, st, note, 'USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    required_upblock1 = [s for s in upblock1_statuses if s != 'NOT_APPLICABLE']
    upblock1_pass = bool(required_upblock1) and all(st == 'PASS' for st in required_upblock1)
    rows.append(row('TIME_VAE_DECODER_UPBLOCK1_BRIDGE_ALIGNMENT', 'PASS' if upblock1_pass else 'PARTIAL', 'deterministic decoder.up_blocks.1 aligned through isolated, decoder-entry, quant-to-decoder and encoder-to-decoder paths; later up blocks and tail are checked separately', 'continue with decoder.up_blocks.2'))
    decoder_upblock2_checks = [
        ('TIME_VAE_DECODER_UPBLOCK2_AUDIT', 'experiments/full_repro/time_vae_alignment/audit_time_vae_decoder_upblock2.json', 'official decoder.up_blocks.2 topology and key audit'),
        ('TIME_VAE_DECODER_UPBLOCK2_ORACLE_TENSORS', 'experiments/full_repro/time_vae_alignment/oracle_tensors_decoder_upblock2/decoder_upblock2_oracle_metadata.json', 'PyTorch decoder.up_blocks.2 oracle tensors exported'),
        ('TIME_VAE_DECODER_UPBLOCK2_RESNET0_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_upblock2_resnet0_alignment.json', 'decoder.up_blocks.2.resnets.0 512->256 shortcut alignment'),
        ('TIME_VAE_DECODER_UPBLOCK2_RESNET1_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_upblock2_resnet1_alignment.json', 'decoder.up_blocks.2.resnets.1 alignment'),
        ('TIME_VAE_DECODER_UPBLOCK2_RESNET2_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_upblock2_resnet2_alignment.json', 'decoder.up_blocks.2.resnets.2 alignment'),
        ('TIME_VAE_DECODER_UPBLOCK2_UPSAMPLER0_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_upblock2_upsampler0_alignment.json', 'decoder.up_blocks.2.upsamplers.0 nearest-2x + conv alignment'),
        ('TIME_VAE_DECODER_UPBLOCK2_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_upblock2_alignment.json', 'full decoder.up_blocks.2 deterministic alignment'),
        ('TIME_VAE_DECODER_UPBLOCK2_SYNTHETIC_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_upblock2_synthetic_alignment.json', 'isolated synthetic hidden -> decoder.up_blocks.2 pressure alignment'),
        ('TIME_VAE_DECODER_UPBLOCKS012_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_upblocks012_alignment.json', 'decoder.up_blocks.0 -> decoder.up_blocks.1 -> decoder.up_blocks.2 bridge alignment'),
        ('TIME_VAE_DECODER_ENTRY_MIDBLOCK_UPBLOCKS012_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_entry_midblock_upblocks012_alignment.json', 'decoder entry -> decoder.mid_block -> decoder.up_blocks.0 -> decoder.up_blocks.1 -> decoder.up_blocks.2 bridge alignment'),
        ('TIME_VAE_QUANT_TO_DECODER_UPBLOCK2_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_quant_to_decoder_upblock2_alignment.json', 'quant moments -> posterior mode -> decoder.up_blocks.2 bridge alignment'),
        ('TIME_VAE_ENCODER_TO_DECODER_UPBLOCK2_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_encoder_to_decoder_upblock2_alignment.json', 'encoder -> quant -> decoder.up_blocks.2 deterministic bridge alignment'),
    ]
    upblock2_statuses = []
    for check, path, note in decoder_upblock2_checks:
        st = status_from_json(path)
        upblock2_statuses.append(st)
        rows.append(row(check, st, note, 'USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    required_upblock2 = [s for s in upblock2_statuses if s != 'NOT_APPLICABLE']
    upblock2_pass = bool(required_upblock2) and all(st == 'PASS' for st in required_upblock2)
    rows.append(row('TIME_VAE_DECODER_UPBLOCK2_BRIDGE_ALIGNMENT', 'PASS' if upblock2_pass else 'PARTIAL', 'deterministic decoder.up_blocks.2 aligned through isolated, decoder-entry, quant-to-decoder and encoder-to-decoder paths; decoder.up_blocks.3 and tail are checked separately', 'continue with decoder.up_blocks.3'))
    decoder_upblock3_checks = [
        ('TIME_VAE_DECODER_UPBLOCK3_AUDIT', 'experiments/full_repro/time_vae_alignment/audit_time_vae_decoder_upblock3.json', 'official decoder.up_blocks.3 topology and key audit'),
        ('TIME_VAE_DECODER_UPBLOCK3_ORACLE_TENSORS', 'experiments/full_repro/time_vae_alignment/oracle_tensors_decoder_upblock3/decoder_upblock3_oracle_metadata.json', 'PyTorch decoder.up_blocks.3 oracle tensors exported'),
        ('TIME_VAE_DECODER_UPBLOCK3_RESNET0_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_upblock3_resnet0_alignment.json', 'decoder.up_blocks.3.resnets.0 256->128 shortcut alignment'),
        ('TIME_VAE_DECODER_UPBLOCK3_RESNET1_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_upblock3_resnet1_alignment.json', 'decoder.up_blocks.3.resnets.1 alignment'),
        ('TIME_VAE_DECODER_UPBLOCK3_RESNET2_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_upblock3_resnet2_alignment.json', 'decoder.up_blocks.3.resnets.2 alignment'),
        ('TIME_VAE_DECODER_UPBLOCK3_UPSAMPLER0_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_upblock3_upsampler0_alignment.json', 'decoder.up_blocks.3 has no official upsampler; NOT_APPLICABLE is expected'),
        ('TIME_VAE_DECODER_UPBLOCK3_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_upblock3_alignment.json', 'full decoder.up_blocks.3 deterministic alignment'),
        ('TIME_VAE_DECODER_UPBLOCK3_SYNTHETIC_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_upblock3_synthetic_alignment.json', 'isolated synthetic hidden -> decoder.up_blocks.3 pressure alignment'),
        ('TIME_VAE_DECODER_UPBLOCKS0123_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_upblocks0123_alignment.json', 'decoder.up_blocks.0 -> decoder.up_blocks.1 -> decoder.up_blocks.2 -> decoder.up_blocks.3 bridge alignment'),
        ('TIME_VAE_DECODER_ENTRY_MIDBLOCK_UPBLOCKS0123_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_entry_midblock_upblocks0123_alignment.json', 'decoder entry -> decoder.mid_block -> decoder.up_blocks.0 -> decoder.up_blocks.1 -> decoder.up_blocks.2 -> decoder.up_blocks.3 bridge alignment'),
        ('TIME_VAE_QUANT_TO_DECODER_UPBLOCK3_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_quant_to_decoder_upblock3_alignment.json', 'quant moments -> posterior mode -> decoder.up_blocks.3 bridge alignment'),
        ('TIME_VAE_ENCODER_TO_DECODER_UPBLOCK3_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_encoder_to_decoder_upblock3_alignment.json', 'encoder -> quant -> decoder.up_blocks.3 deterministic bridge alignment'),
    ]
    upblock3_statuses = []
    for check, path, note in decoder_upblock3_checks:
        st = status_from_json(path)
        upblock3_statuses.append(st)
        rows.append(row(check, st, note, 'USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    required_upblock3 = [s for s in upblock3_statuses if s != 'NOT_APPLICABLE']
    upblock3_pass = bool(required_upblock3) and all(st == 'PASS' for st in required_upblock3)
    rows.append(row('TIME_VAE_DECODER_UPBLOCK3_BRIDGE_ALIGNMENT', 'PASS' if upblock3_pass else 'PARTIAL', 'deterministic decoder.up_blocks.3 aligned through isolated, decoder-entry, quant-to-decoder and encoder-to-decoder paths', 'continue with decoder tail'))
    upblocks_alignment_pass = upblock0_pass and upblock1_pass and upblock2_pass and upblock3_pass
    rows.append(row('TIME_VAE_DECODER_UPBLOCKS_ALIGNMENT', 'PASS' if upblocks_alignment_pass else 'PARTIAL', 'decoder up_blocks.0/1/2/3 are aligned as block-level and bridge-level components', 'continue with decoder tail'))
    decoder_tail_checks = [
        ('TIME_VAE_DECODER_TAIL_AUDIT', 'experiments/full_repro/time_vae_alignment/audit_time_vae_decoder_tail.json', 'official decoder tail topology and key audit'),
        ('TIME_VAE_DECODER_TAIL_ORACLE_TENSORS', 'experiments/full_repro/time_vae_alignment/oracle_tensors_decoder_tail/decoder_tail_oracle_metadata.json', 'PyTorch decoder tail oracle tensors exported'),
        ('TIME_VAE_DECODER_TAIL_NORM_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_tail_norm_alignment.json', 'decoder.conv_norm_out alignment'),
        ('TIME_VAE_DECODER_TAIL_ACT_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_tail_act_alignment.json', 'decoder.conv_act SiLU alignment'),
        ('TIME_VAE_DECODER_TAIL_CONV_OUT_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_tail_conv_out_alignment.json', 'decoder.conv_out alignment'),
        ('TIME_VAE_DECODER_TAIL_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_tail_alignment.json', 'full isolated decoder tail alignment'),
        ('TIME_VAE_DECODER_TAIL_SYNTHETIC_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_tail_synthetic_alignment.json', 'isolated synthetic hidden -> decoder tail alignment'),
        ('TIME_VAE_DECODER_UPBLOCKS0123_TO_TAIL_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_upblocks0123_to_tail_alignment.json', 'decoder.up_blocks.0~3 -> decoder tail bridge alignment'),
        ('TIME_VAE_DECODER_ENTRY_MIDBLOCK_UPBLOCKS0123_TAIL_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_decoder_entry_midblock_upblocks0123_tail_alignment.json', 'decoder entry -> mid_block -> up_blocks.0~3 -> tail bridge alignment'),
        ('TIME_VAE_QUANT_TO_DECODER_TAIL_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_quant_to_decoder_tail_alignment.json', 'quant moments -> posterior mode -> decoder tail bridge alignment'),
        ('TIME_VAE_ENCODER_TO_DECODER_TAIL_ALIGNMENT', 'experiments/full_repro/time_vae_alignment/jittor_encoder_to_decoder_tail_alignment.json', 'encoder -> quant -> deterministic decoder tail bridge alignment'),
    ]
    tail_statuses = []
    for check, path, note in decoder_tail_checks:
        st = status_from_json(path)
        tail_statuses.append(st)
        rows.append(row(check, st, note, 'USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    required_tail = [s for s in tail_statuses if s != 'NOT_APPLICABLE']
    tail_pass = bool(required_tail) and all(st == 'PASS' for st in required_tail)
    rows.append(row('TIME_VAE_DECODER_TAIL_BRIDGE_ALIGNMENT', 'PASS' if tail_pass else 'PARTIAL', 'deterministic decoder tail aligned through isolated, decoder-entry, quant-to-decoder and encoder-to-decoder paths', 'continue deterministic decoder wrapper / UNet only after this stays PASS'))
    deterministic_decoder_pass = decoder_pass and mid_pass and upblocks_alignment_pass and tail_pass
    rows.append(row('TIME_VAE_DETERMINISTIC_DECODER_ALIGNMENT', 'PASS' if deterministic_decoder_pass else 'PARTIAL', 'post_quant_conv -> decoder.conv_in -> decoder.mid_block -> decoder.up_blocks.0~3 -> decoder tail deterministic stack alignment', 'continue full UNet audit/export/port after deterministic decoder stack alignment'))
    rows.append(row('TIME_VAE_FULL_DECODER_ALIGNMENT', 'PASS' if deterministic_decoder_pass else 'NOT_COMPLETE', 'deterministic decoder stack module-level alignment only; not stochastic VAE sampling, not full VAE API, and not full TADSR inference', 'continue UNet/LoRA runtime before full inference'))
    rows.append(row('TIME_VAE_DETERMINISTIC_RECONSTRUCTION_ALIGNMENT', 'PASS' if status_from_json('experiments/full_repro/time_vae_alignment/jittor_encoder_to_decoder_tail_alignment.json') == 'PASS' else 'PARTIAL', 'deterministic encoder -> quant_conv moments -> posterior mode -> decoder stack -> tail alignment; no random sampling', 'continue full UNet audit/export/port'))

    timevae_boundary_audit = load_json('experiments/full_repro/vae_alignment/audit_tadsr_timevae_full_boundary.json')
    timevae_boundary_oracle = load_json('experiments/full_repro/vae_alignment/oracle_tensors_timevae_full_boundary/timevae_full_boundary_oracle_metadata.json')
    timevae_boundary_alignment = load_json('experiments/full_repro/vae_alignment/jittor_timevae_full_boundary_alignment.json')
    timevae_audit_status = str(timevae_boundary_audit.get('status', 'BLOCKED')) if timevae_boundary_audit else 'BLOCKED'
    timevae_contract_status = str(timevae_boundary_audit.get('full_boundary_contract', {}).get('status', 'BLOCKED')) if timevae_boundary_audit else 'BLOCKED'
    timevae_oracle_status = str(timevae_boundary_oracle.get('status', 'BLOCKED')) if timevae_boundary_oracle else 'BLOCKED'
    timevae_boundary_status = str(timevae_boundary_alignment.get('status', 'BLOCKED')) if timevae_boundary_alignment else 'BLOCKED'
    timevae_encode_status = str(timevae_boundary_alignment.get('encode_status', timevae_boundary_status)) if timevae_boundary_alignment else 'BLOCKED'
    timevae_decode_status = str(timevae_boundary_alignment.get('decode_status', timevae_boundary_status)) if timevae_boundary_alignment else 'BLOCKED'
    timevae_post_status = str(timevae_boundary_alignment.get('postprocess_status', 'NOT_APPLICABLE')) if timevae_boundary_alignment else 'BLOCKED'
    timevae_full_alignment_candidate = bool(timevae_boundary_oracle.get('timevae_full_alignment_candidate', False)) if timevae_boundary_oracle else False
    timevae_boundary_sufficient = bool(timevae_boundary_oracle.get('this_boundary_is_sufficient_for_full_tadsr_pipeline', False)) if timevae_boundary_oracle else False
    rows.append(row('TIME_VAE_FULL_API_AUDIT', timevae_audit_status, 'official TimeAwareAutoencoderKL API/method/config audit for full boundary', 'python3 tools/audit_official_tadsr_timevae_full_boundary.py'))
    rows.append(row('TIME_VAE_PIPELINE_USAGE_AUDIT', timevae_audit_status, 'official TADSR pipeline VAE usage audited: encode/sample/scale/decode/clamp plus tiled-hook note', 'python3 tools/audit_official_tadsr_timevae_full_boundary.py'))
    rows.append(row('TIME_VAE_FULL_BOUNDARY_CONTRACT_AUDIT', timevae_contract_status, 'minimal deterministic TimeVAE boundary contract defined from official TADSR usage', 'python3 tools/audit_official_tadsr_timevae_full_boundary.py'))
    rows.append(row('TIME_VAE_FULL_BOUNDARY_ORACLE_TENSORS', timevae_oracle_status, 'official PyTorch oracle tensors for TimeVAE encode/sample/scale/decode/clamp boundary exported', 'python3 tools/export_tadsr_timevae_full_boundary_oracle.py'))
    rows.append(row('TIME_VAE_FULL_BOUNDARY_ENCODE_ALIGNMENT', timevae_encode_status, 'Jittor encoder/quant moments alignment inside the TimeVAE full-boundary tester', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_timevae_full_boundary_alignment.py'))
    rows.append(row('TIME_VAE_FULL_BOUNDARY_DECODE_ALIGNMENT', timevae_decode_status, 'Jittor decode path alignment inside the TimeVAE full-boundary tester', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_timevae_full_boundary_alignment.py'))
    rows.append(row('TIME_VAE_FULL_BOUNDARY_POSTPROCESS_ALIGNMENT', timevae_post_status, 'Jittor clamp[-1,1] postprocess alignment for the TimeVAE boundary', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_timevae_full_boundary_alignment.py'))
    rows.append(row('TIME_VAE_FULL_BOUNDARY_ALIGNMENT', timevae_boundary_status, 'alignment-only non-tiled encode/sample/scale/decode/clamp TimeVAE boundary; no scheduler/full inference', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_timevae_full_boundary_alignment.py'))

    timevae_tiled_audit = load_json('experiments/full_repro/vae_alignment/audit_tadsr_vae_tiled_boundary.json')
    timevae_tiled_oracle = load_json('experiments/full_repro/vae_alignment/oracle_tensors_vae_tiled_boundary/vae_tiled_boundary_oracle_metadata.json')
    timevae_tiled_hook_status = str(timevae_tiled_audit.get('hook_audit', {}).get('status', 'BLOCKED')) if timevae_tiled_audit else 'BLOCKED'
    timevae_tiled_pipeline_status = str(timevae_tiled_audit.get('pipeline_usage_audit', {}).get('status', 'BLOCKED')) if timevae_tiled_audit else 'BLOCKED'
    timevae_tiled_contract_status = str(timevae_tiled_audit.get('tiled_boundary_contract', {}).get('status', 'BLOCKED')) if timevae_tiled_audit else 'BLOCKED'
    timevae_tiled_feasibility_status = str(timevae_tiled_audit.get('oracle_feasibility', {}).get('status', 'BLOCKED')) if timevae_tiled_audit else 'BLOCKED'
    timevae_tiled_oracle_status = str(timevae_tiled_oracle.get('status', 'BLOCKED')) if timevae_tiled_oracle else 'BLOCKED'
    timevae_tiled_vs_nontiled_status = str(timevae_tiled_audit.get('tiled_boundary_contract', {}).get('tiled_vs_nontiled_reference', {}).get('status', 'NOT_APPLICABLE')) if timevae_tiled_audit else 'BLOCKED'
    rows.append(row('TIME_VAE_TILED_HOOK_AUDIT', timevae_tiled_hook_status, 'official VAEHook patch targets and dispatch behavior audited; no Jittor tiled implementation claimed', 'python3 tools/audit_official_tadsr_vae_tiled_boundary.py'))
    rows.append(row('TIME_VAE_TILED_PIPELINE_USAGE_AUDIT', timevae_tiled_pipeline_status, 'official TADSR_test VAEHook usage and encode/decode call paths audited', 'python3 tools/audit_official_tadsr_vae_tiled_boundary.py'))
    rows.append(row('TIME_VAE_TILED_BOUNDARY_CONTRACT_AUDIT', timevae_tiled_contract_status, 'official tiled VAE boundary contract audited, including encoder tiling and decoder hook blocker', 'python3 tools/audit_official_tadsr_vae_tiled_boundary.py'))
    rows.append(row('TIME_VAE_TILED_ORACLE_FEASIBILITY', timevae_tiled_feasibility_status, 'official tiled encode/decode oracle feasibility; BLOCKED is acceptable if contract is explicitly diagnosed', 'python3 tools/audit_official_tadsr_vae_tiled_boundary.py'))
    rows.append(row('TIME_VAE_TILED_ORACLE_TENSORS', timevae_tiled_oracle_status, 'official tiled VAE oracle tensors or blocked metadata exported without running scheduler/full inference', 'python3 tools/export_tadsr_vae_tiled_boundary_oracle.py'))
    rows.append(row('TIME_VAE_TILED_VS_NONTILED_REFERENCE_RECORDED', timevae_tiled_vs_nontiled_status, 'tiled-vs-non-tiled comparison recorded only when a truthful official tiled oracle exists', 'python3 tools/audit_official_tadsr_vae_tiled_boundary.py'))

    timevae_actual_audit = load_json('experiments/full_repro/vae_alignment/audit_tadsr_vae_actual_hook_behavior.json')
    timevae_actual_oracle = load_json('experiments/full_repro/vae_alignment/oracle_tensors_vae_actual_hook_behavior/vae_actual_hook_behavior_oracle_metadata.json')
    timevae_actual_contract = load_json('experiments/full_repro/vae_alignment/jittor_timevae_actual_hook_behavior_oracle_contract.json')
    timevae_actual_alignment = load_json('experiments/full_repro/vae_alignment/jittor_timevae_actual_hook_behavior_alignment.json')
    timevae_actual_decision_status = str(timevae_actual_audit.get('decision_status', 'BLOCKED')) if timevae_actual_audit else 'BLOCKED'
    timevae_actual_audit_status = str(timevae_actual_audit.get('status', 'BLOCKED')) if timevae_actual_audit else 'BLOCKED'
    timevae_actual_policy_status = str(timevae_actual_audit.get('official_mirror_policy_status', 'BLOCKED')) if timevae_actual_audit else 'BLOCKED'
    timevae_decoder_tiled_status = str(timevae_actual_audit.get('decoder_hook', {}).get('tiled_path_status', 'BLOCKED')) if timevae_actual_audit else 'BLOCKED'
    timevae_actual_encoder_feasibility = str(timevae_actual_audit.get('encoder_hook', {}).get('tiled_oracle_feasibility', 'BLOCKED')) if timevae_actual_audit else 'BLOCKED'
    timevae_actual_oracle_status = str(timevae_actual_oracle.get('status', 'BLOCKED')) if timevae_actual_oracle else 'BLOCKED'
    timevae_actual_encoder_oracle_status = 'PASS' if timevae_actual_oracle.get('encoder_tiled_path_triggered') is True else ('BLOCKED' if not timevae_actual_oracle else 'NOT_TRIGGERED')
    timevae_actual_decoder_original_status = 'PASS' if timevae_actual_oracle.get('decoder_original_forward_used') is True and timevae_actual_oracle.get('decoder_tiled_path_triggered') is False else ('BLOCKED' if not timevae_actual_oracle else 'FAIL')
    timevae_actual_contract_status = str(timevae_actual_contract.get('status', 'BLOCKED')) if timevae_actual_contract else 'BLOCKED'
    timevae_actual_encoder_tile_queue_alignment = str(timevae_actual_alignment.get('encoder_tile_queue_alignment_status', 'BLOCKED')) if timevae_actual_alignment else 'BLOCKED'
    timevae_actual_encoder_raw_alignment = str(timevae_actual_alignment.get('encoder_tiled_raw_output_status', 'BLOCKED')) if timevae_actual_alignment else 'BLOCKED'
    timevae_actual_encoder_moments_alignment = str(timevae_actual_alignment.get('encoder_tiled_moments_status', 'BLOCKED')) if timevae_actual_alignment else 'BLOCKED'
    timevae_actual_encoder_posterior_alignment = str(timevae_actual_alignment.get('encoder_tiled_posterior_status', 'BLOCKED')) if timevae_actual_alignment else 'BLOCKED'
    timevae_actual_encoder_alignment = str(timevae_actual_alignment.get('encoder_tiled_alignment_status', 'BLOCKED')) if timevae_actual_alignment else 'BLOCKED'
    timevae_actual_decoder_original_alignment = str(timevae_actual_alignment.get('decoder_original_forward_alignment_status', 'BLOCKED')) if timevae_actual_alignment else 'BLOCKED'
    timevae_actual_decoder_hook_alignment = str(timevae_actual_alignment.get('decoder_hook_behavior_alignment_status', 'BLOCKED')) if timevae_actual_alignment else 'BLOCKED'
    timevae_actual_behavior_alignment = str(timevae_actual_alignment.get('actual_vaehook_behavior_alignment_status', 'BLOCKED')) if timevae_actual_alignment else 'BLOCKED'
    timevae_actual_full_boundary_alignment = str(timevae_actual_alignment.get('actual_vaehook_full_boundary_alignment_status', 'BLOCKED')) if timevae_actual_alignment else 'BLOCKED'
    timevae_actual_tiled_decoder_alignment = str(timevae_actual_alignment.get('tiled_decoder_alignment_status', 'BLOCKED')) if timevae_actual_alignment else 'BLOCKED'
    rows.append(row('TIME_VAE_ACTUAL_VAEHOOK_BEHAVIOR_DECISION', timevae_actual_decision_status, 'design decision: mirror official actual VAEHook behavior rather than corrected tiled decoder', 'docs/timevae_tiled_policy_decision.md'))
    rows.append(row('TIME_VAE_ACTUAL_VAEHOOK_BEHAVIOR_AUDIT', timevae_actual_audit_status, 'official actual VAEHook behavior audited: encoder hook can tile, decoder hook dispatches original_forward', 'python3 tools/audit_official_tadsr_vae_actual_hook_behavior.py'))
    rows.append(row('TIME_VAE_OFFICIAL_MIRROR_POLICY', timevae_actual_policy_status, 'mainline policy mirrors official actual behavior; corrected tiled decoder is beyond official and deferred', 'docs/timevae_tiled_policy_decision.md'))
    rows.append(row('TIME_VAE_DECODER_TILED_PATH_STATUS', timevae_decoder_tiled_status, 'official decoder tiled path remains blocked/not reachable because decoder hook has time_vae=False', 'python3 tools/audit_official_tadsr_vae_actual_hook_behavior.py'))
    rows.append(row('TIME_VAE_ACTUAL_ENCODER_TILED_ORACLE_FEASIBILITY', timevae_actual_encoder_feasibility, 'controlled official actual encoder hook/tiled path feasibility', 'python3 tools/audit_official_tadsr_vae_actual_hook_behavior.py'))
    rows.append(row('TIME_VAE_ACTUAL_VAEHOOK_ORACLE_TENSORS', timevae_actual_oracle_status, 'official actual VAEHook behavior oracle tensors exported; not an ideal corrected tiled decoder oracle', 'python3 tools/export_tadsr_vae_actual_hook_behavior_oracle.py'))
    rows.append(row('TIME_VAE_ACTUAL_ENCODER_TILED_ORACLE_TENSORS', timevae_actual_encoder_oracle_status, 'actual encoder hook oracle tensors with encoder tiled path trigger status', 'python3 tools/export_tadsr_vae_actual_hook_behavior_oracle.py'))
    rows.append(row('TIME_VAE_ACTUAL_DECODER_HOOK_ORIGINAL_FORWARD_ORACLE', timevae_actual_decoder_original_status, 'actual decoder hook oracle confirms installed hook uses original_forward and not tiled decode', 'python3 tools/export_tadsr_vae_actual_hook_behavior_oracle.py'))
    rows.append(row('TIME_VAE_ACTUAL_VAEHOOK_ORACLE_CONTRACT_TEST', timevae_actual_contract_status, 'metadata-only contract test for official actual VAEHook behavior oracle', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_timevae_actual_hook_behavior_oracle_contract.py'))
    rows.append(row('TIME_VAE_ACTUAL_ENCODER_TILE_QUEUE_ALIGNMENT', timevae_actual_encoder_tile_queue_alignment, 'Jittor actual-hook wrapper mirrors official encoder VAEHook task queue, tile split/crop/write semantics, and cross-tile GroupNorm bookkeeping', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_timevae_actual_hook_behavior_alignment.py'))
    rows.append(row('TIME_VAE_ACTUAL_ENCODER_TILED_RAW_OUTPUT_ALIGNMENT', timevae_actual_encoder_raw_alignment, 'Jittor tiled encoder raw output before quant_conv against official actual VAEHook raw encoder oracle', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_timevae_actual_hook_behavior_alignment.py'))
    rows.append(row('TIME_VAE_ACTUAL_ENCODER_TILED_MOMENTS_ALIGNMENT', timevae_actual_encoder_moments_alignment, 'Jittor actual-hook wrapper computes official tiled encoder output plus quant_conv moments parity', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_timevae_actual_hook_behavior_alignment.py'))
    rows.append(row('TIME_VAE_ACTUAL_ENCODER_TILED_POSTERIOR_ALIGNMENT', timevae_actual_encoder_posterior_alignment, 'Jittor actual-hook wrapper posterior tensors against official actual encoder oracle', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_timevae_actual_hook_behavior_alignment.py'))
    rows.append(row('TIME_VAE_ACTUAL_ENCODER_TILED_ALIGNMENT', timevae_actual_encoder_alignment, 'official-actual encoder hook behavior alignment through tiled task queue and quant_conv', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_timevae_actual_hook_behavior_alignment.py'))
    rows.append(row('TIME_VAE_ACTUAL_DECODER_ORIGINAL_FORWARD_ALIGNMENT', timevae_actual_decoder_original_alignment, 'official-actual decoder hook dispatches to original_forward and aligns numerically', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_timevae_actual_hook_behavior_alignment.py'))
    rows.append(row('TIME_VAE_ACTUAL_DECODER_HOOK_BEHAVIOR_ALIGNMENT', timevae_actual_decoder_hook_alignment, 'decoder hook behavior parity: installed hook, time_vae=False, original_forward path', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_timevae_actual_hook_behavior_alignment.py'))
    rows.append(row('TIME_VAE_ACTUAL_VAEHOOK_BEHAVIOR_ALIGNMENT', timevae_actual_behavior_alignment, 'Jittor official-actual VAEHook behavior wrapper/tester status; does not invent a corrected tiled decoder', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_timevae_actual_hook_behavior_alignment.py'))
    rows.append(row('TIME_VAE_ACTUAL_VAEHOOK_FULL_BOUNDARY_ALIGNMENT', timevae_actual_full_boundary_alignment, 'actual hook full boundary from encode/sample/scale/decode/clamp; decoder remains official original_forward and full TADSR inference remains closed', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_timevae_actual_hook_behavior_alignment.py'))
    rows.append(row('TIME_VAE_TILED_DECODER_ALIGNMENT', timevae_actual_tiled_decoder_alignment, 'not applicable for official actual behavior because decoder VAEHook uses original_forward, not vae_tile_forward', 'do not force decoder time_vae=True without a separate corrected-decoder experiment'))


    unet_entry_checks = [
        ('TADSR_UNET_OVERVIEW_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_entry.json', 'official UNet overview audit: config, input contract, LoRA state, entry modules'),
        ('TADSR_UNET_ENTRY_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_entry.json', 'official UNet entry audit: conv_in, time_proj, time_embedding'),
        ('TADSR_UNET_ENTRY_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/unet_entry_oracle_metadata.json', 'PyTorch oracle tensors for UNet entry exported'),
        ('TADSR_UNET_ENTRY_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/unet_entry_oracle_metadata.json', 'effective static conv_in + time_embedding weights exported'),
        ('TADSR_UNET_CENTER_INPUT_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_conv_in_alignment.json', 'center_input_sample bridge alignment'),
        ('TADSR_UNET_CONV_IN_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_conv_in_alignment.json', 'UNet conv_in effective-weight alignment'),
        ('TADSR_UNET_TIME_PROJ_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_time_proj_alignment.json', 'UNet Timesteps positional projection alignment'),
        ('TADSR_UNET_TIME_EMBED_LINEAR1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_time_embedding_alignment.json', 'UNet time_embedding.linear_1 alignment'),
        ('TADSR_UNET_TIME_EMBED_ACT_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_time_embedding_alignment.json', 'UNet time_embedding SiLU activation alignment'),
        ('TADSR_UNET_TIME_EMBED_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_time_embedding_alignment.json', 'UNet full timestep embedding MLP alignment'),
        ('TADSR_UNET_ENTRY_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_alignment.json', 'UNet entry aggregate alignment before down_blocks.0'),
    ]
    unet_entry_statuses = []
    for check, path, note in unet_entry_checks:
        st = status_from_json(path)
        unet_entry_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_entry.py && python3 tools/export_tadsr_unet_entry_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    vae_mode_status = status_from_json('experiments/full_repro/unet_alignment/jittor_unet_entry_vae_mode_alignment.json', default='NOT_APPLICABLE')
    rows.append(row('TADSR_UNET_ENTRY_VAE_MODE_ALIGNMENT', vae_mode_status, 'optional VAE-mode latent bridge; NOT_APPLICABLE until full TimeAware VAE API opens', 'continue full VAE API before enabling this optional bridge'))
    entry_required = [s for s in unet_entry_statuses if s not in {'NOT_APPLICABLE'}]
    unet_entry_pass = bool(entry_required) and all(st == 'PASS' for st in entry_required)
    rows.append(row('TADSR_UNET_ENTRY_ALIGNMENT_AGGREGATE', 'PASS' if unet_entry_pass else 'PARTIAL', 'UNet center input, conv_in, time_proj and time_embedding aligned; full UNet blocks are still unopened', 'Continue TADSR UNet up_blocks.0.upsamplers.0 audit/export/port after up_blocks.0.resnets.2 alignment; keep full UNet forward and full inference NotImplemented until all up_blocks and LoRA runtime integration pass.'))

    resnet0_checks = [
        ('TADSR_UNET_DOWNBLOCK0_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_downblock0_resnet0.json', 'official UNet down_blocks.0 overview audit'),
        ('TADSR_UNET_DOWNBLOCK0_RESNET0_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_downblock0_resnet0.json', 'official down_blocks.0.resnets.0 audit'),
        ('TADSR_UNET_DOWNBLOCK0_RESNET0_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_resnet0/unet_downblock0_resnet0_oracle_metadata.json', 'PyTorch oracle tensors for down_blocks.0.resnets.0 exported'),
        ('TADSR_UNET_DOWNBLOCK0_RESNET0_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_resnet0/unet_downblock0_resnet0_oracle_metadata.json', 'effective static weights for down_blocks.0.resnets.0 exported locally'),
        ('TADSR_UNET_DOWNBLOCK0_RESNET0_NORM1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_resnet0_alignment.json', 'down_blocks.0.resnets.0 norm1 alignment'),
        ('TADSR_UNET_DOWNBLOCK0_RESNET0_CONV1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_resnet0_alignment.json', 'down_blocks.0.resnets.0 conv1 alignment'),
        ('TADSR_UNET_DOWNBLOCK0_RESNET0_TEMB_PROJ_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_resnet0_alignment.json', 'down_blocks.0.resnets.0 time_emb_proj alignment'),
        ('TADSR_UNET_DOWNBLOCK0_RESNET0_CONV2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_resnet0_alignment.json', 'down_blocks.0.resnets.0 conv2 alignment'),
        ('TADSR_UNET_DOWNBLOCK0_RESNET0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_resnet0_alignment.json', 'entry hidden/temb -> down_blocks.0.resnets.0 alignment'),
        ('TADSR_UNET_DOWNBLOCK0_RESNET0_SYNTHETIC_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_resnet0_synthetic_alignment.json', 'isolated synthetic hidden/temb -> down_blocks.0.resnets.0 alignment'),
        ('TADSR_UNET_ENTRY_TO_DOWNBLOCK0_RESNET0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_to_downblock0_resnet0_alignment.json', 'UNet entry -> down_blocks.0.resnets.0 bridge alignment'),
    ]
    resnet0_statuses = []
    for check, path, note in resnet0_checks:
        st = status_from_json(path)
        resnet0_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_downblock0_resnet0.py && python3 tools/export_tadsr_unet_downblock0_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    shortcut_status = status_from_json('experiments/full_repro/unet_alignment/jittor_unet_downblock0_resnet0_shortcut_alignment.json', default='NOT_APPLICABLE')
    rows.append(row('TADSR_UNET_DOWNBLOCK0_RESNET0_SHORTCUT_ALIGNMENT', shortcut_status, 'down_blocks.0.resnets.0 shortcut alignment; NOT_APPLICABLE when identity shortcut is official', 'continue if shortcut appears in later channel-changing ResNet'))
    required_resnet0 = [s for s in resnet0_statuses if s not in {'NOT_APPLICABLE'}]
    resnet0_pass = bool(required_resnet0) and all(st == 'PASS' for st in required_resnet0) and shortcut_status in {'PASS', 'NOT_APPLICABLE'}
    rows.append(row('TADSR_UNET_DOWNBLOCK0_RESNET0_BRIDGE_ALIGNMENT', 'PASS' if resnet0_pass else 'PARTIAL', 'first UNet down_blocks.0 leaf ResNet aligned in isolated and entry-bridge paths only; full down_blocks.0 remains unopened', 'Continue TADSR UNet up_blocks.0.upsamplers.0 audit/export/port after up_blocks.0.resnets.2 alignment; keep full UNet forward and full inference NotImplemented until all up_blocks and LoRA runtime integration pass.'))
    attention0_checks = [
        ('TADSR_UNET_DOWNBLOCK0_ATTENTION0_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_downblock0_attention0.json', 'official down_blocks.0.attentions.0 audit'),
        ('TADSR_UNET_DOWNBLOCK0_ATTENTION0_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_attention0/unet_downblock0_attention0_oracle_metadata.json', 'PyTorch oracle tensors for down_blocks.0.attentions.0 exported'),
        ('TADSR_UNET_DOWNBLOCK0_ATTENTION0_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_attention0/unet_downblock0_attention0_oracle_metadata.json', 'effective static weights for down_blocks.0.attentions.0 exported locally'),
        ('TADSR_UNET_DOWNBLOCK0_ATTENTION0_NORM_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_attention0_proj_alignment.json', 'attention0 top-level GroupNorm alignment'),
        ('TADSR_UNET_DOWNBLOCK0_ATTENTION0_PROJ_IN_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_attention0_proj_alignment.json', 'attention0 proj_in alignment'),
        ('TADSR_UNET_DOWNBLOCK0_ATTENTION0_SEQUENCE_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_attention0_proj_alignment.json', 'attention0 NCHW-to-sequence alignment'),
        ('TADSR_UNET_DOWNBLOCK0_ATTENTION0_ATTN1_QKV_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_attention0_attn1_alignment.json', 'attention0 transformer0 attn1 QKV alignment'),
        ('TADSR_UNET_DOWNBLOCK0_ATTENTION0_ATTN1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_attention0_attn1_alignment.json', 'attention0 transformer0 self-attention output alignment'),
        ('TADSR_UNET_DOWNBLOCK0_ATTENTION0_AFTER_ATTN1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_attention0_attn1_alignment.json', 'attention0 transformer0 after-attn1 residual alignment'),
        ('TADSR_UNET_DOWNBLOCK0_ATTENTION0_ATTN2_QKV_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_attention0_attn2_alignment.json', 'attention0 transformer0 attn2 QKV alignment'),
        ('TADSR_UNET_DOWNBLOCK0_ATTENTION0_ATTN2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_attention0_attn2_alignment.json', 'attention0 transformer0 cross-attention output alignment'),
        ('TADSR_UNET_DOWNBLOCK0_ATTENTION0_AFTER_ATTN2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_attention0_attn2_alignment.json', 'attention0 transformer0 after-attn2 residual alignment'),
        ('TADSR_UNET_DOWNBLOCK0_ATTENTION0_FF_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_attention0_ff_alignment.json', 'attention0 transformer0 feed-forward output alignment'),
        ('TADSR_UNET_DOWNBLOCK0_ATTENTION0_TRANSFORMER0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_attention0_ff_alignment.json', 'attention0 full transformer block alignment'),
        ('TADSR_UNET_DOWNBLOCK0_ATTENTION0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_attention0_alignment.json', 'isolated down_blocks.0.attentions.0 alignment'),
        ('TADSR_UNET_ENTRY_RESNET0_ATTENTION0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_resnet0_attention0_alignment.json', 'UNet entry -> resnet0 -> attention0 bridge alignment'),
    ]
    attention0_statuses = []
    for check, path, note in attention0_checks:
        st = status_from_json(path)
        attention0_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_downblock0_attention0.py && python3 tools/export_tadsr_unet_downblock0_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    required_attention0 = [s for s in attention0_statuses if s not in {'NOT_APPLICABLE'}]
    attention0_pass = bool(required_attention0) and all(st == 'PASS' for st in required_attention0)
    rows.append(row('TADSR_UNET_DOWNBLOCK0_ATTENTION0_BRIDGE_ALIGNMENT', 'PASS' if attention0_pass else 'PARTIAL', 'down_blocks.0.attentions.0 aligned in isolated and entry-resnet0 bridge paths only; full down_blocks.0 remains unopened', 'Continue TADSR UNet up_blocks.0.upsamplers.0 audit/export/port after up_blocks.0.resnets.2 alignment; keep full UNet forward and full inference NotImplemented until all up_blocks and LoRA runtime integration pass.'))


    resnet1_checks = [
        ('TADSR_UNET_DOWNBLOCK0_RESNET1_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_downblock0_resnet1.json', 'official down_blocks.0.resnets.1 audit'),
        ('TADSR_UNET_DOWNBLOCK0_RESNET1_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_resnet1/unet_downblock0_resnet1_oracle_metadata.json', 'PyTorch oracle tensors for down_blocks.0.resnets.1 exported'),
        ('TADSR_UNET_DOWNBLOCK0_RESNET1_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_resnet1/unet_downblock0_resnet1_oracle_metadata.json', 'effective static weights for down_blocks.0.resnets.1 exported locally'),
        ('TADSR_UNET_DOWNBLOCK0_RESNET1_NORM1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_resnet1_alignment.json', 'down_blocks.0.resnets.1 norm1 alignment'),
        ('TADSR_UNET_DOWNBLOCK0_RESNET1_CONV1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_resnet1_alignment.json', 'down_blocks.0.resnets.1 conv1 alignment'),
        ('TADSR_UNET_DOWNBLOCK0_RESNET1_TEMB_PROJ_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_resnet1_alignment.json', 'down_blocks.0.resnets.1 time_emb_proj alignment'),
        ('TADSR_UNET_DOWNBLOCK0_RESNET1_CONV2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_resnet1_alignment.json', 'down_blocks.0.resnets.1 conv2 alignment'),
        ('TADSR_UNET_DOWNBLOCK0_RESNET1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_resnet1_alignment.json', 'entry-attention hidden/temb -> down_blocks.0.resnets.1 alignment'),
        ('TADSR_UNET_DOWNBLOCK0_RESNET1_SYNTHETIC_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_resnet1_synthetic_alignment.json', 'isolated synthetic hidden/temb -> down_blocks.0.resnets.1 alignment'),
        ('TADSR_UNET_ENTRY_RESNET0_ATTENTION0_RESNET1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_resnet0_attention0_resnet1_alignment.json', 'UNet entry -> resnet0 -> attention0 -> resnet1 bridge alignment'),
    ]
    resnet1_statuses = []
    for check, path, note in resnet1_checks:
        st = status_from_json(path)
        resnet1_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_downblock0_resnet1.py && python3 tools/export_tadsr_unet_downblock0_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    resnet1_shortcut_status = status_from_json('experiments/full_repro/unet_alignment/jittor_unet_downblock0_resnet1_shortcut_alignment.json', default='NOT_APPLICABLE')
    rows.append(row('TADSR_UNET_DOWNBLOCK0_RESNET1_SHORTCUT_ALIGNMENT', resnet1_shortcut_status, 'down_blocks.0.resnets.1 shortcut alignment; NOT_APPLICABLE when identity shortcut is official', 'continue if shortcut appears in later channel-changing ResNet'))
    required_resnet1 = [s for s in resnet1_statuses if s not in {'NOT_APPLICABLE'}]
    resnet1_pass = bool(required_resnet1) and all(st == 'PASS' for st in required_resnet1) and resnet1_shortcut_status in {'PASS', 'NOT_APPLICABLE'}
    rows.append(row('TADSR_UNET_DOWNBLOCK0_RESNET1_BRIDGE_ALIGNMENT', 'PASS' if resnet1_pass else 'PARTIAL', 'down_blocks.0.resnets.1 aligned in isolated and entry-resnet0-attention0 bridge paths only; full down_blocks.0 remains unopened', 'Continue TADSR UNet up_blocks.0.upsamplers.0 audit/export/port after up_blocks.0.resnets.2 alignment; keep full UNet forward and full inference NotImplemented until all up_blocks and LoRA runtime integration pass.'))


    attention1_checks = [
        ('TADSR_UNET_DOWNBLOCK0_ATTENTION1_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_downblock0_attention1.json', 'official down_blocks.0.attentions.1 audit'),
        ('TADSR_UNET_DOWNBLOCK0_ATTENTION1_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_attention1/unet_downblock0_attention1_oracle_metadata.json', 'PyTorch oracle tensors for down_blocks.0.attentions.1 exported'),
        ('TADSR_UNET_DOWNBLOCK0_ATTENTION1_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_attention1/unet_downblock0_attention1_oracle_metadata.json', 'effective static weights for down_blocks.0.attentions.1 exported locally'),
        ('TADSR_UNET_DOWNBLOCK0_ATTENTION1_NORM_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_attention1_proj_alignment.json', 'attention1 top-level GroupNorm alignment'),
        ('TADSR_UNET_DOWNBLOCK0_ATTENTION1_PROJ_IN_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_attention1_proj_alignment.json', 'attention1 proj_in alignment'),
        ('TADSR_UNET_DOWNBLOCK0_ATTENTION1_SEQUENCE_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_attention1_proj_alignment.json', 'attention1 NCHW-to-sequence alignment'),
        ('TADSR_UNET_DOWNBLOCK0_ATTENTION1_ATTN1_QKV_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_attention1_attn1_alignment.json', 'attention1 transformer0 attn1 QKV alignment'),
        ('TADSR_UNET_DOWNBLOCK0_ATTENTION1_ATTN1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_attention1_attn1_alignment.json', 'attention1 transformer0 self-attention output alignment'),
        ('TADSR_UNET_DOWNBLOCK0_ATTENTION1_AFTER_ATTN1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_attention1_attn1_alignment.json', 'attention1 transformer0 after-attn1 residual alignment'),
        ('TADSR_UNET_DOWNBLOCK0_ATTENTION1_ATTN2_QKV_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_attention1_attn2_alignment.json', 'attention1 transformer0 attn2 QKV alignment'),
        ('TADSR_UNET_DOWNBLOCK0_ATTENTION1_ATTN2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_attention1_attn2_alignment.json', 'attention1 transformer0 cross-attention output alignment'),
        ('TADSR_UNET_DOWNBLOCK0_ATTENTION1_AFTER_ATTN2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_attention1_attn2_alignment.json', 'attention1 transformer0 after-attn2 residual alignment'),
        ('TADSR_UNET_DOWNBLOCK0_ATTENTION1_FF_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_attention1_ff_alignment.json', 'attention1 transformer0 feed-forward output alignment'),
        ('TADSR_UNET_DOWNBLOCK0_ATTENTION1_TRANSFORMER0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_attention1_ff_alignment.json', 'attention1 full transformer block alignment'),
        ('TADSR_UNET_DOWNBLOCK0_ATTENTION1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_attention1_alignment.json', 'isolated down_blocks.0.attentions.1 alignment'),
        ('TADSR_UNET_ENTRY_RESNET0_ATTENTION0_RESNET1_ATTENTION1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_resnet0_attention0_resnet1_attention1_alignment.json', 'UNet entry -> resnet0 -> attention0 -> resnet1 -> attention1 bridge alignment'),
    ]
    attention1_statuses = []
    for check, path, note in attention1_checks:
        st = status_from_json(path)
        attention1_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_downblock0_attention1.py && python3 tools/export_tadsr_unet_downblock0_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    required_attention1 = [s for s in attention1_statuses if s not in {'NOT_APPLICABLE'}]
    attention1_pass = bool(required_attention1) and all(st == 'PASS' for st in required_attention1)
    rows.append(row('TADSR_UNET_DOWNBLOCK0_ATTENTION1_BRIDGE_ALIGNMENT', 'PASS' if attention1_pass else 'PARTIAL', 'down_blocks.0.attentions.1 aligned in isolated and entry-resnet0-attention0-resnet1 bridge paths only; full down_blocks.0 remains unopened', 'Continue TADSR UNet up_blocks.0.upsamplers.0 audit/export/port after up_blocks.0.resnets.2 alignment; keep full UNet forward and full inference NotImplemented until all up_blocks and LoRA runtime integration pass.'))
    pre_downsampler_pass = resnet0_pass and attention0_pass and resnet1_pass and attention1_pass
    rows.append(row('TADSR_UNET_DOWNBLOCK0_PRE_DOWNSAMPLER_ALIGNMENT', 'PASS' if pre_downsampler_pass else 'PARTIAL', 'down_blocks.0 path through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 is aligned', 'continue with down_blocks.0.downsamplers.0 if this is not PASS'))

    downsampler_checks = [
        ('TADSR_UNET_DOWNBLOCK0_DOWNSAMPLER_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_downblock0_downsampler.json', 'official down_blocks.0.downsamplers.0 audit'),
        ('TADSR_UNET_DOWNBLOCK0_LOCAL_FORWARD_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_downblock0_downsampler.json', 'official local down_blocks.0 manual-chain audit'),
        ('TADSR_UNET_DOWNBLOCK0_DOWNSAMPLER_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_downsampler/unet_downblock0_downsampler_oracle_metadata.json', 'PyTorch oracle tensors for down_blocks.0.downsamplers.0 and local down_blocks.0 exported'),
        ('TADSR_UNET_DOWNBLOCK0_DOWNSAMPLER_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_downsampler/unet_downblock0_downsampler_oracle_metadata.json', 'effective static weights for down_blocks.0.downsamplers.0 exported locally'),
        ('TADSR_UNET_DOWNBLOCK0_DOWNSAMPLER_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_downsampler_alignment.json', 'isolated down_blocks.0.downsamplers.0 alignment'),
        ('TADSR_UNET_DOWNBLOCK0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock0_alignment.json', 'full local UNet down_blocks.0 alignment: entry -> resnet0 -> attention0 -> resnet1 -> attention1 -> downsampler'),
    ]
    downsampler_statuses = []
    for check, path, note in downsampler_checks:
        st = status_from_json(path)
        downsampler_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_downblock0_downsampler.py && python3 tools/export_tadsr_unet_downblock0_downsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    downsampler_pass = all(st == 'PASS' for st in downsampler_statuses)
    rows.append(row('TADSR_UNET_DOWNBLOCK0_BRIDGE_ALIGNMENT', 'PASS' if pre_downsampler_pass and downsampler_pass else 'PARTIAL', 'complete local down_blocks.0 bridge aligned through downsampler; later UNet blocks remain unopened', 'continue with down_blocks.1.attentions.0'))


    downblock1_resnet0_checks = [
        ('TADSR_UNET_DOWNBLOCK1_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_downblock1_resnet0.json', 'official UNet down_blocks.1 overview audit, resnets.0 baseline context'),
        ('TADSR_UNET_DOWNBLOCK1_RESNET0_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_downblock1_resnet0.json', 'official down_blocks.1.resnets.0 audit'),
        ('TADSR_UNET_DOWNBLOCK1_RESNET0_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock1_resnet0/unet_downblock1_resnet0_oracle_metadata.json', 'PyTorch oracle tensors for down_blocks.1.resnets.0 exported'),
        ('TADSR_UNET_DOWNBLOCK1_RESNET0_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock1_resnet0/unet_downblock1_resnet0_oracle_metadata.json', 'effective static weights for down_blocks.1.resnets.0 exported locally'),
        ('TADSR_UNET_DOWNBLOCK1_RESNET0_NORM1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_resnet0_alignment.json', 'down_blocks.1.resnets.0 norm1 alignment'),
        ('TADSR_UNET_DOWNBLOCK1_RESNET0_CONV1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_resnet0_alignment.json', 'down_blocks.1.resnets.0 conv1 alignment'),
        ('TADSR_UNET_DOWNBLOCK1_RESNET0_TEMB_PROJ_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_resnet0_alignment.json', 'down_blocks.1.resnets.0 time_emb_proj alignment'),
        ('TADSR_UNET_DOWNBLOCK1_RESNET0_CONV2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_resnet0_alignment.json', 'down_blocks.1.resnets.0 conv2 alignment'),
        ('TADSR_UNET_DOWNBLOCK1_RESNET0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_resnet0_alignment.json', 'down_blocks.1.resnets.0 local alignment'),
        ('TADSR_UNET_DOWNBLOCK1_RESNET0_SYNTHETIC_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_resnet0_synthetic_alignment.json', 'isolated synthetic hidden/temb -> down_blocks.1.resnets.0 alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_RESNET0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblock0_downblock1_resnet0_alignment.json', 'UNet entry -> full local down_blocks.0 -> down_blocks.1.resnets.0 bridge alignment'),
    ]
    downblock1_resnet0_statuses = []
    for check, path, note in downblock1_resnet0_checks:
        st = status_from_json(path)
        downblock1_resnet0_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_downblock1_resnet0.py && python3 tools/export_tadsr_unet_downblock1_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    downblock1_resnet0_shortcut_status = status_from_json('experiments/full_repro/unet_alignment/jittor_unet_downblock1_resnet0_shortcut_alignment.json', default=status_from_json('experiments/full_repro/unet_alignment/jittor_unet_downblock1_resnet0_alignment.json', default='BLOCKED'))
    rows.append(row('TADSR_UNET_DOWNBLOCK1_RESNET0_SHORTCUT_ALIGNMENT', downblock1_resnet0_shortcut_status, 'down_blocks.1.resnets.0 shortcut alignment; channel-changing ResNet should include conv_shortcut', 'continue only if shortcut remains PASS'))
    required_downblock1_resnet0 = [s for s in downblock1_resnet0_statuses if s not in {'NOT_APPLICABLE'}]
    downblock1_resnet0_pass = bool(required_downblock1_resnet0) and all(st == 'PASS' for st in required_downblock1_resnet0) and downblock1_resnet0_shortcut_status in {'PASS', 'NOT_APPLICABLE'}
    rows.append(row('TADSR_UNET_DOWNBLOCK1_RESNET0_BRIDGE_ALIGNMENT', 'PASS' if downblock1_resnet0_pass else 'PARTIAL', 'down_blocks.1.resnets.0 aligned after complete local down_blocks.0 bridge; remaining down_blocks.1 modules are intentionally unopened', 'continue with down_blocks.1.attentions.0 only'))

    downblock1_attention0_checks = [
        ('TADSR_UNET_DOWNBLOCK1_ATTENTION0_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_downblock1_attention0.json', 'official down_blocks.1.attentions.0 audit'),
        ('TADSR_UNET_DOWNBLOCK1_ATTENTION0_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock1_attention0/unet_downblock1_attention0_oracle_metadata.json', 'PyTorch oracle tensors for down_blocks.1.attentions.0 exported'),
        ('TADSR_UNET_DOWNBLOCK1_ATTENTION0_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock1_attention0/unet_downblock1_attention0_oracle_metadata.json', 'effective static weights for down_blocks.1.attentions.0 exported locally'),
        ('TADSR_UNET_DOWNBLOCK1_ATTENTION0_NORM_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_attention0_proj_alignment.json', 'down_blocks.1.attentions.0 top-level norm alignment'),
        ('TADSR_UNET_DOWNBLOCK1_ATTENTION0_PROJ_IN_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_attention0_proj_alignment.json', 'down_blocks.1.attentions.0 proj_in alignment'),
        ('TADSR_UNET_DOWNBLOCK1_ATTENTION0_SEQUENCE_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_attention0_proj_alignment.json', 'down_blocks.1.attentions.0 NCHW-to-sequence alignment'),
        ('TADSR_UNET_DOWNBLOCK1_ATTENTION0_ATTN1_QKV_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_attention0_attn1_alignment.json', 'down_blocks.1.attentions.0 transformer0 attn1 QKV alignment'),
        ('TADSR_UNET_DOWNBLOCK1_ATTENTION0_ATTN1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_attention0_attn1_alignment.json', 'down_blocks.1.attentions.0 transformer0 self-attention output alignment'),
        ('TADSR_UNET_DOWNBLOCK1_ATTENTION0_AFTER_ATTN1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_attention0_attn1_alignment.json', 'down_blocks.1.attentions.0 after-attn1 residual alignment'),
        ('TADSR_UNET_DOWNBLOCK1_ATTENTION0_ATTN2_QKV_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_attention0_attn2_alignment.json', 'down_blocks.1.attentions.0 transformer0 attn2 QKV alignment'),
        ('TADSR_UNET_DOWNBLOCK1_ATTENTION0_ATTN2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_attention0_attn2_alignment.json', 'down_blocks.1.attentions.0 transformer0 cross-attention output alignment'),
        ('TADSR_UNET_DOWNBLOCK1_ATTENTION0_AFTER_ATTN2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_attention0_attn2_alignment.json', 'down_blocks.1.attentions.0 after-attn2 residual alignment'),
        ('TADSR_UNET_DOWNBLOCK1_ATTENTION0_FF_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_attention0_ff_alignment.json', 'down_blocks.1.attentions.0 feed-forward output alignment'),
        ('TADSR_UNET_DOWNBLOCK1_ATTENTION0_TRANSFORMER0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_attention0_ff_alignment.json', 'down_blocks.1.attentions.0 full transformer block alignment'),
        ('TADSR_UNET_DOWNBLOCK1_ATTENTION0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_attention0_alignment.json', 'isolated down_blocks.1.attentions.0 alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_RESNET0_ATTENTION0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblock0_downblock1_resnet0_attention0_alignment.json', 'UNet entry -> down_blocks.0 -> down_blocks.1.resnets.0 -> attentions.0 bridge alignment'),
    ]
    downblock1_attention0_statuses = []
    for check, path, note in downblock1_attention0_checks:
        st = status_from_json(path)
        downblock1_attention0_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_downblock1_attention0.py && python3 tools/export_tadsr_unet_downblock1_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    required_downblock1_attention0 = [s for s in downblock1_attention0_statuses if s not in {'NOT_APPLICABLE'}]
    downblock1_attention0_pass = bool(required_downblock1_attention0) and all(st == 'PASS' for st in required_downblock1_attention0)
    rows.append(row('TADSR_UNET_DOWNBLOCK1_ATTENTION0_BRIDGE_ALIGNMENT', 'PASS' if downblock1_attention0_pass else 'PARTIAL', 'down_blocks.1.attentions.0 aligned after complete local down_blocks.0 and down_blocks.1.resnets.0 bridge; remaining down_blocks.1 modules are intentionally unopened', 'continue with down_blocks.1.resnets.1 only'))
    downblock1_resnet1_checks = [
        ('TADSR_UNET_DOWNBLOCK1_RESNET1_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_downblock1_resnet1.json', 'official down_blocks.1.resnets.1 audit'),
        ('TADSR_UNET_DOWNBLOCK1_RESNET1_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock1_resnet1/unet_downblock1_resnet1_oracle_metadata.json', 'PyTorch oracle tensors for down_blocks.1.resnets.1 exported'),
        ('TADSR_UNET_DOWNBLOCK1_RESNET1_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock1_resnet1/unet_downblock1_resnet1_oracle_metadata.json', 'effective static weights for down_blocks.1.resnets.1 exported locally'),
        ('TADSR_UNET_DOWNBLOCK1_RESNET1_NORM1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_resnet1_alignment.json', 'down_blocks.1.resnets.1 norm1 alignment'),
        ('TADSR_UNET_DOWNBLOCK1_RESNET1_CONV1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_resnet1_alignment.json', 'down_blocks.1.resnets.1 conv1 alignment'),
        ('TADSR_UNET_DOWNBLOCK1_RESNET1_TEMB_PROJ_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_resnet1_alignment.json', 'down_blocks.1.resnets.1 time_emb_proj alignment'),
        ('TADSR_UNET_DOWNBLOCK1_RESNET1_CONV2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_resnet1_alignment.json', 'down_blocks.1.resnets.1 conv2 alignment'),
        ('TADSR_UNET_DOWNBLOCK1_RESNET1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_resnet1_alignment.json', 'down_blocks.1.resnets.1 local alignment'),
        ('TADSR_UNET_DOWNBLOCK1_RESNET1_SYNTHETIC_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_resnet1_synthetic_alignment.json', 'isolated synthetic hidden/temb -> down_blocks.1.resnets.1 alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_RESNET0_ATTENTION0_RESNET1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblock0_downblock1_resnet0_attention0_resnet1_alignment.json', 'UNet entry -> down_blocks.0 -> down_blocks.1.resnets.0 -> attentions.0 -> resnets.1 bridge alignment'),
    ]
    downblock1_resnet1_statuses = []
    for check, path, note in downblock1_resnet1_checks:
        st = status_from_json(path)
        downblock1_resnet1_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_downblock1_resnet1.py && python3 tools/export_tadsr_unet_downblock1_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    downblock1_resnet1_meta = load_json('experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock1_resnet1/unet_downblock1_resnet1_oracle_metadata.json')
    if downblock1_resnet1_meta.get('resnet1_config', {}).get('has_shortcut') is False:
        downblock1_resnet1_shortcut_status = 'NOT_APPLICABLE'
    else:
        downblock1_resnet1_shortcut_status = status_from_json('experiments/full_repro/unet_alignment/jittor_unet_downblock1_resnet1_shortcut_alignment.json', default=status_from_json('experiments/full_repro/unet_alignment/jittor_unet_downblock1_resnet1_alignment.json', default='BLOCKED'))
    rows.append(row('TADSR_UNET_DOWNBLOCK1_RESNET1_SHORTCUT_ALIGNMENT', downblock1_resnet1_shortcut_status, 'down_blocks.1.resnets.1 shortcut alignment; should be NOT_APPLICABLE/PASS depending on audited shortcut', 'continue only if shortcut remains PASS or NOT_APPLICABLE'))
    required_downblock1_resnet1 = [s for s in downblock1_resnet1_statuses if s not in {'NOT_APPLICABLE'}]
    downblock1_resnet1_pass = bool(required_downblock1_resnet1) and all(st == 'PASS' for st in required_downblock1_resnet1) and downblock1_resnet1_shortcut_status in {'PASS', 'NOT_APPLICABLE'}
    rows.append(row('TADSR_UNET_DOWNBLOCK1_RESNET1_BRIDGE_ALIGNMENT', 'PASS' if downblock1_resnet1_pass else 'PARTIAL', 'down_blocks.1.resnets.1 aligned after complete local down_blocks.0, down_blocks.1.resnets.0 and down_blocks.1.attentions.0 bridge; remaining down_blocks.1 modules are intentionally unopened', 'continue with down_blocks.1.downsamplers.0 only'))
    rows.append(row('TADSR_UNET_DOWNBLOCK1_PRE_ATTENTION1_ALIGNMENT', 'PASS' if downblock1_resnet0_pass and downblock1_attention0_pass and downblock1_resnet1_pass else 'PARTIAL', 'down_blocks.1 path is aligned through resnets.0 -> attentions.0 -> resnets.1 only', 'continue with down_blocks.1.downsamplers.0 only'))
    downblock1_attention1_checks = [
        ('TADSR_UNET_DOWNBLOCK1_ATTENTION1_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_downblock1_attention1.json', 'official down_blocks.1.attentions.1 audit'),
        ('TADSR_UNET_DOWNBLOCK1_ATTENTION1_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock1_attention1/unet_downblock1_attention1_oracle_metadata.json', 'PyTorch oracle tensors for down_blocks.1.attentions.1 exported'),
        ('TADSR_UNET_DOWNBLOCK1_ATTENTION1_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock1_attention1/unet_downblock1_attention1_oracle_metadata.json', 'effective static weights for down_blocks.1.attentions.1 exported locally'),
        ('TADSR_UNET_DOWNBLOCK1_ATTENTION1_NORM_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_attention1_proj_alignment.json', 'down_blocks.1.attentions.1 top-level norm alignment'),
        ('TADSR_UNET_DOWNBLOCK1_ATTENTION1_PROJ_IN_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_attention1_proj_alignment.json', 'down_blocks.1.attentions.1 proj_in alignment'),
        ('TADSR_UNET_DOWNBLOCK1_ATTENTION1_SEQUENCE_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_attention1_proj_alignment.json', 'down_blocks.1.attentions.1 NCHW-to-sequence alignment'),
        ('TADSR_UNET_DOWNBLOCK1_ATTENTION1_ATTN1_QKV_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_attention1_attn1_alignment.json', 'down_blocks.1.attentions.1 transformer0 attn1 QKV alignment'),
        ('TADSR_UNET_DOWNBLOCK1_ATTENTION1_ATTN1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_attention1_attn1_alignment.json', 'down_blocks.1.attentions.1 transformer0 self-attention output alignment'),
        ('TADSR_UNET_DOWNBLOCK1_ATTENTION1_AFTER_ATTN1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_attention1_attn1_alignment.json', 'down_blocks.1.attentions.1 after-attn1 residual alignment'),
        ('TADSR_UNET_DOWNBLOCK1_ATTENTION1_ATTN2_QKV_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_attention1_attn2_alignment.json', 'down_blocks.1.attentions.1 transformer0 attn2 QKV alignment'),
        ('TADSR_UNET_DOWNBLOCK1_ATTENTION1_ATTN2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_attention1_attn2_alignment.json', 'down_blocks.1.attentions.1 transformer0 cross-attention output alignment'),
        ('TADSR_UNET_DOWNBLOCK1_ATTENTION1_AFTER_ATTN2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_attention1_attn2_alignment.json', 'down_blocks.1.attentions.1 after-attn2 residual alignment'),
        ('TADSR_UNET_DOWNBLOCK1_ATTENTION1_FF_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_attention1_ff_alignment.json', 'down_blocks.1.attentions.1 feed-forward output alignment'),
        ('TADSR_UNET_DOWNBLOCK1_ATTENTION1_TRANSFORMER0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_attention1_ff_alignment.json', 'down_blocks.1.attentions.1 full transformer block alignment'),
        ('TADSR_UNET_DOWNBLOCK1_ATTENTION1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_attention1_alignment.json', 'isolated down_blocks.1.attentions.1 alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_RESNET0_ATTENTION0_RESNET1_ATTENTION1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblock0_downblock1_resnet0_attention0_resnet1_attention1_alignment.json', 'UNet entry -> down_blocks.0 -> down_blocks.1 resnet0/attention0/resnet1/attention1 bridge alignment'),
    ]
    downblock1_attention1_statuses = []
    for check, path, note in downblock1_attention1_checks:
        st = status_from_json(path)
        downblock1_attention1_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_downblock1_attention1.py && python3 tools/export_tadsr_unet_downblock1_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    required_downblock1_attention1 = [s for s in downblock1_attention1_statuses if s not in {'NOT_APPLICABLE'}]
    downblock1_attention1_pass = bool(required_downblock1_attention1) and all(st == 'PASS' for st in required_downblock1_attention1)
    rows.append(row('TADSR_UNET_DOWNBLOCK1_ATTENTION1_BRIDGE_ALIGNMENT', 'PASS' if downblock1_attention1_pass else 'PARTIAL', 'down_blocks.1.attentions.1 aligned after complete local down_blocks.0, down_blocks.1.resnets.0, attentions.0 and resnets.1 bridge; downsampler is checked separately below', 'continue with down_blocks.1.downsamplers.0 only'))
    rows.append(row('TADSR_UNET_DOWNBLOCK1_PRE_DOWNSAMPLER_ALIGNMENT', 'PASS' if downblock1_resnet0_pass and downblock1_attention0_pass and downblock1_resnet1_pass and downblock1_attention1_pass else 'PARTIAL', 'down_blocks.1 path is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 only', 'continue with down_blocks.1.downsamplers.0 only'))
    downblock1_downsampler_checks = [
        ('TADSR_UNET_DOWNBLOCK1_DOWNSAMPLER_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_downblock1_downsampler.json', 'official down_blocks.1.downsamplers.0 audit'),
        ('TADSR_UNET_DOWNBLOCK1_DOWNSAMPLER_LORA_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_downblock1_downsampler.json', 'LoRA wrapper/static effective conv audit for down_blocks.1.downsamplers.0'),
        ('TADSR_UNET_DOWNBLOCK1_LOCAL_FORWARD_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_downblock1_downsampler.json', 'official manual-chain local down_blocks.1 audit'),
        ('TADSR_UNET_DOWNBLOCK1_OUTPUT_STATES_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_downblock1_downsampler.json', 'official down_blocks.1 output_states order audit'),
        ('TADSR_UNET_DOWNBLOCK1_DOWNSAMPLER_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock1_downsampler/unet_downblock1_downsampler_oracle_metadata.json', 'PyTorch oracle tensors for down_blocks.1.downsamplers.0 and local down_blocks.1 exported'),
        ('TADSR_UNET_DOWNBLOCK1_DOWNSAMPLER_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock1_downsampler/unet_downblock1_downsampler_oracle_metadata.json', 'effective static weights for down_blocks.1.downsamplers.0 exported locally'),
        ('TADSR_UNET_DOWNBLOCK1_OUTPUT_STATES_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock1_downsampler/unet_downblock1_downsampler_oracle_metadata.json', 'PyTorch oracle tensors for down_blocks.1 output_states exported'),
        ('TADSR_UNET_DOWNBLOCK1_DOWNSAMPLER_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_downsampler_alignment.json', 'isolated down_blocks.1.downsamplers.0 alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_RESNET0_ATTENTION0_RESNET1_ATTENTION1_DOWNSAMPLER_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblock0_downblock1_resnet0_attention0_resnet1_attention1_downsampler_alignment.json', 'UNet entry -> down_blocks.0 -> down_blocks.1 resnet0/attention0/resnet1/attention1/downsampler bridge alignment'),
        ('TADSR_UNET_DOWNBLOCK1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock1_alignment.json', 'full local down_blocks.1 alignment through downsampler and output_states'),
    ]
    downblock1_downsampler_statuses = []
    for check, path, note in downblock1_downsampler_checks:
        st = status_from_json(path)
        downblock1_downsampler_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_downblock1_downsampler.py && python3 tools/export_tadsr_unet_downblock1_downsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    rows.append(row('TADSR_UNET_DOWNBLOCK1_DOWNSAMPLER_PADDING_ALIGNMENT', 'NOT_APPLICABLE', 'official Downsample2D uses Conv2d padding=1 directly; no separate padding op is exported', 'continue with down_blocks.2 after down_blocks.1 stays PASS'))
    downblock1_downsampler_conv_status = status_from_json('experiments/full_repro/unet_alignment/jittor_unet_downblock1_downsampler_alignment.json')
    rows.append(row('TADSR_UNET_DOWNBLOCK1_DOWNSAMPLER_CONV_ALIGNMENT', downblock1_downsampler_conv_status, 'effective static Conv2d alignment for down_blocks.1.downsamplers.0', 'continue only if PASS'))
    downblock1_output_hidden_status = status_from_json('experiments/full_repro/unet_alignment/jittor_unet_downblock1_alignment.json')
    rows.append(row('TADSR_UNET_DOWNBLOCK1_OUTPUT_HIDDEN_ALIGNMENT', downblock1_output_hidden_status, 'down_blocks.1 hidden_states output equals official block.forward output', 'continue only if PASS'))
    downblock1_output_states_status = status_from_json('experiments/full_repro/unet_alignment/jittor_unet_downblock1_alignment.json')
    rows.append(row('TADSR_UNET_DOWNBLOCK1_OUTPUT_STATES_ALIGNMENT', downblock1_output_states_status, 'down_blocks.1 output_states tuple order and tensors match official block.forward', 'continue only if PASS'))
    downblock1_downsampler_pass = all(st == 'PASS' for st in downblock1_downsampler_statuses) and downblock1_downsampler_conv_status == 'PASS' and downblock1_output_hidden_status == 'PASS' and downblock1_output_states_status == 'PASS'
    rows.append(row('TADSR_UNET_DOWNBLOCK1_BRIDGE_ALIGNMENT', 'PASS' if downblock1_resnet0_pass and downblock1_attention0_pass and downblock1_resnet1_pass and downblock1_attention1_pass and downblock1_downsampler_pass else 'PARTIAL', 'complete local down_blocks.1 bridge aligned through downsampler and output_states; later UNet blocks remain unopened', 'continue with down_blocks.2.resnets.0'))

    downblock2_resnet0_checks = [
        ('TADSR_UNET_DOWNBLOCK2_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_downblock2_resnet0.json', 'official UNet down_blocks.2 overview audit, resnets.0 baseline context'),
        ('TADSR_UNET_DOWNBLOCK2_RESNET0_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_downblock2_resnet0.json', 'official down_blocks.2.resnets.0 audit'),
        ('TADSR_UNET_DOWNBLOCK2_RESNET0_LORA_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_downblock2_resnet0.json', 'LoRA/effective weight state for down_blocks.2.resnets.0'),
        ('TADSR_UNET_DOWNBLOCK2_RESNET0_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock2_resnet0/unet_downblock2_resnet0_oracle_metadata.json', 'PyTorch oracle tensors for down_blocks.2.resnets.0 exported'),
        ('TADSR_UNET_DOWNBLOCK2_RESNET0_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock2_resnet0/unet_downblock2_resnet0_oracle_metadata.json', 'effective static weights for down_blocks.2.resnets.0 exported locally'),
        ('TADSR_UNET_DOWNBLOCK2_RESNET0_NORM1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_resnet0_alignment.json', 'down_blocks.2.resnets.0 norm1 alignment'),
        ('TADSR_UNET_DOWNBLOCK2_RESNET0_CONV1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_resnet0_alignment.json', 'down_blocks.2.resnets.0 conv1 alignment'),
        ('TADSR_UNET_DOWNBLOCK2_RESNET0_TEMB_PROJ_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_resnet0_alignment.json', 'down_blocks.2.resnets.0 time_emb_proj alignment'),
        ('TADSR_UNET_DOWNBLOCK2_RESNET0_CONV2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_resnet0_alignment.json', 'down_blocks.2.resnets.0 conv2 alignment'),
        ('TADSR_UNET_DOWNBLOCK2_RESNET0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_resnet0_alignment.json', 'down_blocks.2.resnets.0 local alignment'),
        ('TADSR_UNET_DOWNBLOCK2_RESNET0_SYNTHETIC_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_resnet0_synthetic_alignment.json', 'isolated synthetic hidden/temb -> down_blocks.2.resnets.0 alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_DOWNBLOCK2_RESNET0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblock0_downblock1_downblock2_resnet0_alignment.json', 'UNet entry -> full local down_blocks.0 -> full local down_blocks.1 -> down_blocks.2.resnets.0 bridge alignment'),
    ]
    downblock2_resnet0_statuses = []
    for check, path, note in downblock2_resnet0_checks:
        st = status_from_json(path)
        downblock2_resnet0_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_downblock2_resnet0.py && python3 tools/export_tadsr_unet_downblock2_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    downblock2_resnet0_meta = load_json('experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock2_resnet0/unet_downblock2_resnet0_oracle_metadata.json')
    if downblock2_resnet0_meta.get('resnet0_config', {}).get('has_shortcut') is False:
        downblock2_resnet0_shortcut_status = 'NOT_APPLICABLE'
    else:
        downblock2_resnet0_shortcut_status = status_from_json('experiments/full_repro/unet_alignment/jittor_unet_downblock2_resnet0_alignment.json', default='BLOCKED')
    rows.append(row('TADSR_UNET_DOWNBLOCK2_RESNET0_SHORTCUT_ALIGNMENT', downblock2_resnet0_shortcut_status, 'down_blocks.2.resnets.0 shortcut alignment; PASS if conv_shortcut exists, NOT_APPLICABLE if official shortcut is absent', 'continue only if shortcut remains PASS or NOT_APPLICABLE'))
    required_downblock2_resnet0 = [s for s in downblock2_resnet0_statuses if s not in {'NOT_APPLICABLE'}]
    downblock2_resnet0_pass = bool(required_downblock2_resnet0) and all(st == 'PASS' for st in required_downblock2_resnet0) and downblock2_resnet0_shortcut_status in {'PASS', 'NOT_APPLICABLE'}
    rows.append(row('TADSR_UNET_DOWNBLOCK2_RESNET0_BRIDGE_ALIGNMENT', 'PASS' if downblock2_resnet0_pass else 'PARTIAL', 'down_blocks.2.resnets.0 aligned after complete local down_blocks.0 and down_blocks.1 bridge; remaining down_blocks.2 modules are intentionally unopened', 'continue with down_blocks.2.attentions.0 only'))

    downblock2_attention0_checks = [
        ('TADSR_UNET_DOWNBLOCK2_ATTENTION0_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_downblock2_attention0.json', 'official down_blocks.2.attentions.0 audit'),
        ('TADSR_UNET_DOWNBLOCK2_ATTENTION0_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock2_attention0/unet_downblock2_attention0_oracle_metadata.json', 'PyTorch oracle tensors for down_blocks.2.attentions.0 exported'),
        ('TADSR_UNET_DOWNBLOCK2_ATTENTION0_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock2_attention0/unet_downblock2_attention0_oracle_metadata.json', 'effective static weights for down_blocks.2.attentions.0 exported locally'),
        ('TADSR_UNET_DOWNBLOCK2_ATTENTION0_NORM_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_attention0_proj_alignment.json', 'down_blocks.2.attentions.0 top-level norm alignment'),
        ('TADSR_UNET_DOWNBLOCK2_ATTENTION0_PROJ_IN_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_attention0_proj_alignment.json', 'down_blocks.2.attentions.0 proj_in alignment'),
        ('TADSR_UNET_DOWNBLOCK2_ATTENTION0_SEQUENCE_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_attention0_proj_alignment.json', 'down_blocks.2.attentions.0 NCHW-to-sequence alignment'),
        ('TADSR_UNET_DOWNBLOCK2_ATTENTION0_ATTN1_QKV_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_attention0_attn1_alignment.json', 'down_blocks.2.attentions.0 transformer0 attn1 QKV alignment'),
        ('TADSR_UNET_DOWNBLOCK2_ATTENTION0_ATTN1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_attention0_attn1_alignment.json', 'down_blocks.2.attentions.0 transformer0 self-attention output alignment'),
        ('TADSR_UNET_DOWNBLOCK2_ATTENTION0_AFTER_ATTN1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_attention0_attn1_alignment.json', 'down_blocks.2.attentions.0 after-attn1 residual alignment'),
        ('TADSR_UNET_DOWNBLOCK2_ATTENTION0_ATTN2_QKV_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_attention0_attn2_alignment.json', 'down_blocks.2.attentions.0 transformer0 attn2 QKV alignment'),
        ('TADSR_UNET_DOWNBLOCK2_ATTENTION0_ATTN2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_attention0_attn2_alignment.json', 'down_blocks.2.attentions.0 transformer0 cross-attention output alignment'),
        ('TADSR_UNET_DOWNBLOCK2_ATTENTION0_AFTER_ATTN2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_attention0_attn2_alignment.json', 'down_blocks.2.attentions.0 after-attn2 residual alignment'),
        ('TADSR_UNET_DOWNBLOCK2_ATTENTION0_FF_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_attention0_ff_alignment.json', 'down_blocks.2.attentions.0 feed-forward output alignment'),
        ('TADSR_UNET_DOWNBLOCK2_ATTENTION0_TRANSFORMER0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_attention0_ff_alignment.json', 'down_blocks.2.attentions.0 full transformer block alignment'),
        ('TADSR_UNET_DOWNBLOCK2_ATTENTION0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_attention0_alignment.json', 'isolated down_blocks.2.attentions.0 alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_DOWNBLOCK2_RESNET0_ATTENTION0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblock0_downblock1_downblock2_resnet0_attention0_alignment.json', 'UNet entry -> down_blocks.0 -> down_blocks.1 -> down_blocks.2.resnets.0 -> attentions.0 bridge alignment'),
    ]
    downblock2_attention0_statuses = []
    for check, path, note in downblock2_attention0_checks:
        st = status_from_json(path)
        downblock2_attention0_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_downblock2_attention0.py && python3 tools/export_tadsr_unet_downblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    required_downblock2_attention0 = [s for s in downblock2_attention0_statuses if s not in {'NOT_APPLICABLE'}]
    downblock2_attention0_pass = bool(required_downblock2_attention0) and all(st == 'PASS' for st in required_downblock2_attention0)
    rows.append(row('TADSR_UNET_DOWNBLOCK2_ATTENTION0_BRIDGE_ALIGNMENT', 'PASS' if downblock2_attention0_pass else 'PARTIAL', 'down_blocks.2.attentions.0 aligned after complete local down_blocks.0/down_blocks.1/down_blocks.2.resnets.0 bridge; remaining down_blocks.2 modules are intentionally unopened', 'continue with down_blocks.2.resnets.1 only'))
    rows.append(row('TADSR_UNET_DOWNBLOCK2_PRE_RESNET1_ALIGNMENT', 'PASS' if downblock2_resnet0_pass and downblock2_attention0_pass else 'PARTIAL', 'down_blocks.2 path is aligned through resnets.0 -> attentions.0 only', 'continue with down_blocks.2.resnets.1 only'))

    downblock2_resnet1_checks = [
        ('TADSR_UNET_DOWNBLOCK2_RESNET1_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_downblock2_resnet1.json', 'official down_blocks.2.resnets.1 audit'),
        ('TADSR_UNET_DOWNBLOCK2_RESNET1_LORA_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_downblock2_resnet1.json', 'LoRA/effective weight state for down_blocks.2.resnets.1'),
        ('TADSR_UNET_DOWNBLOCK2_RESNET1_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock2_resnet1/unet_downblock2_resnet1_oracle_metadata.json', 'PyTorch oracle tensors for down_blocks.2.resnets.1 exported'),
        ('TADSR_UNET_DOWNBLOCK2_RESNET1_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock2_resnet1/unet_downblock2_resnet1_oracle_metadata.json', 'effective static weights for down_blocks.2.resnets.1 exported locally'),
        ('TADSR_UNET_DOWNBLOCK2_RESNET1_NORM1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_resnet1_alignment.json', 'down_blocks.2.resnets.1 norm1 alignment'),
        ('TADSR_UNET_DOWNBLOCK2_RESNET1_CONV1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_resnet1_alignment.json', 'down_blocks.2.resnets.1 conv1 alignment'),
        ('TADSR_UNET_DOWNBLOCK2_RESNET1_TEMB_PROJ_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_resnet1_alignment.json', 'down_blocks.2.resnets.1 time_emb_proj alignment'),
        ('TADSR_UNET_DOWNBLOCK2_RESNET1_CONV2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_resnet1_alignment.json', 'down_blocks.2.resnets.1 conv2 alignment'),
        ('TADSR_UNET_DOWNBLOCK2_RESNET1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_resnet1_alignment.json', 'down_blocks.2.resnets.1 local alignment'),
        ('TADSR_UNET_DOWNBLOCK2_RESNET1_SYNTHETIC_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_resnet1_synthetic_alignment.json', 'isolated synthetic hidden/temb -> down_blocks.2.resnets.1 alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_DOWNBLOCK2_RESNET0_ATTENTION0_RESNET1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblock0_downblock1_downblock2_resnet0_attention0_resnet1_alignment.json', 'UNet entry -> full local down_blocks.0/down_blocks.1 -> down_blocks.2.resnets.0 -> attentions.0 -> resnets.1 bridge alignment'),
    ]
    downblock2_resnet1_statuses = []
    for check, path, note in downblock2_resnet1_checks:
        st = status_from_json(path)
        downblock2_resnet1_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_downblock2_resnet1.py && python3 tools/export_tadsr_unet_downblock2_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    downblock2_resnet1_meta = load_json('experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock2_resnet1/unet_downblock2_resnet1_oracle_metadata.json')
    if downblock2_resnet1_meta.get('resnet1_config', {}).get('has_shortcut') is False:
        downblock2_resnet1_shortcut_status = 'NOT_APPLICABLE'
    else:
        downblock2_resnet1_shortcut_status = status_from_json('experiments/full_repro/unet_alignment/jittor_unet_downblock2_resnet1_alignment.json', default='BLOCKED')
    rows.append(row('TADSR_UNET_DOWNBLOCK2_RESNET1_SHORTCUT_ALIGNMENT', downblock2_resnet1_shortcut_status, 'down_blocks.2.resnets.1 shortcut alignment; PASS if conv_shortcut exists, NOT_APPLICABLE if official shortcut is absent', 'continue only if shortcut remains PASS or NOT_APPLICABLE'))
    required_downblock2_resnet1 = [s for s in downblock2_resnet1_statuses if s not in {'NOT_APPLICABLE'}]
    downblock2_resnet1_pass = bool(required_downblock2_resnet1) and all(st == 'PASS' for st in required_downblock2_resnet1) and downblock2_resnet1_shortcut_status in {'PASS', 'NOT_APPLICABLE'}
    rows.append(row('TADSR_UNET_DOWNBLOCK2_RESNET1_BRIDGE_ALIGNMENT', 'PASS' if downblock2_resnet1_pass else 'PARTIAL', 'down_blocks.2.resnets.1 aligned after complete local down_blocks.0/down_blocks.1/down_blocks.2.resnets.0/down_blocks.2.attentions.0 bridge; remaining down_blocks.2 modules are intentionally unopened', 'continue with down_blocks.2.attentions.1 only'))
    rows.append(row('TADSR_UNET_DOWNBLOCK2_PRE_ATTENTION1_ALIGNMENT', 'PASS' if downblock2_resnet0_pass and downblock2_attention0_pass and downblock2_resnet1_pass else 'PARTIAL', 'down_blocks.2 path is aligned through resnets.0 -> attentions.0 -> resnets.1 only', 'continue with down_blocks.2.attentions.1 only'))

    downblock2_attention1_checks = [
        ('TADSR_UNET_DOWNBLOCK2_ATTENTION1_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_downblock2_attention1.json', 'official down_blocks.2.attentions.1 audit'),
        ('TADSR_UNET_DOWNBLOCK2_ATTENTION1_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock2_attention1/unet_downblock2_attention1_oracle_metadata.json', 'PyTorch oracle tensors for down_blocks.2.attentions.1 exported'),
        ('TADSR_UNET_DOWNBLOCK2_ATTENTION1_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock2_attention1/unet_downblock2_attention1_oracle_metadata.json', 'effective static weights for down_blocks.2.attentions.1 exported locally'),
        ('TADSR_UNET_DOWNBLOCK2_ATTENTION1_NORM_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_attention1_proj_alignment.json', 'down_blocks.2.attentions.1 top-level norm alignment'),
        ('TADSR_UNET_DOWNBLOCK2_ATTENTION1_PROJ_IN_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_attention1_proj_alignment.json', 'down_blocks.2.attentions.1 proj_in alignment'),
        ('TADSR_UNET_DOWNBLOCK2_ATTENTION1_SEQUENCE_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_attention1_proj_alignment.json', 'down_blocks.2.attentions.1 NCHW-to-sequence alignment'),
        ('TADSR_UNET_DOWNBLOCK2_ATTENTION1_ATTN1_QKV_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_attention1_attn1_alignment.json', 'down_blocks.2.attentions.1 transformer0 attn1 QKV alignment'),
        ('TADSR_UNET_DOWNBLOCK2_ATTENTION1_ATTN1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_attention1_attn1_alignment.json', 'down_blocks.2.attentions.1 transformer0 self-attention output alignment'),
        ('TADSR_UNET_DOWNBLOCK2_ATTENTION1_AFTER_ATTN1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_attention1_attn1_alignment.json', 'down_blocks.2.attentions.1 after-attn1 residual alignment'),
        ('TADSR_UNET_DOWNBLOCK2_ATTENTION1_ATTN2_QKV_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_attention1_attn2_alignment.json', 'down_blocks.2.attentions.1 transformer0 attn2 QKV alignment'),
        ('TADSR_UNET_DOWNBLOCK2_ATTENTION1_ATTN2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_attention1_attn2_alignment.json', 'down_blocks.2.attentions.1 transformer0 cross-attention output alignment'),
        ('TADSR_UNET_DOWNBLOCK2_ATTENTION1_AFTER_ATTN2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_attention1_attn2_alignment.json', 'down_blocks.2.attentions.1 after-attn2 residual alignment'),
        ('TADSR_UNET_DOWNBLOCK2_ATTENTION1_FF_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_attention1_ff_alignment.json', 'down_blocks.2.attentions.1 feed-forward output alignment'),
        ('TADSR_UNET_DOWNBLOCK2_ATTENTION1_TRANSFORMER0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_attention1_ff_alignment.json', 'down_blocks.2.attentions.1 full transformer block alignment'),
        ('TADSR_UNET_DOWNBLOCK2_ATTENTION1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_attention1_alignment.json', 'isolated down_blocks.2.attentions.1 alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_DOWNBLOCK2_RESNET0_ATTENTION0_RESNET1_ATTENTION1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblock0_downblock1_downblock2_resnet0_attention0_resnet1_attention1_alignment.json', 'UNet entry -> down_blocks.0 -> down_blocks.1 -> down_blocks.2.resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 bridge alignment'),
    ]
    downblock2_attention1_statuses = []
    for check, path, note in downblock2_attention1_checks:
        st = status_from_json(path)
        downblock2_attention1_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_downblock2_attention1.py && python3 tools/export_tadsr_unet_downblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    required_downblock2_attention1 = [s for s in downblock2_attention1_statuses if s not in {'NOT_APPLICABLE'}]
    downblock2_attention1_pass = bool(required_downblock2_attention1) and all(st == 'PASS' for st in required_downblock2_attention1)
    rows.append(row('TADSR_UNET_DOWNBLOCK2_ATTENTION1_BRIDGE_ALIGNMENT', 'PASS' if downblock2_attention1_pass else 'PARTIAL', 'down_blocks.2.attentions.1 aligned after complete local down_blocks.0/down_blocks.1/down_blocks.2.resnets.0/down_blocks.2.attentions.0/down_blocks.2.resnets.1 bridge; downsampler remains intentionally unopened', 'continue with down_blocks.2.downsamplers.0 only'))
    rows.append(row('TADSR_UNET_DOWNBLOCK2_PRE_DOWNSAMPLER_ALIGNMENT', 'PASS' if downblock2_resnet0_pass and downblock2_attention0_pass and downblock2_resnet1_pass and downblock2_attention1_pass else 'PARTIAL', 'down_blocks.2 path is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 only', 'continue with down_blocks.2.downsamplers.0 only'))

    downblock2_downsampler_checks = [
        ('TADSR_UNET_DOWNBLOCK2_DOWNSAMPLER_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_downblock2_downsampler.json', 'official down_blocks.2.downsamplers.0 audit'),
        ('TADSR_UNET_DOWNBLOCK2_DOWNSAMPLER_LORA_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_downblock2_downsampler.json', 'LoRA/effective weight state for down_blocks.2.downsamplers.0'),
        ('TADSR_UNET_DOWNBLOCK2_LOCAL_FORWARD_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_downblock2_downsampler.json', 'manual local down_blocks.2 chain matches official block.forward'),
        ('TADSR_UNET_DOWNBLOCK2_OUTPUT_STATES_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_downblock2_downsampler.json', 'official down_blocks.2 residual output_states tuple audited'),
        ('TADSR_UNET_DOWNBLOCK2_DOWNSAMPLER_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock2_downsampler/unet_downblock2_downsampler_oracle_metadata.json', 'PyTorch oracle tensors for down_blocks.2.downsamplers.0 exported'),
        ('TADSR_UNET_DOWNBLOCK2_DOWNSAMPLER_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock2_downsampler/unet_downblock2_downsampler_oracle_metadata.json', 'effective static weights for down_blocks.2.downsamplers.0 exported locally'),
        ('TADSR_UNET_DOWNBLOCK2_DOWNSAMPLER_PADDING_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_downsampler_alignment.json', 'down_blocks.2.downsampler padding alignment; NOT_APPLICABLE for direct conv path'),
        ('TADSR_UNET_DOWNBLOCK2_DOWNSAMPLER_CONV_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_downsampler_alignment.json', 'down_blocks.2.downsampler conv output alignment'),
        ('TADSR_UNET_DOWNBLOCK2_DOWNSAMPLER_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_downsampler_alignment.json', 'isolated down_blocks.2.downsampler alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_DOWNBLOCK2_RESNET0_ATTENTION0_RESNET1_ATTENTION1_DOWNSAMPLER_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblock0_downblock1_downblock2_resnet0_attention0_resnet1_attention1_downsampler_alignment.json', 'UNet entry -> down_blocks.0 -> down_blocks.1 -> down_blocks.2 local chain through downsampler bridge alignment'),
        ('TADSR_UNET_DOWNBLOCK2_OUTPUT_HIDDEN_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_alignment.json', 'down_blocks.2 final hidden_states output alignment'),
        ('TADSR_UNET_DOWNBLOCK2_OUTPUT_STATES_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_alignment.json', 'down_blocks.2 residual output_states tuple alignment'),
        ('TADSR_UNET_DOWNBLOCK2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock2_alignment.json', 'full local down_blocks.2 alignment through output_states tuple'),
    ]
    downblock2_downsampler_statuses = []
    for check, path, note in downblock2_downsampler_checks:
        st = status_from_json(path)
        downblock2_downsampler_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_downblock2_downsampler.py && python3 tools/export_tadsr_unet_downblock2_downsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    required_downblock2_downsampler = [s for s in downblock2_downsampler_statuses if s not in {'NOT_APPLICABLE'}]
    downblock2_downsampler_pass = bool(required_downblock2_downsampler) and all(st == 'PASS' for st in required_downblock2_downsampler)
    rows.append(row('TADSR_UNET_DOWNBLOCK2_BRIDGE_ALIGNMENT', 'PASS' if downblock2_resnet0_pass and downblock2_attention0_pass and downblock2_resnet1_pass and downblock2_attention1_pass and downblock2_downsampler_pass else 'PARTIAL', 'complete local down_blocks.2 bridge aligned through downsampler and output_states; later UNet blocks remain unopened', 'continue with down_blocks.3.resnets.0 only'))
    downblock3_resnet0_checks = [
        ('TADSR_UNET_DOWNBLOCK3_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_downblock3_resnet0.json', 'official UNet down_blocks.3 topology audit'),
        ('TADSR_UNET_DOWNBLOCK3_RESNET0_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_downblock3_resnet0.json', 'official down_blocks.3.resnets.0 audit'),
        ('TADSR_UNET_DOWNBLOCK3_RESNET0_LORA_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_downblock3_resnet0.json', 'LoRA/effective weight state for down_blocks.3.resnets.0'),
        ('TADSR_UNET_DOWNBLOCK3_RESNET0_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock3_resnet0/unet_downblock3_resnet0_oracle_metadata.json', 'PyTorch oracle tensors for down_blocks.3.resnets.0 exported'),
        ('TADSR_UNET_DOWNBLOCK3_RESNET0_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock3_resnet0/unet_downblock3_resnet0_oracle_metadata.json', 'effective static weights for down_blocks.3.resnets.0 exported locally'),
        ('TADSR_UNET_DOWNBLOCK3_RESNET0_NORM1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock3_resnet0_alignment.json', 'down_blocks.3.resnets.0 norm1 alignment'),
        ('TADSR_UNET_DOWNBLOCK3_RESNET0_CONV1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock3_resnet0_alignment.json', 'down_blocks.3.resnets.0 conv1 alignment'),
        ('TADSR_UNET_DOWNBLOCK3_RESNET0_TEMB_PROJ_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock3_resnet0_alignment.json', 'down_blocks.3.resnets.0 time_emb_proj alignment'),
        ('TADSR_UNET_DOWNBLOCK3_RESNET0_CONV2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock3_resnet0_alignment.json', 'down_blocks.3.resnets.0 conv2 alignment'),
        ('TADSR_UNET_DOWNBLOCK3_RESNET0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock3_resnet0_alignment.json', 'down_blocks.3.resnets.0 local alignment'),
        ('TADSR_UNET_DOWNBLOCK3_RESNET0_SYNTHETIC_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock3_resnet0_synthetic_alignment.json', 'isolated synthetic hidden/temb -> down_blocks.3.resnets.0 alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_DOWNBLOCK2_DOWNBLOCK3_RESNET0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblock0_downblock1_downblock2_downblock3_resnet0_alignment.json', 'UNet entry -> down_blocks.0 -> down_blocks.1 -> down_blocks.2 -> down_blocks.3.resnets.0 bridge alignment'),
    ]
    downblock3_resnet0_statuses = []
    for check, path, note in downblock3_resnet0_checks:
        st = status_from_json(path)
        downblock3_resnet0_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_downblock3_resnet0.py && python3 tools/export_tadsr_unet_downblock3_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    downblock3_resnet0_meta = load_json('experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock3_resnet0/unet_downblock3_resnet0_oracle_metadata.json')
    if downblock3_resnet0_meta.get('resnet0_config', {}).get('has_shortcut') is False:
        downblock3_resnet0_shortcut_status = 'NOT_APPLICABLE'
    else:
        downblock3_resnet0_shortcut_status = status_from_json('experiments/full_repro/unet_alignment/jittor_unet_downblock3_resnet0_alignment.json', default='BLOCKED')
    rows.append(row('TADSR_UNET_DOWNBLOCK3_RESNET0_SHORTCUT_ALIGNMENT', downblock3_resnet0_shortcut_status, 'down_blocks.3.resnets.0 shortcut alignment; PASS if conv_shortcut exists, NOT_APPLICABLE if official shortcut is absent', 'continue only if shortcut remains PASS or NOT_APPLICABLE'))
    required_downblock3_resnet0 = [s for s in downblock3_resnet0_statuses if s not in {'NOT_APPLICABLE'}]
    downblock3_resnet0_pass = bool(required_downblock3_resnet0) and all(st == 'PASS' for st in required_downblock3_resnet0) and downblock3_resnet0_shortcut_status in {'PASS', 'NOT_APPLICABLE'}
    rows.append(row('TADSR_UNET_DOWNBLOCK3_RESNET0_BRIDGE_ALIGNMENT', 'PASS' if downblock3_resnet0_pass else 'PARTIAL', 'down_blocks.3.resnets.0 aligned after complete local down_blocks.0/1/2 bridge; remaining down_blocks.3 modules are intentionally unopened', 'continue with down_blocks.3.resnets.1 only'))
    downblock3_resnet1_checks = [
        ('TADSR_UNET_DOWNBLOCK3_RESNET1_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_downblock3_resnet1.json', 'official down_blocks.3.resnets.1 audit'),
        ('TADSR_UNET_DOWNBLOCK3_LOCAL_FORWARD_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_downblock3_resnet1.json', 'official down_blocks.3 local forward hidden output audited'),
        ('TADSR_UNET_DOWNBLOCK3_OUTPUT_STATES_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_downblock3_resnet1.json', 'official down_blocks.3 residual output_states tuple audited'),
        ('TADSR_UNET_DOWNBLOCK3_RESNET1_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock3_resnet1/unet_downblock3_resnet1_oracle_metadata.json', 'PyTorch oracle tensors for down_blocks.3.resnets.1 exported'),
        ('TADSR_UNET_DOWNBLOCK3_RESNET1_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock3_resnet1/unet_downblock3_resnet1_oracle_metadata.json', 'effective static weights for down_blocks.3.resnets.1 exported locally'),
        ('TADSR_UNET_DOWNBLOCK3_RESNET1_NORM1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock3_resnet1_alignment.json', 'down_blocks.3.resnets.1 norm1 alignment'),
        ('TADSR_UNET_DOWNBLOCK3_RESNET1_CONV1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock3_resnet1_alignment.json', 'down_blocks.3.resnets.1 conv1 alignment'),
        ('TADSR_UNET_DOWNBLOCK3_RESNET1_TEMB_PROJ_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock3_resnet1_alignment.json', 'down_blocks.3.resnets.1 time_emb_proj alignment'),
        ('TADSR_UNET_DOWNBLOCK3_RESNET1_CONV2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock3_resnet1_alignment.json', 'down_blocks.3.resnets.1 conv2 alignment'),
        ('TADSR_UNET_DOWNBLOCK3_RESNET1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock3_resnet1_alignment.json', 'down_blocks.3.resnets.1 local alignment'),
        ('TADSR_UNET_DOWNBLOCK3_RESNET1_SYNTHETIC_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock3_resnet1_synthetic_alignment.json', 'isolated synthetic hidden/temb -> down_blocks.3.resnets.1 alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_DOWNBLOCK2_DOWNBLOCK3_RESNET0_RESNET1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblock0_downblock1_downblock2_downblock3_resnet0_resnet1_alignment.json', 'UNet entry -> down_blocks.0 -> down_blocks.1 -> down_blocks.2 -> down_blocks.3.resnets.0 -> resnets.1 bridge alignment'),
        ('TADSR_UNET_DOWNBLOCK3_OUTPUT_HIDDEN_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock3_alignment.json', 'down_blocks.3 final hidden_states output alignment'),
        ('TADSR_UNET_DOWNBLOCK3_OUTPUT_STATES_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock3_alignment.json', 'down_blocks.3 residual output_states tuple alignment'),
        ('TADSR_UNET_DOWNBLOCK3_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_downblock3_alignment.json', 'full local down_blocks.3 alignment through output_states tuple'),
    ]
    downblock3_resnet1_statuses = []
    for check, path, note in downblock3_resnet1_checks:
        st = status_from_json(path)
        downblock3_resnet1_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_downblock3_resnet1.py && python3 tools/export_tadsr_unet_downblock3_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    downblock3_resnet1_meta = load_json('experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock3_resnet1/unet_downblock3_resnet1_oracle_metadata.json')
    if downblock3_resnet1_meta.get('resnet1_config', {}).get('has_shortcut') is False:
        downblock3_resnet1_shortcut_status = 'NOT_APPLICABLE'
    else:
        downblock3_resnet1_shortcut_status = status_from_json('experiments/full_repro/unet_alignment/jittor_unet_downblock3_resnet1_alignment.json', default='BLOCKED')
    rows.append(row('TADSR_UNET_DOWNBLOCK3_RESNET1_SHORTCUT_ALIGNMENT', downblock3_resnet1_shortcut_status, 'down_blocks.3.resnets.1 shortcut alignment; PASS if conv_shortcut exists, NOT_APPLICABLE if official shortcut is absent', 'continue only if shortcut remains PASS or NOT_APPLICABLE'))
    required_downblock3_resnet1 = [s for s in downblock3_resnet1_statuses if s not in {'NOT_APPLICABLE'}]
    downblock3_resnet1_pass = bool(required_downblock3_resnet1) and all(st == 'PASS' for st in required_downblock3_resnet1) and downblock3_resnet1_shortcut_status in {'PASS', 'NOT_APPLICABLE'}
    rows.append(row('TADSR_UNET_DOWNBLOCK3_BRIDGE_ALIGNMENT', 'PASS' if downblock3_resnet0_pass and downblock3_resnet1_pass else 'PARTIAL', 'complete local down_blocks.3 bridge aligned through resnet0 -> resnet1 and output_states; mid_block remains unopened', 'continue with mid_block.resnets.0 only'))
    midblock_resnet0_checks = [
        ('TADSR_UNET_MIDBLOCK_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_midblock_resnet0.json', 'official UNet mid_block topology audit'),
        ('TADSR_UNET_MIDBLOCK_RESNET0_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_midblock_resnet0.json', 'official mid_block.resnets.0 audit'),
        ('TADSR_UNET_MIDBLOCK_RESNET0_LORA_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_midblock_resnet0.json', 'LoRA/effective weight state for mid_block.resnets.0'),
        ('TADSR_UNET_MIDBLOCK_RESNET0_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_midblock_resnet0/unet_midblock_resnet0_oracle_metadata.json', 'PyTorch oracle tensors for mid_block.resnets.0 exported'),
        ('TADSR_UNET_MIDBLOCK_RESNET0_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_midblock_resnet0/unet_midblock_resnet0_oracle_metadata.json', 'effective static weights for mid_block.resnets.0 exported locally'),
        ('TADSR_UNET_MIDBLOCK_RESNET0_NORM1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_midblock_resnet0_alignment.json', 'mid_block.resnets.0 norm1 alignment'),
        ('TADSR_UNET_MIDBLOCK_RESNET0_CONV1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_midblock_resnet0_alignment.json', 'mid_block.resnets.0 conv1 alignment'),
        ('TADSR_UNET_MIDBLOCK_RESNET0_TEMB_PROJ_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_midblock_resnet0_alignment.json', 'mid_block.resnets.0 time_emb_proj alignment'),
        ('TADSR_UNET_MIDBLOCK_RESNET0_CONV2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_midblock_resnet0_alignment.json', 'mid_block.resnets.0 conv2 alignment'),
        ('TADSR_UNET_MIDBLOCK_RESNET0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_midblock_resnet0_alignment.json', 'mid_block.resnets.0 local alignment'),
        ('TADSR_UNET_MIDBLOCK_RESNET0_SYNTHETIC_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_midblock_resnet0_synthetic_alignment.json', 'isolated synthetic hidden/temb -> mid_block.resnets.0 alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_RESNET0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblocks_midblock_resnet0_alignment.json', 'UNet entry -> down_blocks.0/1/2/3 -> mid_block.resnets.0 bridge alignment'),
    ]
    midblock_resnet0_statuses = []
    for check, path, note in midblock_resnet0_checks:
        st = status_from_json(path)
        midblock_resnet0_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_midblock_resnet0.py && python3 tools/export_tadsr_unet_midblock_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    midblock_resnet0_meta = load_json('experiments/full_repro/unet_alignment/oracle_tensors_unet_midblock_resnet0/unet_midblock_resnet0_oracle_metadata.json')
    if midblock_resnet0_meta.get('resnet0_config', {}).get('has_shortcut') is False:
        midblock_resnet0_shortcut_status = 'NOT_APPLICABLE'
    else:
        midblock_resnet0_shortcut_status = status_from_json('experiments/full_repro/unet_alignment/jittor_unet_midblock_resnet0_alignment.json', default='BLOCKED')
    rows.append(row('TADSR_UNET_MIDBLOCK_RESNET0_SHORTCUT_ALIGNMENT', midblock_resnet0_shortcut_status, 'mid_block.resnets.0 shortcut alignment; PASS if conv_shortcut exists, NOT_APPLICABLE if official shortcut is absent', 'continue only if shortcut remains PASS or NOT_APPLICABLE'))
    required_midblock_resnet0 = [s for s in midblock_resnet0_statuses if s not in {'NOT_APPLICABLE'}]
    midblock_resnet0_pass = bool(required_midblock_resnet0) and all(st == 'PASS' for st in required_midblock_resnet0) and midblock_resnet0_shortcut_status in {'PASS', 'NOT_APPLICABLE'}
    rows.append(row('TADSR_UNET_MIDBLOCK_RESNET0_BRIDGE_ALIGNMENT', 'PASS' if midblock_resnet0_pass else 'PARTIAL', 'mid_block.resnets.0 aligned after complete local down_blocks.0/1/2/3 bridge; attention/resnet1 remain unopened', 'continue with mid_block.attentions.0 if present'))

    midblock_attention0_checks = [
        ('TADSR_UNET_MIDBLOCK_ATTENTION0_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_midblock_attention0.json', 'official mid_block.attentions.0 audit'),
        ('TADSR_UNET_MIDBLOCK_ATTENTION0_LORA_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_midblock_attention0.json', 'LoRA/effective weight state for mid_block.attentions.0'),
        ('TADSR_UNET_MIDBLOCK_ATTENTION0_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_midblock_attention0/unet_midblock_attention0_oracle_metadata.json', 'PyTorch oracle tensors for mid_block.attentions.0 exported'),
        ('TADSR_UNET_MIDBLOCK_ATTENTION0_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_midblock_attention0/unet_midblock_attention0_oracle_metadata.json', 'effective static weights for mid_block.attentions.0 exported locally'),
        ('TADSR_UNET_MIDBLOCK_ATTENTION0_NORM_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_midblock_attention0_proj_alignment.json', 'mid_block.attentions.0 top-level norm alignment'),
        ('TADSR_UNET_MIDBLOCK_ATTENTION0_PROJ_IN_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_midblock_attention0_proj_alignment.json', 'mid_block.attentions.0 proj_in alignment'),
        ('TADSR_UNET_MIDBLOCK_ATTENTION0_SEQUENCE_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_midblock_attention0_proj_alignment.json', 'mid_block.attentions.0 NCHW-to-sequence alignment'),
        ('TADSR_UNET_MIDBLOCK_ATTENTION0_ATTN1_QKV_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_midblock_attention0_attn1_alignment.json', 'mid_block.attentions.0 self-attention QKV alignment'),
        ('TADSR_UNET_MIDBLOCK_ATTENTION0_ATTN1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_midblock_attention0_attn1_alignment.json', 'mid_block.attentions.0 self-attention output alignment'),
        ('TADSR_UNET_MIDBLOCK_ATTENTION0_AFTER_ATTN1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_midblock_attention0_attn1_alignment.json', 'mid_block.attentions.0 after-attn1 residual alignment'),
        ('TADSR_UNET_MIDBLOCK_ATTENTION0_ATTN2_QKV_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_midblock_attention0_attn2_alignment.json', 'mid_block.attentions.0 cross-attention QKV alignment'),
        ('TADSR_UNET_MIDBLOCK_ATTENTION0_ATTN2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_midblock_attention0_attn2_alignment.json', 'mid_block.attentions.0 cross-attention output alignment'),
        ('TADSR_UNET_MIDBLOCK_ATTENTION0_AFTER_ATTN2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_midblock_attention0_attn2_alignment.json', 'mid_block.attentions.0 after-attn2 residual alignment'),
        ('TADSR_UNET_MIDBLOCK_ATTENTION0_FF_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_midblock_attention0_ff_alignment.json', 'mid_block.attentions.0 feed-forward output alignment'),
        ('TADSR_UNET_MIDBLOCK_ATTENTION0_TRANSFORMER0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_midblock_attention0_ff_alignment.json', 'mid_block.attentions.0 transformer block alignment'),
        ('TADSR_UNET_MIDBLOCK_ATTENTION0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_midblock_attention0_alignment.json', 'isolated mid_block.attentions.0 alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_RESNET0_ATTENTION0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblocks_midblock_resnet0_attention0_alignment.json', 'UNet entry -> down_blocks.0/1/2/3 -> mid_block.resnets.0 -> mid_block.attentions.0 bridge alignment'),
    ]
    midblock_attention0_statuses = []
    for check, path, note in midblock_attention0_checks:
        st = status_from_json(path)
        midblock_attention0_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_midblock_attention0.py && python3 tools/export_tadsr_unet_midblock_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    midblock_attention0_pass = bool(midblock_attention0_statuses) and all(st == 'PASS' for st in midblock_attention0_statuses)
    rows.append(row('TADSR_UNET_MIDBLOCK_ATTENTION0_BRIDGE_ALIGNMENT', 'PASS' if midblock_attention0_pass else 'PARTIAL', 'mid_block.attentions.0 aligned in isolated and complete pre-attention bridge paths; resnet1 remains unopened', 'continue with mid_block.resnets.1'))
    rows.append(row('TADSR_UNET_MIDBLOCK_PRE_RESNET1_ALIGNMENT', 'PASS' if (midblock_resnet0_pass and midblock_attention0_pass) else 'PARTIAL', 'UNet path through mid_block.resnets.0 and mid_block.attentions.0 is aligned; mid_block.resnets.1 is evaluated separately below', 'continue with mid_block.resnets.1'))

    midblock_resnet1_checks = [
        ('TADSR_UNET_MIDBLOCK_RESNET1_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_midblock_resnet1.json', 'official mid_block.resnets.1 audit'),
        ('TADSR_UNET_MIDBLOCK_LOCAL_FORWARD_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_midblock_resnet1.json', 'official local mid_block forward audit'),
        ('TADSR_UNET_MIDBLOCK_RESNET1_LORA_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_midblock_resnet1.json', 'LoRA/effective weight state for mid_block.resnets.1'),
        ('TADSR_UNET_MIDBLOCK_RESNET1_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_midblock_resnet1/unet_midblock_resnet1_oracle_metadata.json', 'PyTorch oracle tensors for mid_block.resnets.1 exported'),
        ('TADSR_UNET_MIDBLOCK_RESNET1_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_midblock_resnet1/unet_midblock_resnet1_oracle_metadata.json', 'effective static weights for mid_block.resnets.1 exported locally'),
        ('TADSR_UNET_MIDBLOCK_RESNET1_NORM1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_midblock_resnet1_alignment.json', 'mid_block.resnets.1 norm1 alignment'),
        ('TADSR_UNET_MIDBLOCK_RESNET1_CONV1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_midblock_resnet1_alignment.json', 'mid_block.resnets.1 conv1 alignment'),
        ('TADSR_UNET_MIDBLOCK_RESNET1_TEMB_PROJ_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_midblock_resnet1_alignment.json', 'mid_block.resnets.1 time_emb_proj alignment'),
        ('TADSR_UNET_MIDBLOCK_RESNET1_CONV2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_midblock_resnet1_alignment.json', 'mid_block.resnets.1 conv2 alignment'),
        ('TADSR_UNET_MIDBLOCK_RESNET1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_midblock_resnet1_alignment.json', 'isolated entry-input mid_block.resnets.1 alignment'),
        ('TADSR_UNET_MIDBLOCK_RESNET1_SYNTHETIC_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_midblock_resnet1_synthetic_alignment.json', 'synthetic hidden/temb -> mid_block.resnets.1 alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_RESNET0_ATTENTION0_RESNET1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblocks_midblock_resnet0_attention0_resnet1_alignment.json', 'UNet entry -> down_blocks.0/1/2/3 -> mid_block.resnets.0 -> attention0 -> resnets.1 bridge alignment'),
        ('TADSR_UNET_MIDBLOCK_OUTPUT_HIDDEN_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_midblock_local_alignment.json', 'manual local mid_block hidden output vs official local mid_block forward'),
    ]
    midblock_resnet1_statuses = []
    for check, path, note in midblock_resnet1_checks:
        st = status_from_json(path)
        midblock_resnet1_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_midblock_resnet1.py && python3 tools/export_tadsr_unet_midblock_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    midblock_resnet1_meta = load_json('experiments/full_repro/unet_alignment/oracle_tensors_unet_midblock_resnet1/unet_midblock_resnet1_oracle_metadata.json')
    if midblock_resnet1_meta.get('resnet1_config', {}).get('has_shortcut') is False:
        midblock_resnet1_shortcut_status = 'NOT_APPLICABLE'
    else:
        midblock_resnet1_shortcut_status = status_from_json('experiments/full_repro/unet_alignment/jittor_unet_midblock_resnet1_alignment.json', default='BLOCKED')
    rows.append(row('TADSR_UNET_MIDBLOCK_RESNET1_SHORTCUT_ALIGNMENT', midblock_resnet1_shortcut_status, 'mid_block.resnets.1 shortcut alignment; PASS if conv_shortcut exists, NOT_APPLICABLE if official shortcut is absent', 'continue only if shortcut remains PASS or NOT_APPLICABLE'))
    local_midblock = load_json('experiments/full_repro/unet_alignment/jittor_unet_midblock_local_alignment.json')
    output_states_status = local_midblock.get('output_states_status', midblock_resnet1_meta.get('local_midblock_output_states_status', 'BLOCKED'))
    rows.append(row('TADSR_UNET_MIDBLOCK_OUTPUT_STATES_ALIGNMENT', output_states_status, 'official local mid_block output_states status; NOT_APPLICABLE when mid_block returns hidden_states only', 'do not invent output_states'))
    midblock_resnet1_required = [s for s in midblock_resnet1_statuses if s not in {'NOT_APPLICABLE'}]
    midblock_resnet1_pass = bool(midblock_resnet1_required) and all(st == 'PASS' for st in midblock_resnet1_required) and midblock_resnet1_shortcut_status in {'PASS', 'NOT_APPLICABLE'} and output_states_status in {'PASS', 'NOT_APPLICABLE'}
    rows.append(row('TADSR_UNET_MIDBLOCK_BRIDGE_ALIGNMENT', 'PASS' if (midblock_resnet0_pass and midblock_attention0_pass and midblock_resnet1_pass) else 'PARTIAL', 'complete local mid_block chain resnets.0 -> attentions.0 -> resnets.1 aligned; up_blocks remain unopened', 'continue with up_blocks.0 if PASS'))
    rows.append(row('TADSR_UNET_MIDBLOCK_ALIGNMENT', 'PASS' if (midblock_resnet0_pass and midblock_attention0_pass and midblock_resnet1_pass) else 'PARTIAL', 'local mid_block hidden output aligned through resnets.1; this is not full UNet forward', 'continue with up_blocks.0'))


    upblock0_resnet0_checks = [
        ('TADSR_UNET_UPBLOCK0_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock0_resnet0.json', 'official up_blocks.0 topology audit'),
        ('TADSR_UNET_UPBLOCK0_RESNET0_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock0_resnet0.json', 'official up_blocks.0.resnets.0 audit'),
        ('TADSR_UNET_UPBLOCK0_RESNET0_RESIDUAL_CONTRACT_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock0_resnet0.json', 'official residual pop/concat contract audit'),
        ('TADSR_UNET_UPBLOCK0_RESNET0_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock0_resnet0/unet_upblock0_resnet0_oracle_metadata.json', 'PyTorch oracle tensors for up_blocks.0.resnets.0 exported'),
        ('TADSR_UNET_UPBLOCK0_RESNET0_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock0_resnet0/unet_upblock0_resnet0_oracle_metadata.json', 'effective static weights for up_blocks.0.resnets.0 exported locally'),
        ('TADSR_UNET_UPBLOCK0_RESNET0_RESIDUAL_CONCAT_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock0_resnet0_alignment.json', 'up_blocks.0.resnets.0 hidden/residual concat alignment'),
        ('TADSR_UNET_UPBLOCK0_RESNET0_NORM1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock0_resnet0_alignment.json', 'up_blocks.0.resnets.0 norm1 alignment'),
        ('TADSR_UNET_UPBLOCK0_RESNET0_CONV1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock0_resnet0_alignment.json', 'up_blocks.0.resnets.0 conv1 alignment'),
        ('TADSR_UNET_UPBLOCK0_RESNET0_TEMB_PROJ_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock0_resnet0_alignment.json', 'up_blocks.0.resnets.0 time_emb_proj alignment'),
        ('TADSR_UNET_UPBLOCK0_RESNET0_CONV2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock0_resnet0_alignment.json', 'up_blocks.0.resnets.0 conv2 alignment'),
        ('TADSR_UNET_UPBLOCK0_RESNET0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock0_resnet0_alignment.json', 'isolated up_blocks.0.resnets.0 after official residual concat alignment'),
        ('TADSR_UNET_UPBLOCK0_RESNET0_SYNTHETIC_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock0_resnet0_synthetic_alignment.json', 'synthetic hidden/residual -> up_blocks.0.resnets.0 alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_RESNET0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblocks_midblock_upblock0_resnet0_alignment.json', 'UNet entry -> down_blocks.0/1/2/3 -> mid_block -> up_blocks.0.resnets.0 bridge alignment'),
    ]
    upblock0_resnet0_statuses = []
    for check, path, note in upblock0_resnet0_checks:
        st = status_from_json(path)
        upblock0_resnet0_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_upblock0_resnet0.py && python3 tools/export_tadsr_unet_upblock0_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    upblock0_resnet0_meta = load_json('experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock0_resnet0/unet_upblock0_resnet0_oracle_metadata.json')
    if upblock0_resnet0_meta.get('resnet0_config', {}).get('has_shortcut') is False:
        upblock0_resnet0_shortcut_status = 'NOT_APPLICABLE'
    else:
        upblock0_resnet0_shortcut_status = status_from_json('experiments/full_repro/unet_alignment/jittor_unet_upblock0_resnet0_alignment.json', default='BLOCKED')
    rows.append(row('TADSR_UNET_UPBLOCK0_RESNET0_SHORTCUT_ALIGNMENT', upblock0_resnet0_shortcut_status, 'up_blocks.0.resnets.0 shortcut alignment; PASS if conv_shortcut exists, NOT_APPLICABLE if official shortcut is absent', 'continue only if shortcut remains PASS or NOT_APPLICABLE'))
    upblock0_resnet0_required = [s for s in upblock0_resnet0_statuses if s not in {'NOT_APPLICABLE'}]
    upblock0_resnet0_pass = bool(upblock0_resnet0_required) and all(st == 'PASS' for st in upblock0_resnet0_required) and upblock0_resnet0_shortcut_status in {'PASS', 'NOT_APPLICABLE'}
    rows.append(row('TADSR_UNET_UPBLOCK0_RESNET0_BRIDGE_ALIGNMENT', 'PASS' if upblock0_resnet0_pass else 'PARTIAL', 'first up path leaf up_blocks.0.resnets.0 aligned after official residual consumption; full up_blocks.0 remains unopened', 'continue with actual next module after up_blocks.0.resnets.0'))


    upblock0_resnet1_checks = [
        ('TADSR_UNET_UPBLOCK0_RESNET1_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock0_resnet1.json', 'official up_blocks.0.resnets.1 audit'),
        ('TADSR_UNET_UPBLOCK0_RESNET1_RESIDUAL_CONTRACT_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock0_resnet1.json', 'official residual pop/concat contract audit for second upblock residual'),
        ('TADSR_UNET_UPBLOCK0_RESNET1_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock0_resnet1/unet_upblock0_resnet1_oracle_metadata.json', 'PyTorch oracle tensors for up_blocks.0.resnets.1 exported'),
        ('TADSR_UNET_UPBLOCK0_RESNET1_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock0_resnet1/unet_upblock0_resnet1_oracle_metadata.json', 'effective static weights for up_blocks.0.resnets.1 exported locally'),
        ('TADSR_UNET_UPBLOCK0_RESNET1_RESIDUAL_CONCAT_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock0_resnet1_alignment.json', 'up_blocks.0.resnets.1 hidden/residual concat alignment'),
        ('TADSR_UNET_UPBLOCK0_RESNET1_NORM1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock0_resnet1_alignment.json', 'up_blocks.0.resnets.1 norm1 alignment'),
        ('TADSR_UNET_UPBLOCK0_RESNET1_CONV1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock0_resnet1_alignment.json', 'up_blocks.0.resnets.1 conv1 alignment'),
        ('TADSR_UNET_UPBLOCK0_RESNET1_TEMB_PROJ_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock0_resnet1_alignment.json', 'up_blocks.0.resnets.1 time_emb_proj alignment'),
        ('TADSR_UNET_UPBLOCK0_RESNET1_CONV2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock0_resnet1_alignment.json', 'up_blocks.0.resnets.1 conv2 alignment'),
        ('TADSR_UNET_UPBLOCK0_RESNET1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock0_resnet1_alignment.json', 'isolated up_blocks.0.resnets.1 after official residual concat alignment'),
        ('TADSR_UNET_UPBLOCK0_RESNET1_SYNTHETIC_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock0_resnet1_synthetic_alignment.json', 'synthetic hidden/residual -> up_blocks.0.resnets.1 alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_RESNET0_RESNET1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblocks_midblock_upblock0_resnet0_resnet1_alignment.json', 'UNet entry -> down_blocks.0/1/2/3 -> mid_block -> up_blocks.0.resnets.0 -> resnets.1 bridge alignment'),
    ]
    upblock0_resnet1_statuses = []
    for check, path, note in upblock0_resnet1_checks:
        st = status_from_json(path)
        upblock0_resnet1_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_upblock0_resnet1.py && python3 tools/export_tadsr_unet_upblock0_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    upblock0_resnet1_meta = load_json('experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock0_resnet1/unet_upblock0_resnet1_oracle_metadata.json')
    if upblock0_resnet1_meta.get('resnet1_config', {}).get('has_shortcut') is False:
        upblock0_resnet1_shortcut_status = 'NOT_APPLICABLE'
    else:
        upblock0_resnet1_shortcut_status = status_from_json('experiments/full_repro/unet_alignment/jittor_unet_upblock0_resnet1_alignment.json', default='BLOCKED')
    rows.append(row('TADSR_UNET_UPBLOCK0_RESNET1_SHORTCUT_ALIGNMENT', upblock0_resnet1_shortcut_status, 'up_blocks.0.resnets.1 shortcut alignment; PASS if conv_shortcut exists, NOT_APPLICABLE if official shortcut is absent', 'continue only if shortcut remains PASS or NOT_APPLICABLE'))
    upblock0_resnet1_required = [s for s in upblock0_resnet1_statuses if s not in {'NOT_APPLICABLE'}]
    upblock0_resnet1_pass = bool(upblock0_resnet1_required) and all(st == 'PASS' for st in upblock0_resnet1_required) and upblock0_resnet1_shortcut_status in {'PASS', 'NOT_APPLICABLE'}
    rows.append(row('TADSR_UNET_UPBLOCK0_RESNET1_BRIDGE_ALIGNMENT', 'PASS' if (upblock0_resnet0_pass and upblock0_resnet1_pass) else 'PARTIAL', 'up_blocks.0.resnets.1 aligned after official second residual consumption; full up_blocks.0 remains unopened', 'continue with actual next module after up_blocks.0.resnets.1'))
    rows.append(row('TADSR_UNET_UPBLOCK0_PRE_RESNET2_ALIGNMENT', 'PASS' if (upblock0_resnet0_pass and upblock0_resnet1_pass) else 'PARTIAL', 'up_blocks.0 path is aligned through resnets.0 -> resnets.1 only', 'continue with up_blocks.0.resnets.2'))

    upblock0_resnet2_checks = [
        ('TADSR_UNET_UPBLOCK0_RESNET2_AUDIT', 'experiments/full_repro/unet_alignment/audit_official_tadsr_unet_upblock0_resnet2.json', 'official up_blocks.0.resnets.2 audit'),
        ('TADSR_UNET_UPBLOCK0_RESNET2_RESIDUAL_CONTRACT_AUDIT', 'experiments/full_repro/unet_alignment/audit_official_tadsr_unet_upblock0_resnet2.json', 'official residual pop/concat contract audit for third upblock residual'),
        ('TADSR_UNET_UPBLOCK0_RESNET2_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock0_resnet2/unet_upblock0_resnet2_oracle_metadata.json', 'PyTorch oracle tensors for up_blocks.0.resnets.2 exported'),
        ('TADSR_UNET_UPBLOCK0_RESNET2_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock0_resnet2/unet_upblock0_resnet2_oracle_metadata.json', 'effective static weights for up_blocks.0.resnets.2 exported locally'),
        ('TADSR_UNET_UPBLOCK0_RESNET2_RESIDUAL_CONCAT_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock0_resnet2_alignment.json', 'up_blocks.0.resnets.2 hidden/residual concat alignment'),
        ('TADSR_UNET_UPBLOCK0_RESNET2_NORM1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock0_resnet2_alignment.json', 'up_blocks.0.resnets.2 norm1 alignment'),
        ('TADSR_UNET_UPBLOCK0_RESNET2_CONV1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock0_resnet2_alignment.json', 'up_blocks.0.resnets.2 conv1 alignment'),
        ('TADSR_UNET_UPBLOCK0_RESNET2_TEMB_PROJ_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock0_resnet2_alignment.json', 'up_blocks.0.resnets.2 time_emb_proj alignment'),
        ('TADSR_UNET_UPBLOCK0_RESNET2_CONV2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock0_resnet2_alignment.json', 'up_blocks.0.resnets.2 conv2 alignment'),
        ('TADSR_UNET_UPBLOCK0_RESNET2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock0_resnet2_alignment.json', 'isolated up_blocks.0.resnets.2 after official residual concat alignment'),
        ('TADSR_UNET_UPBLOCK0_RESNET2_SYNTHETIC_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock0_resnet2_synthetic_alignment.json', 'synthetic hidden/residual -> up_blocks.0.resnets.2 alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_RESNET0_RESNET1_RESNET2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblocks_midblock_upblock0_resnet0_resnet1_resnet2_alignment.json', 'UNet entry -> down_blocks.0/1/2/3 -> mid_block -> up_blocks.0.resnets.0 -> resnets.1 -> resnets.2 bridge alignment'),
    ]
    upblock0_resnet2_statuses = []
    for check, path, note in upblock0_resnet2_checks:
        st = status_from_json(path)
        upblock0_resnet2_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_upblock0_resnet2.py && python3 tools/export_tadsr_unet_upblock0_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    upblock0_resnet2_meta = load_json('experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock0_resnet2/unet_upblock0_resnet2_oracle_metadata.json')
    if upblock0_resnet2_meta.get('resnet2_config', {}).get('has_shortcut') is False:
        upblock0_resnet2_shortcut_status = 'NOT_APPLICABLE'
    else:
        upblock0_resnet2_shortcut_status = status_from_json('experiments/full_repro/unet_alignment/jittor_unet_upblock0_resnet2_alignment.json', default='BLOCKED')
    rows.append(row('TADSR_UNET_UPBLOCK0_RESNET2_SHORTCUT_ALIGNMENT', upblock0_resnet2_shortcut_status, 'up_blocks.0.resnets.2 shortcut alignment; PASS if conv_shortcut exists, NOT_APPLICABLE if official shortcut is absent', 'continue only if shortcut remains PASS or NOT_APPLICABLE'))
    upblock0_resnet2_required = [s for s in upblock0_resnet2_statuses if s not in {'NOT_APPLICABLE'}]
    upblock0_resnet2_pass = bool(upblock0_resnet2_required) and all(st == 'PASS' for st in upblock0_resnet2_required) and upblock0_resnet2_shortcut_status in {'PASS', 'NOT_APPLICABLE'}
    rows.append(row('TADSR_UNET_UPBLOCK0_RESNET2_BRIDGE_ALIGNMENT', 'PASS' if (upblock0_resnet0_pass and upblock0_resnet1_pass and upblock0_resnet2_pass) else 'PARTIAL', 'up_blocks.0.resnets.2 aligned after official third residual consumption; full up_blocks.0 remains unopened', 'continue with up_blocks.0.upsamplers.0'))
    rows.append(row('TADSR_UNET_UPBLOCK0_PRE_UPSAMPLER_ALIGNMENT', 'PASS' if (upblock0_resnet0_pass and upblock0_resnet1_pass and upblock0_resnet2_pass) else 'PARTIAL', 'up_blocks.0 path is aligned through resnets.0 -> resnets.1 -> resnets.2 only', 'continue with up_blocks.0.upsamplers.0'))

    upblock0_upsampler_checks = [
        ('TADSR_UNET_UPBLOCK0_UPSAMPLER_AUDIT', 'experiments/full_repro/unet_alignment/audit_official_tadsr_unet_upblock0_upsampler.json', 'official up_blocks.0.upsamplers.0 audit'),
        ('TADSR_UNET_UPBLOCK0_LOCAL_FORWARD_AUDIT', 'experiments/full_repro/unet_alignment/audit_official_tadsr_unet_upblock0_upsampler.json', 'safe official local up_blocks.0 hidden-state forward audit'),
        ('TADSR_UNET_UPBLOCK0_UPSAMPLER_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock0_upsampler/unet_upblock0_upsampler_oracle_metadata.json', 'PyTorch oracle tensors for up_blocks.0.upsamplers.0 exported'),
        ('TADSR_UNET_UPBLOCK0_UPSAMPLER_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock0_upsampler/unet_upblock0_upsampler_oracle_metadata.json', 'effective static weights for up_blocks.0.upsamplers.0 exported locally'),
        ('TADSR_UNET_UPBLOCK0_LOCAL_FORWARD_ORACLE', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock0_upsampler/unet_upblock0_upsampler_oracle_metadata.json', 'PyTorch oracle for local up_blocks.0 hidden-state output exported'),
        ('TADSR_UNET_UPBLOCK0_UPSAMPLER_INTERPOLATION_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock0_upsampler_alignment.json', 'nearest-2x interpolation input to upsampler conv alignment'),
        ('TADSR_UNET_UPBLOCK0_UPSAMPLER_CONV_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock0_upsampler_alignment.json', 'effective upsampler conv output alignment'),
        ('TADSR_UNET_UPBLOCK0_UPSAMPLER_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock0_upsampler_alignment.json', 'isolated up_blocks.0.upsamplers.0 alignment'),
        ('TADSR_UNET_UPBLOCK0_UPSAMPLER_SYNTHETIC_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock0_upsampler_synthetic_alignment.json', 'synthetic hidden -> up_blocks.0.upsamplers.0 alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_RESNETS012_UPSAMPLER_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblocks_midblock_upblock0_resnets012_upsampler_alignment.json', 'UNet entry -> all down_blocks -> full local mid_block -> up_blocks.0.resnets.0/1/2 -> upsamplers.0 bridge alignment'),
        ('TADSR_UNET_UPBLOCK0_OUTPUT_HIDDEN_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock0_local_alignment.json', 'full local up_blocks.0 hidden-state output alignment'),
        ('TADSR_UNET_UPBLOCK0_OUTPUT_STATES_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock0_local_alignment.json', 'up_blocks.0 output-state contract alignment; official UpBlock2D returns hidden states only'),
    ]
    upblock0_upsampler_statuses = []
    for check, path, note in upblock0_upsampler_checks:
        st = status_from_json(path)
        upblock0_upsampler_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_upblock0_upsampler.py && python3 tools/export_tadsr_unet_upblock0_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    upblock0_upsampler_required = [s for s in upblock0_upsampler_statuses if s not in {'NOT_APPLICABLE'}]
    upblock0_upsampler_pass = bool(upblock0_upsampler_required) and all(st == 'PASS' for st in upblock0_upsampler_required)
    upblock0_local_pass = status_from_json('experiments/full_repro/unet_alignment/jittor_unet_upblock0_local_alignment.json') == 'PASS'
    rows.append(row('TADSR_UNET_UPBLOCK0_BRIDGE_ALIGNMENT', 'PASS' if (upblock0_resnet0_pass and upblock0_resnet1_pass and upblock0_resnet2_pass and upblock0_upsampler_pass) else 'PARTIAL', 'up_blocks.0 path is aligned through resnets.0 -> resnets.1 -> resnets.2 -> upsamplers.0', 'continue with up_blocks.1.resnets.0'))
    rows.append(row('TADSR_UNET_UPBLOCK0_ALIGNMENT', 'PASS' if (upblock0_resnet0_pass and upblock0_resnet1_pass and upblock0_resnet2_pass and upblock0_upsampler_pass and upblock0_local_pass) else 'PARTIAL', 'Full local up_blocks.0 hidden output is aligned; later up_blocks remain unopened', 'continue with up_blocks.1.resnets.0'))

    upblock1_resnet0_checks = [
        ('TADSR_UNET_UPBLOCK1_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock1_resnet0.json', 'official up_blocks.1 topology audit through resnets.0 only'),
        ('TADSR_UNET_UPBLOCK1_RESNET0_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock1_resnet0.json', 'official up_blocks.1.resnets.0 audit'),
        ('TADSR_UNET_UPBLOCK1_RESNET0_RESIDUAL_CONTRACT_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock1_resnet0.json', 'official residual pop/concat contract audit after local up_blocks.0'),
        ('TADSR_UNET_UPBLOCK1_RESNET0_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock1_resnet0/unet_upblock1_resnet0_oracle_metadata.json', 'PyTorch oracle tensors for up_blocks.1.resnets.0 exported'),
        ('TADSR_UNET_UPBLOCK1_RESNET0_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock1_resnet0/unet_upblock1_resnet0_oracle_metadata.json', 'effective static weights for up_blocks.1.resnets.0 exported locally'),
        ('TADSR_UNET_UPBLOCK1_RESNET0_RESIDUAL_CONCAT_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_resnet0_alignment.json', 'up_blocks.1.resnets.0 hidden/residual concat alignment'),
        ('TADSR_UNET_UPBLOCK1_RESNET0_NORM1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_resnet0_alignment.json', 'up_blocks.1.resnets.0 norm1 alignment'),
        ('TADSR_UNET_UPBLOCK1_RESNET0_CONV1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_resnet0_alignment.json', 'up_blocks.1.resnets.0 conv1 alignment'),
        ('TADSR_UNET_UPBLOCK1_RESNET0_TEMB_PROJ_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_resnet0_alignment.json', 'up_blocks.1.resnets.0 time_emb_proj alignment'),
        ('TADSR_UNET_UPBLOCK1_RESNET0_CONV2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_resnet0_alignment.json', 'up_blocks.1.resnets.0 conv2 alignment'),
        ('TADSR_UNET_UPBLOCK1_RESNET0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_resnet0_alignment.json', 'isolated up_blocks.1.resnets.0 after official residual concat alignment'),
        ('TADSR_UNET_UPBLOCK1_RESNET0_SYNTHETIC_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_resnet0_synthetic_alignment.json', 'synthetic hidden/residual -> up_blocks.1.resnets.0 alignment'),
        ('TADSR_UNET_UPBLOCK1_RESNET0_BRIDGE_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblocks_midblock_upblock0_upblock1_resnet0_alignment.json', 'UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> up_blocks.1.resnets.0 bridge alignment'),
    ]
    upblock1_resnet0_statuses = []
    for check, path, note in upblock1_resnet0_checks:
        st = status_from_json(path)
        upblock1_resnet0_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_upblock1_resnet0.py && python3 tools/export_tadsr_unet_upblock1_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    upblock1_resnet0_meta = load_json('experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock1_resnet0/unet_upblock1_resnet0_oracle_metadata.json')
    if upblock1_resnet0_meta.get('resnet0_config', {}).get('has_shortcut') is False:
        upblock1_resnet0_shortcut_status = 'NOT_APPLICABLE'
    else:
        upblock1_resnet0_shortcut_status = status_from_json('experiments/full_repro/unet_alignment/jittor_unet_upblock1_resnet0_alignment.json', default='BLOCKED')
    rows.append(row('TADSR_UNET_UPBLOCK1_RESNET0_SHORTCUT_ALIGNMENT', upblock1_resnet0_shortcut_status, 'up_blocks.1.resnets.0 shortcut alignment; PASS if conv_shortcut exists, NOT_APPLICABLE if official shortcut is absent', 'continue only if shortcut remains PASS or NOT_APPLICABLE'))
    upblock1_resnet0_required = [s for s in upblock1_resnet0_statuses if s not in {'NOT_APPLICABLE'}]
    upblock1_resnet0_pass = bool(upblock1_resnet0_required) and all(st == 'PASS' for st in upblock1_resnet0_required) and upblock1_resnet0_shortcut_status in {'PASS', 'NOT_APPLICABLE'}
    rows.append(row('TADSR_UNET_UPBLOCK1_PRE_ATTENTION0_ALIGNMENT', 'PASS' if (upblock0_resnet0_pass and upblock0_resnet1_pass and upblock0_resnet2_pass and upblock0_upsampler_pass and upblock0_local_pass and upblock1_resnet0_pass) else 'PARTIAL', 'up_blocks.1 path is aligned through resnets.0 only', 'continue with up_blocks.1.attentions.0'))

    upblock1_attention0_checks = [
        ('TADSR_UNET_UPBLOCK1_ATTENTION0_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock1_attention0.json', 'official up_blocks.1.attentions.0 topology/LoRA audit'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION0_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock1_attention0/unet_upblock1_attention0_oracle_metadata.json', 'PyTorch oracle tensors for up_blocks.1.attentions.0 exported'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION0_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock1_attention0/unet_upblock1_attention0_oracle_metadata.json', 'effective static weights for up_blocks.1.attentions.0 exported locally'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION0_NORM_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_attention0_proj_alignment.json', 'up_blocks.1.attentions.0 group norm alignment'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION0_SEQUENCE_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_attention0_proj_alignment.json', 'up_blocks.1.attentions.0 NCHW to sequence reshape alignment'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION0_PROJ_IN_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_attention0_proj_alignment.json', 'up_blocks.1.attentions.0 proj_in alignment'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION0_ATTN1_QKV_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_attention0_attn1_alignment.json', 'up_blocks.1.attentions.0 self-attention q/k/v alignment'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION0_ATTN1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_attention0_attn1_alignment.json', 'up_blocks.1.attentions.0 self-attention output alignment'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION0_AFTER_ATTN1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_attention0_attn1_alignment.json', 'up_blocks.1.attentions.0 residual after attn1 alignment'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION0_ATTN2_QKV_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_attention0_attn2_alignment.json', 'up_blocks.1.attentions.0 cross-attention q/k/v alignment'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION0_ATTN2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_attention0_attn2_alignment.json', 'up_blocks.1.attentions.0 cross-attention output alignment'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION0_AFTER_ATTN2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_attention0_attn2_alignment.json', 'up_blocks.1.attentions.0 residual after attn2 alignment'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION0_FF_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_attention0_ff_alignment.json', 'up_blocks.1.attentions.0 feed-forward output alignment'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION0_TRANSFORMER0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_attention0_ff_alignment.json', 'up_blocks.1.attentions.0 transformer block output alignment'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_attention0_alignment.json', 'isolated up_blocks.1.attentions.0 output alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_RESNET0_ATTENTION0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblocks_midblock_upblock0_upblock1_resnet0_attention0_alignment.json', 'UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> up_blocks.1.resnets.0 -> up_blocks.1.attentions.0 bridge alignment'),
    ]
    upblock1_attention0_statuses = []
    for check, path, note in upblock1_attention0_checks:
        st = status_from_json(path)
        upblock1_attention0_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_upblock1_attention0.py && python3 tools/export_tadsr_unet_upblock1_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    upblock1_attention0_pass = bool(upblock1_attention0_statuses) and all(st == 'PASS' for st in upblock1_attention0_statuses)
    rows.append(row('TADSR_UNET_UPBLOCK1_ATTENTION0_BRIDGE_ALIGNMENT', 'PASS' if upblock1_attention0_pass else 'PARTIAL', 'up_blocks.1 path is aligned through resnets.0 -> attentions.0 only', 'continue with up_blocks.1.resnets.1'))
    rows.append(row('TADSR_UNET_UPBLOCK1_PRE_RESNET1_ALIGNMENT', 'PASS' if upblock1_attention0_pass else 'PARTIAL', 'input to next unopened up_blocks.1.resnets.1 is now aligned', 'continue with up_blocks.1.resnets.1'))

    upblock1_resnet1_checks = [
        ('TADSR_UNET_UPBLOCK1_RESNET1_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock1_resnet1.json', 'official up_blocks.1.resnets.1 audit'),
        ('TADSR_UNET_UPBLOCK1_RESNET1_RESIDUAL_CONTRACT_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock1_resnet1.json', 'official residual pop/concat contract audit after up_blocks.1.attentions.0'),
        ('TADSR_UNET_UPBLOCK1_RESNET1_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock1_resnet1/unet_upblock1_resnet1_oracle_metadata.json', 'PyTorch oracle tensors for up_blocks.1.resnets.1 exported'),
        ('TADSR_UNET_UPBLOCK1_RESNET1_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock1_resnet1/unet_upblock1_resnet1_oracle_metadata.json', 'effective static weights for up_blocks.1.resnets.1 exported locally'),
        ('TADSR_UNET_UPBLOCK1_RESNET1_RESIDUAL_CONCAT_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_resnet1_alignment.json', 'up_blocks.1.resnets.1 hidden/residual concat alignment'),
        ('TADSR_UNET_UPBLOCK1_RESNET1_NORM1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_resnet1_alignment.json', 'up_blocks.1.resnets.1 norm1 alignment'),
        ('TADSR_UNET_UPBLOCK1_RESNET1_CONV1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_resnet1_alignment.json', 'up_blocks.1.resnets.1 conv1 alignment'),
        ('TADSR_UNET_UPBLOCK1_RESNET1_TEMB_PROJ_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_resnet1_alignment.json', 'up_blocks.1.resnets.1 time_emb_proj alignment'),
        ('TADSR_UNET_UPBLOCK1_RESNET1_CONV2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_resnet1_alignment.json', 'up_blocks.1.resnets.1 conv2 alignment'),
        ('TADSR_UNET_UPBLOCK1_RESNET1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_resnet1_alignment.json', 'isolated up_blocks.1.resnets.1 after official residual concat alignment'),
        ('TADSR_UNET_UPBLOCK1_RESNET1_SYNTHETIC_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_resnet1_synthetic_alignment.json', 'synthetic hidden/residual -> up_blocks.1.resnets.1 alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_RESNET0_ATTENTION0_RESNET1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblocks_midblock_upblock0_upblock1_resnet0_attention0_resnet1_alignment.json', 'UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> up_blocks.1.resnets.0 -> up_blocks.1.attentions.0 -> up_blocks.1.resnets.1 bridge alignment'),
    ]
    upblock1_resnet1_statuses = []
    for check, path, note in upblock1_resnet1_checks:
        st = status_from_json(path)
        upblock1_resnet1_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_upblock1_resnet1.py && python3 tools/export_tadsr_unet_upblock1_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    upblock1_resnet1_meta = load_json('experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock1_resnet1/unet_upblock1_resnet1_oracle_metadata.json')
    if upblock1_resnet1_meta.get('resnet1_config', {}).get('has_shortcut') is False:
        upblock1_resnet1_shortcut_status = 'NOT_APPLICABLE'
    else:
        upblock1_resnet1_shortcut_status = status_from_json('experiments/full_repro/unet_alignment/jittor_unet_upblock1_resnet1_alignment.json', default='BLOCKED')
    rows.append(row('TADSR_UNET_UPBLOCK1_RESNET1_SHORTCUT_ALIGNMENT', upblock1_resnet1_shortcut_status, 'up_blocks.1.resnets.1 shortcut alignment; PASS if conv_shortcut exists, NOT_APPLICABLE if official shortcut is absent', 'continue only if shortcut remains PASS or NOT_APPLICABLE'))
    upblock1_resnet1_required = [s for s in upblock1_resnet1_statuses if s not in {'NOT_APPLICABLE'}]
    upblock1_resnet1_pass = bool(upblock1_resnet1_required) and all(st == 'PASS' for st in upblock1_resnet1_required) and upblock1_resnet1_shortcut_status in {'PASS', 'NOT_APPLICABLE'}
    rows.append(row('TADSR_UNET_UPBLOCK1_RESNET1_BRIDGE_ALIGNMENT', 'PASS' if (upblock1_attention0_pass and upblock1_resnet1_pass) else 'PARTIAL', 'up_blocks.1 path is aligned through resnets.0 -> attentions.0 -> resnets.1 only', 'continue with up_blocks.1.attentions.1'))
    rows.append(row('TADSR_UNET_UPBLOCK1_PRE_ATTENTION1_ALIGNMENT', 'PASS' if (upblock1_attention0_pass and upblock1_resnet1_pass) else 'PARTIAL', 'input to next unopened up_blocks.1.attentions.1 is now aligned', 'continue with up_blocks.1.attentions.1'))

    upblock1_attention1_checks = [
        ('TADSR_UNET_UPBLOCK1_ATTENTION1_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock1_attention1.json', 'official up_blocks.1.attentions.1 topology/LoRA audit'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION1_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock1_attention1/unet_upblock1_attention1_oracle_metadata.json', 'PyTorch oracle tensors for up_blocks.1.attentions.1 exported'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION1_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock1_attention1/unet_upblock1_attention1_oracle_metadata.json', 'effective static weights for up_blocks.1.attentions.1 exported locally'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION1_NORM_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_attention1_proj_alignment.json', 'up_blocks.1.attentions.1 group norm alignment'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION1_SEQUENCE_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_attention1_proj_alignment.json', 'up_blocks.1.attentions.1 NCHW to sequence reshape alignment'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION1_PROJ_IN_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_attention1_proj_alignment.json', 'up_blocks.1.attentions.1 proj_in alignment'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION1_ATTN1_QKV_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_attention1_attn1_alignment.json', 'up_blocks.1.attentions.1 self-attention q/k/v alignment'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION1_ATTN1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_attention1_attn1_alignment.json', 'up_blocks.1.attentions.1 self-attention output alignment'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION1_AFTER_ATTN1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_attention1_attn1_alignment.json', 'up_blocks.1.attentions.1 residual after attn1 alignment'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION1_ATTN2_QKV_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_attention1_attn2_alignment.json', 'up_blocks.1.attentions.1 cross-attention q/k/v alignment'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION1_ATTN2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_attention1_attn2_alignment.json', 'up_blocks.1.attentions.1 cross-attention output alignment'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION1_AFTER_ATTN2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_attention1_attn2_alignment.json', 'up_blocks.1.attentions.1 residual after attn2 alignment'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION1_FF_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_attention1_ff_alignment.json', 'up_blocks.1.attentions.1 feed-forward output alignment'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION1_TRANSFORMER0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_attention1_ff_alignment.json', 'up_blocks.1.attentions.1 transformer block output alignment'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_attention1_alignment.json', 'isolated up_blocks.1.attentions.1 output alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_RESNET0_ATTENTION0_RESNET1_ATTENTION1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblocks_midblock_upblock0_upblock1_resnet0_attention0_resnet1_attention1_alignment.json', 'UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> up_blocks.1.resnets.0 -> up_blocks.1.attentions.0 -> up_blocks.1.resnets.1 -> up_blocks.1.attentions.1 bridge alignment'),
    ]
    upblock1_attention1_statuses = []
    for check, path, note in upblock1_attention1_checks:
        st = status_from_json(path)
        upblock1_attention1_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_upblock1_attention1.py && python3 tools/export_tadsr_unet_upblock1_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    upblock1_attention1_pass = bool(upblock1_attention1_statuses) and all(st == 'PASS' for st in upblock1_attention1_statuses)
    rows.append(row('TADSR_UNET_UPBLOCK1_ATTENTION1_BRIDGE_ALIGNMENT', 'PASS' if (upblock1_attention0_pass and upblock1_resnet1_pass and upblock1_attention1_pass) else 'PARTIAL', 'up_blocks.1 path is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 only', 'continue with up_blocks.1.resnets.2'))
    rows.append(row('TADSR_UNET_UPBLOCK1_PRE_RESNET2_ALIGNMENT', 'PASS' if (upblock1_attention0_pass and upblock1_resnet1_pass and upblock1_attention1_pass) else 'PARTIAL', 'input to next unopened up_blocks.1.resnets.2 is now aligned', 'continue with up_blocks.1.resnets.2'))

    upblock1_resnet2_checks = [
        ('TADSR_UNET_UPBLOCK1_RESNET2_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock1_resnet2.json', 'official up_blocks.1.resnets.2 audit after up_blocks.1.attentions.1'),
        ('TADSR_UNET_UPBLOCK1_RESNET2_RESIDUAL_CONTRACT_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock1_resnet2.json', 'official residual pop/concat contract audit after up_blocks.1.attentions.1'),
        ('TADSR_UNET_UPBLOCK1_RESNET2_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock1_resnet2/unet_upblock1_resnet2_oracle_metadata.json', 'PyTorch oracle tensors for up_blocks.1.resnets.2 exported'),
        ('TADSR_UNET_UPBLOCK1_RESNET2_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock1_resnet2/unet_upblock1_resnet2_oracle_metadata.json', 'effective static weights for up_blocks.1.resnets.2 exported locally'),
        ('TADSR_UNET_UPBLOCK1_RESNET2_RESIDUAL_CONCAT_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_resnet2_alignment.json', 'up_blocks.1.resnets.2 hidden/residual concat alignment'),
        ('TADSR_UNET_UPBLOCK1_RESNET2_NORM1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_resnet2_alignment.json', 'up_blocks.1.resnets.2 norm1 alignment'),
        ('TADSR_UNET_UPBLOCK1_RESNET2_CONV1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_resnet2_alignment.json', 'up_blocks.1.resnets.2 conv1 alignment'),
        ('TADSR_UNET_UPBLOCK1_RESNET2_TEMB_PROJ_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_resnet2_alignment.json', 'up_blocks.1.resnets.2 time_emb_proj alignment'),
        ('TADSR_UNET_UPBLOCK1_RESNET2_CONV2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_resnet2_alignment.json', 'up_blocks.1.resnets.2 conv2 alignment'),
        ('TADSR_UNET_UPBLOCK1_RESNET2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_resnet2_alignment.json', 'isolated up_blocks.1.resnets.2 after attention1 output and official residual concat alignment'),
        ('TADSR_UNET_UPBLOCK1_RESNET2_SYNTHETIC_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_resnet2_synthetic_alignment.json', 'synthetic hidden/residual -> up_blocks.1.resnets.2 alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_RESNET0_ATTENTION0_RESNET1_ATTENTION1_RESNET2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblocks_midblock_upblock0_upblock1_resnet0_attention0_resnet1_attention1_resnet2_alignment.json', 'UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> up_blocks.1.resnets.0 -> up_blocks.1.attentions.0 -> up_blocks.1.resnets.1 -> up_blocks.1.attentions.1 -> up_blocks.1.resnets.2 bridge alignment'),
    ]
    upblock1_resnet2_statuses = []
    for check, path, note in upblock1_resnet2_checks:
        st = status_from_json(path)
        upblock1_resnet2_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_upblock1_resnet2.py && python3 tools/export_tadsr_unet_upblock1_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    upblock1_resnet2_meta = load_json('experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock1_resnet2/unet_upblock1_resnet2_oracle_metadata.json')
    if upblock1_resnet2_meta.get('resnet2_config', {}).get('has_shortcut') is False:
        upblock1_resnet2_shortcut_status = 'NOT_APPLICABLE'
    else:
        upblock1_resnet2_shortcut_status = status_from_json('experiments/full_repro/unet_alignment/jittor_unet_upblock1_resnet2_alignment.json', default='BLOCKED')
    rows.append(row('TADSR_UNET_UPBLOCK1_RESNET2_SHORTCUT_ALIGNMENT', upblock1_resnet2_shortcut_status, 'up_blocks.1.resnets.2 shortcut alignment; PASS if conv_shortcut exists, NOT_APPLICABLE if official shortcut is absent', 'continue only if shortcut remains PASS or NOT_APPLICABLE'))
    upblock1_resnet2_required = [s for s in upblock1_resnet2_statuses if s not in {'NOT_APPLICABLE'}]
    upblock1_resnet2_pass = bool(upblock1_resnet2_required) and all(st == 'PASS' for st in upblock1_resnet2_required) and upblock1_resnet2_shortcut_status in {'PASS', 'NOT_APPLICABLE'}
    rows.append(row('TADSR_UNET_UPBLOCK1_RESNET2_BRIDGE_ALIGNMENT', 'PASS' if (upblock1_attention0_pass and upblock1_resnet1_pass and upblock1_attention1_pass and upblock1_resnet2_pass) else 'PARTIAL', 'up_blocks.1 path is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 -> resnets.2 only', 'continue with up_blocks.1.attentions.2'))
    rows.append(row('TADSR_UNET_UPBLOCK1_PRE_ATTENTION2_ALIGNMENT', 'PASS' if (upblock1_attention0_pass and upblock1_resnet1_pass and upblock1_attention1_pass and upblock1_resnet2_pass) else 'PARTIAL', 'input to next unopened up_blocks.1.attentions.2 is now aligned', 'continue with up_blocks.1.attentions.2'))

    upblock1_attention2_checks = [
        ('TADSR_UNET_UPBLOCK1_ATTENTION2_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock1_attention2.json', 'official up_blocks.1.attentions.2 topology/LoRA audit'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION2_RESIDUAL_CONTRACT_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock1_attention2.json', 'official no-residual-consumption contract audit for up_blocks.1.attentions.2'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION2_LORA_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock1_attention2.json', 'LoRA/effective static weight audit for up_blocks.1.attentions.2'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION2_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock1_attention2/unet_upblock1_attention2_oracle_metadata.json', 'PyTorch oracle tensors for up_blocks.1.attentions.2 exported'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION2_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock1_attention2/unet_upblock1_attention2_oracle_metadata.json', 'effective static weights for up_blocks.1.attentions.2 exported locally'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION2_RESIDUAL_CONTRACT_ORACLE', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock1_attention2/unet_upblock1_attention2_oracle_metadata.json', 'oracle records attention2 consumes no accumulated residuals'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION2_NORM_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_attention2_proj_alignment.json', 'up_blocks.1.attentions.2 group norm alignment'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION2_SEQUENCE_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_attention2_proj_alignment.json', 'up_blocks.1.attentions.2 NCHW to sequence reshape alignment'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION2_PROJ_IN_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_attention2_proj_alignment.json', 'up_blocks.1.attentions.2 proj_in alignment'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION2_ATTN1_QKV_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_attention2_attn1_alignment.json', 'up_blocks.1.attentions.2 self-attention q/k/v alignment'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION2_ATTN1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_attention2_attn1_alignment.json', 'up_blocks.1.attentions.2 self-attention output alignment'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION2_AFTER_ATTN1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_attention2_attn1_alignment.json', 'up_blocks.1.attentions.2 residual after attn1 alignment'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION2_ATTN2_QKV_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_attention2_attn2_alignment.json', 'up_blocks.1.attentions.2 cross-attention q/k/v alignment'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION2_ATTN2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_attention2_attn2_alignment.json', 'up_blocks.1.attentions.2 cross-attention output alignment'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION2_AFTER_ATTN2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_attention2_attn2_alignment.json', 'up_blocks.1.attentions.2 residual after attn2 alignment'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION2_FF_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_attention2_ff_alignment.json', 'up_blocks.1.attentions.2 feed-forward output alignment'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION2_TRANSFORMER0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_attention2_ff_alignment.json', 'up_blocks.1.attentions.2 transformer block output alignment'),
        ('TADSR_UNET_UPBLOCK1_ATTENTION2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_attention2_alignment.json', 'isolated up_blocks.1.attentions.2 output alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_RESNET0_ATTENTION0_RESNET1_ATTENTION1_RESNET2_ATTENTION2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblocks_midblock_upblock0_upblock1_resnet0_attention0_resnet1_attention1_resnet2_attention2_alignment.json', 'UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> up_blocks.1.resnets.0 -> up_blocks.1.attentions.0 -> up_blocks.1.resnets.1 -> up_blocks.1.attentions.1 -> up_blocks.1.resnets.2 -> up_blocks.1.attentions.2 bridge alignment'),
    ]
    upblock1_attention2_statuses = []
    for check, path, note in upblock1_attention2_checks:
        st = status_from_json(path)
        upblock1_attention2_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_upblock1_attention2.py && python3 tools/export_tadsr_unet_upblock1_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    upblock1_attention2_pass = bool(upblock1_attention2_statuses) and all(st == 'PASS' for st in upblock1_attention2_statuses)
    rows.append(row('TADSR_UNET_UPBLOCK1_ATTENTION2_BRIDGE_ALIGNMENT', 'PASS' if (upblock1_attention0_pass and upblock1_resnet1_pass and upblock1_attention1_pass and upblock1_resnet2_pass and upblock1_attention2_pass) else 'PARTIAL', 'up_blocks.1 path is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 -> resnets.2 -> attentions.2 only', 'continue with up_blocks.1.upsamplers.0'))
    rows.append(row('TADSR_UNET_UPBLOCK1_PRE_UPSAMPLER_ALIGNMENT', 'PASS' if (upblock1_attention0_pass and upblock1_resnet1_pass and upblock1_attention1_pass and upblock1_resnet2_pass and upblock1_attention2_pass) else 'PARTIAL', 'input to next unopened up_blocks.1.upsamplers.0 is now aligned', 'continue with up_blocks.1.upsamplers.0'))

    upblock1_upsampler_checks = [
        ('TADSR_UNET_UPBLOCK1_UPSAMPLER_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock1_upsampler.json', 'official up_blocks.1.upsamplers.0 topology/operation audit'),
        ('TADSR_UNET_UPBLOCK1_UPSAMPLER_LORA_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock1_upsampler.json', 'LoRA/effective static weight audit for up_blocks.1.upsamplers.0'),
        ('TADSR_UNET_UPBLOCK1_UPSAMPLER_RESIDUAL_CONTRACT_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock1_upsampler.json', 'official upsampler consumes no accumulated residuals'),
        ('TADSR_UNET_UPBLOCK1_UPSAMPLER_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock1_upsampler/unet_upblock1_upsampler_oracle_metadata.json', 'PyTorch oracle tensors for up_blocks.1.upsamplers.0 exported'),
        ('TADSR_UNET_UPBLOCK1_UPSAMPLER_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock1_upsampler/unet_upblock1_upsampler_oracle_metadata.json', 'effective static weights for up_blocks.1.upsamplers.0 exported locally'),
        ('TADSR_UNET_UPBLOCK1_UPSAMPLER_RESIDUAL_CONTRACT_ORACLE', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock1_upsampler/unet_upblock1_upsampler_oracle_metadata.json', 'oracle records upsampler consumes no accumulated residuals'),
        ('TADSR_UNET_UPBLOCK1_UPSAMPLER_INTERPOLATION_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_upsampler_alignment.json', 'up_blocks.1.upsamplers.0 nearest interpolation alignment'),
        ('TADSR_UNET_UPBLOCK1_UPSAMPLER_CONV_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_upsampler_alignment.json', 'up_blocks.1.upsamplers.0 exported effective conv alignment'),
        ('TADSR_UNET_UPBLOCK1_UPSAMPLER_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_upsampler_alignment.json', 'isolated up_blocks.1.upsamplers.0 output alignment'),
        ('TADSR_UNET_UPBLOCK1_UPSAMPLER_SYNTHETIC_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_upsampler_synthetic_alignment.json', 'synthetic hidden_states -> up_blocks.1.upsamplers.0 alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_THROUGH_UPSAMPLER_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblocks_midblock_upblock0_upblock1_through_upsampler_alignment.json', 'UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> up_blocks.1 through upsamplers.0 bridge alignment'),
    ]
    upblock1_upsampler_statuses = []
    for check, path, note in upblock1_upsampler_checks:
        st = status_from_json(path)
        upblock1_upsampler_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_upblock1_upsampler.py && python3 tools/export_tadsr_unet_upblock1_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    upblock1_upsampler_required = [s for s in upblock1_upsampler_statuses if s not in {'NOT_APPLICABLE'}]
    upblock1_upsampler_pass = bool(upblock1_upsampler_required) and all(st == 'PASS' for st in upblock1_upsampler_required)
    rows.append(row('TADSR_UNET_UPBLOCK1_UPSAMPLER_BRIDGE_ALIGNMENT', 'PASS' if (upblock1_attention0_pass and upblock1_resnet1_pass and upblock1_attention1_pass and upblock1_resnet2_pass and upblock1_attention2_pass and upblock1_upsampler_pass) else 'PARTIAL', 'up_blocks.1 path is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 -> resnets.2 -> attentions.2 -> upsamplers.0 only; full aggregate is intentionally deferred', 'complete full local up_blocks.1 aggregate verification in the next stage'))

    upblock1_local_checks = [
        ('TADSR_UNET_UPBLOCK1_LOCAL_FORWARD_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock1_local.json', 'official local up_blocks.1 forward equals manual resnet/attention/upsampler chain'),
        ('TADSR_UNET_UPBLOCK1_RESIDUAL_CONTRACT_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock1_local.json', 'official local up_blocks.1 residual tuple contract audited'),
        ('TADSR_UNET_UPBLOCK1_LOCAL_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock1_local/unet_upblock1_local_oracle_metadata.json', 'PyTorch oracle tensors for full local up_blocks.1 aggregate exported'),
        ('TADSR_UNET_UPBLOCK1_OUTPUT_HIDDEN_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock1_local_alignment.json', 'Jittor full local up_blocks.1 hidden output alignment'),
    ]
    upblock1_local_statuses = []
    for check, path, note in upblock1_local_checks:
        st = status_from_json(path)
        upblock1_local_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_upblock1_local.py && python3 tools/export_tadsr_unet_upblock1_local_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    upblock1_local_alignment = load_json('experiments/full_repro/unet_alignment/jittor_unet_upblock1_local_alignment.json')
    upblock1_output_states_status = str(upblock1_local_alignment.get('output_states_status') or load_json('experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock1_local/unet_upblock1_local_oracle_metadata.json').get('output_states_status') or 'BLOCKED')
    if upblock1_output_states_status not in {'PASS', 'NOT_APPLICABLE'}:
        upblock1_output_states_status = 'BLOCKED' if upblock1_output_states_status == 'UNKNOWN_TUPLE' else upblock1_output_states_status
    rows.append(row('TADSR_UNET_UPBLOCK1_OUTPUT_STATES_ALIGNMENT', upblock1_output_states_status, 'official local up_blocks.1 output_states are not returned in this config, or aligned if present', 'continue only if PASS or NOT_APPLICABLE'))
    upblock1_local_pass = bool(upblock1_local_statuses) and all(st == 'PASS' for st in upblock1_local_statuses) and upblock1_output_states_status in {'PASS', 'NOT_APPLICABLE'}
    upblock1_alignment_pass = (
        upblock1_attention0_pass
        and upblock1_resnet1_pass
        and upblock1_attention1_pass
        and upblock1_resnet2_pass
        and upblock1_attention2_pass
        and upblock1_upsampler_pass
        and upblock1_local_pass
        and upblock1_resnet0_pass
    )
    rows.append(row('TADSR_UNET_UPBLOCK1_ALIGNMENT', 'PASS' if upblock1_alignment_pass else 'PARTIAL', 'full local up_blocks.1 aggregate is aligned through output hidden; execution still stops before up_blocks.2', 'continue with up_blocks.2.resnets.0 only after this remains PASS'))

    upblock2_resnet0_checks = [
        ('TADSR_UNET_UPBLOCK2_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock2_resnet0.json', 'official up_blocks.2 topology audited only through resnets.0'),
        ('TADSR_UNET_UPBLOCK2_RESNET0_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock2_resnet0.json', 'official up_blocks.2.resnets.0 config audited'),
        ('TADSR_UNET_UPBLOCK2_RESNET0_RESIDUAL_CONTRACT_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock2_resnet0.json', 'official residual consumption for up_blocks.2.resnets.0 audited'),
        ('TADSR_UNET_UPBLOCK2_RESNET0_LORA_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock2_resnet0.json', 'LoRA/effective static weight audit for up_blocks.2.resnets.0'),
        ('TADSR_UNET_UPBLOCK2_RESNET0_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock2_resnet0/unet_upblock2_resnet0_oracle_metadata.json', 'PyTorch oracle tensors for up_blocks.2.resnets.0 exported'),
        ('TADSR_UNET_UPBLOCK2_RESNET0_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock2_resnet0/unet_upblock2_resnet0_oracle_metadata.json', 'effective static weights for up_blocks.2.resnets.0 exported locally'),
        ('TADSR_UNET_UPBLOCK2_RESNET0_RESIDUAL_CONTRACT_ORACLE', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock2_resnet0/unet_upblock2_resnet0_oracle_metadata.json', 'oracle records the exact residual consumed by up_blocks.2.resnets.0'),
        ('TADSR_UNET_UPBLOCK2_RESNET0_RESIDUAL_CONCAT_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_resnet0_alignment.json', 'up_blocks.2.resnets.0 residual concat alignment'),
        ('TADSR_UNET_UPBLOCK2_RESNET0_NORM1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_resnet0_alignment.json', 'up_blocks.2.resnets.0 norm1 alignment'),
        ('TADSR_UNET_UPBLOCK2_RESNET0_CONV1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_resnet0_alignment.json', 'up_blocks.2.resnets.0 conv1 alignment'),
        ('TADSR_UNET_UPBLOCK2_RESNET0_TEMB_PROJ_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_resnet0_alignment.json', 'up_blocks.2.resnets.0 time embedding projection alignment'),
        ('TADSR_UNET_UPBLOCK2_RESNET0_CONV2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_resnet0_alignment.json', 'up_blocks.2.resnets.0 conv2 alignment'),
        ('TADSR_UNET_UPBLOCK2_RESNET0_SHORTCUT_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_resnet0_alignment.json', 'up_blocks.2.resnets.0 shortcut alignment'),
        ('TADSR_UNET_UPBLOCK2_RESNET0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_resnet0_alignment.json', 'isolated entry hidden/residual -> up_blocks.2.resnets.0 output alignment'),
        ('TADSR_UNET_UPBLOCK2_RESNET0_SYNTHETIC_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_resnet0_synthetic_alignment.json', 'synthetic hidden/residual -> up_blocks.2.resnets.0 alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_RESNET0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblocks_midblock_upblock0_upblock1_upblock2_resnet0_alignment.json', 'UNet entry -> down_blocks -> mid_block -> up_blocks.0 -> up_blocks.1 -> up_blocks.2.resnets.0 bridge alignment'),
    ]
    upblock2_resnet0_statuses = []
    for check, path, note in upblock2_resnet0_checks:
        st = status_from_json(path)
        upblock2_resnet0_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_upblock2_resnet0.py && python3 tools/export_tadsr_unet_upblock2_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    upblock2_resnet0_pass = bool(upblock2_resnet0_statuses) and all(st == 'PASS' for st in upblock2_resnet0_statuses)
    rows.append(row('TADSR_UNET_UPBLOCK2_RESNET0_BRIDGE_ALIGNMENT', 'PASS' if (upblock1_alignment_pass and upblock2_resnet0_pass) else 'PARTIAL', 'up_blocks.2 is aligned only through resnets.0 and stops before attentions.0', 'continue with up_blocks.2.attentions.0; keep full up_blocks.2/full UNet/full inference incomplete'))
    rows.append(row('TADSR_UNET_UPBLOCK2_PRE_ATTENTION0_ALIGNMENT', 'PASS' if (upblock1_alignment_pass and upblock2_resnet0_pass) else 'PARTIAL', 'execution reaches the input boundary before up_blocks.2.attentions.0 without entering attention0', 'next target is up_blocks.2.attentions.0'))

    upblock2_attention0_checks = [
        ('TADSR_UNET_UPBLOCK2_ATTENTION0_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock2_attention0.json', 'official up_blocks.2.attentions.0 topology/LoRA audit'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION0_RESIDUAL_CONTRACT_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock2_attention0.json', 'official no-residual-consumption contract audit for up_blocks.2.attentions.0'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION0_LORA_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock2_attention0.json', 'LoRA/effective static weight audit for up_blocks.2.attentions.0'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION0_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock2_attention0/unet_upblock2_attention0_oracle_metadata.json', 'PyTorch oracle tensors for up_blocks.2.attentions.0 exported'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION0_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock2_attention0/unet_upblock2_attention0_oracle_metadata.json', 'effective static weights for up_blocks.2.attentions.0 exported locally'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION0_RESIDUAL_CONTRACT_ORACLE', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock2_attention0/unet_upblock2_attention0_oracle_metadata.json', 'oracle records attention0 consumes no accumulated residuals'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION0_NORM_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_attention0_proj_alignment.json', 'up_blocks.2.attentions.0 group norm alignment'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION0_SEQUENCE_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_attention0_proj_alignment.json', 'up_blocks.2.attentions.0 NCHW to sequence reshape alignment'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION0_PROJ_IN_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_attention0_proj_alignment.json', 'up_blocks.2.attentions.0 proj_in alignment'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION0_ATTN1_QKV_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_attention0_attn1_alignment.json', 'up_blocks.2.attentions.0 self-attention q/k/v alignment'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION0_ATTN1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_attention0_attn1_alignment.json', 'up_blocks.2.attentions.0 self-attention output alignment'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION0_AFTER_ATTN1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_attention0_attn1_alignment.json', 'up_blocks.2.attentions.0 residual after attn1 alignment'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION0_ATTN2_QKV_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_attention0_attn2_alignment.json', 'up_blocks.2.attentions.0 cross-attention q/k/v alignment'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION0_ATTN2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_attention0_attn2_alignment.json', 'up_blocks.2.attentions.0 cross-attention output alignment'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION0_AFTER_ATTN2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_attention0_attn2_alignment.json', 'up_blocks.2.attentions.0 residual after attn2 alignment'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION0_FF_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_attention0_ff_alignment.json', 'up_blocks.2.attentions.0 feed-forward output alignment'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION0_TRANSFORMER0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_attention0_ff_alignment.json', 'up_blocks.2.attentions.0 transformer block output alignment'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_attention0_alignment.json', 'isolated up_blocks.2.attentions.0 output alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_RESNET0_ATTENTION0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblocks_midblock_upblock0_upblock1_upblock2_resnet0_attention0_alignment.json', 'UNet entry -> down_blocks -> mid_block -> up_blocks.0 -> up_blocks.1 -> up_blocks.2.resnets.0 -> up_blocks.2.attentions.0 bridge alignment'),
    ]
    upblock2_attention0_statuses = []
    for check, path, note in upblock2_attention0_checks:
        st = status_from_json(path)
        upblock2_attention0_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_upblock2_attention0.py && python3 tools/export_tadsr_unet_upblock2_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    upblock2_attention0_pass = bool(upblock2_attention0_statuses) and all(st == 'PASS' for st in upblock2_attention0_statuses)
    rows.append(row('TADSR_UNET_UPBLOCK2_ATTENTION0_BRIDGE_ALIGNMENT', 'PASS' if (upblock1_alignment_pass and upblock2_resnet0_pass and upblock2_attention0_pass) else 'PARTIAL', 'up_blocks.2 is aligned through resnets.0 -> attentions.0 only and stops before resnets.1', 'continue with up_blocks.2.resnets.1; keep full up_blocks.2/full UNet/full inference incomplete'))
    rows.append(row('TADSR_UNET_UPBLOCK2_PRE_RESNET1_ALIGNMENT', 'PASS' if (upblock1_alignment_pass and upblock2_resnet0_pass and upblock2_attention0_pass) else 'PARTIAL', 'execution reaches the input boundary before up_blocks.2.resnets.1 without entering resnet1', 'next target is up_blocks.2.resnets.1'))

    upblock2_resnet1_checks = [
        ('TADSR_UNET_UPBLOCK2_RESNET1_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock2_resnet1.json', 'official up_blocks.2.resnets.1 config audited'),
        ('TADSR_UNET_UPBLOCK2_RESNET1_RESIDUAL_CONTRACT_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock2_resnet1.json', 'official residual consumption for up_blocks.2.resnets.1 audited'),
        ('TADSR_UNET_UPBLOCK2_RESNET1_LORA_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock2_resnet1.json', 'LoRA/effective static weight audit for up_blocks.2.resnets.1'),
        ('TADSR_UNET_UPBLOCK2_RESNET1_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock2_resnet1/unet_upblock2_resnet1_oracle_metadata.json', 'PyTorch oracle tensors for up_blocks.2.resnets.1 exported'),
        ('TADSR_UNET_UPBLOCK2_RESNET1_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock2_resnet1/unet_upblock2_resnet1_oracle_metadata.json', 'effective static weights for up_blocks.2.resnets.1 exported locally'),
        ('TADSR_UNET_UPBLOCK2_RESNET1_RESIDUAL_CONTRACT_ORACLE', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock2_resnet1/unet_upblock2_resnet1_oracle_metadata.json', 'oracle records the exact residual consumed by up_blocks.2.resnets.1'),
        ('TADSR_UNET_UPBLOCK2_RESNET1_RESIDUAL_CONCAT_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_resnet1_alignment.json', 'up_blocks.2.resnets.1 residual concat alignment'),
        ('TADSR_UNET_UPBLOCK2_RESNET1_NORM1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_resnet1_alignment.json', 'up_blocks.2.resnets.1 norm1 alignment'),
        ('TADSR_UNET_UPBLOCK2_RESNET1_CONV1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_resnet1_alignment.json', 'up_blocks.2.resnets.1 conv1 alignment'),
        ('TADSR_UNET_UPBLOCK2_RESNET1_TEMB_PROJ_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_resnet1_alignment.json', 'up_blocks.2.resnets.1 time embedding projection alignment'),
        ('TADSR_UNET_UPBLOCK2_RESNET1_CONV2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_resnet1_alignment.json', 'up_blocks.2.resnets.1 conv2 alignment'),
        ('TADSR_UNET_UPBLOCK2_RESNET1_SHORTCUT_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_resnet1_alignment.json', 'up_blocks.2.resnets.1 shortcut alignment'),
        ('TADSR_UNET_UPBLOCK2_RESNET1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_resnet1_alignment.json', 'isolated attention0 hidden/residual -> up_blocks.2.resnets.1 output alignment'),
        ('TADSR_UNET_UPBLOCK2_RESNET1_SYNTHETIC_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_resnet1_synthetic_alignment.json', 'synthetic hidden/residual -> up_blocks.2.resnets.1 alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_RESNET0_ATTENTION0_RESNET1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblocks_midblock_upblock0_upblock1_upblock2_resnet0_attention0_resnet1_alignment.json', 'UNet entry -> down_blocks -> mid_block -> up_blocks.0 -> up_blocks.1 -> up_blocks.2.resnets.0 -> up_blocks.2.attentions.0 -> up_blocks.2.resnets.1 bridge alignment'),
    ]
    upblock2_resnet1_statuses = []
    for check, path, note in upblock2_resnet1_checks:
        st = status_from_json(path)
        upblock2_resnet1_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_upblock2_resnet1.py && python3 tools/export_tadsr_unet_upblock2_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    upblock2_resnet1_pass = bool(upblock2_resnet1_statuses) and all(st == 'PASS' for st in upblock2_resnet1_statuses)
    rows.append(row('TADSR_UNET_UPBLOCK2_RESNET1_BRIDGE_ALIGNMENT', 'PASS' if (upblock1_alignment_pass and upblock2_resnet0_pass and upblock2_attention0_pass and upblock2_resnet1_pass) else 'PARTIAL', 'up_blocks.2 is aligned through resnets.0 -> attentions.0 -> resnets.1 only and stops before attentions.1', 'continue with up_blocks.2.attentions.1; keep full up_blocks.2/full UNet/full inference incomplete'))
    rows.append(row('TADSR_UNET_UPBLOCK2_PRE_ATTENTION1_ALIGNMENT', 'PASS' if (upblock1_alignment_pass and upblock2_resnet0_pass and upblock2_attention0_pass and upblock2_resnet1_pass) else 'PARTIAL', 'execution reaches the input boundary before up_blocks.2.attentions.1 without entering attention1', 'next target is up_blocks.2.attentions.1'))
    upblock2_attention1_checks = [
        ('TADSR_UNET_UPBLOCK2_ATTENTION1_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock2_attention1.json', 'official up_blocks.2.attentions.1 topology/LoRA audit'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION1_RESIDUAL_CONTRACT_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock2_attention1.json', 'official no-residual-consumption contract audit for up_blocks.2.attentions.1'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION1_LORA_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock2_attention1.json', 'LoRA/effective static weight audit for up_blocks.2.attentions.1'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION1_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock2_attention1/unet_upblock2_attention1_oracle_metadata.json', 'PyTorch oracle tensors for up_blocks.2.attentions.1 exported'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION1_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock2_attention1/unet_upblock2_attention1_oracle_metadata.json', 'effective static weights for up_blocks.2.attentions.1 exported locally'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION1_RESIDUAL_CONTRACT_ORACLE', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock2_attention1/unet_upblock2_attention1_oracle_metadata.json', 'oracle records attention1 consumes no accumulated residuals'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION1_NORM_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_attention1_proj_alignment.json', 'up_blocks.2.attentions.1 group norm alignment'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION1_SEQUENCE_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_attention1_proj_alignment.json', 'up_blocks.2.attentions.1 NCHW to sequence reshape alignment'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION1_PROJ_IN_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_attention1_proj_alignment.json', 'up_blocks.2.attentions.1 proj_in alignment'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION1_ATTN1_QKV_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_attention1_attn1_alignment.json', 'up_blocks.2.attentions.1 self-attention q/k/v alignment'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION1_ATTN1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_attention1_attn1_alignment.json', 'up_blocks.2.attentions.1 self-attention output alignment'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION1_AFTER_ATTN1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_attention1_attn1_alignment.json', 'up_blocks.2.attentions.1 residual after attn1 alignment'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION1_ATTN2_QKV_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_attention1_attn2_alignment.json', 'up_blocks.2.attentions.1 cross-attention q/k/v alignment'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION1_ATTN2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_attention1_attn2_alignment.json', 'up_blocks.2.attentions.1 cross-attention output alignment'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION1_AFTER_ATTN2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_attention1_attn2_alignment.json', 'up_blocks.2.attentions.1 residual after attn2 alignment'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION1_FF_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_attention1_ff_alignment.json', 'up_blocks.2.attentions.1 feed-forward output alignment'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION1_TRANSFORMER0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_attention1_ff_alignment.json', 'up_blocks.2.attentions.1 transformer block output alignment'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_attention1_alignment.json', 'isolated up_blocks.2.attentions.1 output alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_RESNET0_ATTENTION0_RESNET1_ATTENTION1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblocks_midblock_upblock0_upblock1_upblock2_resnet0_attention0_resnet1_attention1_alignment.json', 'UNet entry -> down_blocks -> mid_block -> up_blocks.0 -> up_blocks.1 -> up_blocks.2.resnets.0 -> up_blocks.2.attentions.0 -> up_blocks.2.resnets.1 -> up_blocks.2.attentions.1 bridge alignment'),
    ]
    upblock2_attention1_statuses = []
    for check, path, note in upblock2_attention1_checks:
        st = status_from_json(path)
        upblock2_attention1_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_upblock2_attention1.py && python3 tools/export_tadsr_unet_upblock2_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    upblock2_attention1_pass = bool(upblock2_attention1_statuses) and all(st == 'PASS' for st in upblock2_attention1_statuses)
    rows.append(row('TADSR_UNET_UPBLOCK2_ATTENTION1_BRIDGE_ALIGNMENT', 'PASS' if (upblock1_alignment_pass and upblock2_resnet0_pass and upblock2_attention0_pass and upblock2_resnet1_pass and upblock2_attention1_pass) else 'PARTIAL', 'up_blocks.2 is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 only and stops before resnets.2', 'continue with up_blocks.2.resnets.2; keep full up_blocks.2/full UNet/full inference incomplete'))
    rows.append(row('TADSR_UNET_UPBLOCK2_PRE_RESNET2_ALIGNMENT', 'PASS' if (upblock1_alignment_pass and upblock2_resnet0_pass and upblock2_attention0_pass and upblock2_resnet1_pass and upblock2_attention1_pass) else 'PARTIAL', 'execution reaches the input boundary before up_blocks.2.resnets.2 without entering resnet2', 'next target is up_blocks.2.resnets.2'))

    upblock2_resnet2_checks = [
        ('TADSR_UNET_UPBLOCK2_RESNET2_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock2_resnet2.json', 'official up_blocks.2.resnets.2 config audited'),
        ('TADSR_UNET_UPBLOCK2_RESNET2_RESIDUAL_CONTRACT_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock2_resnet2.json', 'official residual consumption for up_blocks.2.resnets.2 audited'),
        ('TADSR_UNET_UPBLOCK2_RESNET2_LORA_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock2_resnet2.json', 'LoRA/effective static weight audit for up_blocks.2.resnets.2'),
        ('TADSR_UNET_UPBLOCK2_RESNET2_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock2_resnet2/unet_upblock2_resnet2_oracle_metadata.json', 'PyTorch oracle tensors for up_blocks.2.resnets.2 exported'),
        ('TADSR_UNET_UPBLOCK2_RESNET2_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock2_resnet2/unet_upblock2_resnet2_oracle_metadata.json', 'effective static weights for up_blocks.2.resnets.2 exported locally'),
        ('TADSR_UNET_UPBLOCK2_RESNET2_RESIDUAL_CONTRACT_ORACLE', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock2_resnet2/unet_upblock2_resnet2_oracle_metadata.json', 'oracle records the exact residual consumed by up_blocks.2.resnets.2'),
        ('TADSR_UNET_UPBLOCK2_RESNET2_RESIDUAL_CONCAT_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_resnet2_alignment.json', 'up_blocks.2.resnets.2 residual concat alignment'),
        ('TADSR_UNET_UPBLOCK2_RESNET2_NORM1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_resnet2_alignment.json', 'up_blocks.2.resnets.2 norm1 alignment'),
        ('TADSR_UNET_UPBLOCK2_RESNET2_CONV1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_resnet2_alignment.json', 'up_blocks.2.resnets.2 conv1 alignment'),
        ('TADSR_UNET_UPBLOCK2_RESNET2_TEMB_PROJ_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_resnet2_alignment.json', 'up_blocks.2.resnets.2 time embedding projection alignment'),
        ('TADSR_UNET_UPBLOCK2_RESNET2_CONV2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_resnet2_alignment.json', 'up_blocks.2.resnets.2 conv2 alignment'),
        ('TADSR_UNET_UPBLOCK2_RESNET2_SHORTCUT_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_resnet2_alignment.json', 'up_blocks.2.resnets.2 shortcut alignment'),
        ('TADSR_UNET_UPBLOCK2_RESNET2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_resnet2_alignment.json', 'isolated attention1 hidden/residual -> up_blocks.2.resnets.2 output alignment'),
        ('TADSR_UNET_UPBLOCK2_RESNET2_SYNTHETIC_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_resnet2_synthetic_alignment.json', 'synthetic hidden/residual -> up_blocks.2.resnets.2 alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_RESNET0_ATTENTION0_RESNET1_ATTENTION1_RESNET2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblocks_midblock_upblock0_upblock1_upblock2_resnet0_attention0_resnet1_attention1_resnet2_alignment.json', 'UNet entry -> down_blocks -> mid_block -> up_blocks.0 -> up_blocks.1 -> up_blocks.2.resnets.0 -> up_blocks.2.attentions.0 -> up_blocks.2.resnets.1 -> up_blocks.2.attentions.1 -> up_blocks.2.resnets.2 bridge alignment'),
    ]
    upblock2_resnet2_statuses = []
    for check, path, note in upblock2_resnet2_checks:
        st = status_from_json(path)
        upblock2_resnet2_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_upblock2_resnet2.py && python3 tools/export_tadsr_unet_upblock2_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    upblock2_resnet2_pass = bool(upblock2_resnet2_statuses) and all(st == 'PASS' for st in upblock2_resnet2_statuses)
    rows.append(row('TADSR_UNET_UPBLOCK2_RESNET2_BRIDGE_ALIGNMENT', 'PASS' if (upblock1_alignment_pass and upblock2_resnet0_pass and upblock2_attention0_pass and upblock2_resnet1_pass and upblock2_attention1_pass and upblock2_resnet2_pass) else 'PARTIAL', 'up_blocks.2 is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 -> resnets.2 only and stops before attentions.2', 'continue with up_blocks.2.attentions.2; keep full up_blocks.2/full UNet/full inference incomplete'))
    rows.append(row('TADSR_UNET_UPBLOCK2_PRE_ATTENTION2_ALIGNMENT', 'PASS' if (upblock1_alignment_pass and upblock2_resnet0_pass and upblock2_attention0_pass and upblock2_resnet1_pass and upblock2_attention1_pass and upblock2_resnet2_pass) else 'PARTIAL', 'execution reaches the input boundary before up_blocks.2.attentions.2 without entering attention2', 'next target is up_blocks.2.attentions.2'))
    upblock2_attention2_checks = [
        ('TADSR_UNET_UPBLOCK2_ATTENTION2_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock2_attention2.json', 'official up_blocks.2.attentions.2 topology/LoRA audit'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION2_RESIDUAL_CONTRACT_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock2_attention2.json', 'official no-residual-consumption contract audit for up_blocks.2.attentions.2'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION2_LORA_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock2_attention2.json', 'LoRA/effective static weight audit for up_blocks.2.attentions.2'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION2_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock2_attention2/unet_upblock2_attention2_oracle_metadata.json', 'PyTorch oracle tensors for up_blocks.2.attentions.2 exported'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION2_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock2_attention2/unet_upblock2_attention2_oracle_metadata.json', 'effective static weights for up_blocks.2.attentions.2 exported locally'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION2_RESIDUAL_CONTRACT_ORACLE', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock2_attention2/unet_upblock2_attention2_oracle_metadata.json', 'oracle records attention2 consumes no accumulated residuals'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION2_NORM_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_attention2_proj_alignment.json', 'up_blocks.2.attentions.2 group norm alignment'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION2_SEQUENCE_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_attention2_proj_alignment.json', 'up_blocks.2.attentions.2 NCHW to sequence reshape alignment'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION2_PROJ_IN_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_attention2_proj_alignment.json', 'up_blocks.2.attentions.2 proj_in alignment'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION2_ATTN1_QKV_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_attention2_attn1_alignment.json', 'up_blocks.2.attentions.2 self-attention q/k/v alignment'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION2_ATTN1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_attention2_attn1_alignment.json', 'up_blocks.2.attentions.2 self-attention output alignment'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION2_AFTER_ATTN1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_attention2_attn1_alignment.json', 'up_blocks.2.attentions.2 residual after attn1 alignment'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION2_ATTN2_QKV_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_attention2_attn2_alignment.json', 'up_blocks.2.attentions.2 cross-attention q/k/v alignment'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION2_ATTN2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_attention2_attn2_alignment.json', 'up_blocks.2.attentions.2 cross-attention output alignment'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION2_AFTER_ATTN2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_attention2_attn2_alignment.json', 'up_blocks.2.attentions.2 residual after attn2 alignment'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION2_FF_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_attention2_ff_alignment.json', 'up_blocks.2.attentions.2 feed-forward output alignment'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION2_TRANSFORMER0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_attention2_ff_alignment.json', 'up_blocks.2.attentions.2 transformer block output alignment'),
        ('TADSR_UNET_UPBLOCK2_ATTENTION2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_attention2_alignment.json', 'isolated up_blocks.2.attentions.2 output alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_RESNET0_ATTENTION0_RESNET1_ATTENTION1_RESNET2_ATTENTION2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblocks_midblock_upblock0_upblock1_upblock2_resnet0_attention0_resnet1_attention1_resnet2_attention2_alignment.json', 'UNet entry -> down_blocks -> mid_block -> up_blocks.0 -> up_blocks.1 -> up_blocks.2.resnets.0 -> up_blocks.2.attentions.0 -> up_blocks.2.resnets.1 -> up_blocks.2.attentions.1 -> up_blocks.2.resnets.2 -> up_blocks.2.attentions.2 bridge alignment'),
    ]
    upblock2_attention2_statuses = []
    for check, path, note in upblock2_attention2_checks:
        st = status_from_json(path)
        upblock2_attention2_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_upblock2_attention2.py && python3 tools/export_tadsr_unet_upblock2_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    upblock2_attention2_pass = bool(upblock2_attention2_statuses) and all(st == 'PASS' for st in upblock2_attention2_statuses)
    rows.append(row('TADSR_UNET_UPBLOCK2_ATTENTION2_BRIDGE_ALIGNMENT', 'PASS' if (upblock1_alignment_pass and upblock2_resnet0_pass and upblock2_attention0_pass and upblock2_resnet1_pass and upblock2_attention1_pass and upblock2_resnet2_pass and upblock2_attention2_pass) else 'PARTIAL', 'up_blocks.2 is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 -> resnets.2 -> attentions.2 only and stops before upsamplers.0', 'continue with up_blocks.2.upsamplers.0; keep full up_blocks.2/full UNet/full inference incomplete'))
    rows.append(row('TADSR_UNET_UPBLOCK2_PRE_UPSAMPLER_ALIGNMENT', 'PASS' if (upblock1_alignment_pass and upblock2_resnet0_pass and upblock2_attention0_pass and upblock2_resnet1_pass and upblock2_attention1_pass and upblock2_resnet2_pass and upblock2_attention2_pass) else 'PARTIAL', 'execution reaches the input boundary before up_blocks.2.upsamplers.0 without entering upsampler', 'next target is up_blocks.2.upsamplers.0'))
    upblock2_upsampler_checks = [
        ('TADSR_UNET_UPBLOCK2_UPSAMPLER_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock2_upsampler.json', 'official up_blocks.2.upsamplers.0 topology/operation audit'),
        ('TADSR_UNET_UPBLOCK2_UPSAMPLER_LORA_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock2_upsampler.json', 'LoRA/effective static weight audit for up_blocks.2.upsamplers.0'),
        ('TADSR_UNET_UPBLOCK2_UPSAMPLER_RESIDUAL_CONTRACT_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock2_upsampler.json', 'official upsampler consumes no accumulated residuals'),
        ('TADSR_UNET_UPBLOCK2_UPSAMPLER_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock2_upsampler/unet_upblock2_upsampler_oracle_metadata.json', 'PyTorch oracle tensors for up_blocks.2.upsamplers.0 exported'),
        ('TADSR_UNET_UPBLOCK2_UPSAMPLER_EFFECTIVE_WEIGHTS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock2_upsampler/unet_upblock2_upsampler_oracle_metadata.json', 'effective static weights for up_blocks.2.upsamplers.0 exported locally'),
        ('TADSR_UNET_UPBLOCK2_UPSAMPLER_RESIDUAL_CONTRACT_ORACLE', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock2_upsampler/unet_upblock2_upsampler_oracle_metadata.json', 'oracle records upsampler consumes no accumulated residuals'),
        ('TADSR_UNET_UPBLOCK2_UPSAMPLER_INTERPOLATION_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_upsampler_alignment.json', 'up_blocks.2.upsamplers.0 nearest interpolation alignment'),
        ('TADSR_UNET_UPBLOCK2_UPSAMPLER_CONV_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_upsampler_alignment.json', 'up_blocks.2.upsamplers.0 exported effective conv alignment'),
        ('TADSR_UNET_UPBLOCK2_UPSAMPLER_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_upsampler_alignment.json', 'isolated up_blocks.2.upsamplers.0 output alignment'),
        ('TADSR_UNET_UPBLOCK2_UPSAMPLER_SYNTHETIC_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock2_upsampler_synthetic_alignment.json', 'synthetic hidden_states -> up_blocks.2.upsamplers.0 alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_THROUGH_UPSAMPLER_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblocks_midblock_upblock0_upblock1_upblock2_through_upsampler_alignment.json', 'UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> full local up_blocks.1 -> up_blocks.2 through upsamplers.0 bridge alignment'),
    ]
    upblock2_upsampler_statuses = []
    for check, path, note in upblock2_upsampler_checks:
        st = status_from_json(path)
        upblock2_upsampler_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_upblock2_upsampler.py && python3 tools/export_tadsr_unet_upblock2_upsampler_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    upblock2_upsampler_required = [s for s in upblock2_upsampler_statuses if s not in {'NOT_APPLICABLE'}]
    upblock2_upsampler_pass = bool(upblock2_upsampler_required) and all(st == 'PASS' for st in upblock2_upsampler_required)
    upblock2_upsampler_bridge_pass = (
        upblock1_alignment_pass
        and upblock2_resnet0_pass
        and upblock2_attention0_pass
        and upblock2_resnet1_pass
        and upblock2_attention1_pass
        and upblock2_resnet2_pass
        and upblock2_attention2_pass
        and upblock2_upsampler_pass
    )
    rows.append(row('TADSR_UNET_UPBLOCK2_UPSAMPLER_BRIDGE_ALIGNMENT', 'PASS' if upblock2_upsampler_bridge_pass else 'PARTIAL', 'up_blocks.2 path is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 -> resnets.2 -> attentions.2 -> upsamplers.0 only; full aggregate is checked separately', 'complete full local up_blocks.2 aggregate verification in the next stage'))
    upblock2_local_checks = [
        ('TADSR_UNET_UPBLOCK2_LOCAL_FORWARD_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock2_local.json', 'official full local up_blocks.2 forward output matches manual resnet/attention/upsampler chain'),
        ('TADSR_UNET_UPBLOCK2_RESIDUAL_CONTRACT_AUDIT', 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock2_local.json', 'official residual consumption contract for full local up_blocks.2 audited'),
        ('TADSR_UNET_UPBLOCK2_LOCAL_ORACLE_TENSORS', 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock2_local/unet_upblock2_local_oracle_metadata.json', 'PyTorch oracle tensors for full local up_blocks.2 aggregate exported'),
    ]
    upblock2_local_statuses = []
    for check, path, note in upblock2_local_checks:
        st = status_from_json(path)
        upblock2_local_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_upblock2_local.py && python3 tools/export_tadsr_unet_upblock2_local_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    upblock2_local_alignment = load_json('experiments/full_repro/unet_alignment/jittor_unet_upblock2_local_alignment.json')
    upblock2_output_hidden_status = str(upblock2_local_alignment.get('output_hidden_status', status_from_json('experiments/full_repro/unet_alignment/jittor_unet_upblock2_local_alignment.json')))
    upblock2_output_states_status = str(upblock2_local_alignment.get('output_states_status', 'BLOCKED'))
    rows.append(row('TADSR_UNET_UPBLOCK2_OUTPUT_HIDDEN_ALIGNMENT', upblock2_output_hidden_status, 'full local up_blocks.2 hidden-state output alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_upblock2_local_alignment.py'))
    rows.append(row('TADSR_UNET_UPBLOCK2_OUTPUT_STATES_ALIGNMENT', upblock2_output_states_status, 'up_blocks.2 output-state contract alignment; official CrossAttnUpBlock2D returns hidden states only in this path', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_upblock2_local_alignment.py'))
    upblock2_local_required = [s for s in upblock2_local_statuses if s not in {'NOT_APPLICABLE'}]
    upblock2_local_pass = (
        upblock2_upsampler_bridge_pass
        and bool(upblock2_local_required)
        and all(st == 'PASS' for st in upblock2_local_required)
        and upblock2_output_hidden_status == 'PASS'
        and upblock2_output_states_status in {'PASS', 'NOT_APPLICABLE'}
        and status_from_json('experiments/full_repro/unet_alignment/jittor_unet_upblock2_local_alignment.json') == 'PASS'
    )
    rows.append(row('TADSR_UNET_UPBLOCK2_ALIGNMENT', 'PASS' if upblock2_local_pass else 'PARTIAL', 'full local up_blocks.2 aggregate is aligned through output hidden; execution still stops before up_blocks.3', 'continue with up_blocks.3.resnets.0 only after this remains PASS'))
    upblock3_audit_path = 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock3_resnet0.json'
    upblock3_oracle_path = 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock3_resnet0/unet_upblock3_resnet0_oracle_metadata.json'
    upblock3_attention0_audit_path = 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock3_attention0.json'
    upblock3_attention0_oracle_path = 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock3_attention0/unet_upblock3_attention0_oracle_metadata.json'
    upblock3_resnet1_audit_path = 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock3_resnet1.json'
    upblock3_resnet1_oracle_path = 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock3_resnet1/unet_upblock3_resnet1_oracle_metadata.json'
    upblock3_attention1_audit_path = 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock3_attention1.json'
    upblock3_attention1_oracle_path = 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock3_attention1/unet_upblock3_attention1_oracle_metadata.json'
    upblock3_resnet2_audit_path = 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock3_resnet2.json'
    upblock3_resnet2_oracle_path = 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock3_resnet2/unet_upblock3_resnet2_oracle_metadata.json'
    upblock3_attention2_audit_path = 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock3_attention2.json'
    upblock3_attention2_oracle_path = 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock3_attention2/unet_upblock3_attention2_oracle_metadata.json'
    upblock3_local_audit_path = 'experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock3_local.json'
    upblock3_local_oracle_path = 'experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock3_local/unet_upblock3_local_oracle_metadata.json'
    upblock3_local_alignment_path = 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_local_alignment.json'
    output_tail_audit_path = 'experiments/full_repro/unet_alignment/audit_tadsr_unet_output_tail.json'
    output_tail_oracle_path = 'experiments/full_repro/unet_alignment/oracle_tensors_unet_output_tail/unet_output_tail_oracle_metadata.json'
    output_tail_alignment_path = 'experiments/full_repro/unet_alignment/jittor_unet_output_tail_alignment.json'
    output_tail_synthetic_alignment_path = 'experiments/full_repro/unet_alignment/jittor_unet_output_tail_synthetic_alignment.json'
    entry_to_output_tail_alignment_path = 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblocks_midblock_upblocks_output_tail_alignment.json'
    manual_wrapper_audit_path = 'experiments/full_repro/unet_alignment/audit_tadsr_unet_manual_wrapper.json'
    manual_wrapper_oracle_path = 'experiments/full_repro/unet_alignment/oracle_tensors_unet_manual_wrapper/unet_manual_wrapper_oracle_metadata.json'
    manual_wrapper_alignment_path = 'experiments/full_repro/unet_alignment/jittor_unet_manual_wrapper_alignment.json'
    full_forward_audit_path = 'experiments/full_repro/unet_alignment/audit_tadsr_unet_full_forward.json'
    full_forward_oracle_path = 'experiments/full_repro/unet_alignment/oracle_tensors_unet_full_forward/unet_full_forward_oracle_metadata.json'
    full_forward_alignment_path = 'experiments/full_repro/unet_alignment/jittor_unet_full_forward_alignment.json'
    upblock3_resnet0_checks = [
        ('TADSR_UNET_UPBLOCK3_AUDIT', upblock3_audit_path, 'official up_blocks.3 topology audit'),
        ('TADSR_UNET_UPBLOCK3_RESNET0_AUDIT', upblock3_audit_path, 'official up_blocks.3.resnets.0 module audit'),
        ('TADSR_UNET_UPBLOCK3_RESNET0_RESIDUAL_CONTRACT_AUDIT', upblock3_audit_path, 'official residual consumption contract for up_blocks.3.resnets.0'),
        ('TADSR_UNET_UPBLOCK3_RESNET0_LORA_AUDIT', upblock3_audit_path, 'LoRA/effective static weight audit for up_blocks.3.resnets.0'),
        ('TADSR_UNET_UPBLOCK3_RESNET0_ORACLE_TENSORS', upblock3_oracle_path, 'PyTorch oracle tensors for up_blocks.3.resnets.0 exported'),
        ('TADSR_UNET_UPBLOCK3_RESNET0_EFFECTIVE_WEIGHTS', upblock3_oracle_path, 'effective static weights for up_blocks.3.resnets.0 exported locally'),
        ('TADSR_UNET_UPBLOCK3_RESNET0_RESIDUAL_CONTRACT_ORACLE', upblock3_oracle_path, 'oracle records up_blocks.3.resnets.0 residual pop/concat contract'),
        ('TADSR_UNET_UPBLOCK3_RESNET0_RESIDUAL_CONCAT_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_resnet0_alignment.json', 'up_blocks.3.resnets.0 residual concat alignment'),
        ('TADSR_UNET_UPBLOCK3_RESNET0_NORM1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_resnet0_alignment.json', 'up_blocks.3.resnets.0 norm1 alignment'),
        ('TADSR_UNET_UPBLOCK3_RESNET0_CONV1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_resnet0_alignment.json', 'up_blocks.3.resnets.0 conv1 alignment'),
        ('TADSR_UNET_UPBLOCK3_RESNET0_TEMB_PROJ_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_resnet0_alignment.json', 'up_blocks.3.resnets.0 time embedding projection alignment'),
        ('TADSR_UNET_UPBLOCK3_RESNET0_CONV2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_resnet0_alignment.json', 'up_blocks.3.resnets.0 conv2 alignment'),
        ('TADSR_UNET_UPBLOCK3_RESNET0_SHORTCUT_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_resnet0_alignment.json', 'up_blocks.3.resnets.0 shortcut alignment'),
        ('TADSR_UNET_UPBLOCK3_RESNET0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_resnet0_alignment.json', 'isolated up_blocks.3.resnets.0 output alignment'),
        ('TADSR_UNET_UPBLOCK3_RESNET0_SYNTHETIC_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_resnet0_synthetic_alignment.json', 'synthetic hidden/residual/temb -> up_blocks.3.resnets.0 alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_UPBLOCK3_RESNET0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblocks_midblock_upblock0_upblock1_upblock2_upblock3_resnet0_alignment.json', 'UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> full local up_blocks.1 -> full local up_blocks.2 -> up_blocks.3.resnets.0 bridge alignment'),
    ]
    upblock3_resnet0_statuses = []
    for check, path, note in upblock3_resnet0_checks:
        st = status_from_json(path)
        upblock3_resnet0_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_upblock3_resnet0.py && python3 tools/export_tadsr_unet_upblock3_resnet0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    upblock3_resnet0_pass = (
        upblock2_local_pass
        and bool(upblock3_resnet0_statuses)
        and all(st == 'PASS' for st in upblock3_resnet0_statuses)
    )
    up3_audit = load_json(upblock3_audit_path)
    up3_next = (
        up3_audit.get('upblock3_topology', {}).get('actual_next_module_after_resnet0')
        or up3_audit.get('next_module_preview', {}).get('name')
        or 'up_blocks.3 next module'
    )
    rows.append(row('TADSR_UNET_UPBLOCK3_RESNET0_BRIDGE_ALIGNMENT', 'PASS' if upblock3_resnet0_pass else 'PARTIAL', 'up_blocks.3 path is aligned through resnets.0 only and deliberately stops before the next up_blocks.3 module', 'continue with the next official up_blocks.3 module'))
    if up3_next == 'up_blocks.3.attentions.0':
        rows.append(row('TADSR_UNET_UPBLOCK3_PRE_ATTENTION0_ALIGNMENT', 'PASS' if upblock3_resnet0_pass else 'PARTIAL', 'execution reaches the input boundary before up_blocks.3.attentions.0 without entering attention0', 'next target is up_blocks.3.attentions.0'))
    elif up3_next == 'up_blocks.3.resnets.1':
        rows.append(row('TADSR_UNET_UPBLOCK3_PRE_RESNET1_ALIGNMENT', 'PASS' if upblock3_resnet0_pass else 'PARTIAL', 'execution reaches the input boundary before up_blocks.3.resnets.1 without entering resnet1', 'next target is up_blocks.3.resnets.1'))
    else:
        rows.append(row('TADSR_UNET_UPBLOCK3_PRE_NEXT_ALIGNMENT', 'PASS' if upblock3_resnet0_pass else 'PARTIAL', f'execution reaches the input boundary before {up3_next} without entering it', f'next target is {up3_next}'))
    upblock3_attention0_checks = [
        ('TADSR_UNET_UPBLOCK3_ATTENTION0_AUDIT', upblock3_attention0_audit_path, 'official up_blocks.3.attentions.0 top-level audit'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION0_TRANSFORMER_AUDIT', upblock3_attention0_audit_path, 'official transformer block audit for up_blocks.3.attentions.0'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION0_LORA_AUDIT', upblock3_attention0_audit_path, 'LoRA/effective static weight audit for up_blocks.3.attentions.0'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION0_RESIDUAL_CONTRACT_AUDIT', upblock3_attention0_audit_path, 'official residual contract: attention0 consumes no residual'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION0_ORACLE_TENSORS', upblock3_attention0_oracle_path, 'PyTorch oracle tensors for up_blocks.3.attentions.0 exported'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION0_EFFECTIVE_WEIGHTS', upblock3_attention0_oracle_path, 'effective static weights for up_blocks.3.attentions.0 exported locally'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION0_RESIDUAL_CONTRACT_ORACLE', upblock3_attention0_oracle_path, 'oracle records unchanged residual tuple across up_blocks.3.attentions.0'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION0_NORM_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_attention0_proj_alignment.json', 'up_blocks.3.attentions.0 top norm alignment'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION0_PROJ_IN_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_attention0_proj_alignment.json', 'up_blocks.3.attentions.0 proj_in alignment'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION0_SEQUENCE_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_attention0_proj_alignment.json', 'up_blocks.3.attentions.0 NCHW-to-sequence contract alignment'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION0_ATTN1_QKV_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_attention0_attn1_alignment.json', 'up_blocks.3.attentions.0 self-attention Q/K/V alignment'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION0_ATTN1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_attention0_attn1_alignment.json', 'up_blocks.3.attentions.0 self-attention output alignment'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION0_AFTER_ATTN1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_attention0_attn1_alignment.json', 'up_blocks.3.attentions.0 residual after self-attention alignment'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION0_ATTN2_QKV_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_attention0_attn2_alignment.json', 'up_blocks.3.attentions.0 cross-attention Q/K/V alignment'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION0_ATTN2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_attention0_attn2_alignment.json', 'up_blocks.3.attentions.0 cross-attention output alignment'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION0_AFTER_ATTN2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_attention0_attn2_alignment.json', 'up_blocks.3.attentions.0 residual after cross-attention alignment'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION0_FF_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_attention0_ff_alignment.json', 'up_blocks.3.attentions.0 feed-forward alignment'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION0_TRANSFORMER0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_attention0_ff_alignment.json', 'up_blocks.3.attentions.0 transformer block output alignment'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_attention0_alignment.json', 'isolated up_blocks.3.attentions.0 output alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_UPBLOCK3_RESNET0_ATTENTION0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblocks_midblock_upblock0_upblock1_upblock2_upblock3_resnet0_attention0_alignment.json', 'UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> full local up_blocks.1 -> full local up_blocks.2 -> up_blocks.3.resnets.0 -> up_blocks.3.attentions.0 bridge alignment'),
    ]
    upblock3_attention0_statuses = []
    for check, path, note in upblock3_attention0_checks:
        st = status_from_json(path)
        upblock3_attention0_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_upblock3_attention0.py && python3 tools/export_tadsr_unet_upblock3_attention0_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    upblock3_attention0_pass = (
        upblock3_resnet0_pass
        and bool(upblock3_attention0_statuses)
        and all(st == 'PASS' for st in upblock3_attention0_statuses)
    )
    up3_attention0_audit = load_json(upblock3_attention0_audit_path)
    up3_after_attention0_next = (
        up3_attention0_audit.get('upblock3_overview', {}).get('actual_next_module_after_attention0')
        or up3_attention0_audit.get('upblock3_topology', {}).get('actual_next_module_after_attention0')
        or up3_attention0_audit.get('next_module_preview', {}).get('name')
        or 'up_blocks.3.resnets.1'
    )
    rows.append(row('TADSR_UNET_UPBLOCK3_ATTENTION0_BRIDGE_ALIGNMENT', 'PASS' if upblock3_attention0_pass else 'PARTIAL', 'up_blocks.3 path is aligned through attentions.0 only and deliberately stops before the next up_blocks.3 module', 'continue with the next official up_blocks.3 module'))
    if up3_after_attention0_next == 'up_blocks.3.resnets.1':
        rows.append(row('TADSR_UNET_UPBLOCK3_PRE_RESNET1_ALIGNMENT', 'PASS' if upblock3_attention0_pass else 'PARTIAL', 'execution reaches the input boundary before up_blocks.3.resnets.1 without entering resnet1', 'next target is up_blocks.3.resnets.1'))
    else:
        rows.append(row('TADSR_UNET_UPBLOCK3_PRE_NEXT_AFTER_ATTENTION0_ALIGNMENT', 'PASS' if upblock3_attention0_pass else 'PARTIAL', f'execution reaches the input boundary before {up3_after_attention0_next} without entering it', f'next target is {up3_after_attention0_next}'))
    upblock3_resnet1_checks = [
        ('TADSR_UNET_UPBLOCK3_RESNET1_AUDIT', upblock3_resnet1_audit_path, 'official up_blocks.3.resnets.1 module audit'),
        ('TADSR_UNET_UPBLOCK3_RESNET1_RESIDUAL_CONTRACT_AUDIT', upblock3_resnet1_audit_path, 'official residual consumption contract for up_blocks.3.resnets.1'),
        ('TADSR_UNET_UPBLOCK3_RESNET1_LORA_AUDIT', upblock3_resnet1_audit_path, 'LoRA/effective static weight audit for up_blocks.3.resnets.1'),
        ('TADSR_UNET_UPBLOCK3_RESNET1_ORACLE_TENSORS', upblock3_resnet1_oracle_path, 'PyTorch oracle tensors for up_blocks.3.resnets.1 exported'),
        ('TADSR_UNET_UPBLOCK3_RESNET1_EFFECTIVE_WEIGHTS', upblock3_resnet1_oracle_path, 'effective static weights for up_blocks.3.resnets.1 exported locally'),
        ('TADSR_UNET_UPBLOCK3_RESNET1_RESIDUAL_CONTRACT_ORACLE', upblock3_resnet1_oracle_path, 'oracle records up_blocks.3.resnets.1 residual pop/concat contract'),
        ('TADSR_UNET_UPBLOCK3_RESNET1_RESIDUAL_CONCAT_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_resnet1_alignment.json', 'up_blocks.3.resnets.1 residual concat alignment'),
        ('TADSR_UNET_UPBLOCK3_RESNET1_NORM1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_resnet1_alignment.json', 'up_blocks.3.resnets.1 norm1 alignment'),
        ('TADSR_UNET_UPBLOCK3_RESNET1_CONV1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_resnet1_alignment.json', 'up_blocks.3.resnets.1 conv1 alignment'),
        ('TADSR_UNET_UPBLOCK3_RESNET1_TEMB_PROJ_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_resnet1_alignment.json', 'up_blocks.3.resnets.1 time embedding projection alignment'),
        ('TADSR_UNET_UPBLOCK3_RESNET1_CONV2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_resnet1_alignment.json', 'up_blocks.3.resnets.1 conv2 alignment'),
        ('TADSR_UNET_UPBLOCK3_RESNET1_SHORTCUT_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_resnet1_alignment.json', 'up_blocks.3.resnets.1 shortcut alignment'),
        ('TADSR_UNET_UPBLOCK3_RESNET1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_resnet1_alignment.json', 'isolated up_blocks.3.resnets.1 output alignment'),
        ('TADSR_UNET_UPBLOCK3_RESNET1_SYNTHETIC_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_resnet1_synthetic_alignment.json', 'synthetic hidden/residual/temb -> up_blocks.3.resnets.1 alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_UPBLOCK3_RESNET0_ATTENTION0_RESNET1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblocks_midblock_upblock0_upblock1_upblock2_upblock3_resnet0_attention0_resnet1_alignment.json', 'UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> full local up_blocks.1 -> full local up_blocks.2 -> up_blocks.3.resnets.0 -> up_blocks.3.attentions.0 -> up_blocks.3.resnets.1 bridge alignment'),
    ]
    upblock3_resnet1_statuses = []
    for check, path, note in upblock3_resnet1_checks:
        st = status_from_json(path)
        upblock3_resnet1_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_upblock3_resnet1.py && python3 tools/export_tadsr_unet_upblock3_resnet1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    upblock3_resnet1_pass = (
        upblock3_attention0_pass
        and bool(upblock3_resnet1_statuses)
        and all(st == 'PASS' for st in upblock3_resnet1_statuses)
    )
    up3_resnet1_audit = load_json(upblock3_resnet1_audit_path)
    up3_after_resnet1_next = (
        up3_resnet1_audit.get('upblock3_topology', {}).get('actual_next_module_after_resnet1')
        or up3_resnet1_audit.get('next_module_preview', {}).get('name')
        or 'up_blocks.3.attentions.1'
    )
    rows.append(row('TADSR_UNET_UPBLOCK3_RESNET1_BRIDGE_ALIGNMENT', 'PASS' if upblock3_resnet1_pass else 'PARTIAL', 'up_blocks.3 path is aligned through resnets.0 -> attentions.0 -> resnets.1 only and deliberately stops before the next up_blocks.3 module', 'continue with the next official up_blocks.3 module'))
    if up3_after_resnet1_next == 'up_blocks.3.attentions.1':
        rows.append(row('TADSR_UNET_UPBLOCK3_PRE_ATTENTION1_ALIGNMENT', 'PASS' if upblock3_resnet1_pass else 'PARTIAL', 'execution reaches the input boundary before up_blocks.3.attentions.1 without entering attention1', 'next target is up_blocks.3.attentions.1'))
    else:
        rows.append(row('TADSR_UNET_UPBLOCK3_PRE_NEXT_AFTER_RESNET1_ALIGNMENT', 'PASS' if upblock3_resnet1_pass else 'PARTIAL', f'execution reaches the input boundary before {up3_after_resnet1_next} without entering it', f'next target is {up3_after_resnet1_next}'))
    upblock3_attention1_checks = [
        ('TADSR_UNET_UPBLOCK3_ATTENTION1_AUDIT', upblock3_attention1_audit_path, 'official up_blocks.3.attentions.1 top-level audit'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION1_TRANSFORMER_AUDIT', upblock3_attention1_audit_path, 'official transformer block audit for up_blocks.3.attentions.1'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION1_LORA_AUDIT', upblock3_attention1_audit_path, 'LoRA/effective static weight audit for up_blocks.3.attentions.1'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION1_RESIDUAL_CONTRACT_AUDIT', upblock3_attention1_audit_path, 'official residual contract: attention1 consumes no residual'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION1_ORACLE_TENSORS', upblock3_attention1_oracle_path, 'PyTorch oracle tensors for up_blocks.3.attentions.1 exported'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION1_EFFECTIVE_WEIGHTS', upblock3_attention1_oracle_path, 'effective static weights for up_blocks.3.attentions.1 exported locally'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION1_RESIDUAL_CONTRACT_ORACLE', upblock3_attention1_oracle_path, 'oracle records unchanged residual tuple across up_blocks.3.attentions.1'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION1_NORM_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_attention1_proj_alignment.json', 'up_blocks.3.attentions.1 top norm alignment'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION1_PROJ_IN_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_attention1_proj_alignment.json', 'up_blocks.3.attentions.1 proj_in alignment'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION1_SEQUENCE_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_attention1_proj_alignment.json', 'up_blocks.3.attentions.1 NCHW-to-sequence contract alignment'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION1_ATTN1_QKV_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_attention1_attn1_alignment.json', 'up_blocks.3.attentions.1 self-attention Q/K/V alignment'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION1_ATTN1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_attention1_attn1_alignment.json', 'up_blocks.3.attentions.1 self-attention output alignment'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION1_AFTER_ATTN1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_attention1_attn1_alignment.json', 'up_blocks.3.attentions.1 residual after self-attention alignment'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION1_ATTN2_QKV_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_attention1_attn2_alignment.json', 'up_blocks.3.attentions.1 cross-attention Q/K/V alignment'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION1_ATTN2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_attention1_attn2_alignment.json', 'up_blocks.3.attentions.1 cross-attention output alignment'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION1_AFTER_ATTN2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_attention1_attn2_alignment.json', 'up_blocks.3.attentions.1 residual after cross-attention alignment'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION1_FF_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_attention1_ff_alignment.json', 'up_blocks.3.attentions.1 feed-forward alignment'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION1_TRANSFORMER0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_attention1_ff_alignment.json', 'up_blocks.3.attentions.1 transformer block output alignment'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_attention1_alignment.json', 'isolated up_blocks.3.attentions.1 output alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_UPBLOCK3_RESNET0_ATTENTION0_RESNET1_ATTENTION1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblocks_midblock_upblock0_upblock1_upblock2_upblock3_resnet0_attention0_resnet1_attention1_alignment.json', 'UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> full local up_blocks.1 -> full local up_blocks.2 -> up_blocks.3.resnets.0 -> up_blocks.3.attentions.0 -> up_blocks.3.resnets.1 -> up_blocks.3.attentions.1 bridge alignment'),
    ]
    upblock3_attention1_statuses = []
    for check, path, note in upblock3_attention1_checks:
        st = status_from_json(path)
        upblock3_attention1_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_upblock3_attention1.py && python3 tools/export_tadsr_unet_upblock3_attention1_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    upblock3_attention1_pass = (
        upblock3_resnet1_pass
        and bool(upblock3_attention1_statuses)
        and all(st == 'PASS' for st in upblock3_attention1_statuses)
    )
    up3_attention1_audit = load_json(upblock3_attention1_audit_path)
    up3_after_attention1_next = (
        up3_attention1_audit.get('upblock3_overview', {}).get('actual_next_module_after_attention1')
        or up3_attention1_audit.get('upblock3_topology', {}).get('actual_next_module_after_attention1')
        or up3_attention1_audit.get('next_module_preview', {}).get('name')
        or 'up_blocks.3.resnets.2'
    )
    rows.append(row('TADSR_UNET_UPBLOCK3_ATTENTION1_BRIDGE_ALIGNMENT', 'PASS' if upblock3_attention1_pass else 'PARTIAL', 'up_blocks.3 path is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 only and deliberately stops before the next up_blocks.3 module', 'continue with the next official up_blocks.3 module'))
    if up3_after_attention1_next == 'up_blocks.3.resnets.2':
        rows.append(row('TADSR_UNET_UPBLOCK3_PRE_RESNET2_ALIGNMENT', 'PASS' if upblock3_attention1_pass else 'PARTIAL', 'execution reaches the input boundary before up_blocks.3.resnets.2 without entering resnet2', 'next target is up_blocks.3.resnets.2'))
    else:
        rows.append(row('TADSR_UNET_UPBLOCK3_PRE_NEXT_AFTER_ATTENTION1_ALIGNMENT', 'PASS' if upblock3_attention1_pass else 'PARTIAL', f'execution reaches the input boundary before {up3_after_attention1_next} without entering it', f'next target is {up3_after_attention1_next}'))
    upblock3_resnet2_checks = [
        ('TADSR_UNET_UPBLOCK3_RESNET2_AUDIT', upblock3_resnet2_audit_path, 'official up_blocks.3.resnets.2 module audit'),
        ('TADSR_UNET_UPBLOCK3_RESNET2_RESIDUAL_CONTRACT_AUDIT', upblock3_resnet2_audit_path, 'official residual consumption contract for up_blocks.3.resnets.2'),
        ('TADSR_UNET_UPBLOCK3_RESNET2_LORA_AUDIT', upblock3_resnet2_audit_path, 'LoRA/effective static weight audit for up_blocks.3.resnets.2'),
        ('TADSR_UNET_UPBLOCK3_RESNET2_ORACLE_TENSORS', upblock3_resnet2_oracle_path, 'PyTorch oracle tensors for up_blocks.3.resnets.2 exported'),
        ('TADSR_UNET_UPBLOCK3_RESNET2_EFFECTIVE_WEIGHTS', upblock3_resnet2_oracle_path, 'effective static weights for up_blocks.3.resnets.2 exported locally'),
        ('TADSR_UNET_UPBLOCK3_RESNET2_RESIDUAL_CONTRACT_ORACLE', upblock3_resnet2_oracle_path, 'oracle records up_blocks.3.resnets.2 residual pop/concat contract'),
        ('TADSR_UNET_UPBLOCK3_RESNET2_RESIDUAL_CONCAT_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_resnet2_alignment.json', 'up_blocks.3.resnets.2 residual concat alignment'),
        ('TADSR_UNET_UPBLOCK3_RESNET2_NORM1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_resnet2_alignment.json', 'up_blocks.3.resnets.2 norm1 alignment'),
        ('TADSR_UNET_UPBLOCK3_RESNET2_CONV1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_resnet2_alignment.json', 'up_blocks.3.resnets.2 conv1 alignment'),
        ('TADSR_UNET_UPBLOCK3_RESNET2_TEMB_PROJ_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_resnet2_alignment.json', 'up_blocks.3.resnets.2 time embedding projection alignment'),
        ('TADSR_UNET_UPBLOCK3_RESNET2_CONV2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_resnet2_alignment.json', 'up_blocks.3.resnets.2 conv2 alignment'),
        ('TADSR_UNET_UPBLOCK3_RESNET2_SHORTCUT_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_resnet2_alignment.json', 'up_blocks.3.resnets.2 shortcut alignment'),
        ('TADSR_UNET_UPBLOCK3_RESNET2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_resnet2_alignment.json', 'isolated up_blocks.3.resnets.2 output alignment'),
        ('TADSR_UNET_UPBLOCK3_RESNET2_SYNTHETIC_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_resnet2_synthetic_alignment.json', 'synthetic hidden/residual/temb -> up_blocks.3.resnets.2 alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_UPBLOCK3_RESNET0_ATTENTION0_RESNET1_ATTENTION1_RESNET2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblocks_midblock_upblock0_upblock1_upblock2_upblock3_resnet0_attention0_resnet1_attention1_resnet2_alignment.json', 'UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> full local up_blocks.1 -> full local up_blocks.2 -> up_blocks.3.resnets.0 -> up_blocks.3.attentions.0 -> up_blocks.3.resnets.1 -> up_blocks.3.attentions.1 -> up_blocks.3.resnets.2 bridge alignment'),
    ]
    upblock3_resnet2_statuses = []
    for check, path, note in upblock3_resnet2_checks:
        st = status_from_json(path)
        upblock3_resnet2_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_upblock3_resnet2.py && python3 tools/export_tadsr_unet_upblock3_resnet2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    upblock3_resnet2_pass = (
        upblock3_attention1_pass
        and bool(upblock3_resnet2_statuses)
        and all(st == 'PASS' for st in upblock3_resnet2_statuses)
    )
    up3_resnet2_audit = load_json(upblock3_resnet2_audit_path)
    up3_after_resnet2_next = (
        up3_resnet2_audit.get('upblock3_topology', {}).get('actual_next_module_after_resnet2')
        or up3_resnet2_audit.get('downstream_preview_only', {}).get('actual_next_module_after_resnet2')
        or up3_resnet2_audit.get('next_module_preview', {}).get('name')
        or 'up_blocks.3.attentions.2'
    )
    rows.append(row('TADSR_UNET_UPBLOCK3_RESNET2_BRIDGE_ALIGNMENT', 'PASS' if upblock3_resnet2_pass else 'PARTIAL', 'up_blocks.3 path is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 -> resnets.2 only and deliberately stops before the next up_blocks.3 module', 'continue with the next official up_blocks.3 module'))
    if up3_after_resnet2_next == 'up_blocks.3.attentions.2':
        rows.append(row('TADSR_UNET_UPBLOCK3_PRE_ATTENTION2_ALIGNMENT', 'PASS' if upblock3_resnet2_pass else 'PARTIAL', 'execution reaches the input boundary before up_blocks.3.attentions.2 without entering attention2', 'next target is up_blocks.3.attentions.2'))
    else:
        rows.append(row('TADSR_UNET_UPBLOCK3_PRE_NEXT_AFTER_RESNET2_ALIGNMENT', 'PASS' if upblock3_resnet2_pass else 'PARTIAL', f'execution reaches the input boundary before {up3_after_resnet2_next} without entering it', f'next target is {up3_after_resnet2_next}'))
    upblock3_attention2_checks = [
        ('TADSR_UNET_UPBLOCK3_ATTENTION2_AUDIT', upblock3_attention2_audit_path, 'official up_blocks.3.attentions.2 module audit'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION2_TRANSFORMER_AUDIT', upblock3_attention2_audit_path, 'official Transformer2DModel/BasicTransformerBlock audit for up_blocks.3.attentions.2'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION2_LORA_AUDIT', upblock3_attention2_audit_path, 'LoRA/effective static weight audit for up_blocks.3.attentions.2'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION2_RESIDUAL_CONTRACT_AUDIT', upblock3_attention2_audit_path, 'official residual non-consumption contract for up_blocks.3.attentions.2'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION2_ORACLE_TENSORS', upblock3_attention2_oracle_path, 'PyTorch oracle tensors for up_blocks.3.attentions.2 exported'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION2_EFFECTIVE_WEIGHTS', upblock3_attention2_oracle_path, 'effective static weights for up_blocks.3.attentions.2 exported locally'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION2_RESIDUAL_CONTRACT_ORACLE', upblock3_attention2_oracle_path, 'oracle records up_blocks.3.attentions.2 residual non-consumption contract'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION2_NORM_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_attention2_proj_alignment.json', 'attention2 top-level GroupNorm alignment'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION2_PROJ_IN_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_attention2_proj_alignment.json', 'attention2 proj_in alignment'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION2_SEQUENCE_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_attention2_proj_alignment.json', 'attention2 NCHW-to-sequence alignment'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION2_ATTN1_QKV_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_attention2_attn1_alignment.json', 'attention2 transformer0 self-attention QKV alignment'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION2_ATTN1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_attention2_attn1_alignment.json', 'attention2 transformer0 self-attention output alignment'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION2_AFTER_ATTN1_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_attention2_attn1_alignment.json', 'attention2 transformer0 after-attn1 residual alignment'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION2_ATTN2_QKV_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_attention2_attn2_alignment.json', 'attention2 transformer0 cross-attention QKV alignment'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION2_ATTN2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_attention2_attn2_alignment.json', 'attention2 transformer0 cross-attention output alignment'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION2_AFTER_ATTN2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_attention2_attn2_alignment.json', 'attention2 transformer0 after-attn2 residual alignment'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION2_FF_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_attention2_ff_alignment.json', 'attention2 feed-forward output alignment'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION2_TRANSFORMER0_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_attention2_ff_alignment.json', 'attention2 full transformer block alignment'),
        ('TADSR_UNET_UPBLOCK3_ATTENTION2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_upblock3_attention2_alignment.json', 'isolated up_blocks.3.attentions.2 output alignment'),
        ('TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_UPBLOCK3_RESNET0_ATTENTION0_RESNET1_ATTENTION1_RESNET2_ATTENTION2_ALIGNMENT', 'experiments/full_repro/unet_alignment/jittor_unet_entry_downblocks_midblock_upblock0_upblock1_upblock2_upblock3_resnet0_attention0_resnet1_attention1_resnet2_attention2_alignment.json', 'UNet entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> full local up_blocks.1 -> full local up_blocks.2 -> up_blocks.3.resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 -> resnets.2 -> attentions.2 bridge alignment'),
    ]
    upblock3_attention2_statuses = []
    for check, path, note in upblock3_attention2_checks:
        st = status_from_json(path)
        upblock3_attention2_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_upblock3_attention2.py && python3 tools/export_tadsr_unet_upblock3_attention2_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    upblock3_attention2_pass = (
        upblock3_resnet2_pass
        and bool(upblock3_attention2_statuses)
        and all(st == 'PASS' for st in upblock3_attention2_statuses)
    )
    rows.append(row('TADSR_UNET_UPBLOCK3_ATTENTION2_BRIDGE_ALIGNMENT', 'PASS' if upblock3_attention2_pass else 'PARTIAL', 'up_blocks.3 path is aligned through resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 -> resnets.2 -> attentions.2 and deliberately stops before output tail/full local aggregate', 'next target is the full local up_blocks.3 aggregate / output-tail-boundary verification'))
    rows.append(row('TADSR_UNET_UPBLOCK3_PRE_LOCAL_AGGREGATE_ALIGNMENT', 'PASS' if upblock3_attention2_pass else 'PARTIAL', 'execution reaches the boundary before output tail / full local up_blocks.3 aggregate without entering full UNet forward', 'next target is full local up_blocks.3 aggregate verification only'))
    upblock3_local_checks = [
        ('TADSR_UNET_UPBLOCK3_LOCAL_FORWARD_AUDIT', upblock3_local_audit_path, 'official local up_blocks.3 forward agrees with the audited manual resnet/attention chain'),
        ('TADSR_UNET_UPBLOCK3_RESIDUAL_CONTRACT_AUDIT', upblock3_local_audit_path, 'official up_blocks.3 local residual tuple consumption contract audited'),
        ('TADSR_UNET_UPBLOCK3_LOCAL_ORACLE_TENSORS', upblock3_local_oracle_path, 'PyTorch oracle tensors for full local up_blocks.3 aggregate exported'),
        ('TADSR_UNET_UPBLOCK3_RESIDUAL_CONTRACT_ORACLE', upblock3_local_oracle_path, 'oracle records up_blocks.3 residual consumption and zero residuals after attention2'),
        ('TADSR_UNET_OUTPUT_TAIL_BOUNDARY_AUDIT', upblock3_local_audit_path, 'official output-tail boundary after up_blocks.3 audited without executing the tail'),
    ]
    upblock3_local_statuses = []
    for check, path, note in upblock3_local_checks:
        st = status_from_json(path)
        upblock3_local_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_upblock3_local.py && python3 tools/export_tadsr_unet_upblock3_local_oracle.py && USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_upblock3_local_alignment.py'))
    upblock3_local_alignment = load_json(upblock3_local_alignment_path)
    upblock3_output_hidden_status = upblock3_local_alignment.get('output_hidden_status', status_from_json(upblock3_local_alignment_path))
    upblock3_output_states_status = upblock3_local_alignment.get('output_states_status', 'BLOCKED')
    output_tail_boundary_alignment = upblock3_local_alignment.get('output_tail_boundary_status', status_from_json(upblock3_local_alignment_path))
    rows.append(row('TADSR_UNET_UPBLOCK3_OUTPUT_HIDDEN_ALIGNMENT', upblock3_output_hidden_status, 'Jittor full local up_blocks.3 hidden output matches the PyTorch local oracle', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_upblock3_local_alignment.py'))
    rows.append(row('TADSR_UNET_UPBLOCK3_OUTPUT_STATES_ALIGNMENT', upblock3_output_states_status, 'official up_blocks.3 output_states contract; NOT_APPLICABLE if it returns a hidden tensor only', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_upblock3_local_alignment.py'))
    rows.append(row('TADSR_UNET_OUTPUT_TAIL_BOUNDARY_ALIGNMENT', output_tail_boundary_alignment, 'Jittor bridge stops at the audited output-tail boundary without running conv_norm_out/conv_act/conv_out', 'next target is output tail audit/export/port'))
    upblock3_local_pass = (
        upblock3_attention2_pass
        and bool(upblock3_local_statuses)
        and all(st == 'PASS' for st in upblock3_local_statuses)
        and upblock3_output_hidden_status == 'PASS'
        and upblock3_output_states_status in {'PASS', 'NOT_APPLICABLE'}
        and output_tail_boundary_alignment == 'PASS'
    )
    rows.append(row('TADSR_UNET_UPBLOCK3_ALIGNMENT', 'PASS' if upblock3_local_pass else 'PARTIAL', 'up_blocks.3 leaves plus full local aggregate hidden output are aligned and the output-tail boundary is audited', 'continue with output tail audit/export/port before full UNet forward'))
    output_tail_checks = [
        ('TADSR_UNET_OUTPUT_TAIL_AUDIT', output_tail_audit_path, 'official output tail conv_norm_out -> conv_act -> conv_out audit'),
        ('TADSR_UNET_OUTPUT_TAIL_TOPOLOGY_AUDIT', output_tail_audit_path, 'official output tail topology/config audit'),
        ('TADSR_UNET_OUTPUT_TAIL_LORA_AUDIT', output_tail_audit_path, 'output tail conv_out LoRA/effective static weight audit'),
        ('TADSR_UNET_OUTPUT_TAIL_LOCAL_EXECUTION_AUDIT', output_tail_audit_path, 'official local output tail execution without official unet.forward'),
        ('TADSR_UNET_OUTPUT_TAIL_ORACLE_TENSORS', output_tail_oracle_path, 'PyTorch oracle tensors for output tail exported'),
        ('TADSR_UNET_OUTPUT_TAIL_EFFECTIVE_WEIGHTS', output_tail_oracle_path, 'conv_norm_out raw affine parameters and conv_out LoRA-merged effective weights exported'),
    ]
    output_tail_statuses = []
    for check, path, note in output_tail_checks:
        st = status_from_json(path)
        output_tail_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_output_tail.py && python3 tools/export_tadsr_unet_output_tail_oracle.py && USE_CUDA=0 nvcc_path="" bash scripts/run_jittor_alignment_tests.sh'))
    output_tail_alignment = load_json(output_tail_alignment_path)
    output_tail_norm_status = output_tail_alignment.get('norm_status', status_from_json(output_tail_alignment_path))
    output_tail_act_status = output_tail_alignment.get('act_status', status_from_json(output_tail_alignment_path))
    output_tail_conv_out_status = output_tail_alignment.get('conv_out_status', status_from_json(output_tail_alignment_path))
    output_tail_alignment_status = status_from_json(output_tail_alignment_path)
    output_tail_synthetic_status = status_from_json(output_tail_synthetic_alignment_path)
    entry_to_output_tail_alignment = load_json(entry_to_output_tail_alignment_path)
    entry_to_output_tail_status = status_from_json(entry_to_output_tail_alignment_path)
    manual_blocks_to_tail_status = entry_to_output_tail_alignment.get('status', entry_to_output_tail_status)
    rows.append(row('TADSR_UNET_OUTPUT_TAIL_NORM_ALIGNMENT', output_tail_norm_status, 'conv_norm_out GroupNorm alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_output_tail_alignment.py'))
    rows.append(row('TADSR_UNET_OUTPUT_TAIL_ACT_ALIGNMENT', output_tail_act_status, 'conv_act SiLU alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_output_tail_alignment.py'))
    rows.append(row('TADSR_UNET_OUTPUT_TAIL_CONV_OUT_ALIGNMENT', output_tail_conv_out_status, 'conv_out LoRA-merged effective convolution alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_output_tail_alignment.py'))
    rows.append(row('TADSR_UNET_OUTPUT_TAIL_ALIGNMENT', output_tail_alignment_status, 'isolated output tail alignment from the PyTorch up_blocks.3 output tensor', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_output_tail_alignment.py'))
    rows.append(row('TADSR_UNET_OUTPUT_TAIL_SYNTHETIC_ALIGNMENT', output_tail_synthetic_status, 'synthetic hidden tensor -> output tail alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_output_tail_synthetic_alignment.py'))
    rows.append(row('TADSR_UNET_ENTRY_TO_OUTPUT_TAIL_ALIGNMENT', entry_to_output_tail_status, 'entry -> all down_blocks -> full local mid_block -> full local up_blocks.0/1/2/3 -> output tail bridge alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_entry_downblocks_midblock_upblocks_output_tail_alignment.py'))
    rows.append(row('TADSR_UNET_MANUAL_BLOCKS_TO_TAIL_ALIGNMENT', manual_blocks_to_tail_status, 'manual block composition through output tail is aligned without official UNet.forward', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_entry_downblocks_midblock_upblocks_output_tail_alignment.py'))
    output_tail_pass = (
        upblock3_local_pass
        and bool(output_tail_statuses)
        and all(st == 'PASS' for st in output_tail_statuses)
        and output_tail_norm_status == 'PASS'
        and output_tail_act_status == 'PASS'
        and output_tail_conv_out_status == 'PASS'
        and output_tail_alignment_status == 'PASS'
        and output_tail_synthetic_status == 'PASS'
        and entry_to_output_tail_status == 'PASS'
        and manual_blocks_to_tail_status == 'PASS'
    )
    manual_wrapper_checks = [
        ('TADSR_UNET_MANUAL_FULL_WRAPPER_AUDIT', manual_wrapper_audit_path, 'official manual wrapper contract audit without official UNet.forward'),
        ('TADSR_UNET_MANUAL_FULL_WRAPPER_CONTRACT_AUDIT', manual_wrapper_audit_path, 'manual wrapper input/output contract audit'),
        ('TADSR_UNET_MANUAL_FULL_CHAIN_TO_TAIL_AUDIT', manual_wrapper_audit_path, 'official module chain audit through output tail without full forward'),
        ('TADSR_UNET_MANUAL_FULL_WRAPPER_ORACLE_TENSORS', manual_wrapper_oracle_path, 'PyTorch oracle tensors for alignment-only manual full-chain wrapper exported'),
    ]
    manual_wrapper_statuses = []
    for check, path, note in manual_wrapper_checks:
        st = status_from_json(path)
        manual_wrapper_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_manual_wrapper.py && python3 tools/export_tadsr_unet_manual_wrapper_oracle.py'))
    manual_wrapper_alignment = load_json(manual_wrapper_alignment_path)
    manual_wrapper_entry_status = manual_wrapper_alignment.get('entry_status', status_from_json(manual_wrapper_alignment_path))
    manual_wrapper_downblocks_status = manual_wrapper_alignment.get('downblocks_status', status_from_json(manual_wrapper_alignment_path))
    manual_wrapper_midblock_status = manual_wrapper_alignment.get('midblock_status', status_from_json(manual_wrapper_alignment_path))
    manual_wrapper_upblocks_status = manual_wrapper_alignment.get('upblocks_status', status_from_json(manual_wrapper_alignment_path))
    manual_wrapper_output_tail_status = manual_wrapper_alignment.get('output_tail_status', status_from_json(manual_wrapper_alignment_path))
    manual_wrapper_alignment_status = status_from_json(manual_wrapper_alignment_path)
    rows.append(row('TADSR_UNET_MANUAL_WRAPPER_ENTRY_ALIGNMENT', manual_wrapper_entry_status, 'manual wrapper entry path alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_manual_wrapper_alignment.py'))
    rows.append(row('TADSR_UNET_MANUAL_WRAPPER_DOWNBLOCKS_ALIGNMENT', manual_wrapper_downblocks_status, 'manual wrapper down_blocks alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_manual_wrapper_alignment.py'))
    rows.append(row('TADSR_UNET_MANUAL_WRAPPER_MIDBLOCK_ALIGNMENT', manual_wrapper_midblock_status, 'manual wrapper mid_block alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_manual_wrapper_alignment.py'))
    rows.append(row('TADSR_UNET_MANUAL_WRAPPER_UPBLOCKS_ALIGNMENT', manual_wrapper_upblocks_status, 'manual wrapper up_blocks.0/1/2/3 alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_manual_wrapper_alignment.py'))
    rows.append(row('TADSR_UNET_MANUAL_WRAPPER_OUTPUT_TAIL_ALIGNMENT', manual_wrapper_output_tail_status, 'manual wrapper output-tail alignment', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_manual_wrapper_alignment.py'))
    rows.append(row('TADSR_UNET_MANUAL_FULL_WRAPPER_ALIGNMENT', manual_wrapper_alignment_status, 'alignment-only manual UNet chain wrapper output matches the PyTorch manual oracle', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_manual_wrapper_alignment.py'))
    rows.append(row('TADSR_UNET_MANUAL_FULL_CHAIN_ALIGNMENT', manual_wrapper_alignment_status, 'manual full chain from UNet inputs through output tail is aligned; this is not official full forward', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_manual_wrapper_alignment.py'))
    manual_wrapper_pass = (
        output_tail_pass
        and bool(manual_wrapper_statuses)
        and all(st == 'PASS' for st in manual_wrapper_statuses)
        and manual_wrapper_entry_status == 'PASS'
        and manual_wrapper_downblocks_status == 'PASS'
        and manual_wrapper_midblock_status == 'PASS'
        and manual_wrapper_upblocks_status == 'PASS'
        and manual_wrapper_output_tail_status == 'PASS'
        and manual_wrapper_alignment_status == 'PASS'
    )
    full_forward_checks = [
        ('TADSR_UNET_OFFICIAL_FULL_FORWARD_AUDIT', full_forward_audit_path, 'official PyTorch UNet.forward contract audit'),
        ('TADSR_UNET_FULL_FORWARD_CONTRACT_AUDIT', full_forward_audit_path, 'official full forward input/output and return contract audit'),
        ('TADSR_UNET_MANUAL_VS_OFFICIAL_FULL_FORWARD_AUDIT', full_forward_audit_path, 'PyTorch manual wrapper output matches official full forward output'),
        ('TADSR_UNET_OFFICIAL_FULL_FORWARD_ORACLE_TENSORS', full_forward_oracle_path, 'official full forward oracle tensors exported'),
        ('TADSR_UNET_MANUAL_VS_OFFICIAL_FULL_FORWARD_ALIGNMENT', full_forward_oracle_path, 'manual wrapper oracle matches official full forward oracle'),
    ]
    full_forward_statuses = []
    for check, path, note in full_forward_checks:
        st = status_from_json(path)
        full_forward_statuses.append(st)
        rows.append(row(check, st, note, 'python3 tools/audit_official_tadsr_unet_full_forward.py && python3 tools/export_tadsr_unet_full_forward_oracle.py'))
    full_forward_alignment = load_json(full_forward_alignment_path)
    jittor_vs_official_status = full_forward_alignment.get('jittor_vs_official_status', status_from_json(full_forward_alignment_path))
    jittor_vs_manual_status = full_forward_alignment.get('jittor_vs_manual_status', status_from_json(full_forward_alignment_path))
    return_contract_status = full_forward_alignment.get('return_contract_status', status_from_json(full_forward_alignment_path))
    full_forward_alignment_status = status_from_json(full_forward_alignment_path)
    rows.append(row('TADSR_UNET_JITTOR_VS_OFFICIAL_FULL_FORWARD_ALIGNMENT', jittor_vs_official_status, 'Jittor alignment-only full forward output matches official PyTorch UNet.forward oracle', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_full_forward_alignment.py'))
    rows.append(row('TADSR_UNET_JITTOR_VS_MANUAL_WRAPPER_ALIGNMENT', jittor_vs_manual_status, 'Jittor alignment-only full forward output matches PyTorch manual wrapper oracle', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_full_forward_alignment.py'))
    rows.append(row('TADSR_UNET_FULL_FORWARD_RETURN_CONTRACT_ALIGNMENT', return_contract_status, 'alignment-only return_dict/tensor contract matches the test contract', 'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_unet_full_forward_alignment.py'))
    full_forward_pass = (
        jittor_vs_official_status == 'PASS'
        and jittor_vs_manual_status == 'PASS'
        and return_contract_status in {'PASS', 'NOT_APPLICABLE'}
        and full_forward_alignment_status == 'PASS'
    )
    rows.append(row('TADSR_UNET_UPBLOCKS_ALIGNMENT', 'PASS' if upblock3_local_pass else 'NOT_COMPLETE', 'UNet up_blocks.0/1/2/3 local aggregate path is aligned; output tail is checked separately and full UNet forward remains unopened', 'continue with full UNet forward-wrapper audit/export after output tail stays PASS'))
    rows.append(row('TADSR_UNET_CROSS_ATTENTION_ALIGNMENT', 'PASS' if upblock3_local_pass else 'NOT_COMPLETE', 'all local UNet cross-attention modules through up_blocks.3 are aligned; global full-forward integration remains incomplete until full forward wrapper is compared', 'continue with full-forward migration'))
    rows.append(row('TADSR_UNET_FULL_FORWARD_ALIGNMENT', 'PASS' if full_forward_pass else 'NOT_COMPLETE', 'UNet full forward numerical alignment against official PyTorch UNet.forward; not full TADSR inference', 'continue pipeline boundary planning only after keeping inference guard intact'))
    rows.append(row('TADSR_UNET_LORA_RUNTIME_INTEGRATION', 'PARTIAL', 'LoRA-bearing modules are exported as static effective weights; generic runtime LoRA is not implemented', 'implement runtime LoRA only if a later audit proves dynamic adapter behavior is required'))
    lora_policy = load_json('experiments/full_repro/lora_alignment/audit_tadsr_lora_policy.json')
    lora_coverage = load_json('experiments/full_repro/lora_alignment/jittor_effective_lora_coverage.json')
    lora_contract = load_json('experiments/full_repro/lora_alignment/lora_policy_contract_test.json')
    timevae_lora_effective = load_json('experiments/full_repro/lora_alignment/timevae_lora_effective_weights/timevae_lora_effective_weights_metadata.json')
    timevae_lora_effective_test = load_json('experiments/full_repro/lora_alignment/timevae_lora_effective_weight_coverage_test.json')
    rows.append(row(
        'TADSR_LORA_POLICY_AUDIT',
        str(lora_policy.get('status', 'BLOCKED')) if lora_policy else 'BLOCKED',
        'official TADSR LoRA/PEFT source usage, active module inventory, and inference-time adapter policy audited',
        'python3 tools/audit_official_tadsr_lora_policy.py',
    ))
    rows.append(row(
        'TADSR_STATIC_EFFECTIVE_LORA_POLICY_AUDIT',
        str(lora_policy.get('static_effective_lora_policy_audit', 'BLOCKED')) if lora_policy else 'BLOCKED',
        'static effective weights are the selected alignment policy when official inference uses fixed active adapters',
        'docs/lora_policy_decision.md',
    ))
    rows.append(row(
        'TADSR_RUNTIME_DYNAMIC_LORA_REQUIREMENT_AUDIT',
        str(lora_policy.get('runtime_dynamic_lora_requirement_audit', 'BLOCKED')) if lora_policy else 'BLOCKED',
        'generic runtime LoRA is required only if official inference dynamically changes adapters or scale',
        'python3 tools/audit_official_tadsr_lora_policy.py',
    ))
    rows.append(row(
        'TADSR_LORA_MODULE_INVENTORY_AUDIT',
        str(lora_policy.get('module_inventory_status', 'BLOCKED')) if lora_policy else 'BLOCKED',
        'active LoRA A/B module inventory from the official TADSR checkpoint and source policy audit',
        'python3 tools/audit_official_tadsr_lora_policy.py',
    ))
    rows.append(row(
        'TIME_VAE_LORA_EFFECTIVE_WEIGHTS_AUDIT',
        str(timevae_lora_effective.get('timevae_lora_effective_weights_audit', 'BLOCKED')) if timevae_lora_effective else 'BLOCKED',
        'official active TimeVAE LoRA pair inventory is exported as static effective weights',
        'python3 tools/export_tadsr_timevae_lora_effective_weights.py',
    ))
    rows.append(row(
        'TIME_VAE_LORA_EFFECTIVE_WEIGHTS_EXPORT',
        str(timevae_lora_effective.get('timevae_lora_effective_weights_export', 'BLOCKED')) if timevae_lora_effective else 'BLOCKED',
        'converted_timevae_lora_effective_weights.npz exists as a local git-ignored static effective-weight artifact',
        'python3 tools/export_tadsr_timevae_lora_effective_weights.py',
    ))
    rows.append(row(
        'TIME_VAE_LORA_EFFECTIVE_WEIGHT_MANUAL_VERIFY',
        str(timevae_lora_effective.get('timevae_lora_effective_weight_manual_verify', 'BLOCKED')) if timevae_lora_effective else 'BLOCKED',
        'official active LoRA module forward is compared with manual effective-weight forward for each TimeVAE pair',
        'python3 tools/export_tadsr_timevae_lora_effective_weights.py',
    ))
    rows.append(row(
        'TIME_VAE_LORA_EFFECTIVE_ARTIFACT_COVERAGE',
        str(lora_coverage.get('timevae_lora_effective_artifact_coverage', 'BLOCKED')) if lora_coverage else 'BLOCKED',
        'all active TimeVAE LoRA pairs have static effective-weight artifact metadata and verification evidence',
        'python3 tools/audit_jittor_tadsr_effective_lora_coverage.py',
    ))
    rows.append(row(
        'TIME_VAE_ACTIVE_LORA_MODULE_COVERAGE',
        str(lora_coverage.get('timevae_active_lora_module_coverage', 'BLOCKED')) if lora_coverage else 'BLOCKED',
        'TimeVAE active LoRA module coverage in the project-wide static effective-weight policy audit',
        'python3 tools/audit_jittor_tadsr_effective_lora_coverage.py',
    ))
    rows.append(row(
        'TADSR_STATIC_EFFECTIVE_LORA_COVERAGE_AUDIT',
        str(lora_coverage.get('static_effective_lora_coverage_audit', 'BLOCKED')) if lora_coverage else 'BLOCKED',
        'Jittor static effective-weight artifacts are checked against official active LoRA modules',
        'python3 tools/audit_jittor_tadsr_effective_lora_coverage.py',
    ))
    rows.append(row(
        'TADSR_ACTIVE_LORA_MODULE_COVERAGE',
        str(lora_coverage.get('active_lora_module_coverage', 'BLOCKED')) if lora_coverage else 'BLOCKED',
        'active official LoRA modules covered by exported effective weights or explicitly reported as partial/missing',
        'python3 tools/audit_jittor_tadsr_effective_lora_coverage.py',
    ))
    rows.append(row(
        'TADSR_EFFECTIVE_WEIGHT_ARTIFACT_COVERAGE',
        str(lora_coverage.get('effective_weight_artifact_coverage', 'BLOCKED')) if lora_coverage else 'BLOCKED',
        'effective-weight artifacts exist and large NPZ files remain local artifacts rather than committed payloads',
        'python3 tools/audit_jittor_tadsr_effective_lora_coverage.py',
    ))
    rows.append(row(
        'TADSR_DYNAMIC_RUNTIME_LORA_IMPLEMENTATION',
        str(lora_coverage.get('dynamic_runtime_lora_implementation', 'NOT_IMPLEMENTED_BY_DESIGN')) if lora_coverage else 'NOT_IMPLEMENTED_BY_DESIGN',
        'generic runtime LoRA adapter loading/switching/scale is deferred and not claimed complete',
        'dedicated runtime LoRA stage only if required by a future official behavior audit',
    ))
    rows.append(row(
        'TADSR_LORA_POLICY_CONTRACT_TEST',
        str(lora_contract.get('status', 'BLOCKED')) if lora_contract else 'BLOCKED',
        'metadata-only contract test for LoRA policy/coverage evidence and full-inference guard',
        'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_lora_policy_coverage.py',
    ))
    rows.append(row(
        'TIME_VAE_LORA_EFFECTIVE_WEIGHT_COVERAGE_TEST',
        str(timevae_lora_effective_test.get('status', 'BLOCKED')) if timevae_lora_effective_test else 'BLOCKED',
        'metadata-only contract test for TimeVAE LoRA effective artifact coverage and full-inference guard',
        'USE_CUDA=0 nvcc_path="" python3 tests_jittor_alignment/test_timevae_lora_effective_weight_coverage.py',
    ))

    rows.append(row('TIME_VAE_ENCODER_TO_QUANT_ALIGNMENT', status_from_json('experiments/full_repro/time_vae_alignment/jittor_encoder_stage0123_mid_tail_quant_alignment.json'), 'TimeAware VAE encoder-side path to quant_conv aligned; deterministic decoder stack through tail aligned separately', 'continue full UNet audit/export/port'))
    timevae_full_pass = (
        deterministic_decoder_pass
        and timevae_audit_status == 'PASS'
        and timevae_contract_status == 'PASS'
        and timevae_oracle_status == 'PASS'
        and timevae_boundary_status == 'PASS'
        and timevae_full_alignment_candidate
        and timevae_boundary_sufficient
    )
    timevae_full_note = (
        'TimeVAE full-boundary alignment is PASS, but full TimeVAE alignment remains NOT_COMPLETE because audit metadata says the boundary is not sufficient for full TADSR pipeline: VAEHook tiling, internal stochastic sampling policy, runtime LoRA, and full inference remain unopened.'
        if timevae_boundary_status == 'PASS' and not timevae_full_pass
        else 'full TimeVAE required boundary is complete and aligned'
    )
    rows.append(row('TIME_VAE_FULL_ALIGNMENT', 'PASS' if timevae_full_pass else 'NOT_COMPLETE', timevae_full_note, 'audit VAEHook tiled path / runtime LoRA before full inference' if not timevae_full_pass else 'continue scheduler / minimal pipeline dry-run planning while keeping full inference guard intact'))
    rows.append(row('JITTOR_FULL_INFERENCE', 'NOT_COMPLETE', 'full VAE/UNet/LoRA inference is intentionally NotImplemented', 'continue block-level migration'))
    rows.append(row('JITTOR_MIGRATION_REPORT', 'PASS' if Path('experiments/full_repro/jittor_alignment/jittor_migration_report.md').exists() else 'BLOCKED', 'migration evidence report exists' if Path('experiments/full_repro/jittor_alignment/jittor_migration_report.md').exists() else 'missing migration report', 'python3 scripts/make_jittor_migration_report.py'))
    rows.append(row('JITTOR_FULL_PORT', 'PARTIAL' if Path('jittor_tadsr_full/README_FULL_PORT_STATUS.md').exists() else 'NOT_STARTED', 'skeleton/check CLI only; full VAE/UNet not complete', 'continue module port'))
    readme = Path('README.md').read_text(encoding='utf-8') if Path('README.md').exists() else ''
    rok = all(s in readme for s in ['Project Goal', 'What Is Actually Implemented in Jittor', 'Jittor Module Alignment', 'Still Not Claimed'])
    rows.append(row('FINAL_README', 'PASS' if rok else 'FAIL', 'README has Jittor migration evidence sections' if rok else 'README missing Jittor migration evidence sections', 'update README'))

    pre_packaging_status = {n: s for n, s, _, _ in rows}
    evidence_manifest = load_json('experiments/final_evidence_manifest.json')
    evidence_manifest_exists = Path('experiments/final_evidence_manifest.json').exists() and Path('experiments/final_evidence_manifest.md').exists()
    evidence_manifest_status = str(evidence_manifest.get('status', 'BLOCKED')) if evidence_manifest else 'BLOCKED'
    rows.append(row(
        'TADSR_FINAL_EVIDENCE_MANIFEST',
        evidence_manifest_status if evidence_manifest_exists else 'BLOCKED',
        'final evidence manifest indexes committed audit/report files' if evidence_manifest_exists else 'final evidence manifest missing',
        'python3 scripts/collect_final_evidence_manifest.py',
    ))
    production_memo = Path('docs/production_cli_design_audit.md')
    memo_text = production_memo.read_text(encoding='utf-8') if production_memo.exists() else ''
    production_memo_ok = production_memo.exists() and all(s in memo_text for s in [
        'NotImplementedError',
        'Full denoising-loop contract',
        'Minimal decode tensor boundary does not mean image generation is complete',
        'Static effective LoRA coverage does not mean generic runtime LoRA',
    ])
    rows.append(row(
        'TADSR_PRODUCTION_CLI_DESIGN_AUDIT',
        'PASS' if production_memo_ok else 'BLOCKED',
        'production CLI design memo exists and keeps full inference guarded' if production_memo_ok else 'production CLI design memo missing required guardrail language',
        'update docs/production_cli_design_audit.md',
    ))
    checklist = Path('docs/final_submission_checklist.md')
    checklist_text = checklist.read_text(encoding='utf-8') if checklist.exists() else ''
    checklist_ok = checklist.exists() and all(s in checklist_text for s in [
        'UNet full forward alignment',
        'Minimal one-step decode tensor dry-run',
        'Full inference remains',
        'Known Limitations',
    ])
    rows.append(row(
        'TADSR_FINAL_SUBMISSION_CHECKLIST',
        'PASS' if checklist_ok else 'BLOCKED',
        'final submission checklist exists with status, commands, evidence and limitations' if checklist_ok else 'final submission checklist missing required sections',
        'update docs/final_submission_checklist.md',
    ))
    reporting_files = [
        Path('README.md'),
        Path('jittor_tadsr_full/README_FULL_PORT_STATUS.md'),
        Path('docs/next_jittor_full_migration_tasks.md'),
        Path('docs/03_ppt_outline.md'),
        Path('docs/04_video_script.md'),
        Path('experiments/full_repro/jittor_alignment/jittor_migration_report.md'),
    ]
    reporting_text = '\n'.join(p.read_text(encoding='utf-8') for p in reporting_files if p.exists())
    reporting_ok = all(p.exists() for p in reporting_files) and all(s in reporting_text for s in [
        'Minimal one-step decode',
        'JITTOR_FULL_INFERENCE',
        'NOT_COMPLETE',
        'production full',
    ])
    false_claims = [
        'full TADSR inference complete',
        'production pipeline complete',
        'image/video generation complete',
        'runtime dynamic LoRA complete',
    ]
    false_claim_hits = []
    for line in reporting_text.splitlines():
        low = line.lower()
        if any(guard in low for guard in ['not ', 'no ', 'do not', 'does not', 'without', 'never']):
            continue
        for claim in false_claims:
            if claim in line:
                false_claim_hits.append(line.strip())
    reporting_status = 'PASS' if reporting_ok and not false_claim_hits else ('PARTIAL' if reporting_ok else 'BLOCKED')
    rows.append(row(
        'TADSR_REPORTING_READINESS_AUDIT',
        reporting_status,
        'README/docs/report consistently document completed boundaries and honest gaps' if reporting_status == 'PASS' else f'reporting needs review; false_claim_hits={false_claim_hits}',
        'update README/docs/report wording',
    ))
    gitignore = Path('.gitignore').read_text(encoding='utf-8') if Path('.gitignore').exists() else ''
    large_policy_ok = all(s in gitignore for s in ['*.npy', '*.npz']) and Path('experiments/final_evidence_manifest.md').exists()
    rows.append(row(
        'TADSR_LARGE_ARTIFACT_POLICY_AUDIT',
        'PASS' if large_policy_ok else 'PARTIAL',
        'large tensor artifacts are ignored; metadata/report files are committed' if large_policy_ok else 'large artifact ignore policy needs review',
        'check .gitignore and staged files before commit',
    ))
    technical_ready = all(pre_packaging_status.get(k) == 'PASS' for k in [
        'TADSR_UNET_FULL_FORWARD_ALIGNMENT',
        'TIME_VAE_ACTUAL_VAEHOOK_FULL_BOUNDARY_ALIGNMENT',
        'TADSR_STATIC_EFFECTIVE_LORA_COVERAGE_AUDIT',
        'TADSR_SCHEDULER_BOUNDARY_ALIGNMENT',
        'TADSR_MINIMAL_ONE_STEP_DECODE_DRY_RUN',
    ])
    honest_gaps = (
        pre_packaging_status.get('JITTOR_FULL_INFERENCE') == 'NOT_COMPLETE'
        and pre_packaging_status.get('JITTOR_FULL_PORT') == 'PARTIAL'
        and pre_packaging_status.get('TIME_VAE_FULL_ALIGNMENT') == 'NOT_COMPLETE'
        and pre_packaging_status.get('TADSR_DYNAMIC_RUNTIME_LORA_IMPLEMENTATION') == 'NOT_IMPLEMENTED_BY_DESIGN'
    )
    final_packaging_ready = (
        evidence_manifest_exists
        and evidence_manifest_status in {'PASS', 'PARTIAL'}
        and production_memo_ok
        and checklist_ok
        and reporting_status == 'PASS'
        and large_policy_ok
        and Path('experiments/final_audit_report.json').exists()
        and technical_ready
        and honest_gaps
    )
    rows.append(row(
        'TADSR_FINAL_PACKAGING_READINESS',
        'PASS' if final_packaging_ready else 'PARTIAL',
        'final evidence package is ready for presentation/repository handoff while full inference remains guarded' if final_packaging_ready else 'fix missing evidence/docs before new technical work',
        'prepare final presentation/video/repository handoff' if final_packaging_ready else 'repair final evidence package',
    ))

    presentation_validation = load_json('experiments/final_presentation_package_validation.json')
    presentation_markers = presentation_validation.get('markers', {}) if presentation_validation else {}
    presentation_package_status = str(presentation_markers.get('TADSR_FINAL_PRESENTATION_PACKAGE', 'BLOCKED'))
    video_script_status = str(presentation_markers.get('TADSR_VIDEO_SCRIPT_READY', 'BLOCKED'))
    demo_runbook_status = str(presentation_markers.get('TADSR_DEMO_RUNBOOK_READY', 'BLOCKED'))
    repository_handoff_status = str(presentation_markers.get('TADSR_REPOSITORY_HANDOFF_READY', 'BLOCKED'))
    rows.append(row(
        'TADSR_FINAL_PRESENTATION_PACKAGE',
        presentation_package_status,
        'final PPT outline and presentation handoff docs are ready' if presentation_package_status == 'PASS' else 'presentation package validation missing or failed',
        'python3 scripts/validate_final_presentation_package.py',
    ))
    rows.append(row(
        'TADSR_VIDEO_SCRIPT_READY',
        video_script_status,
        'final video script is ready and avoids full-inference/image-generation claims' if video_script_status == 'PASS' else 'video script needs review',
        'update docs/04_video_script.md',
    ))
    rows.append(row(
        'TADSR_DEMO_RUNBOOK_READY',
        demo_runbook_status,
        'demo runbook lists reproducible audit/alignment commands and expected markers' if demo_runbook_status == 'PASS' else 'demo runbook missing or incomplete',
        'update docs/final_demo_runbook.md',
    ))
    rows.append(row(
        'TADSR_REPOSITORY_HANDOFF_READY',
        repository_handoff_status,
        'repository handoff guide maps code, evidence, large-file policy and limitations' if repository_handoff_status == 'PASS' else 'repository handoff guide missing or incomplete',
        'update docs/repository_handoff_guide.md',
    ))
    final_handoff_ready = (
        final_packaging_ready
        and presentation_package_status == 'PASS'
        and video_script_status == 'PASS'
        and demo_runbook_status == 'PASS'
        and repository_handoff_status == 'PASS'
        and honest_gaps
    )
    rows.append(row(
        'TADSR_FINAL_HANDOFF_READINESS',
        'PASS' if final_handoff_ready else 'PARTIAL',
        'repository is ready for final presentation/video recording and submission handoff' if final_handoff_ready else 'fix missing presentation/demo/handoff docs',
        'record final presentation/video; future technical work should be controlled production CLI validation' if final_handoff_ready else 'run python3 scripts/validate_final_presentation_package.py and repair docs',
    ))

    final_deliverables_validation = load_json('experiments/final_deliverables_validation.json')
    final_deliverable_markers = final_deliverables_validation.get('markers', {}) if final_deliverables_validation else {}

    def final_deliverable_status(name: str) -> str:
        return str(final_deliverable_markers.get(name, 'BLOCKED'))

    final_deliverable_rows = [
        (
            'TADSR_FINAL_PPT_READY',
            'final PPTX presentation exists and is non-empty',
            'generate deliverables/TADSR-Jittor_final_presentation.pptx',
        ),
        (
            'TADSR_FINAL_PDF_READY',
            'final PDF presentation export exists and is non-empty',
            'generate deliverables/TADSR-Jittor_final_presentation.pdf',
        ),
        (
            'TADSR_FINAL_VIDEO_RECORDING_PACKAGE_READY',
            'video recording guide exists with safe demo commands and honest limitations',
            'update deliverables/TADSR-Jittor_video_recording_guide.md',
        ),
        (
            'TADSR_FINAL_SUBMISSION_README_READY',
            'submission README exists and indexes final deliverables/evidence',
            'update deliverables/TADSR-Jittor_submission_readme.md',
        ),
        (
            'TADSR_FINAL_DELIVERABLE_SIZE_AUDIT',
            'final deliverables stay below the 100 MB submission budget',
            'remove large generated artifacts from deliverables if needed',
        ),
        (
            'TADSR_FINAL_DELIVERABLES_INCLUDE_SMOKE_TRAINING',
            'final PPT/video/submission README include small-data PyTorch-vs-Jittor smoke-training evidence',
            'update final deliverables with smoke-training logs, curves and alignment summaries',
        ),
        (
            'TADSR_FINAL_PPT_INCLUDES_SMOKE_TRAINING',
            'final PPT Markdown/PPTX source includes the small-data smoke-training evidence slide',
            'update docs/03_ppt_outline.md and deliverables/TADSR-Jittor_final_presentation.md',
        ),
        (
            'TADSR_VIDEO_SCRIPT_INCLUDES_SMOKE_TRAINING',
            'final video script/recording guide includes the smoke-training segment and demo command',
            'update docs/04_video_script.md and deliverables/TADSR-Jittor_video_recording_guide.md',
        ),
        (
            'TADSR_SUBMISSION_README_INCLUDES_SMOKE_TRAINING',
            'submission README indexes smoke-training logs, loss curves, prediction visualizations and multi-seed evidence',
            'update deliverables/TADSR-Jittor_submission_readme.md',
        ),
        (
            'TADSR_FINAL_DELIVERABLES_READY',
            'PPT/PDF/video guide/submission README are ready and free of misleading full-inference claims',
            'python3 scripts/validate_final_deliverables.py',
        ),
    ]
    for marker, pass_note, action in final_deliverable_rows:
        marker_status = final_deliverable_status(marker)
        rows.append(row(
            marker,
            marker_status,
            pass_note if marker_status == 'PASS' else 'final deliverables validation missing or failed',
            action,
        ))
    final_submission_ready = (
        final_handoff_ready
        and final_deliverable_status('TADSR_FINAL_DELIVERABLES_READY') == 'PASS'
        and final_deliverable_status('TADSR_FINAL_DELIVERABLES_INCLUDE_SMOKE_TRAINING') == 'PASS'
        and honest_gaps
    )
    rows.append(row(
        'TADSR_FINAL_SUBMISSION_READY',
        'PASS' if final_submission_ready else 'PARTIAL',
        'final deliverables are ready for submission while full inference remains explicitly guarded' if final_submission_ready else 'run final deliverables validation and keep full inference NotImplemented',
        'record final video and submit the repository package' if final_submission_ready else 'python3 scripts/validate_final_deliverables.py',
    ))

    submission_content_validation = load_json('experiments/final_submission_content_validation.json')
    submission_content_markers = submission_content_validation.get('markers', {}) if submission_content_validation else {}

    def submission_content_status(name: str) -> str:
        return str(submission_content_markers.get(name, 'BLOCKED'))

    submission_content_rows = [
        (
            'TADSR_FINAL_SUBMISSION_CONTENT_VALIDATION',
            'PPT/video/submission wording contains required final markers and avoids misleading claims',
            'python3 scripts/validate_final_submission_content.py',
        ),
        (
            'TADSR_FINAL_PPT_CONTENT_GUARDRAIL',
            'PPT Markdown and outline preserve honest full-inference limitations',
            'update docs/03_ppt_outline.md and deliverables/TADSR-Jittor_final_presentation.md',
        ),
        (
            'TADSR_FINAL_VIDEO_SCRIPT_GUARDRAIL',
            'video script and recording guide avoid full-inference/image-generation/runtime-LoRA over-claims',
            'update docs/04_video_script.md and deliverables/TADSR-Jittor_video_recording_guide.md',
        ),
        (
            'TADSR_FINAL_SUBMISSION_README_GUARDRAIL',
            'submission README indexes evidence and keeps guarded-scope wording',
            'update deliverables/TADSR-Jittor_submission_readme.md',
        ),
    ]
    for marker, pass_note, action in submission_content_rows:
        marker_status = submission_content_status(marker)
        rows.append(row(
            marker,
            marker_status,
            pass_note if marker_status == 'PASS' else 'final submission content validation missing or failed',
            action,
        ))

    repository_handoff_validation = load_json('experiments/repository_handoff_validation.json')
    repository_handoff_markers = repository_handoff_validation.get('markers', {}) if repository_handoff_validation else {}

    def repository_handoff_marker_status(name: str) -> str:
        return str(repository_handoff_markers.get(name, 'BLOCKED'))

    repository_handoff_rows = [
        (
            'TADSR_REPOSITORY_HANDOFF_VALIDATION',
            'repository handoff files, git readability and large artifact policy are validated',
            'python3 scripts/validate_repository_handoff.py',
        ),
        (
            'TADSR_GITHUB_SUBMISSION_GUIDE_READY',
            'GitHub handoff guide exists with remote/push commands and no automatic push policy',
            'update docs/github_submission_handoff.md',
        ),
        (
            'TADSR_TRACKED_LARGE_ARTIFACT_AUDIT',
            'no new, staged or out-of-policy .npy/.npz files are present; historical oracle tensors are separately reported',
            'check git ls-files and git status for .npy/.npz',
        ),
    ]
    for marker, pass_note, action in repository_handoff_rows:
        marker_status = repository_handoff_marker_status(marker)
        rows.append(row(
            marker,
            marker_status,
            pass_note if marker_status == 'PASS' else 'repository handoff validation missing or failed',
            action,
        ))

    video_preflight = Path('docs/video_recording_preflight_checklist.md')
    video_preflight_text = video_preflight.read_text(encoding='utf-8') if video_preflight.exists() else ''
    video_preflight_ok = video_preflight.exists() and all(s in video_preflight_text for s in [
        'Full inference remains guarded',
        'No generated final image/video output is claimed',
        'Generic runtime LoRA is not implemented',
        'python3 scripts/final_audit.py',
        '20-30 Minute',
    ])
    rows.append(row(
        'TADSR_VIDEO_RECORDING_PREFLIGHT_CHECKLIST',
        'PASS' if video_preflight_ok else 'PARTIAL',
        'recording checklist exists with windows, commands, limits and timing' if video_preflight_ok else 'video recording preflight checklist missing required content',
        'update docs/video_recording_preflight_checklist.md',
    ))

    release_candidate_qa_ready = (
        final_submission_ready
        and submission_content_status('TADSR_FINAL_SUBMISSION_CONTENT_VALIDATION') == 'PASS'
        and repository_handoff_marker_status('TADSR_REPOSITORY_HANDOFF_VALIDATION') == 'PASS'
        and repository_handoff_marker_status('TADSR_GITHUB_SUBMISSION_GUIDE_READY') == 'PASS'
        and video_preflight_ok
        and honest_gaps
    )
    rows.append(row(
        'TADSR_RELEASE_CANDIDATE_QA_READINESS',
        'PASS' if release_candidate_qa_ready else 'PARTIAL',
        'release-candidate QA package is ready for human GitHub handoff and video recording' if release_candidate_qa_ready else 'fix missing docs, repository handoff, or misleading content before submission',
        'human can record final video, upload/push repository to GitHub, and submit deliverables' if release_candidate_qa_ready else 'run final content/repository validators and repair issues',
    ))

    smoke_root = Path('experiments/smoke_training/output_tail')
    smoke_metadata = load_json(smoke_root / 'smoke_training_data_metadata.json')
    pytorch_smoke = load_json(smoke_root / 'pytorch/final_metrics.json')
    jittor_smoke = load_json(smoke_root / 'jittor/final_metrics.json')
    smoke_comparison = load_json(smoke_root / 'smoke_training_comparison.json')
    smoke_alignment = load_json(smoke_root / 'smoke_training_alignment_metrics.json')
    smoke_multiseed = load_json(smoke_root / 'multiseed/multiseed_summary.json')
    smoke_artifacts = load_json(smoke_root / 'smoke_training_artifacts_test.json')

    def false_flags_ok(d: dict) -> bool:
        return bool(d) and all(d.get(k) is False for k in [
            'full_tadsr_training',
            'full_inference_executed',
            'image_saved',
            'video_saved',
        ])

    smoke_data_status = 'PASS' if (
        smoke_metadata.get('task_name') == 'tadsr_output_tail_conv_out_smoke_training'
        and false_flags_ok(smoke_metadata)
        and Path(smoke_root / 'smoke_training_data_summary.md').exists()
    ) else 'BLOCKED'
    train_val_split_status = 'PASS' if (
        smoke_metadata.get('num_samples') == 32
        and smoke_metadata.get('train_samples') == 24
        and smoke_metadata.get('val_samples') == 8
        and len(smoke_metadata.get('train_indices', [])) == 24
        and len(smoke_metadata.get('val_indices', [])) == 8
    ) else 'BLOCKED'
    rows.append(row(
        'TADSR_SMOKE_TRAINING_DATA_PREP',
        smoke_data_status,
        'deterministic output-tail conv_out feature-target pairs metadata exists; raw .npy tensors stay ignored' if smoke_data_status == 'PASS' else 'smoke training data metadata missing or unsafe',
        'python3 tools/export_tadsr_smoke_training_data.py --num-samples 32 --train-samples 24 --val-samples 8 --seed 1234',
    ))
    rows.append(row(
        'TADSR_SMOKE_TRAINING_TRAIN_VAL_SPLIT',
        train_val_split_status,
        'smoke training data has a deterministic 24/8 train-validation split over 32 output-tail samples' if train_val_split_status == 'PASS' else 'train/validation split metadata missing or incomplete',
        'python3 tools/export_tadsr_smoke_training_data.py --num-samples 32 --train-samples 24 --val-samples 8 --seed 1234',
    ))

    pytorch_status = str(pytorch_smoke.get('status', 'BLOCKED')) if pytorch_smoke else 'BLOCKED'
    pytorch_loss_status = str(pytorch_smoke.get('loss_decrease_status', 'BLOCKED')) if pytorch_smoke else 'BLOCKED'
    rows.append(row(
        'TADSR_PYTORCH_SMOKE_TRAINING',
        pytorch_status,
        'PyTorch reference output-tail conv_out smoke training ran with real optimizer steps' if pytorch_status == 'PASS' else 'PyTorch smoke training metrics missing or failed',
        '/mnt/data/sj/venvs/tadsr_official_strict_cu118/bin/python tools/train_smoke_pytorch_output_tail.py',
    ))
    rows.append(row(
        'TADSR_PYTORCH_SMOKE_TRAINING_LOSS_DECREASE',
        pytorch_loss_status,
        'PyTorch smoke-training loss decreased without NaN/Inf' if pytorch_loss_status == 'PASS' else 'PyTorch loss decrease is missing, partial, or failed',
        'inspect experiments/smoke_training/output_tail/pytorch/loss.csv',
    ))

    jittor_status = str(jittor_smoke.get('status', 'BLOCKED')) if jittor_smoke else 'BLOCKED'
    jittor_loss_status = str(jittor_smoke.get('loss_decrease_status', 'BLOCKED')) if jittor_smoke else 'BLOCKED'
    rows.append(row(
        'TADSR_JITTOR_SMOKE_TRAINING',
        jittor_status,
        'Jittor output-tail conv_out smoke training ran with real optimizer steps' if jittor_status == 'PASS' else 'Jittor smoke training metrics missing or failed',
        'USE_CUDA=0 nvcc_path="" python3 scripts/train_smoke_jittor_output_tail.py',
    ))
    rows.append(row(
        'TADSR_JITTOR_SMOKE_TRAINING_LOSS_DECREASE',
        jittor_loss_status,
        'Jittor smoke-training loss decreased without NaN/Inf' if jittor_loss_status == 'PASS' else 'Jittor loss decrease is missing, partial, or failed',
        'inspect experiments/smoke_training/output_tail/jittor/loss.csv',
    ))

    loss_log_status = 'PASS' if (
        Path(smoke_root / 'pytorch/loss.csv').exists()
        and Path(smoke_root / 'jittor/loss.csv').exists()
    ) else 'BLOCKED'
    validation_log_status = 'PASS' if (
        Path(smoke_root / 'pytorch/validation_loss.csv').exists()
        and Path(smoke_root / 'jittor/validation_loss.csv').exists()
    ) else 'BLOCKED'
    loss_curve_status = 'PASS' if (
        Path(smoke_root / 'loss_curve.png').exists()
        and Path(smoke_root / 'loss_curve.png').stat().st_size > 0
    ) else 'BLOCKED'
    train_val_curve_status = 'PASS' if (
        Path(smoke_root / 'visualizations/train_val_loss_curve.png').exists()
        and Path(smoke_root / 'visualizations/train_val_loss_curve.png').stat().st_size > 0
    ) else 'BLOCKED'
    loss_gap_curve_status = 'PASS' if (
        Path(smoke_root / 'visualizations/loss_gap_curve.png').exists()
        and Path(smoke_root / 'visualizations/relative_loss_gap_curve.png').exists()
        and Path(smoke_root / 'visualizations/loss_gap_curve.png').stat().st_size > 0
        and Path(smoke_root / 'visualizations/relative_loss_gap_curve.png').stat().st_size > 0
    ) else 'BLOCKED'
    perf_status = 'PASS' if (
        Path(smoke_root / 'pytorch/performance_log.csv').exists()
        and Path(smoke_root / 'jittor/performance_log.csv').exists()
    ) else 'BLOCKED'
    performance_vis_status = 'PASS' if (
        Path(smoke_root / 'visualizations/performance_step_time.png').exists()
        and Path(smoke_root / 'visualizations/performance_samples_per_sec.png').exists()
        and Path(smoke_root / 'visualizations/performance_step_time.png').stat().st_size > 0
        and Path(smoke_root / 'visualizations/performance_samples_per_sec.png').stat().st_size > 0
    ) else 'BLOCKED'
    tensor_vis_status = 'PASS' if all(
        Path(smoke_root / p).exists() and Path(smoke_root / p).stat().st_size > 0
        for p in [
            'visualizations/prediction_target_heatmap.png',
            'visualizations/pytorch_jittor_prediction_heatmap.png',
            'visualizations/prediction_error_heatmap.png',
        ]
    ) else 'BLOCKED'
    alignment_status = str(smoke_comparison.get('trend_alignment_status', 'BLOCKED')) if smoke_comparison else 'BLOCKED'
    prediction_alignment_status = str(smoke_alignment.get('prediction_alignment_status', 'BLOCKED')) if smoke_alignment else 'BLOCKED'
    validation_alignment_status = str(smoke_alignment.get('validation_alignment_status', 'BLOCKED')) if smoke_alignment else 'BLOCKED'
    multiseed_status = str(smoke_multiseed.get('status', 'BLOCKED')) if smoke_multiseed else 'BLOCKED'
    artifacts_status = str(smoke_artifacts.get('status', 'BLOCKED')) if smoke_artifacts else 'BLOCKED'
    rows.append(row(
        'TADSR_SMOKE_TRAINING_LOSS_LOG',
        loss_log_status,
        'PyTorch and Jittor smoke-training loss.csv files exist' if loss_log_status == 'PASS' else 'loss logs missing',
        'run PyTorch/Jittor smoke training scripts',
    ))
    rows.append(row(
        'TADSR_SMOKE_TRAINING_VALIDATION_LOSS_LOG',
        validation_log_status,
        'PyTorch and Jittor validation_loss.csv files exist for small-data validation evidence' if validation_log_status == 'PASS' else 'validation loss logs missing',
        'run PyTorch/Jittor smoke training scripts with --eval-interval 10',
    ))
    rows.append(row(
        'TADSR_SMOKE_TRAINING_LOSS_CURVE',
        loss_curve_status,
        'loss_curve.png generated from both smoke-training logs' if loss_curve_status == 'PASS' else 'loss curve missing',
        'python3 scripts/plot_smoke_training_curves.py',
    ))
    rows.append(row(
        'TADSR_SMOKE_TRAINING_TRAIN_VAL_LOSS_CURVE',
        train_val_curve_status,
        'train/validation loss curve visualizes both PyTorch and Jittor smoke training' if train_val_curve_status == 'PASS' else 'train/validation loss curve missing',
        'python3 scripts/visualize_smoke_training_alignment.py --data-dir experiments/smoke_training/output_tail',
    ))
    rows.append(row(
        'TADSR_SMOKE_TRAINING_LOSS_GAP_CURVE',
        loss_gap_curve_status,
        'absolute and relative PyTorch/Jittor loss gap curves are generated' if loss_gap_curve_status == 'PASS' else 'loss gap curves missing',
        'python3 scripts/visualize_smoke_training_alignment.py --data-dir experiments/smoke_training/output_tail',
    ))
    rows.append(row(
        'TADSR_SMOKE_TRAINING_PERFORMANCE_LOG',
        perf_status,
        'performance_log.csv exists for both PyTorch and Jittor smoke training' if perf_status == 'PASS' else 'performance logs missing',
        'run PyTorch/Jittor smoke training scripts',
    ))
    rows.append(row(
        'TADSR_SMOKE_TRAINING_PERFORMANCE_VISUALIZATION',
        performance_vis_status,
        'step-time and samples/sec visualizations exist for PyTorch/Jittor smoke training' if performance_vis_status == 'PASS' else 'performance visualizations missing',
        'python3 scripts/visualize_smoke_training_alignment.py --data-dir experiments/smoke_training/output_tail',
    ))
    rows.append(row(
        'TADSR_SMOKE_TRAINING_PYTORCH_JITTOR_LOSS_ALIGNMENT',
        alignment_status,
        'PyTorch and Jittor output-tail smoke losses both decrease with aligned trend' if alignment_status == 'PASS' else 'loss trend alignment is missing, partial, or failed',
        'python3 scripts/plot_smoke_training_curves.py',
    ))
    rows.append(row(
        'TADSR_SMOKE_TRAINING_PREDICTION_ALIGNMENT',
        prediction_alignment_status,
        'PyTorch and Jittor final validation prediction tensors align for the output-tail smoke task' if prediction_alignment_status == 'PASS' else 'prediction alignment metrics missing, partial, or failed',
        'python3 scripts/analyze_smoke_training_alignment.py --data-dir experiments/smoke_training/output_tail',
    ))
    rows.append(row(
        'TADSR_SMOKE_TRAINING_VALIDATION_ALIGNMENT',
        validation_alignment_status,
        'validation loss curves and prediction summaries align between PyTorch and Jittor' if validation_alignment_status == 'PASS' else 'validation alignment metrics missing, partial, or failed',
        'python3 scripts/analyze_smoke_training_alignment.py --data-dir experiments/smoke_training/output_tail',
    ))
    rows.append(row(
        'TADSR_SMOKE_TRAINING_TENSOR_VISUALIZATION',
        tensor_vis_status,
        'prediction/target/error tensor heatmaps are generated as diagnostics, not restored images' if tensor_vis_status == 'PASS' else 'tensor diagnostic heatmaps missing',
        'python3 scripts/visualize_smoke_training_alignment.py --data-dir experiments/smoke_training/output_tail',
    ))
    rows.append(row(
        'TADSR_SMOKE_TRAINING_MULTI_SEED_STABILITY',
        multiseed_status,
        'multi-seed small-data output-tail training stability summary exists' if multiseed_status == 'PASS' else ('multi-seed stability is documented as partial' if multiseed_status == 'PARTIAL' else 'multi-seed stability summary missing or failed'),
        'python3 scripts/run_smoke_training_multiseed.py --seeds 1234 2025 42 --steps 200',
    ))
    rows.append(row(
        'TADSR_SMOKE_TRAINING_ARTIFACTS_TEST',
        artifacts_status,
        'smoke-training artifacts are present and do not claim full training/inference/image/video generation' if artifacts_status == 'PASS' else 'smoke-training artifact test missing or failed',
        'python3 tests_jittor_alignment/test_smoke_training_artifacts.py',
    ))
    smoke_core_statuses = [
        smoke_data_status,
        train_val_split_status,
        pytorch_status,
        jittor_status,
        loss_log_status,
        validation_log_status,
        loss_curve_status,
        train_val_curve_status,
        loss_gap_curve_status,
        perf_status,
        performance_vis_status,
        tensor_vis_status,
        artifacts_status,
    ]
    alignment_statuses = [alignment_status, prediction_alignment_status, validation_alignment_status]
    if all(st == 'PASS' for st in smoke_core_statuses) and all(st == 'PASS' for st in alignment_statuses) and multiseed_status == 'PASS':
        smoke_readiness = 'PASS'
    elif all(st == 'PASS' for st in smoke_core_statuses) and not any(st == 'FAIL' for st in alignment_statuses) and multiseed_status in {'PASS', 'PARTIAL'}:
        smoke_readiness = 'PASS_WITH_PARTIAL_ALIGNMENT'
    elif any(st == 'FAIL' for st in smoke_core_statuses + alignment_statuses + [multiseed_status]):
        smoke_readiness = 'FAIL'
    else:
        smoke_readiness = 'PARTIAL'
    rows.append(row(
        'TADSR_SMALL_DATA_TRAINING_READINESS',
        smoke_readiness,
        'small-data training pipeline evidence is ready; this is output-tail smoke training, not full TADSR training' if smoke_readiness == 'PASS' else 'small-data training evidence needs review or documentation',
        'update final presentation/video materials with smoke-training evidence' if smoke_readiness == 'PASS' else 'tune lr/steps or document partial smoke-training alignment',
    ))

    github_release_audit = load_json('experiments/github_release_readiness_audit.json')
    github_release_markers = github_release_audit.get('markers', {}) if github_release_audit else {}

    def github_release_status(name: str) -> str:
        return str(github_release_markers.get(name, 'BLOCKED'))

    github_release_rows = [
        (
            'TADSR_GITHUB_HEAD_ARTIFACT_AUDIT',
            'tracked tensor artifacts are counted and classified without deleting evidence files',
            'python3 scripts/audit_github_release_readiness.py',
        ),
        (
            'TADSR_GIT_HISTORY_LARGE_BLOB_AUDIT',
            'Git history blobs are scanned for >100MB hard-limit and >50MB warning risks',
            'python3 scripts/audit_github_release_readiness.py',
        ),
        (
            'TADSR_GITHUB_DELIVERABLE_SIZE_AUDIT',
            'final deliverables remain below the 100MB submission budget',
            'python3 scripts/audit_github_release_readiness.py',
        ),
        (
            'TADSR_GITHUB_WORKTREE_LARGE_ARTIFACT_AUDIT',
            'worktree/staged tensor artifact risks are reported without adding new tensor files',
            'python3 scripts/audit_github_release_readiness.py',
        ),
        (
            'TADSR_EVIDENCE_DEPENDENCY_AUDIT',
            'final evidence manifest depends on metadata/reports rather than requiring raw oracle tensors',
            'python3 scripts/audit_github_release_readiness.py',
        ),
        (
            'TADSR_GITHUB_RELEASE_SLIMMING_DECISION',
            'release slimming decision memo exists and records direct-push vs cleanup options',
            'update docs/github_release_slimming_decision.md',
        ),
        (
            'TADSR_GITHUB_RELEASE_READINESS_AUDIT',
            'GitHub release readiness audit reports no hard-limit risk or explains remaining size risk',
            'python3 scripts/audit_github_release_readiness.py',
        ),
        (
            'TADSR_GITHUB_RELEASE_AFTER_PHASE5B_READY',
            'Phase 5-B summary and diagnostic plan are present in GitHub release readiness evidence',
            'python3 scripts/audit_github_release_readiness.py',
        ),
    ]
    for marker, pass_note, action in github_release_rows:
        marker_status = github_release_status(marker)
        rows.append(row(
            marker,
            marker_status,
            pass_note if marker_status == 'PASS' else 'GitHub release readiness audit missing, partial, or failed',
            action,
        ))

    feasibility_rows = [
        (
            'TADSR_JITTOR_MIGRATION_FEASIBILITY_VALIDATION',
            'matrix-style migration feasibility validation exists and passes',
            'python3 scripts/validate_jittor_migration_feasibility.py',
        ),
        (
            'TADSR_MODULE_COVERAGE_MATRIX',
            'UNet / TimeVAE / LoRA / scheduler / training / deliverables coverage matrix is ready',
            'python3 scripts/validate_jittor_migration_feasibility.py',
        ),
        (
            'TADSR_WEIGHT_LOADING_COVERAGE_MATRIX',
            'weight loading and static effective LoRA coverage matrix is ready',
            'python3 scripts/validate_jittor_migration_feasibility.py',
        ),
        (
            'TADSR_LORA_EFFECTIVE_WEIGHT_COVERAGE_MATRIX',
            'all active static effective LoRA modules are covered by evidence',
            'python3 scripts/validate_jittor_migration_feasibility.py',
        ),
        (
            'TADSR_NUMERICAL_ALIGNMENT_COVERAGE_MATRIX',
            'numerical alignment evidence matrix is ready',
            'python3 scripts/validate_jittor_migration_feasibility.py',
        ),
        (
            'TADSR_INTEGRATION_PATH_COVERAGE_MATRIX',
            'UNet + TimeVAE + LoRA + scheduler + minimal integration paths are summarized',
            'python3 scripts/validate_jittor_migration_feasibility.py',
        ),
        (
            'TADSR_TRAINING_PATH_FEASIBILITY_MATRIX',
            'small-data PyTorch-vs-Jittor training feasibility matrix is ready',
            'python3 scripts/validate_jittor_migration_feasibility.py',
        ),
        (
            'TADSR_BOUNDARY_LEVEL_REPRODUCTION',
            'boundary-level Jittor migration evidence is sufficient for the final report',
            'python3 scripts/validate_jittor_migration_feasibility.py',
        ),
        (
            'TADSR_SMALL_DATA_TRAINING_ALIGNMENT',
            'small-data PyTorch-vs-Jittor loss/prediction/multi-seed alignment is recorded',
            'python3 scripts/validate_jittor_migration_feasibility.py',
        ),
        (
            'TADSR_FULL_INFERENCE_GUARD_VALIDATION',
            'full inference CLI still raises NotImplementedError and remains guarded',
            'python3 scripts/validate_jittor_migration_feasibility.py',
        ),
        (
            'TADSR_FULL_INFERENCE_GAP_ANALYSIS',
            'full inference gap is analyzed and documented while JITTOR_FULL_INFERENCE stays NOT_COMPLETE',
            'python3 scripts/validate_jittor_migration_feasibility.py',
        ),
        (
            'TADSR_TIMEVAE_FULL_ALIGNMENT_GAP_ANALYSIS',
            'TimeVAE full-alignment gap is analyzed while TIME_VAE_FULL_ALIGNMENT stays NOT_COMPLETE',
            'python3 scripts/validate_jittor_migration_feasibility.py',
        ),
        (
            'TADSR_LORA_RUNTIME_GAP_ANALYSIS',
            'static effective LoRA vs dynamic runtime LoRA boundary is documented',
            'python3 scripts/validate_jittor_migration_feasibility.py',
        ),
        (
            'TADSR_RUNTIME_LORA_LAYER_FORMULA_ALIGNMENT',
            'fixed-adapter/fixed-scale LoRA formula equivalence is checked without claiming runtime adapter switching',
            'python3 scripts/validate_lora_layer_formula_alignment.py',
        ),
        (
            'TADSR_SUBMISSION_FACING_STATUS_SUMMARY',
            'teacher-facing final status summary exists',
            'python3 scripts/validate_jittor_migration_feasibility.py',
        ),
        (
            'TADSR_ENVIRONMENT_BLOCKERS_EXPLAINED',
            'resource and environment blockers are explicitly explained in the teacher-facing summary',
            'python3 scripts/validate_jittor_migration_feasibility.py',
        ),
        (
            'TADSR_GAP_ANALYSIS_READINESS',
            'all known gaps are analyzed, scoped, and guarded against false completion claims',
            'python3 scripts/validate_jittor_migration_feasibility.py',
        ),
    ]
    for marker, pass_note, action in feasibility_rows:
        marker_status = feasibility_marker_status(marker)
        rows.append(row(
            marker,
            marker_status,
            pass_note if marker_status == 'PASS' else 'migration feasibility summary missing, partial, or failed',
            action,
        ))

    production_completion_rows = [
        (
            'TADSR_PRODUCTION_COMPLETION_READINESS',
            'production completion branch/readiness checks are generated without opening full inference',
            'python3 scripts/validate_production_completion_readiness.py',
        ),
        (
            'TADSR_PRODUCTION_COMPLETION_BRANCH_READY',
            'codex/tadsr-production-completion branch and submission-freeze ref are available',
            'git branch --show-current',
        ),
        (
            'TADSR_PRODUCTION_COMPLETION_BASELINE_PRESERVED',
            'baseline submission markers and NotImplemented full-inference guard are preserved',
            'python3 scripts/validate_production_completion_readiness.py',
        ),
        (
            'TADSR_TIMEVAE_FULL_PRODUCTION_PATH_AUDIT',
            'TimeVAE production path audit report exists; PARTIAL means live official runtime remains a blocker',
            'python3 tools/audit_timevae_full_production_path.py --metadata-only 1',
        ),
        (
            'TADSR_OFFICIAL_RUNTIME_LORA_BEHAVIOR_AUDIT',
            'official runtime LoRA behavior audit report exists; PARTIAL means existing reports were used',
            'python3 tools/audit_official_runtime_lora_behavior.py --metadata-only 1',
        ),
        (
            'TADSR_DYNAMIC_RUNTIME_LORA_NOT_REQUIRED_FOR_OFFICIAL_INFERENCE',
            'existing fixed-adapter evidence indicates dynamic LoRA is not required for the audited fixed inference path',
            'python3 tools/audit_official_runtime_lora_behavior.py --metadata-only 1',
        ),
        (
            'TADSR_FULL_INFERENCE_CONTROLLED_VALIDATION_PLAN',
            'controlled full-inference validation plan exists; stages were not executed in this phase',
            'inspect docs/production_completion/full_inference_controlled_validation_plan.md',
        ),
        (
            'TADSR_PRODUCTION_COMPLETION_BLOCKER_REPORT',
            'blocker report documents official-env and future-stage limits without failing current submission',
            'inspect experiments/production_completion/blockers/production_completion_blockers.md',
        ),
        (
            'TADSR_PRODUCTION_COMPLETION_PHASE1_VALIDATION',
            'Phase 1 validation summarizes readiness/audits/plan/blockers without upgrading NOT_COMPLETE markers',
            'python3 scripts/validate_production_completion_phase1.py',
        ),
        (
            'TADSR_PRODUCTION_OFFICIAL_ENV_RESOLUTION',
            'Phase 2 official repo/weights/python availability is resolved without loading models',
            'python3 scripts/resolve_production_official_env.py',
        ),
        (
            'TADSR_TIMEVAE_PRODUCTION_ORACLE_METADATA',
            'TimeVAE production oracle metadata report exists or is explicitly blocked by official environment',
            'python3 tools/export_timevae_production_metadata_oracle.py --metadata-only 1',
        ),
        (
            'TADSR_TIMEVAE_METADATA_PARTIAL_DIAGNOSIS',
            'TimeVAE metadata PARTIAL reason, missing fields and next repair actions are reported',
            'python scripts/diagnose_timevae_metadata_partial.py',
        ),
        (
            'TADSR_OFFICIAL_DIFFUSERS_OVERLAY_READY',
            'isolated diffusers overlay readiness is recorded without modifying the official strict venv',
            'python3 scripts/prepare_official_runtime_dependency_overlay.py --execute 1',
        ),
        (
            'TADSR_OFFICIAL_RUNTIME_DEPENDENCY_REPAIR_READINESS',
            'official dependency repair readiness is recorded from overlay dry-run or execution',
            'python3 scripts/prepare_official_runtime_dependency_overlay.py --execute 1',
        ),
        (
            'TADSR_OFFICIAL_RUNTIME_DEPENDENCY_OVERLAY_ACTIVE',
            'official dependency diagnosis records whether PYTHONPATH overlay was active for imports',
            'python3 scripts/diagnose_official_runtime_dependencies.py --pythonpath-overlay <overlay>',
        ),
        (
            'TADSR_OFFICIAL_RUNTIME_DEPENDENCY_DIAGNOSIS',
            'official Python runtime dependency diagnosis is recorded without modifying the environment',
            'python3 scripts/diagnose_official_runtime_dependencies.py',
        ),
        (
            'TADSR_TIMEVAE_METADATA_REPAIR_ATTEMPT',
            'TimeVAE production metadata repair attempt is recorded without running full inference',
            'python3 tools/export_timevae_production_metadata_oracle.py --metadata-only 1',
        ),
        (
            'TADSR_TIMEVAE_METADATA_FIELDS_COMPLETE',
            'TimeVAE required metadata field completeness is evaluated explicitly',
            'python3 tests_jittor_alignment/test_timevae_production_alignment_preflight.py',
        ),
        (
            'TADSR_TIMEVAE_METADATA_READY_FOR_ONE_STEP',
            'TimeVAE metadata readiness for controlled one-step tensor alignment is evaluated separately',
            'python3 scripts/validate_full_inference_metadata_contract.py',
        ),
        (
            'TADSR_TIMEVAE_LIVE_METADATA_COMPLETION',
            'controlled TimeVAE live encode/decode metadata completion is validated from JSON only',
            'python3 scripts/validate_timevae_live_metadata_completion.py',
        ),
        (
            'TADSR_TIMEVAE_LIVE_ENCODE_METADATA',
            'controlled live TimeVAE encode/posterior/latent metadata fields are complete',
            'python3 scripts/validate_timevae_live_metadata_completion.py',
        ),
        (
            'TADSR_TIMEVAE_LIVE_DECODE_METADATA',
            'controlled live TimeVAE decode/clamp metadata fields are complete',
            'python3 scripts/validate_timevae_live_metadata_completion.py',
        ),
        (
            'TADSR_TIMEVAE_LIVE_SAFETY_FLAGS',
            'controlled live TimeVAE metadata export proves no full inference, scheduler loop, UNet call, image/video generation or raw tensor commit',
            'python3 scripts/validate_timevae_live_metadata_completion.py',
        ),
        (
            'TADSR_TIMEVAE_PRODUCTION_ALIGNMENT_PREFLIGHT',
            'Jittor TimeVAE production alignment preflight checks metadata contracts without upgrading full alignment',
            'python3 tests_jittor_alignment/test_timevae_production_alignment_preflight.py',
        ),
        (
            'TADSR_FULL_INFERENCE_METADATA_DRY_RUN_CONTRACT',
            'full-inference stage contract is documented without running denoising loop or image generation',
            'python3 scripts/validate_full_inference_metadata_contract.py',
        ),
        (
            'TADSR_PRODUCTION_COMPLETION_PHASE2_VALIDATION',
            'Phase 2 validation aggregates official env resolution, TimeVAE metadata, LoRA audit and metadata contract',
            'python3 scripts/validate_production_completion_phase2.py',
        ),
        (
            'TADSR_PRODUCTION_COMPLETION_PHASE3_VALIDATION',
            'Phase 3 validation aggregates live official-env readiness, TimeVAE metadata export, LoRA live audit and the full-inference metadata contract',
            'python3 scripts/validate_production_completion_phase3.py',
        ),
        (
            'TADSR_TIMEVAE_LIVE_METADATA_EXPORT',
            'live TimeVAE metadata export is PASS only when official runtime metadata oracle is actually available',
            'python3 scripts/validate_production_completion_phase3.py',
        ),
        (
            'TADSR_LORA_LIVE_RUNTIME_AUDIT',
            'live LoRA runtime audit is PASS only when official runtime evidence is available; PARTIAL means existing reports were used',
            'python3 scripts/validate_production_completion_phase3.py',
        ),
        (
            'TADSR_RUNTIME_LORA_DECISION_FINALIZED_FOR_FIXED_INFERENCE',
            'fixed-adapter official inference path does not require dynamic runtime LoRA; dynamic runtime LoRA remains future work',
            'inspect docs/lora_runtime_gap_analysis.md',
        ),
        (
            'TADSR_SMOKE_TRAINING_SUBMISSION_SUMMARY',
            'small-data PyTorch-vs-Jittor smoke-training evidence is consolidated for submission',
            'python scripts/summarize_smoke_training_evidence.py',
        ),
        (
            'TADSR_FULL_INFERENCE_CONTRACT_READY_FOR_ONE_STEP',
            'one-step tensor alignment readiness is separate from full inference execution and remains blocked until Phase 3 prerequisites pass',
            'python3 scripts/validate_production_completion_phase3.py',
        ),
        (
            'TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT_PLAN',
            'controlled one-step tensor alignment plan is generated only after the live metadata and readiness gates pass',
            'inspect docs/production_completion/one_step_tensor_alignment_plan.md and experiments/production_completion/full_inference/one_step_tensor_alignment_plan.md',
        ),
        (
            'TADSR_PHASE3B_LINUX_MANUAL_RUNBOOK_READY',
            'Linux Phase 3-B manual execution runbook exists and contains safe instructions',
            'inspect docs/production_completion/linux_phase3b_manual_runbook.md',
        ),
        (
            'TADSR_PHASE3B_LINUX_LIVE_AUDIT_SCRIPT_READY',
            'Linux one-click Phase 3-B live audit script exists and remains metadata/audit-only',
            'bash -n scripts/run_phase3b_live_audit_linux.sh',
        ),
        (
            'TADSR_PHASE3B_RESULT_PACKAGER_READY',
            'Phase 3-B result packager exists and only packages JSON/Markdown/log/txt evidence',
            'python3 scripts/package_phase3b_live_results.py --repo-root . --output experiments/production_completion/phase3b_live_results_package.zip',
        ),
        (
            'TADSR_PHASE3B_RESULT_IMPORT_VALIDATOR_READY',
            'Windows/local Phase 3-B result import validator exists and rejects raw tensors',
            'python scripts/import_phase3b_live_results.py --package phase3b_live_results_package.zip --repo-root .',
        ),
        (
            'TADSR_PHASE3B_MANUAL_EXECUTION_KIT_READY',
            'manual Linux execution kit is ready while live audit remains blocked by authentication / official runtime gates',
            'manual login: ssh -p 10022 sj@10.195.21.2',
        ),
        (
            'TADSR_PHASE3B_FINAL_MANUAL_HANDOFF_READY',
            'manual handoff validation confirms Linux runbook, env template, live-audit script, packager and importer are ready',
            'python scripts/validate_phase3b_manual_handoff.py',
        ),
        (
            'TADSR_PHASE3B_LIVE_RESULTS_IMPORT_VALIDATION',
            'live result import validation is PASS only after a Linux package is safely imported',
            'python scripts/import_phase3b_live_results.py --package phase3b_live_results_package.zip --repo-root .',
        ),
        (
            'TADSR_PHASE3B_LIVE_RESULTS_NO_RAW_TENSORS',
            'live result import package contains no .npy/.npz/local_tensors/weights/checkpoints',
            'inspect experiments/production_completion/phase3_import_validation.md',
        ),
        (
            'TADSR_PHASE3D_LIVE_RESULTS_PACKAGE_FOUND',
            'Phase 3-D found the Linux live-audit result package in an allowed local candidate path',
            'inspect experiments/production_completion/phase3d_import_gate_status.md',
        ),
        (
            'TADSR_PHASE3D_LIVE_RESULTS_PACKAGE_SECURITY',
            'Phase 3-D package security passes only after safe dry-run / no-raw-tensor validation',
            'python scripts/import_phase3b_live_results.py --package phase3b_live_results_package.zip --repo-root . --dry-run 1',
        ),
        (
            'TADSR_PHASE3D_IMPORT_GATE',
            'Phase 3-D import gate passes only after live results are safely imported',
            'inspect experiments/production_completion/phase3d_import_gate_status.md',
        ),
        (
            'TADSR_PHASE3D_ONE_STEP_GATE_DECISION',
            'one-step tensor alignment gate decision is separate from full inference and remains blocked until prerequisites pass',
            'inspect experiments/production_completion/phase3_import_decision.md',
        ),
        (
            'TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT_PROTOCOL_READY',
            'controlled one-step protocol is ready only after import gate and one-step prerequisites pass',
            'inspect experiments/production_completion/full_inference/one_step_tensor_alignment_protocol.md',
        ),
        (
            'TADSR_FULL_INFERENCE_ONE_STEP_GATE_BLOCKER_SUMMARY',
            'blocker summary explains why controlled one-step tensor alignment is not yet allowed',
            'inspect experiments/production_completion/full_inference/one_step_gate_blocker_summary.md',
        ),
        (
            'TADSR_ONE_STEP_OFFICIAL_PATH_AUDIT',
            'official controlled one-step tensor path is audited without running the full denoising loop or image/video generation',
            'python3 tools/audit_official_tadsr_one_step_tensor_path.py',
        ),
        (
            'TADSR_ONE_STEP_OFFICIAL_ORACLE_TENSORS',
            'official PyTorch one-step oracle tensor metadata is exported; raw tensors remain local_tensors-only and ignored',
            'python3 tools/export_tadsr_one_step_tensor_oracle.py',
        ),
        (
            'TADSR_ONE_STEP_JITTOR_TENSOR_RUN',
            'Jittor executes the corresponding controlled one-step tensor path when official local tensors are available',
            'python3 scripts/run_jittor_one_step_tensor_alignment.py',
        ),
        (
            'TADSR_ONE_STEP_TENSOR_STAGE_ALIGNMENT',
            'per-stage shape/range/error/cosine alignment passes only for controlled one-step tensor stages',
            'python3 scripts/run_jittor_one_step_tensor_alignment.py',
        ),
        (
            'TADSR_ONE_STEP_TENSOR_ALIGNMENT_VALIDATION',
            'metadata-only validator checks one-step artifacts, safety flags, and raw tensor policy',
            'python3 scripts/validate_one_step_tensor_alignment.py',
        ),
        (
            'TADSR_ONE_STEP_NO_IMAGE_VIDEO_GUARD',
            'one-step phase did not generate restored images or videos',
            'python3 scripts/validate_one_step_tensor_alignment.py',
        ),
        (
            'TADSR_ONE_STEP_RAW_TENSOR_POLICY',
            'one-step raw .npy/.npz tensors are not tracked or staged',
            'python3 scripts/validate_one_step_tensor_alignment.py',
        ),
        (
            'TADSR_ONE_STEP_FULL_INFERENCE_GUARD_PRESERVED',
            'production full-inference CLI remains guarded by NotImplementedError',
            'python3 scripts/validate_one_step_tensor_alignment.py',
        ),
        (
            'TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT',
            'controlled one-step tensor alignment status; this is not full TADSR inference completion',
            'python3 scripts/validate_one_step_tensor_alignment.py',
        ),
        (
            'TADSR_OFFICIAL_ACTUAL_INFERENCE_PATH_AUDIT',
            'official TADSR actual inference path is audited without executing production full inference',
            'python3 tools/audit_official_tadsr_actual_inference_path.py',
        ),
        (
            'TADSR_OFFICIAL_INFERENCE_MULTISTEP_REQUIREMENT_AUDIT',
            'single-step vs multi-step requirement is determined from the official actual path',
            'python3 tools/audit_official_tadsr_actual_inference_path.py',
        ),
        (
            'TADSR_OFFICIAL_ACTUAL_PATH_SINGLE_STEP_CONTRACT',
            'official actual inference path is documented as a single-step/get_x0_from_res contract when applicable',
            'python3 tools/audit_official_tadsr_actual_inference_path.py',
        ),
        (
            'TADSR_POSTPROCESS_CONTRACT_AUDIT',
            'postprocess/output tensor-to-image contract is documented without image/video generation',
            'python3 tools/audit_official_tadsr_postprocess_contract.py',
        ),
        (
            'TADSR_OUTPUT_IMAGE_SAVE_POLICY_AUDIT',
            'official image save policy is audited but not executed',
            'python3 tools/audit_official_tadsr_postprocess_contract.py',
        ),
        (
            'TADSR_POSTPROCESS_NOT_EXECUTED_GUARD',
            'postprocess image save remains documented_not_executed in this project',
            'python3 tools/audit_official_tadsr_postprocess_contract.py',
        ),
        (
            'TADSR_TINY_MULTISTEP_ALIGNMENT_APPLICABILITY',
            'decision marker records whether tiny multi-step is required by the official actual path',
            'python3 scripts/decide_multistep_alignment_applicability.py',
        ),
        (
            'TADSR_TINY_MULTISTEP_ALIGNMENT_DECISION',
            'multi-step applicability decision is generated without running multi-step alignment',
            'python3 scripts/decide_multistep_alignment_applicability.py',
        ),
        (
            'TADSR_EXPERIMENTAL_CLI_METADATA_ONLY_PLAN',
            'future experimental metadata-only CLI plan is documented without implementation',
            'inspect docs/production_completion/experimental_cli_metadata_only_plan.md',
        ),
        (
            'TADSR_PHASE5B_SUBMISSION_FREEZE_SUMMARY',
            'Phase 5-B final submission freeze summary exists and preserves honest scope',
            'inspect docs/final_phase5b_submission_freeze_summary.md',
        ),
        (
            'TADSR_DIAGNOSTIC_IMAGE_SMOKE_PLAN',
            'future diagnostic image-smoke plan is documented without execution',
            'inspect docs/production_completion/diagnostic_image_smoke_plan.md',
        ),
        (
            'TADSR_DIAGNOSTIC_IMAGE_SMOKE_EXECUTED',
            'diagnostic image-smoke execution status is recorded honestly',
            'python3 scripts/validate_diagnostic_image_smoke.py',
        ),
        (
            'TADSR_DIAGNOSTIC_IMAGE_SMOKE_READY',
            'diagnostic image-smoke readiness is recorded as PASS or honest PARTIAL/BLOCKED',
            'python3 scripts/generate_diagnostic_image_smoke.py',
        ),
        (
            'TADSR_DIAGNOSTIC_IMAGE_SMOKE_ALIGNMENT',
            'diagnostic image-smoke alignment is PASS only when local tensors and PNG evidence exist',
            'python3 scripts/generate_diagnostic_image_smoke.py',
        ),
        (
            'TADSR_DIAGNOSTIC_IMAGE_SMOKE_NOT_FULL_INFERENCE',
            'diagnostic image-smoke explicitly does not run full inference',
            'python3 scripts/validate_diagnostic_image_smoke.py',
        ),
        (
            'TADSR_DIAGNOSTIC_IMAGE_SMOKE_NO_RAW_TENSORS',
            'diagnostic image-smoke does not stage raw .npy/.npz tensors',
            'python3 scripts/validate_diagnostic_image_smoke.py',
        ),
        (
            'TADSR_DIAGNOSTIC_IMAGE_SMOKE_ARTIFACTS_READY',
            'diagnostic image-smoke artifacts are available or honestly partial',
            'python3 scripts/validate_diagnostic_image_smoke.py',
        ),
        (
            'TADSR_DIAGNOSTIC_IMAGE_SMOKE_VALIDATION',
            'diagnostic image-smoke validator checks artifacts and guardrails',
            'python3 scripts/validate_diagnostic_image_smoke.py',
        ),
        (
            'TADSR_DIAGNOSTIC_IMAGE_SMOKE_FALSE_CLAIM_GUARD',
            'diagnostic image-smoke wording avoids false full-inference/image-generation claims',
            'python3 scripts/validate_diagnostic_image_smoke.py',
        ),
        (
            'TADSR_DIAGNOSTIC_IMAGE_SMOKE_ARTIFACT_POLICY',
            'diagnostic image-smoke artifact policy rejects staged raw tensors',
            'python3 scripts/validate_diagnostic_image_smoke.py',
        ),
        (
            'TADSR_ONE_STEP_DIAGNOSTIC_CLI_READY',
            'metadata-only one-step diagnostic CLI is ready',
            'python3 scripts/validate_one_step_diagnostic_cli.py',
        ),
        (
            'TADSR_ONE_STEP_DIAGNOSTIC_CLI_NOT_FULL_INFERENCE',
            'one-step diagnostic CLI keeps full inference guarded',
            'python3 scripts/validate_one_step_diagnostic_cli.py',
        ),
        (
            'TADSR_TIMEVAE_FULL_ALIGNMENT_CLOSURE_SUMMARY',
            'TimeVAE closure summary exists without upgrading full alignment',
            'python3 scripts/validate_timevae_closure.py',
        ),
        (
            'TADSR_TIMEVAE_FULL_ALIGNMENT_REMAINS_SCOPED',
            'TimeVAE full alignment remains scoped and NOT_COMPLETE',
            'python3 scripts/validate_timevae_closure.py',
        ),
        (
            'TADSR_RUNTIME_LORA_FINAL_DECISION_PROOF',
            'runtime LoRA final decision proof is documented',
            'python3 scripts/validate_runtime_lora_decision.py',
        ),
        (
            'TADSR_DYNAMIC_RUNTIME_LORA_NOT_REQUIRED_FOR_AUDITED_PATH',
            'dynamic runtime LoRA is not required for the audited fixed path',
            'python3 scripts/validate_runtime_lora_decision.py',
        ),
        (
            'TADSR_OFFICIAL_ACTUAL_SINGLE_STEP_PATH_DOCUMENTED',
            'official actual path is documented as single-step/get_x0_from_res',
            'python3 scripts/validate_production_completion_phase3.py',
        ),
        (
            'TADSR_MULTISTEP_NOT_REQUIRED_FOR_OFFICIAL_ACTUAL_PATH',
            'tiny multi-step is documented as not required for the official actual path',
            'python3 scripts/validate_production_completion_phase3.py',
        ),
        (
            'TADSR_CONTROLLED_ONE_STEP_EVIDENCE_DOCUMENTED',
            'controlled one-step tensor evidence and key metrics are documented',
            'python3 scripts/validate_production_completion_phase3.py',
        ),
        (
            'TADSR_FINAL_SUBMISSION_AFTER_PHASE5B_READY',
            'final submission materials are ready after Phase 5-B without opening full inference',
            'python3 scripts/validate_production_completion_phase3.py',
        ),
        (
            'TADSR_FINAL_SUBMISSION_WITH_DIAGNOSTIC_SMOKE_READY',
            'final submission materials include Phase 6-A diagnostic smoke/CLI/closure proof status',
            'python3 scripts/validate_production_completion_phase3.py',
        ),
    ]
    for marker, pass_note, action in production_completion_rows:
        marker_status = production_completion_marker_status(marker)
        accepted_statuses = {'PASS', 'PARTIAL', 'PASS_WITH_BLOCKERS', 'BLOCKED', 'NOT_REQUIRED_FOR_OFFICIAL_ACTUAL_INFERENCE', 'REQUIRED_NEXT', 'PARTIAL_UNKNOWN', 'NOT_EXECUTED'}
        rows.append(row(
            marker,
            marker_status,
            pass_note if marker_status in accepted_statuses else 'production completion evidence missing or failed',
            action,
        ))

    perfection_sources = [
        ('experiments/course_requirement_compliance_matrix.json', [
            ('TADSR_COURSE_REQUIREMENT_COMPLIANCE', 'course requirement matrix covers the submitted evidence', 'python3 scripts/validate_course_requirement_compliance.py'),
            ('TADSR_COURSE_REQUIREMENT_EVIDENCE_MATRIX', 'course requirement evidence matrix is generated', 'python3 scripts/validate_course_requirement_compliance.py'),
            ('TADSR_COURSE_TRAINING_REQUIREMENT_EVIDENCE', 'training logs/loss curves/performance evidence are indexed for course requirements', 'python3 scripts/validate_course_requirement_compliance.py'),
            ('TADSR_COURSE_VISUALIZATION_REQUIREMENT_EVIDENCE', 'visualization evidence is indexed for course requirements', 'python3 scripts/validate_course_requirement_compliance.py'),
            ('TADSR_COURSE_GITHUB_REQUIREMENT_EVIDENCE', 'GitHub submission readiness evidence is indexed for course requirements', 'python3 scripts/validate_course_requirement_compliance.py'),
            ('TADSR_COURSE_PPT_VIDEO_REQUIREMENT_EVIDENCE', 'PPT/PDF/video guide evidence is indexed for course requirements', 'python3 scripts/validate_course_requirement_compliance.py'),
        ]),
        ('experiments/final_evidence_index.json', [
            ('TADSR_FINAL_EVIDENCE_INDEX', 'teacher-readable evidence index is generated', 'python3 scripts/build_final_evidence_index.py'),
            ('TADSR_FINAL_EVIDENCE_INDEX_TEACHER_READABLE', 'final evidence index is readable by course reviewers', 'python3 scripts/build_final_evidence_index.py'),
        ]),
        ('experiments/smoke_training/output_tail/training_alignment_evidence_validation.json', [
            ('TADSR_TRAINING_ALIGNMENT_EVIDENCE_VALIDATION', 'small-data training evidence completeness is audited', 'python3 scripts/validate_training_alignment_evidence.py'),
            ('TADSR_TRAINING_LOSS_CURVE_EVIDENCE', 'training and validation loss curves are present', 'python3 scripts/validate_training_alignment_evidence.py'),
            ('TADSR_TRAINING_PERFORMANCE_LOG_EVIDENCE', 'training performance logs are present', 'python3 scripts/validate_training_alignment_evidence.py'),
            ('TADSR_TRAINING_OUTPUT_ALIGNMENT_EVIDENCE', 'training prediction/output alignment visualizations are present', 'python3 scripts/validate_training_alignment_evidence.py'),
            ('TADSR_TRAINING_GRAD_PARAM_UPDATE_EVIDENCE', 'parameter-update evidence exists; gradient norm may be optional if not logged originally', 'python3 scripts/validate_training_alignment_evidence.py'),
            ('TADSR_TRAINING_EVIDENCE_TEACHER_READY', 'training evidence is ready for teacher review without claiming full TADSR training', 'python3 scripts/validate_training_alignment_evidence.py'),
        ]),
        ('experiments/final_claims_consistency_validation.json', [
            ('TADSR_FINAL_CLAIMS_CONSISTENCY', 'final materials preserve scope consistency', 'python3 scripts/validate_final_claims_consistency.py'),
            ('TADSR_FINAL_FALSE_CLAIM_GUARD', 'final materials avoid misleading full-inference/image-generation/runtime-LoRA claims', 'python3 scripts/validate_final_claims_consistency.py'),
            ('TADSR_FINAL_SCOPE_CONSISTENCY', 'required NOT_COMPLETE / NOT_IMPLEMENTED statuses remain visible', 'python3 scripts/validate_final_claims_consistency.py'),
        ]),
        ('experiments/defense_risk_response_pack_validation.json', [
            ('TADSR_DEFENSE_RISK_RESPONSE_PACK', 'Chinese defense risk-response pack covers high-risk reviewer questions', 'python3 scripts/validate_defense_risk_response_pack.py'),
            ('TADSR_DEFENSE_SHORT_ANSWERS_READY', 'short Chinese defense answers are ready for live Q&A', 'python3 scripts/validate_defense_risk_response_pack.py'),
            ('TADSR_DEFENSE_LONG_ANSWERS_READY', 'long Chinese defense answers are ready for detailed reviewer follow-up', 'python3 scripts/validate_defense_risk_response_pack.py'),
            ('TADSR_DEFENSE_FALSE_CLAIM_GUARD', 'defense materials avoid unguarded full-inference/image-generation/runtime-LoRA overclaims', 'python3 scripts/validate_defense_risk_response_pack.py'),
            ('TADSR_DEFENSE_EVIDENCE_LOOKUP_READY', 'defense evidence lookup table maps claims to files and demo commands', 'python3 scripts/validate_defense_risk_response_pack.py'),
        ]),
        ('experiments/final_release_candidate_signoff.json', [
            ('TADSR_FINAL_RELEASE_CANDIDATE_SIGNOFF', 'final release candidate signoff report is present', 'docs/final_release_candidate_signoff.md'),
            ('TADSR_FINAL_RELEASE_CANDIDATE_TECHNICAL_EVIDENCE', 'final signoff summarizes core technical evidence', 'docs/final_release_candidate_signoff.md'),
            ('TADSR_FINAL_RELEASE_CANDIDATE_SCOPE_GUARD', 'final signoff preserves NOT_COMPLETE / NOT_IMPLEMENTED scope guards', 'docs/final_release_candidate_signoff.md'),
        ]),
        ('experiments/final_links_and_paths_validation.json', [
            ('TADSR_FINAL_LINKS_AND_PATHS_VALIDATION', 'final critical links and paths are validated', 'python3 scripts/validate_final_links_and_paths.py'),
            ('TADSR_FINAL_MARKDOWN_LINKS_VALIDATION', 'final Markdown links are scanned; noncritical warnings are recorded', 'python3 scripts/validate_final_links_and_paths.py'),
            ('TADSR_FINAL_PLACEHOLDER_SCAN', 'final materials are scanned for obvious placeholders', 'python3 scripts/validate_final_links_and_paths.py'),
        ]),
        ('experiments/final_chinese_materials_validation.json', [
            ('TADSR_FINAL_CHINESE_MATERIALS_VALIDATION', 'final reviewer-facing materials are primarily Chinese', 'python3 scripts/validate_final_chinese_materials.py'),
            ('TADSR_FINAL_MOJIBAKE_SCAN', 'final Chinese materials are scanned for obvious mojibake', 'python3 scripts/validate_final_chinese_materials.py'),
            ('TADSR_FINAL_CHINESE_READABILITY_READY', 'final Chinese materials are ready for human review', 'python3 scripts/validate_final_chinese_materials.py'),
        ]),
        ('experiments/final_command_index.json', [
            ('TADSR_FINAL_COMMAND_INDEX', 'final demo command index is present', 'docs/final_command_index.md'),
            ('TADSR_FINAL_DEMO_COMMANDS_READY', 'teacher-facing demo commands are grouped and explained', 'docs/final_command_index.md'),
        ]),
        ('experiments/final_github_upload_checklist.json', [
            ('TADSR_FINAL_GITHUB_UPLOAD_CHECKLIST', 'GitHub / LMS upload checklist is present', 'docs/final_github_upload_checklist.md'),
            ('TADSR_FINAL_UPLOAD_PACKAGE_READY', 'upload package safety checklist is ready', 'docs/final_github_upload_checklist.md'),
        ]),
        ('experiments/clean_public_release_package_manifest.json', [
            ('TADSR_CLEAN_PUBLIC_RELEASE_PACKAGE_BUILD', 'clean public release tree is built for manual GitHub upload', 'python3 scripts/build_clean_public_release_package.py'),
            ('TADSR_CLEAN_PUBLIC_RELEASE_NO_RAW_TENSORS', 'clean public release tree excludes raw tensors and large private artifacts', 'python3 scripts/build_clean_public_release_package.py'),
            ('TADSR_CLEAN_PUBLIC_RELEASE_SIZE_AUDIT', 'clean public release tree is below the course upload size budget', 'python3 scripts/build_clean_public_release_package.py'),
        ]),
        ('experiments/clean_public_release_package_validation.json', [
            ('TADSR_CLEAN_PUBLIC_RELEASE_PACKAGE_VALIDATION', 'clean public release tree validates as GitHub-ready', 'python3 scripts/validate_clean_public_release_package.py'),
            ('TADSR_CLEAN_PUBLIC_RELEASE_GITHUB_READY', 'clean public release tree can be manually uploaded to GitHub', 'python3 scripts/validate_clean_public_release_package.py'),
            ('TADSR_CLEAN_PUBLIC_RELEASE_FALSE_CLAIM_GUARD', 'clean public release tree avoids misleading completion claims', 'python3 scripts/validate_clean_public_release_package.py'),
        ]),
        ('experiments/final_github_url_status.json', [
            ('TADSR_FINAL_GITHUB_URL_STATUS', 'GitHub URL status is recorded; pending is acceptable before manual repo creation', 'python3 scripts/set_final_github_url.py'),
            ('TADSR_FINAL_GITHUB_URL_UPDATE_SCRIPT_READY', 'GitHub URL update script is ready and does not push or add remotes', 'python3 scripts/set_final_github_url.py'),
        ]),
        ('experiments/final_human_submission_instructions.json', [
            ('TADSR_FINAL_HUMAN_SUBMISSION_INSTRUCTIONS', 'final human submission instructions are present', 'docs/final_human_submission_instructions.md'),
        ]),
        ('experiments/final_video_rehearsal_checklist.json', [
            ('TADSR_FINAL_VIDEO_REHEARSAL_CHECKLIST', 'final video rehearsal checklist is present', 'docs/final_video_rehearsal_checklist.md'),
            ('TADSR_FINAL_VIDEO_RECORDING_READY', 'video recording flow is ready for human execution', 'docs/final_video_rehearsal_checklist.md'),
        ]),
        ('experiments/final_submission_freeze_tag.json', [
            ('TADSR_FINAL_SUBMISSION_FREEZE_TAG_DOC', 'final freeze/tag documentation is present without auto-tagging or pushing', 'docs/final_submission_freeze_tag.md'),
        ]),
        ('experiments/final_human_submission_lock_report.json', [
            ('TADSR_FINAL_HUMAN_SUBMISSION_LOCK_REPORT', 'final human submission lock report is present', 'docs/final_human_submission_lock_report.md'),
            ('TADSR_FINAL_HUMAN_SUBMISSION_LOCK_READY', 'final human submission lock is ready; only manual submission actions remain', 'docs/final_human_submission_lock_report.md'),
        ]),
        ('experiments/final_video_submission_check.json', [
            ('TADSR_FINAL_VIDEO_SUBMISSION_CHECK', 'final video submission check is present', 'docs/final_video_submission_check.md'),
            ('TADSR_FINAL_VIDEO_RECORDING_READY', 'video recording flow is ready for human execution', 'docs/final_video_submission_check.md'),
        ]),
        ('experiments/full_engineering_completion_roadmap.json', [
            ('TADSR_FULL_ENGINEERING_COMPLETION_ROADMAP', 'future full-engineering completion roadmap is documented without execution', 'docs/full_engineering_completion_roadmap.md'),
            ('TADSR_CONTROLLED_EXTENSION_PLAN_READY', 'controlled extension plan is ready if future work is approved', 'docs/full_engineering_completion_roadmap.md'),
        ]),
    ]

    def perfection_marker(path: str, marker: str) -> str:
        data = load_json(path)
        markers = data.get('markers', {}) if isinstance(data, dict) else {}
        return str(markers.get(marker, 'BLOCKED'))

    perfection_status = {}
    for src, marker_rows in perfection_sources:
        for marker, pass_note, action in marker_rows:
            marker_status = perfection_marker(src, marker)
            perfection_status[marker] = marker_status
            accepted = {'PASS', 'PARTIAL', 'MISSING_OPTIONAL', 'PARTIAL_GITHUB_URL_PENDING'}
            rows.append(row(
                marker,
                marker_status,
                pass_note if marker_status in accepted else 'Final Perfection evidence missing or failed',
                action,
            ))

    teacher_review_ready = Path('docs/teacher_review_guide.md').exists()
    defense_qa_ready = Path('docs/final_defense_QA.md').exists()
    rows.append(row(
        'TADSR_TEACHER_REVIEW_GUIDE_READY',
        'PASS' if teacher_review_ready else 'BLOCKED',
        'teacher review guide exists and indexes fast review commands/files' if teacher_review_ready else 'create docs/teacher_review_guide.md',
        'update docs/teacher_review_guide.md',
    ))
    rows.append(row(
        'TADSR_FINAL_DEFENSE_QA_READY',
        'PASS' if defense_qa_ready else 'BLOCKED',
        'final defense Q&A exists for high-risk reviewer questions' if defense_qa_ready else 'create docs/final_defense_QA.md',
        'update docs/final_defense_QA.md',
    ))
    final_perfection_ready = (
        perfection_status.get('TADSR_COURSE_REQUIREMENT_COMPLIANCE') == 'PASS'
        and perfection_status.get('TADSR_FINAL_EVIDENCE_INDEX') == 'PASS'
        and perfection_status.get('TADSR_TRAINING_EVIDENCE_TEACHER_READY') == 'PASS'
        and perfection_status.get('TADSR_FINAL_CLAIMS_CONSISTENCY') == 'PASS'
        and perfection_status.get('TADSR_DEFENSE_RISK_RESPONSE_PACK') == 'PASS'
        and perfection_status.get('TADSR_DEFENSE_FALSE_CLAIM_GUARD') == 'PASS'
        and perfection_status.get('TADSR_FULL_ENGINEERING_COMPLETION_ROADMAP') == 'PASS'
        and teacher_review_ready
        and defense_qa_ready
        and production_completion_marker_status('TADSR_DIAGNOSTIC_IMAGE_SMOKE_ALIGNMENT') == 'PASS'
        and production_completion_marker_status('TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT') == 'PASS'
    )
    rows.append(row(
        'TADSR_FINAL_PERFECTION_READINESS',
        'PASS' if final_perfection_ready else 'PARTIAL',
        'course compliance, evidence index, training audit, claims guard, defense guide and roadmap are ready while guarded limitations remain honest' if final_perfection_ready else 'complete Final Perfection evidence/docs before final push',
        'run Final Perfection validators and keep full inference guarded',
    ))
    final_human_submission_ready = (
        final_perfection_ready
        and perfection_status.get('TADSR_FINAL_RELEASE_CANDIDATE_SIGNOFF') == 'PASS'
        and perfection_status.get('TADSR_FINAL_RELEASE_CANDIDATE_SCOPE_GUARD') == 'PASS'
        and perfection_status.get('TADSR_FINAL_LINKS_AND_PATHS_VALIDATION') == 'PASS'
        and perfection_status.get('TADSR_FINAL_CHINESE_MATERIALS_VALIDATION') == 'PASS'
        and perfection_status.get('TADSR_FINAL_COMMAND_INDEX') == 'PASS'
        and perfection_status.get('TADSR_FINAL_GITHUB_UPLOAD_CHECKLIST') == 'PASS'
    )
    rows.append(row(
        'TADSR_FINAL_READY_FOR_HUMAN_SUBMISSION',
        'PASS' if final_human_submission_ready else 'PARTIAL',
        'release candidate signoff, links/path QA, Chinese readability QA, command index and GitHub upload checklist are ready' if final_human_submission_ready else 'complete final release-candidate QA before human submission',
        'run final release-candidate QA validators and keep full inference guarded',
    ))
    github_url_status = perfection_status.get('TADSR_FINAL_GITHUB_URL_STATUS')
    github_url_ok = github_url_status in {'PASS', 'PARTIAL_GITHUB_URL_PENDING'}
    final_public_release_ready = (
        final_human_submission_ready
        and perfection_status.get('TADSR_CLEAN_PUBLIC_RELEASE_PACKAGE_BUILD') == 'PASS'
        and perfection_status.get('TADSR_CLEAN_PUBLIC_RELEASE_PACKAGE_VALIDATION') == 'PASS'
        and perfection_status.get('TADSR_CLEAN_PUBLIC_RELEASE_GITHUB_READY') == 'PASS'
        and perfection_status.get('TADSR_CLEAN_PUBLIC_RELEASE_FALSE_CLAIM_GUARD') == 'PASS'
        and perfection_status.get('TADSR_FINAL_GITHUB_URL_UPDATE_SCRIPT_READY') == 'PASS'
        and github_url_ok
        and perfection_status.get('TADSR_FINAL_HUMAN_SUBMISSION_INSTRUCTIONS') == 'PASS'
        and perfection_status.get('TADSR_FINAL_VIDEO_REHEARSAL_CHECKLIST') == 'PASS'
        and perfection_status.get('TADSR_FINAL_VIDEO_RECORDING_READY') == 'PASS'
        and perfection_status.get('TADSR_FINAL_SUBMISSION_FREEZE_TAG_DOC') == 'PASS'
        and perfection_status.get('TADSR_FINAL_HUMAN_SUBMISSION_LOCK_REPORT') == 'PASS'
        and perfection_status.get('TADSR_FINAL_HUMAN_SUBMISSION_LOCK_READY') == 'PASS'
        and perfection_status.get('TADSR_FINAL_VIDEO_SUBMISSION_CHECK') == 'PASS'
    )
    rows.append(row(
        'TADSR_FINAL_PUBLIC_RELEASE_AND_SUBMISSION_READY',
        'PASS' if final_public_release_ready else 'PARTIAL',
        'clean public release tree, URL setter, human submission instructions, video rehearsal and freeze docs are ready' if final_public_release_ready else 'complete clean public release and human submission lockdown checks',
        'create GitHub repo manually, run set_final_github_url.py with the real URL, record video, then submit',
    ))
    manual_actions_only_remaining = (
        final_public_release_ready
        and status_from_json('experiments/final_submission_bundle_validation.json') == 'PASS'
        and status_from_json('experiments/final_claims_consistency_validation.json') == 'PASS'
        and production_completion_marker_status('TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT') == 'PASS'
        and production_completion_marker_status('TADSR_DIAGNOSTIC_IMAGE_SMOKE_ALIGNMENT') == 'PASS'
    )
    rows.append(row(
        'TADSR_FINAL_MANUAL_ACTIONS_ONLY_REMAINING',
        'PASS' if manual_actions_only_remaining else 'PARTIAL',
        'all technical and material checks are complete; remaining work is GitHub creation, URL writeback, video recording and course submission only' if manual_actions_only_remaining else 'finish final lock, bundle or claims checks before declaring manual-only remainder',
        'create GitHub repo manually, upload clean public release, write back URL, record video and submit',
    ))

    status = {n: s for n, s, _, _ in rows}
    github_release_readiness = status.get('TADSR_GITHUB_RELEASE_READINESS_AUDIT')
    remote_diag = load_json('experiments/production_completion/env/remote_connectivity_diagnostic.json')
    remote_auth_blocked = remote_diag.get('status') == 'AUTH_BLOCKED' or remote_diag.get('ssh_authentication_available') is False
    if status.get('TADSR_FINAL_MANUAL_ACTIONS_ONLY_REMAINING') == 'PASS':
        nxt = 'Manual actions only: create the GitHub repo, upload the clean public release, run set_final_github_url.py with the real URL, record the video, and submit PPT/PDF/video/GitHub URL. Do not expand technical scope further.'
    elif status.get('TADSR_FINAL_PUBLIC_RELEASE_AND_SUBMISSION_READY') == 'PASS':
        nxt = 'Public release and human submission package are ready. Stop technical expansion, manually create the GitHub repo, run set_final_github_url.py with the real URL, record the video, and submit the course materials.'
    elif (
        status.get('TADSR_DIAGNOSTIC_IMAGE_SMOKE_EXECUTED') == 'PASS'
        and status.get('TADSR_DIAGNOSTIC_IMAGE_SMOKE_ALIGNMENT') == 'PASS'
    ):
        nxt = 'Diagnostic one-step image-smoke evidence is available and verified. The final submission may proceed; keep full inference guarded, treat PNGs as diagnostic tensor visualizations only, and do not claim production restored image/video generation.'
    elif status.get('TADSR_TINY_MULTISTEP_ALIGNMENT_APPLICABILITY') == 'NOT_REQUIRED_FOR_OFFICIAL_ACTUAL_INFERENCE':
        nxt = 'Official actual inference path is single-step/get_x0_from_res, so tiny multi-step is not required for official actual inference. Next optional phase is a diagnostic image-smoke plan only after explicit approval; keep full inference guarded and do not generate image/video in audit-only phases.'
    elif status.get('TADSR_TINY_MULTISTEP_ALIGNMENT_APPLICABILITY') == 'REQUIRED_NEXT':
        nxt = 'Official actual inference requires multi-step. Next phase should be tiny two-step latent-only alignment only; still no image/video and no production full inference.'
    elif status.get('TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT') == 'PASS':
        nxt = 'Controlled one-step tensor alignment passed. Next step may be tiny multi-step latent-only alignment; still do not generate image/video, do not open production full inference, and keep JITTOR_FULL_INFERENCE NOT_COMPLETE.'
    elif status.get('TADSR_PRODUCTION_COMPLETION_PHASE3_VALIDATION') == 'PASS' and status.get('TADSR_FULL_INFERENCE_CONTRACT_READY_FOR_ONE_STEP') == 'PASS' and status.get('TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT_PLAN') == 'PASS':
        nxt = 'Production completion Phase 3 passed and one-step tensor alignment is ready. Next step is controlled one-step tensor alignment only; still do not generate image/video, do not open production full inference, and do not upgrade JITTOR_FULL_INFERENCE until the controlled tensor evidence passes.'
    elif status.get('TADSR_PHASE3B_MANUAL_EXECUTION_KIT_READY') == 'PASS' and remote_auth_blocked:
        nxt = 'Phase 3-B manual execution kit is ready. Because 10.195.21.2:10022 is reachable but SSH authentication is unavailable in this automation context, the user should manually login with ssh -p 10022 sj@10.195.21.2, run scripts/run_phase3b_live_audit_linux.sh on Linux, package results with scripts/package_phase3b_live_results.py, then import them with scripts/import_phase3b_live_results.py. Keep JITTOR_FULL_INFERENCE NOT_COMPLETE, TIME_VAE_FULL_ALIGNMENT NOT_COMPLETE, and dynamic runtime LoRA NOT_IMPLEMENTED_BY_DESIGN.'
    elif status.get('TADSR_PRODUCTION_COMPLETION_PHASE3_VALIDATION') == 'PASS_WITH_BLOCKERS':
        nxt = 'Production completion Phase 3 is documented with blockers. Resolve official repo/weights/strict Python or TimeVAE oracle blockers before one-step tensor alignment; keep JITTOR_FULL_INFERENCE NOT_COMPLETE, TIME_VAE_FULL_ALIGNMENT NOT_COMPLETE, and dynamic runtime LoRA NOT_IMPLEMENTED_BY_DESIGN.'
    elif status.get('TADSR_PRODUCTION_COMPLETION_PHASE3_VALIDATION') == 'FAIL':
        nxt = 'Production completion Phase 3 failed a safety or consistency check. Fix the Phase 3 audit before any further production-completion work.'
    elif status.get('TADSR_PRODUCTION_COMPLETION_PHASE2_VALIDATION') == 'PASS':
        nxt = 'Production completion Phase 2 passed with live metadata. Next step is controlled Jittor TimeVAE production tensor alignment; do not upgrade full inference or TimeVAE full alignment until tensor stages pass.'
    elif status.get('TADSR_PRODUCTION_COMPLETION_PHASE2_VALIDATION') == 'PASS_WITH_BLOCKERS':
        nxt = 'Production completion Phase 2 is documented with blockers. Resolve official repo/weights/strict Python before trying one-step full inference; keep JITTOR_FULL_INFERENCE NOT_COMPLETE and TIME_VAE_FULL_ALIGNMENT NOT_COMPLETE.'
    elif status.get('TADSR_PRODUCTION_COMPLETION_PHASE1_VALIDATION') == 'PASS':
        nxt = 'Production completion Phase 1 is ready. If official runtime environment is available, next export TimeVAE production oracle metadata and run controlled one-step validation; keep full inference guard until all stages pass.'
    elif status.get('TADSR_PRODUCTION_COMPLETION_PHASE1_VALIDATION') == 'PASS_WITH_BLOCKERS':
        nxt = 'Production completion Phase 1 is documented with blockers. Resolve official repo/weights/env before implementing full inference; keep JITTOR_FULL_INFERENCE NOT_COMPLETE and dynamic runtime LoRA NOT_IMPLEMENTED_BY_DESIGN.'
    elif status.get('TADSR_SMALL_DATA_TRAINING_READINESS') in {'FAIL', 'PARTIAL', 'PASS_WITH_PARTIAL_ALIGNMENT'}:
        nxt = 'Tune lr/steps if needed or clearly document partial smoke-training alignment; keep full inference CLI NotImplemented.'
    elif status.get('TADSR_RELEASE_CANDIDATE_QA_READINESS') == 'PASS' and github_release_readiness == 'PASS':
        nxt = 'User may manually push/upload repository and record final video. If a lean public repo is desired, run a separate cleanup pass.'
    elif github_release_readiness == 'PARTIAL':
        nxt = 'Decide whether to accept the larger repository or run repository slimming cleanup in a separate authorized pass.'
    elif github_release_readiness == 'FAIL':
        nxt = 'Do not push to GitHub until large history blobs are cleaned in a separate authorized cleanup stage.'
    elif status.get('TADSR_RELEASE_CANDIDATE_QA_READINESS') == 'PASS':
        nxt = 'Human can record final video, upload/push repository to GitHub, and submit deliverables. Future technical work should remain behind explicit controlled production CLI validation.'
    elif status.get('TADSR_FINAL_SUBMISSION_READY') == 'PASS':
        nxt = 'Final deliverables are ready for submission; record the video and submit the repository package. Future technical work should be controlled production CLI validation, not enabled by default.'
    elif status.get('TADSR_FINAL_HANDOFF_READINESS') == 'PASS':
        nxt = 'Repository is ready for final presentation/video recording and submission; run final deliverables validation before release if not already complete.'
    elif status.get('TADSR_FINAL_HANDOFF_READINESS') == 'PARTIAL':
        nxt = 'Fix missing presentation/demo/handoff docs; keep full inference CLI NotImplemented.'
    elif status.get('TADSR_FINAL_PACKAGING_READINESS') == 'PASS':
        nxt = 'Prepare final presentation / video / repository handoff, or optionally plan controlled production CLI validation as a future extension. Keep full inference CLI NotImplemented unless explicitly approved.'
    elif status.get('TADSR_FINAL_PACKAGING_READINESS') == 'PARTIAL':
        nxt = 'Fix missing evidence/docs before any new technical work; keep full inference CLI NotImplemented.'
    elif status.get('JITTOR_PREPROCESS_ALIGNMENT') == 'PASS' and status.get('JITTOR_SCHEDULER_ALIGNMENT') == 'PASS':
        if status.get('TADSR_UNET_FULL_FORWARD_ALIGNMENT') == 'PASS':
            if status.get('TIME_VAE_FULL_BOUNDARY_ALIGNMENT') == 'PASS' and status.get('TIME_VAE_FULL_ALIGNMENT') != 'PASS':
                if status.get('TIME_VAE_ACTUAL_ENCODER_TILED_ALIGNMENT') == 'PARTIAL':
                    nxt = 'Continue with exact Jittor encoder VAEHook.vae_tile_forward task-queue alignment: implement official split_tiles/crop/write behavior and cross-tile GroupNorm statistics. Keep decoder as official original_forward, keep TIME_VAE_TILED_ORACLE_TENSORS blocked by decoder contract, and keep full TADSR inference NotImplemented.'
                elif status.get('TIME_VAE_ACTUAL_VAEHOOK_BEHAVIOR_ALIGNMENT') == 'PASS':
                    if status.get('TADSR_STATIC_EFFECTIVE_LORA_COVERAGE_AUDIT') == 'PASS':
                        if status.get('TADSR_SCHEDULER_BOUNDARY_ALIGNMENT') == 'PASS':
                            if status.get('TADSR_MINIMAL_ONE_STEP_DECODE_DRY_RUN') == 'PASS':
                                nxt = 'Minimal one-step TADSR decode tensor dry-run is aligned; next continue with final packaging/reporting or a controlled production-CLI design audit, while keeping full inference CLI NotImplemented until explicit production pipeline validation is approved.'
                            elif status.get('TADSR_MINIMAL_LATENT_INTEGRATION_DRY_RUN') == 'PASS':
                                nxt = 'Minimal latent-only TADSR integration dry-run is aligned; next align the tensor-only decode/clamp boundary or plan a controlled multi-step latent-only audit, while keeping image/video output path and full inference CLI NotImplemented.'
                            else:
                                nxt = 'Scheduler boundary / one-step contract is audited and aligned; next plan a minimal latent-only TADSR integration dry-run combining UNet full forward + scheduler one-step + VAE boundary, while keeping full inference CLI NotImplemented.'
                        else:
                            nxt = 'Static effective LoRA policy is audited and covered; next audit scheduler boundary / minimal denoising-step contract while keeping full TADSR inference NotImplemented.'
                    elif status.get('TADSR_STATIC_EFFECTIVE_LORA_POLICY_AUDIT') == 'PASS':
                        nxt = 'Static effective LoRA policy is audited, but active coverage is not fully complete; next export/verify remaining TimeVAE LoRA effective weights before scheduler/full pipeline integration. Keep full TADSR inference NotImplemented.'
                    else:
                        nxt = 'Actual official VAEHook behavior is aligned; next audit scheduler integration and runtime LoRA policy without opening full TADSR inference until every boundary is verified.'
                elif status.get('TIME_VAE_ACTUAL_VAEHOOK_ORACLE_TENSORS') == 'PASS':
                    nxt = 'Continue with Jittor official-actual VAEHook behavior wrapper/tester alignment: mirror official behavior, use the exported actual oracle, keep decoder as original_forward because official decoder hook is not tiled, and keep full TADSR inference NotImplemented.'
                elif str(status.get('TIME_VAE_ACTUAL_ENCODER_TILED_ORACLE_FEASIBILITY', '')).startswith('BLOCKED'):
                    nxt = 'Resolve encoder tiled trigger feasibility or document that official actual behavior oracle only covers decoder-original-forward and non-tiled encode; keep full TADSR inference NotImplemented.'
                elif str(status.get('TIME_VAE_TILED_ORACLE_TENSORS', '')).startswith('BLOCKED'):
                    nxt = 'After the VAEHook tiled audit, resolve the official decoder-hook blocker before implementing Jittor tiled VAE: official TADSR_test installs decoder VAEHook with time_vae=False, so a truthful tiled decode oracle was not triggered. Keep full TADSR inference NotImplemented.'
                elif status.get('TIME_VAE_TILED_ORACLE_TENSORS') == 'PASS':
                    nxt = 'Continue with Jittor VAEHook tiled encode/decode wrapper/tester alignment using the exported tiled oracle; keep full TADSR inference NotImplemented until tiled VAE, scheduler integration, and runtime LoRA policy are verified.'
                else:
                    nxt = 'After UNet full forward and TimeVAE non-tiled full-boundary alignment, continue with VAEHook tiled encode/decode audit and runtime VAE/UNet LoRA policy; keep full TADSR inference NotImplemented until TimeVAE full pipeline sufficiency, scheduler integration, and runtime LoRA policy are all verified.'
            else:
                nxt = 'After official TADSR UNet full forward alignment, continue with runtime LoRA integration audit or TADSR pipeline integration planning only after confirming VAE/full pipeline boundaries; keep full TADSR inference NotImplemented until UNet full forward, TimeVAE full alignment, scheduler integration, and runtime LoRA policy are all verified.'
        elif status.get('TADSR_UNET_MANUAL_FULL_WRAPPER_ALIGNMENT') == 'PASS':
            nxt = 'Continue with official TADSR UNet full forward alignment after manual wrapper alignment; keep full TADSR inference and generic runtime LoRA NotImplemented until official full forward is verified.'
        elif status.get('TADSR_UNET_ENTRY_TO_OUTPUT_TAIL_ALIGNMENT') == 'PASS':
            nxt = 'Continue with a manual full-UNet-forward wrapper audit/export/alignment after entry -> all down_blocks -> mid_block -> all up_blocks -> output tail alignment; keep official full UNet forward, runtime LoRA integration, and full inference NotImplemented until that wrapper is verified.'
        elif status.get('TADSR_UNET_UPBLOCK3_ALIGNMENT') == 'PASS':
            nxt = 'Continue TADSR UNet output tail audit/export/port after full local up_blocks.3 aggregate alignment; keep full UNet forward, runtime LoRA integration, and full inference NotImplemented until the output tail is verified.'
        elif status.get('TADSR_UNET_UPBLOCK3_ATTENTION2_BRIDGE_ALIGNMENT') == 'PASS':
            nxt = 'Continue with full local TADSR UNet up_blocks.3 aggregate / output-tail-boundary verification after up_blocks.3.attentions.2 alignment; keep full UNet forward, runtime LoRA integration, and full inference NotImplemented until the aggregate and output tail are verified.'
        elif status.get('TADSR_UNET_UPBLOCK3_RESNET2_BRIDGE_ALIGNMENT') == 'PASS':
            nxt = f'Continue TADSR UNet {up3_after_resnet2_next} audit/export/port after up_blocks.3.resnets.2 alignment; keep full up_blocks.3 aggregate, full UNet forward, runtime LoRA integration, and full inference NotImplemented until all up_blocks and LoRA runtime integration pass.'
        elif status.get('TADSR_UNET_UPBLOCK3_ATTENTION1_BRIDGE_ALIGNMENT') == 'PASS':
            nxt = f'Continue TADSR UNet {up3_after_attention1_next} audit/export/port after up_blocks.3.attentions.1 alignment; keep full up_blocks.3 aggregate, full UNet forward, runtime LoRA integration, and full inference NotImplemented until all up_blocks and LoRA runtime integration pass.'
        elif status.get('TADSR_UNET_UPBLOCK3_RESNET1_BRIDGE_ALIGNMENT') == 'PASS':
            nxt = f'Continue TADSR UNet {up3_after_resnet1_next} audit/export/port after up_blocks.3.resnets.1 alignment; keep full up_blocks.3 aggregate, full UNet forward, runtime LoRA integration, and full inference NotImplemented until all up_blocks and LoRA runtime integration pass.'
        elif status.get('TADSR_UNET_UPBLOCK3_ATTENTION0_BRIDGE_ALIGNMENT') == 'PASS':
            nxt = f'Continue TADSR UNet {up3_after_attention0_next} audit/export/port after up_blocks.3.attentions.0 alignment; keep full up_blocks.3 aggregate, full UNet forward, runtime LoRA integration, and full inference NotImplemented until all up_blocks and LoRA runtime integration pass.'
        elif status.get('TADSR_UNET_UPBLOCK3_RESNET0_BRIDGE_ALIGNMENT') == 'PASS':
            if up3_next == 'up_blocks.3.attentions.0':
                nxt = 'Continue TADSR UNet up_blocks.3.attentions.0 audit/export/port after up_blocks.3.resnets.0 alignment; keep full up_blocks aggregate, full UNet forward, runtime LoRA integration, and full inference NotImplemented until all up_blocks and LoRA runtime integration pass.'
            elif up3_next == 'up_blocks.3.resnets.1':
                nxt = 'Continue TADSR UNet up_blocks.3.resnets.1 audit/export/port after up_blocks.3.resnets.0 alignment; keep full up_blocks aggregate, full UNet forward, runtime LoRA integration, and full inference NotImplemented until all up_blocks and LoRA runtime integration pass.'
            else:
                nxt = f'Continue TADSR UNet {up3_next} audit/export/port after up_blocks.3.resnets.0 alignment; keep full up_blocks aggregate, full UNet forward, runtime LoRA integration, and full inference NotImplemented until all up_blocks and LoRA runtime integration pass.'
        elif status.get('TADSR_UNET_UPBLOCK2_ALIGNMENT') == 'PASS':
            nxt = 'Continue TADSR UNet up_blocks.3.resnets.0 audit/export/port after full local up_blocks.2 alignment; keep full up_blocks aggregate, full UNet forward, runtime LoRA integration, and full inference NotImplemented until all up_blocks and LoRA runtime integration pass.'
        elif status.get('TADSR_UNET_UPBLOCK2_UPSAMPLER_BRIDGE_ALIGNMENT') == 'PASS':
            nxt = 'Complete TADSR UNet up_blocks.2 local aggregate verification after up_blocks.2.upsamplers.0 module alignment; keep full UNet forward and full inference NotImplemented.'
        elif status.get('TADSR_UNET_UPBLOCK2_ATTENTION2_BRIDGE_ALIGNMENT') == 'PASS':
            nxt = 'Continue TADSR UNet up_blocks.2.upsamplers.0 audit/export/port after up_blocks.2.attentions.2 alignment; keep full up_blocks.2, full UNet forward, and full inference NotImplemented until all remaining up_blocks and LoRA runtime integration pass.'
        elif status.get('TADSR_UNET_UPBLOCK2_RESNET2_BRIDGE_ALIGNMENT') == 'PASS':
            nxt = 'Continue TADSR UNet up_blocks.2.attentions.2 audit/export/port after up_blocks.2.resnets.2 alignment; keep full up_blocks.2, full UNet forward, and full inference NotImplemented until all remaining up_blocks and LoRA runtime integration pass.'
        elif status.get('TADSR_UNET_UPBLOCK2_ATTENTION1_BRIDGE_ALIGNMENT') == 'PASS':
            nxt = 'Continue TADSR UNet up_blocks.2.resnets.2 audit/export/port after up_blocks.2.attentions.1 alignment; keep full up_blocks.2, full UNet forward, and full inference NotImplemented until all remaining up_blocks and LoRA runtime integration pass.'
        elif status.get('TADSR_UNET_UPBLOCK2_RESNET1_BRIDGE_ALIGNMENT') == 'PASS':
            nxt = 'Continue TADSR UNet up_blocks.2.attentions.1 audit/export/port after up_blocks.2.resnets.1 alignment; keep full up_blocks.2, full UNet forward, and full inference NotImplemented until all remaining up_blocks and LoRA runtime integration pass.'
        elif status.get('TADSR_UNET_UPBLOCK2_ATTENTION0_BRIDGE_ALIGNMENT') == 'PASS':
            nxt = 'Continue TADSR UNet up_blocks.2.resnets.1 audit/export/port after up_blocks.2.attentions.0 alignment; keep full up_blocks.2, full UNet forward, and full inference NotImplemented until all remaining up_blocks and LoRA runtime integration pass.'
        elif status.get('TADSR_UNET_UPBLOCK2_RESNET0_BRIDGE_ALIGNMENT') == 'PASS':
            nxt = 'Continue TADSR UNet up_blocks.2.attentions.0 audit/export/port after up_blocks.2.resnets.0 alignment; keep full up_blocks.2, full UNet forward, and full inference NotImplemented until all remaining up_blocks and LoRA runtime integration pass.'
        elif status.get('TADSR_UNET_UPBLOCK1_ALIGNMENT') == 'PASS':
            nxt = 'Continue TADSR UNet up_blocks.2.resnets.0 audit/export/port after up_blocks.1 local alignment; keep full UNet forward and full inference NotImplemented until all up_blocks and LoRA runtime integration pass.'
        elif status.get('TADSR_UNET_UPBLOCK1_UPSAMPLER_BRIDGE_ALIGNMENT') == 'PASS':
            nxt = 'Complete TADSR UNet up_blocks.1 local aggregate verification after up_blocks.1.upsamplers.0 module alignment; keep full UNet forward and full inference NotImplemented.'
        elif status.get('TADSR_UNET_UPBLOCK1_ATTENTION2_BRIDGE_ALIGNMENT') == 'PASS':
            nxt = 'Continue TADSR UNet up_blocks.1.upsamplers.0 audit/export/port after up_blocks.1.attentions.2 alignment; keep full UNet forward and full inference NotImplemented until all up_blocks and LoRA runtime integration pass.'
        elif status.get('TADSR_UNET_UPBLOCK1_RESNET2_BRIDGE_ALIGNMENT') == 'PASS':
            nxt = 'Continue TADSR UNet up_blocks.1.attentions.2 audit/export/port after up_blocks.1.resnets.2 alignment; keep full UNet forward and full inference NotImplemented until all up_blocks and LoRA runtime integration pass.'
        elif status.get('TADSR_UNET_UPBLOCK1_ATTENTION1_BRIDGE_ALIGNMENT') == 'PASS':
            nxt = 'Continue TADSR UNet up_blocks.1.resnets.2 audit/export/port after up_blocks.1.attentions.1 alignment; keep full UNet forward and full inference NotImplemented until all up_blocks and LoRA runtime integration pass.'
        elif status.get('TADSR_UNET_UPBLOCK1_RESNET1_BRIDGE_ALIGNMENT') == 'PASS':
            nxt = 'Continue TADSR UNet up_blocks.1.attentions.1 audit/export/port after up_blocks.1.resnets.1 alignment; keep full UNet forward and full inference NotImplemented until all up_blocks and LoRA runtime integration pass.'
        elif status.get('TADSR_UNET_UPBLOCK1_ATTENTION0_BRIDGE_ALIGNMENT') == 'PASS':
            nxt = 'Continue TADSR UNet up_blocks.1.resnets.1 audit/export/port after up_blocks.1.attentions.0; keep full UNet forward and full inference NotImplemented until all up_blocks and LoRA runtime integration pass.'
        elif status.get('TADSR_UNET_UPBLOCK1_RESNET0_BRIDGE_ALIGNMENT') == 'PASS':
            nxt = 'Continue TADSR UNet up_blocks.1.attentions.0 audit/export/port after up_blocks.1.resnets.0; keep full UNet forward and full inference NotImplemented until all up_blocks and LoRA runtime integration pass.'
        else:
            nxt = 'Continue TADSR UNet up_blocks.1.resnets.0 audit/export/port after up_blocks.0 local alignment; keep full UNet forward and full inference NotImplemented until all up_blocks and LoRA runtime integration pass.'
    else:
        nxt = 'Fix preprocess/scheduler alignment first, then continue weight loading and module port.'

    lines = ['# Final Audit Report', '', '```text', 'FINAL AUDIT SUMMARY']
    for n, s, note, _ in rows:
        lines.append(f'{n}: {s} - {note}')
    lines.append(f'NEXT_ACTION: {nxt}')
    lines.append('```')
    lines += ['', '| Check | Status | Note | Next Action |', '|---|---|---|---|']
    for n, s, note, act in rows:
        lines.append(f'| {n} | {s} | {note} | {act} |')
    lines += ['', '## NEXT_ACTION', '', nxt]
    Path('experiments').mkdir(exist_ok=True)
    Path('experiments/final_audit_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    Path('experiments/final_audit_report.json').write_text(json.dumps({'rows': [{'check': n, 'status': s, 'note': note, 'next_action': act} for n, s, note, act in rows], 'next_action': nxt}, indent=2), encoding='utf-8')
    print('FINAL AUDIT SUMMARY')
    for n, s, note, _ in rows:
        print(f'{n}: {s} - {note}')
    print(f'NEXT_ACTION: {nxt}')
    import os
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
