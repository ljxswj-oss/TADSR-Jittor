#!/usr/bin/env python3
from __future__ import annotations

import json
import py_compile
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "experiments" / "production_completion" / "phase3_validation.json"
OUT_MD = ROOT / "experiments" / "production_completion" / "phase3_validation.md"
BLOCKER_JSON = ROOT / "experiments" / "production_completion" / "blockers" / "production_completion_blockers.json"
BLOCKER_MD = ROOT / "experiments" / "production_completion" / "blockers" / "production_completion_blockers.md"
ONE_STEP_PLAN_JSON = ROOT / "experiments" / "production_completion" / "full_inference" / "one_step_tensor_alignment_plan.json"
ONE_STEP_PLAN_MD = ROOT / "experiments" / "production_completion" / "full_inference" / "one_step_tensor_alignment_plan.md"
MANUAL_RUNBOOK = ROOT / "docs" / "production_completion" / "linux_phase3b_manual_runbook.md"
LINUX_SCRIPT = ROOT / "scripts" / "run_phase3b_live_audit_linux.sh"
RESULT_PACKAGER = ROOT / "scripts" / "package_phase3b_live_results.py"
RESULT_IMPORTER = ROOT / "scripts" / "import_phase3b_live_results.py"
IMPORT_VALIDATION_JSON = ROOT / "experiments" / "production_completion" / "phase3_import_validation.json"
PHASE3B_HANDOFF_JSON = ROOT / "experiments" / "production_completion" / "phase3b_manual_handoff.json"
PHASE3D_GATE_JSON = ROOT / "experiments" / "production_completion" / "phase3d_import_gate_status.json"
PHASE3D_GATE_MD = ROOT / "experiments" / "production_completion" / "phase3d_import_gate_status.md"
PHASE3_IMPORT_DECISION_JSON = ROOT / "experiments" / "production_completion" / "phase3_import_decision.json"
PHASE3_IMPORT_DECISION_MD = ROOT / "experiments" / "production_completion" / "phase3_import_decision.md"
ONE_STEP_PROTOCOL_JSON = ROOT / "experiments" / "production_completion" / "full_inference" / "one_step_tensor_alignment_protocol.json"
ONE_STEP_PROTOCOL_MD = ROOT / "experiments" / "production_completion" / "full_inference" / "one_step_tensor_alignment_protocol.md"
ONE_STEP_PROTOCOL_DOC = ROOT / "docs" / "production_completion" / "one_step_tensor_alignment_protocol.md"
ONE_STEP_BLOCKER_JSON = ROOT / "experiments" / "production_completion" / "full_inference" / "one_step_gate_blocker_summary.json"
ONE_STEP_BLOCKER_MD = ROOT / "experiments" / "production_completion" / "full_inference" / "one_step_gate_blocker_summary.md"
ONE_STEP_EXECUTION_DIR = ROOT / "experiments" / "production_completion" / "full_inference" / "one_step"
ONE_STEP_OFFICIAL_AUDIT_JSON = ONE_STEP_EXECUTION_DIR / "one_step_official_path_audit.json"
ONE_STEP_ORACLE_JSON = ONE_STEP_EXECUTION_DIR / "official_one_step_oracle_metadata.json"
ONE_STEP_JITTOR_JSON = ONE_STEP_EXECUTION_DIR / "jittor_one_step_alignment.json"
ONE_STEP_VALIDATION_JSON = ONE_STEP_EXECUTION_DIR / "one_step_tensor_alignment_validation.json"
ACTUAL_INFERENCE_PATH_JSON = ROOT / "experiments" / "production_completion" / "full_inference" / "actual_inference_path_audit.json"
POSTPROCESS_CONTRACT_JSON = ROOT / "experiments" / "production_completion" / "full_inference" / "postprocess_contract_audit.json"
MULTISTEP_DECISION_JSON = ROOT / "experiments" / "production_completion" / "full_inference" / "multistep_applicability_decision.json"
EXPERIMENTAL_CLI_PLAN_JSON = ROOT / "experiments" / "production_completion" / "full_inference" / "experimental_cli_metadata_only_plan.json"
PHASE5B_FREEZE_SUMMARY_JSON = ROOT / "experiments" / "final_phase5b_submission_freeze_summary.json"
DIAGNOSTIC_IMAGE_SMOKE_PLAN_JSON = ROOT / "experiments" / "production_completion" / "full_inference" / "diagnostic_image_smoke_plan.json"
DIAGNOSTIC_IMAGE_SMOKE_METRICS_JSON = ROOT / "experiments" / "production_completion" / "diagnostic_image_smoke" / "diagnostic_image_smoke_metrics.json"
DIAGNOSTIC_IMAGE_SMOKE_VALIDATION_JSON = ROOT / "experiments" / "production_completion" / "diagnostic_image_smoke" / "diagnostic_image_smoke_validation.json"
ONE_STEP_DIAGNOSTIC_CLI_JSON = ROOT / "experiments" / "production_completion" / "diagnostic_cli" / "one_step_diagnostic_cli_test.json"
TIMEVAE_CLOSURE_JSON = ROOT / "experiments" / "timevae_full_alignment_closure_summary.json"
RUNTIME_LORA_DECISION_JSON = ROOT / "experiments" / "runtime_lora_final_decision_proof.json"
TIMEVAE_DIAGNOSIS_JSON = ROOT / "experiments" / "production_completion" / "timevae_full" / "timevae_metadata_partial_diagnosis.json"
TIMEVAE_COMPLETION_JSON = ROOT / "experiments" / "production_completion" / "timevae_full" / "timevae_live_metadata_completion.json"
DEPENDENCY_DIAGNOSIS_JSON = ROOT / "experiments" / "production_completion" / "env" / "official_runtime_dependency_diagnosis.json"
DEPENDENCY_OVERLAY_JSON = ROOT / "experiments" / "production_completion" / "env" / "official_runtime_dependency_overlay.json"
SMOKE_TRAINING_SUMMARY_JSON = ROOT / "experiments" / "smoke_training" / "output_tail" / "smoke_training_submission_summary.json"
PACKAGE_CANDIDATES = [
    ROOT.parent / "phase3b_live_results_package.zip",
    ROOT.parent / "TADSR-Jittor_Final_Materials_20260609_213457" / "phase3b_live_results_package.zip",
    ROOT / "phase3b_live_results_package.zip",
]


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as exc:
        return {"status": "FAIL", "error": repr(exc)}


def status_of(data: dict, default: str = "BLOCKED") -> str:
    if not data:
        return default
    return str(data.get("audit_status", data.get("status", default)))


def python_syntax_ok(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        py_compile.compile(str(path), doraise=True)
        return True
    except Exception:
        return False


def shell_syntax_ok(path: Path) -> bool:
    if not path.exists():
        return False
    bash = shutil.which("bash")
    if bash:
        proc = subprocess.run([bash, "-n", str(path)], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return proc.returncode == 0
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text.startswith("#!/usr/bin/env bash") and "set -euo pipefail" in text


def safe_text_file(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="ignore")
    forbidden = [
        "BEGIN OPENSSH PRIVATE KEY",
        "BEGIN RSA PRIVATE KEY",
        "BEGIN DSA PRIVATE KEY",
        "BEGIN EC PRIVATE KEY",
        "HUGGINGFACE_TOKEN=",
        "HF_TOKEN=",
        "password=",
        "PASSWORD=",
        "token=",
        "TOKEN=",
    ]
    return not any(item in text for item in forbidden)


def phase3b_manual_kit_status() -> dict:
    runbook_exists = MANUAL_RUNBOOK.exists()
    linux_script_exists = LINUX_SCRIPT.exists()
    packager_exists = RESULT_PACKAGER.exists()
    importer_exists = RESULT_IMPORTER.exists()
    packager_syntax = python_syntax_ok(RESULT_PACKAGER)
    importer_syntax = python_syntax_ok(RESULT_IMPORTER)
    shell_syntax = shell_syntax_ok(LINUX_SCRIPT)
    text_files_safe = all(
        safe_text_file(path)
        for path in [MANUAL_RUNBOOK, LINUX_SCRIPT, RESULT_PACKAGER, RESULT_IMPORTER]
        if path.exists()
    )
    no_raw_paths = all(
        not any(part == "local_tensors" for part in path.parts)
        and path.suffix.lower() not in {".npy", ".npz"}
        for path in [MANUAL_RUNBOOK, LINUX_SCRIPT, RESULT_PACKAGER, RESULT_IMPORTER]
    )
    return {
        "runbook_exists": runbook_exists,
        "linux_script_exists": linux_script_exists,
        "packager_exists": packager_exists,
        "importer_exists": importer_exists,
        "packager_syntax_ok": packager_syntax,
        "importer_syntax_ok": importer_syntax,
        "linux_script_syntax_ok": shell_syntax,
        "text_files_safe": text_files_safe,
        "no_raw_tensor_paths": no_raw_paths,
        "runbook_status": "PASS" if runbook_exists and safe_text_file(MANUAL_RUNBOOK) else "BLOCKED",
        "linux_script_status": "PASS" if linux_script_exists and shell_syntax and safe_text_file(LINUX_SCRIPT) else "BLOCKED",
        "packager_status": "PASS" if packager_exists and packager_syntax and safe_text_file(RESULT_PACKAGER) else "BLOCKED",
        "importer_status": "PASS" if importer_exists and importer_syntax and safe_text_file(RESULT_IMPORTER) else "BLOCKED",
    }


def package_candidates_status() -> list[dict]:
    rows = []
    for path in PACKAGE_CANDIDATES:
        sha_path = path.with_suffix(".sha256.txt")
        exists = path.exists()
        rows.append(
            {
                "path": str(path),
                "exists": exists,
                "size_bytes": path.stat().st_size if exists else None,
                "sha256_path": str(sha_path),
                "sha256_exists": sha_path.exists(),
            }
        )
    return rows


def phase3d_gate_status(import_validation: dict, ready_for_one_step: bool, phase3_blockers: list[str]) -> dict:
    candidates = package_candidates_status()
    package_found = any(row["exists"] for row in candidates)
    import_status = status_of(import_validation, "BLOCKED")
    no_raw_status = str(import_validation.get("no_raw_tensors_status", "BLOCKED")) if import_validation else "BLOCKED"
    package_security = "PASS" if import_status in {"PASS", "PARTIAL"} and no_raw_status == "PASS" else "BLOCKED"
    import_gate = "PASS" if import_status == "PASS" and no_raw_status == "PASS" else "BLOCKED"
    one_step_gate = "PASS" if ready_for_one_step and import_gate == "PASS" else "BLOCKED"
    protocol_ready = "PASS" if one_step_gate == "PASS" and ONE_STEP_PROTOCOL_JSON.exists() and ONE_STEP_PROTOCOL_MD.exists() else "BLOCKED"
    blocker_summary = "PASS" if one_step_gate != "PASS" or ONE_STEP_BLOCKER_JSON.exists() else "BLOCKED"

    blockers = list(phase3_blockers)
    if not package_found:
        blockers.append("phase3b_live_results_package.zip is not present in any local candidate path")
    if package_found and package_security != "PASS":
        blockers.append("Phase 3-B package exists but has not passed import dry-run / no-raw-tensor validation")
    if import_gate != "PASS":
        blockers.append("Phase 3-B live results have not been safely imported")
    if not ready_for_one_step:
        blockers.append("one-step tensor alignment prerequisites are not satisfied")

    next_action = (
        "Manually run Linux Phase 3-B live audit, copy phase3b_live_results_package.zip back to Windows, "
        "then import it with scripts/import_phase3b_live_results.py."
        if not package_found
        else (
            "Run import_phase3b_live_results.py dry-run and import, then rerun Phase 3 validation."
            if import_gate != "PASS"
            else (
                "Next round may prepare controlled one-step tensor alignment; keep full inference guarded."
                if one_step_gate == "PASS"
                else "Resolve one-step prerequisites before any tensor alignment execution."
            )
        )
    )
    markers = {
        "TADSR_PHASE3D_LIVE_RESULTS_PACKAGE_FOUND": "PASS" if package_found else "BLOCKED",
        "TADSR_PHASE3D_LIVE_RESULTS_PACKAGE_SECURITY": package_security,
        "TADSR_PHASE3D_IMPORT_GATE": import_gate,
        "TADSR_PHASE3D_ONE_STEP_GATE_DECISION": one_step_gate,
        "TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT_PROTOCOL_READY": protocol_ready,
        "TADSR_FULL_INFERENCE_ONE_STEP_GATE_BLOCKER_SUMMARY": blocker_summary,
    }
    return {
        "status_marker": "TADSR_PHASE3D_IMPORT_GATE",
        "status": import_gate,
        "markers": markers,
        "package_candidates": candidates,
        "package_found": package_found,
        "package_security_status": package_security,
        "phase3b_import_validation_status": import_status,
        "phase3b_import_no_raw_tensors_status": no_raw_status,
        "one_step_allowed_next": one_step_gate == "PASS",
        "one_step_gate_decision": one_step_gate,
        "blockers": blockers,
        "next_required_action": next_action,
    }


def write_phase3d_reports(payload: dict) -> None:
    PHASE3D_GATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    PHASE3D_GATE_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Phase 3-D import gate status", ""]
    for key, value in payload["markers"].items():
        lines.append(f"`{key}: {value}`")
    lines += ["", "## Package candidates", ""]
    for row in payload["package_candidates"]:
        lines.append(f"- `{row['path']}`: exists=`{row['exists']}`, sha256_exists=`{row['sha256_exists']}`")
    lines += [
        "",
        "## Decision",
        "",
        f"- `one_step_allowed_next`: `{payload['one_step_allowed_next']}`",
        f"- `next_required_action`: {payload['next_required_action']}",
        "",
        "## Blockers",
        "",
    ]
    for blocker in payload["blockers"]:
        lines.append(f"- {blocker}")
    PHASE3D_GATE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_import_decision_reports(payload: dict) -> None:
    decision = {
        "status_marker": "TADSR_PHASE3D_ONE_STEP_GATE_DECISION",
        "status": payload["markers"]["TADSR_PHASE3D_ONE_STEP_GATE_DECISION"],
        "one_step_allowed_next": payload["one_step_allowed_next"],
        "import_gate": payload["markers"]["TADSR_PHASE3D_IMPORT_GATE"],
        "package_found": payload["package_found"],
        "package_security_status": payload["package_security_status"],
        "phase3b_import_validation_status": payload["phase3b_import_validation_status"],
        "phase3b_import_no_raw_tensors_status": payload["phase3b_import_no_raw_tensors_status"],
        "blockers": payload["blockers"],
        "next_required_action": payload["next_required_action"],
    }
    PHASE3_IMPORT_DECISION_JSON.parent.mkdir(parents=True, exist_ok=True)
    PHASE3_IMPORT_DECISION_JSON.write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Phase 3 import decision",
        "",
        f"`TADSR_PHASE3D_ONE_STEP_GATE_DECISION: {decision['status']}`",
        "",
        f"- `one_step_allowed_next`: `{decision['one_step_allowed_next']}`",
        f"- `import_gate`: `{decision['import_gate']}`",
        f"- `package_found`: `{decision['package_found']}`",
        "",
        "## Next action",
        "",
        decision["next_required_action"],
    ]
    if decision["blockers"]:
        lines += ["", "## Blockers", ""]
        for blocker in decision["blockers"]:
            lines.append(f"- {blocker}")
    PHASE3_IMPORT_DECISION_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_one_step_blocker_summary(payload: dict) -> None:
    summary = {
        "status_marker": "TADSR_FULL_INFERENCE_ONE_STEP_GATE_BLOCKER_SUMMARY",
        "status": "PASS",
        "one_step_allowed_next": payload["one_step_allowed_next"],
        "blockers": payload["blockers"],
        "recommended_next_action": payload["next_required_action"],
        "full_inference_status": "NOT_COMPLETE",
        "note": "This blocker summary does not execute one-step tensor alignment and does not open full inference.",
    }
    ONE_STEP_BLOCKER_JSON.parent.mkdir(parents=True, exist_ok=True)
    ONE_STEP_BLOCKER_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# One-step tensor alignment gate blocker summary",
        "",
        "`TADSR_FULL_INFERENCE_ONE_STEP_GATE_BLOCKER_SUMMARY: PASS`",
        "",
        f"- `one_step_allowed_next`: `{summary['one_step_allowed_next']}`",
        f"- `full_inference_status`: `{summary['full_inference_status']}`",
        "",
        "## Blockers",
        "",
    ]
    for blocker in summary["blockers"]:
        lines.append(f"- {blocker}")
    lines += ["", "## Recommended next action", "", summary["recommended_next_action"]]
    ONE_STEP_BLOCKER_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_one_step_protocol() -> None:
    protocol = {
        "status_marker": "TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT_PROTOCOL_READY",
        "status": "PASS",
        "executed": False,
        "full_inference_executed": False,
        "denoising_loop_executed": False,
        "image_or_video_generated": False,
        "raw_tensor_policy": "protocol only; no raw tensors are produced or committed in this phase",
        "scope": "controlled one-step tensor alignment protocol for the next phase",
        "official_pytorch_oracle_stage": [
            "load official PyTorch TADSR assets in strict environment",
            "prepare one in-memory low-resolution input tensor with fixed seed",
            "run exactly one TimeVAE encode metadata-compatible boundary",
            "run exactly one UNet forward tensor boundary with fixed scheduler timestep",
            "run exactly one scheduler step tensor boundary",
            "run one decode/clamp tensor boundary only if prior tensor gates pass",
        ],
        "jittor_stage": [
            "load converted Jittor weights already covered by existing reports",
            "mirror the same one-step tensor inputs and timestep",
            "compare tensor outputs at stage boundaries with absolute/relative/cosine metrics",
        ],
        "required_metrics": [
            "shape match",
            "dtype policy match",
            "max absolute error",
            "mean absolute error",
            "relative L2 error",
            "cosine similarity",
            "finite value check",
        ],
        "stop_conditions": [
            "any shape mismatch",
            "any non-finite tensor statistics",
            "unexpected scheduler loop or multi-step denoising",
            "attempted final image/video generation",
            "attempted raw .npy/.npz commit",
            "full inference guard removed or bypassed",
        ],
        "must_remain_statuses": {
            "JITTOR_FULL_INFERENCE": "NOT_COMPLETE",
            "JITTOR_FULL_PORT": "PARTIAL",
            "TIME_VAE_FULL_ALIGNMENT": "NOT_COMPLETE",
            "TADSR_DYNAMIC_RUNTIME_LORA_IMPLEMENTATION": "NOT_IMPLEMENTED_BY_DESIGN",
        },
    }
    ONE_STEP_PROTOCOL_JSON.parent.mkdir(parents=True, exist_ok=True)
    ONE_STEP_PROTOCOL_JSON.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Controlled one-step tensor alignment protocol",
        "",
        "`TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT_PROTOCOL_READY: PASS`",
        "",
        "This file is a protocol for the next phase. It does not execute one-step tensor alignment, does not run production full inference, does not run a full denoising loop, and does not generate images or videos.",
        "",
        "## Scope",
        "",
        protocol["scope"],
        "",
        "## Official PyTorch oracle stage",
        "",
    ]
    for item in protocol["official_pytorch_oracle_stage"]:
        lines.append(f"- {item}")
    lines += ["", "## Jittor stage", ""]
    for item in protocol["jittor_stage"]:
        lines.append(f"- {item}")
    lines += ["", "## Metrics", ""]
    for item in protocol["required_metrics"]:
        lines.append(f"- {item}")
    lines += ["", "## Stop conditions", ""]
    for item in protocol["stop_conditions"]:
        lines.append(f"- {item}")
    lines += [
        "",
        "## Must-remain status markers",
        "",
    ]
    for key, value in protocol["must_remain_statuses"].items():
        lines.append(f"- `{key}: {value}`")
    text = "\n".join(lines) + "\n"
    ONE_STEP_PROTOCOL_MD.write_text(text, encoding="utf-8")
    ONE_STEP_PROTOCOL_DOC.parent.mkdir(parents=True, exist_ok=True)
    ONE_STEP_PROTOCOL_DOC.write_text(text, encoding="utf-8")


def nested_marker(data: dict, marker: str, default: str = "BLOCKED") -> str:
    markers = data.get("markers", {}) if isinstance(data, dict) else {}
    if isinstance(markers, dict) and marker in markers:
        return str(markers.get(marker, default))
    if data.get("status_marker") == marker:
        return str(data.get("audit_status", data.get("status", default)))
    return default


def final_statuses() -> dict[str, str]:
    data = load_json(ROOT / "experiments" / "final_audit_report.json")
    return {str(row.get("check")): str(row.get("status")) for row in data.get("rows", [])}


def write_md(path: Path, title: str, payload: dict, markers: dict[str, str]) -> None:
    lines = [f"# {title}", ""]
    for key, value in markers.items():
        lines.append(f"`{key}: {value}`")
    lines += ["", "## Summary", ""]
    for key in [
        "official_env_resolution_status",
        "live_timevae_audit_status",
        "timevae_oracle_metadata_status",
        "timevae_preflight_status",
        "lora_live_audit_status",
        "full_inference_metadata_contract_status",
        "remote_connectivity_status",
        "remote_tcp_10022_succeeded",
        "remote_ssh_authentication_available",
        "one_step_alignment_plan_status",
    ]:
        lines.append(f"- `{key}`: `{payload.get(key)}`")
    lines += ["", f"- `ready_for_one_step_tensor_alignment`: `{payload.get('ready_for_one_step_tensor_alignment')}`"]
    lines += ["", "## Next action", "", str(payload.get("next_required_action", ""))]
    blockers = payload.get("blockers", [])
    if blockers:
        lines += ["", "## Blockers", ""]
        for item in blockers:
            lines.append(f"- {item}")
    phase3_blockers = payload.get("phase3_blockers", [])
    if phase3_blockers:
        lines += ["", "## Phase 3 blockers", ""]
        for item in phase3_blockers:
            lines.append(f"- {item}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    env = load_json(ROOT / "experiments" / "production_completion" / "env" / "official_env_resolution.json")
    remote = load_json(ROOT / "experiments" / "production_completion" / "env" / "remote_connectivity_diagnostic.json")
    audit = load_json(ROOT / "experiments" / "production_completion" / "timevae_full" / "official_timevae_full_path_audit.json")
    oracle = load_json(ROOT / "experiments" / "production_completion" / "timevae_full" / "timevae_production_oracle_metadata.json")
    preflight = load_json(ROOT / "experiments" / "production_completion" / "timevae_full" / "jittor_timevae_production_alignment_preflight.json")
    lora = load_json(ROOT / "experiments" / "production_completion" / "runtime_lora" / "official_runtime_lora_behavior_audit.json")
    contract = load_json(ROOT / "experiments" / "production_completion" / "full_inference" / "metadata_dry_run_contract.json")
    one_step_plan = load_json(ONE_STEP_PLAN_JSON)
    import_validation = load_json(IMPORT_VALIDATION_JSON)
    manual_handoff = load_json(PHASE3B_HANDOFF_JSON)
    timevae_diagnosis = load_json(TIMEVAE_DIAGNOSIS_JSON)
    timevae_completion = load_json(TIMEVAE_COMPLETION_JSON)
    dependency_diagnosis = load_json(DEPENDENCY_DIAGNOSIS_JSON)
    dependency_overlay = load_json(DEPENDENCY_OVERLAY_JSON)
    smoke_training_summary = load_json(SMOKE_TRAINING_SUMMARY_JSON)
    final = final_statuses()
    manual_kit = phase3b_manual_kit_status()

    env_status = status_of(env)
    timevae_audit_status = status_of(audit)
    oracle_status = status_of(oracle)
    preflight_status = status_of(preflight)
    lora_status = status_of(lora)
    contract_status = status_of(contract)
    one_step_plan_status = status_of(one_step_plan)
    must_remain = {
        "JITTOR_FULL_INFERENCE": final.get("JITTOR_FULL_INFERENCE", "MISSING"),
        "JITTOR_FULL_PORT": final.get("JITTOR_FULL_PORT", "MISSING"),
        "TIME_VAE_FULL_ALIGNMENT": final.get("TIME_VAE_FULL_ALIGNMENT", "MISSING"),
        "TADSR_DYNAMIC_RUNTIME_LORA_IMPLEMENTATION": final.get("TADSR_DYNAMIC_RUNTIME_LORA_IMPLEMENTATION", "MISSING"),
    }
    must_remain_ok = (
        must_remain["JITTOR_FULL_INFERENCE"] == "NOT_COMPLETE"
        and must_remain["JITTOR_FULL_PORT"] == "PARTIAL"
        and must_remain["TIME_VAE_FULL_ALIGNMENT"] == "NOT_COMPLETE"
        and must_remain["TADSR_DYNAMIC_RUNTIME_LORA_IMPLEMENTATION"] == "NOT_IMPLEMENTED_BY_DESIGN"
    )
    no_execution_claims = (
        oracle.get("full_inference_executed") is False
        and oracle.get("denoising_loop_executed") is False
        and oracle.get("image_or_video_generated", oracle.get("image_or_video_output_generated")) is False
        and contract.get("full_inference_executed") is False
        and contract.get("denoising_loop_executed") is False
        and contract.get("image_or_video_output_generated") is False
    )
    timevae_completion_status = status_of(timevae_completion, "BLOCKED")
    ready_for_one_step = bool(
        env_status == "PASS"
        and oracle_status == "PASS"
        and timevae_completion_status == "PASS"
        and preflight.get("ready_for_one_step_tensor_alignment") is True
        and contract.get("ready_for_one_step_tensor_alignment") is True
        and must_remain_ok
        and no_execution_claims
    )
    blockers: list[str] = []
    if env_status != "PASS":
        blockers.append("official repo / weights / strict Python are not fully available in this run")
    if remote:
        if remote.get("tcp_10022_succeeded") is False:
            blockers.append("10.195.21.2:10022 is unreachable from the current Windows machine")
        elif remote.get("ssh_authentication_available") is False:
            blockers.append("10.195.21.2:10022 is reachable, but SSH authentication is unavailable in the current automation context")
    if oracle_status != "PASS":
        blockers.append("TimeVAE production oracle metadata is not exported live")
    if preflight_status not in {"PASS", "PARTIAL"}:
        blockers.append("Jittor TimeVAE production alignment preflight is not available")
    if not must_remain_ok:
        blockers.append("one or more must-remain NOT_COMPLETE / NOT_IMPLEMENTED markers changed")
    if not no_execution_claims:
        blockers.append("metadata-only safety fields indicate unexpected execution claim")
    if ready_for_one_step and one_step_plan_status != "PASS":
        blockers.append("one-step tensor alignment gate is ready, but the one-step alignment plan has not been generated")
    manual_kit_ready = bool(
        manual_kit["runbook_status"] == "PASS"
        and manual_kit["linux_script_status"] == "PASS"
        and manual_kit["packager_status"] == "PASS"
        and manual_kit["importer_status"] == "PASS"
        and manual_kit["text_files_safe"]
        and manual_kit["no_raw_tensor_paths"]
        and must_remain_ok
    )
    if not manual_kit_ready:
        blockers.append("Phase 3-B manual execution kit is not fully ready")

    if ready_for_one_step and one_step_plan_status == "PASS":
        phase3_status = "PASS"
        next_action = "Proceed to controlled one-step tensor alignment only; still do not generate final image/video or open full production inference."
    elif ready_for_one_step:
        phase3_status = "PASS_WITH_BLOCKERS"
        next_action = "Generate and review the controlled one-step tensor alignment plan before executing any one-step tensor comparison."
    elif env_status == "PASS" and must_remain_ok and no_execution_claims:
        phase3_status = "PASS_WITH_BLOCKERS"
        next_action = "Resolve TimeVAE oracle / preflight blockers before one-step tensor alignment."
    elif must_remain_ok and no_execution_claims:
        phase3_status = "PASS_WITH_BLOCKERS"
        next_action = "Run this Phase 3-A package on Linux with official repo, weights and strict Python; do not continue to one-step tensor alignment yet."
    else:
        phase3_status = "FAIL"
        next_action = "Fix safety-marker regression before any further production-completion work."

    live_metadata_marker = "PASS" if oracle_status == "PASS" else oracle_status
    lora_live_marker = "PASS" if lora_status == "PASS" else lora_status
    one_step_marker = "PASS" if ready_for_one_step else "BLOCKED"
    one_step_plan_marker = "PASS" if one_step_plan_status == "PASS" else "BLOCKED"
    markers = {
        "TADSR_PRODUCTION_COMPLETION_PHASE3_VALIDATION": phase3_status,
        "TADSR_OFFICIAL_DIFFUSERS_OVERLAY_READY": nested_marker(dependency_overlay, "TADSR_OFFICIAL_DIFFUSERS_OVERLAY_READY", "BLOCKED"),
        "TADSR_OFFICIAL_RUNTIME_DEPENDENCY_REPAIR_READINESS": nested_marker(dependency_overlay, "TADSR_OFFICIAL_RUNTIME_DEPENDENCY_REPAIR_READINESS", "BLOCKED"),
        "TADSR_OFFICIAL_RUNTIME_DEPENDENCY_OVERLAY_ACTIVE": nested_marker(dependency_diagnosis, "TADSR_OFFICIAL_RUNTIME_DEPENDENCY_OVERLAY_ACTIVE", "BLOCKED"),
        "TADSR_TIMEVAE_LIVE_METADATA_EXPORT": live_metadata_marker,
        "TADSR_LORA_LIVE_RUNTIME_AUDIT": lora_live_marker,
        "TADSR_FULL_INFERENCE_CONTRACT_READY_FOR_ONE_STEP": one_step_marker,
        "TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT_PLAN": one_step_plan_marker,
        "TADSR_TIMEVAE_METADATA_PARTIAL_DIAGNOSIS": status_of(timevae_diagnosis, "BLOCKED"),
        "TADSR_TIMEVAE_LIVE_METADATA_COMPLETION": nested_marker(timevae_completion, "TADSR_TIMEVAE_LIVE_METADATA_COMPLETION", timevae_completion_status),
        "TADSR_TIMEVAE_LIVE_ENCODE_METADATA": nested_marker(timevae_completion, "TADSR_TIMEVAE_LIVE_ENCODE_METADATA", "BLOCKED"),
        "TADSR_TIMEVAE_LIVE_DECODE_METADATA": nested_marker(timevae_completion, "TADSR_TIMEVAE_LIVE_DECODE_METADATA", "BLOCKED"),
        "TADSR_TIMEVAE_LIVE_SAFETY_FLAGS": nested_marker(timevae_completion, "TADSR_TIMEVAE_LIVE_SAFETY_FLAGS", "BLOCKED"),
        "TADSR_OFFICIAL_RUNTIME_DEPENDENCY_DIAGNOSIS": status_of(dependency_diagnosis, "BLOCKED"),
        "TADSR_TIMEVAE_METADATA_REPAIR_ATTEMPT": nested_marker(preflight, "TADSR_TIMEVAE_METADATA_REPAIR_ATTEMPT", nested_marker(contract, "TADSR_TIMEVAE_METADATA_REPAIR_ATTEMPT", "BLOCKED")),
        "TADSR_TIMEVAE_METADATA_FIELDS_COMPLETE": nested_marker(preflight, "TADSR_TIMEVAE_METADATA_FIELDS_COMPLETE", nested_marker(contract, "TADSR_TIMEVAE_METADATA_FIELDS_COMPLETE", "BLOCKED")),
        "TADSR_TIMEVAE_METADATA_READY_FOR_ONE_STEP": nested_marker(preflight, "TADSR_TIMEVAE_METADATA_READY_FOR_ONE_STEP", nested_marker(contract, "TADSR_TIMEVAE_METADATA_READY_FOR_ONE_STEP", "BLOCKED")),
        "TADSR_RUNTIME_LORA_DECISION_FINALIZED_FOR_FIXED_INFERENCE": "PASS" if lora.get("dynamic_runtime_lora_not_required_marker") == "PASS" or lora.get("dynamic_runtime_lora_required_for_official_inference") is False else "PARTIAL",
        "TADSR_SMOKE_TRAINING_SUBMISSION_SUMMARY": status_of(smoke_training_summary, "BLOCKED"),
        "TADSR_PHASE3B_LINUX_MANUAL_RUNBOOK_READY": manual_kit["runbook_status"],
        "TADSR_PHASE3B_LINUX_LIVE_AUDIT_SCRIPT_READY": manual_kit["linux_script_status"],
        "TADSR_PHASE3B_RESULT_PACKAGER_READY": manual_kit["packager_status"],
        "TADSR_PHASE3B_RESULT_IMPORT_VALIDATOR_READY": manual_kit["importer_status"],
        "TADSR_PHASE3B_MANUAL_EXECUTION_KIT_READY": "PASS" if manual_kit_ready else "BLOCKED",
        "TADSR_PHASE3B_FINAL_MANUAL_HANDOFF_READY": status_of(manual_handoff, "PASS" if manual_kit_ready else "BLOCKED"),
        "TADSR_PHASE3B_LIVE_RESULTS_IMPORT_VALIDATION": status_of(import_validation, "BLOCKED"),
        "TADSR_PHASE3B_LIVE_RESULTS_NO_RAW_TENSORS": str(import_validation.get("no_raw_tensors_status", "BLOCKED")) if import_validation else "BLOCKED",
    }
    if ready_for_one_step:
        write_one_step_protocol()
        one_step_plan_status = "PASS"
        one_step_plan_marker = "PASS"
        markers["TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT_PLAN"] = "PASS"
        markers["TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT_PROTOCOL_READY"] = "PASS"
        blockers = [item for item in blockers if item != "one-step tensor alignment gate is ready, but the one-step alignment plan has not been generated"]
        if not blockers:
            phase3_status = "PASS"
            next_action = "Next round may execute controlled one-step tensor alignment only; keep full inference guarded and do not generate images or videos."
            markers["TADSR_PRODUCTION_COMPLETION_PHASE3_VALIDATION"] = "PASS"
    phase3d = phase3d_gate_status(import_validation, ready_for_one_step, blockers)
    write_phase3d_reports(phase3d)
    write_import_decision_reports(phase3d)
    write_one_step_blocker_summary(phase3d)
    markers.update(phase3d["markers"])
    one_step_official_audit = load_json(ONE_STEP_OFFICIAL_AUDIT_JSON)
    one_step_oracle = load_json(ONE_STEP_ORACLE_JSON)
    one_step_jittor = load_json(ONE_STEP_JITTOR_JSON)
    one_step_validation = load_json(ONE_STEP_VALIDATION_JSON)
    one_step_markers = {
        "TADSR_ONE_STEP_OFFICIAL_PATH_AUDIT": status_of(one_step_official_audit, "NOT_RUN"),
        "TADSR_ONE_STEP_OFFICIAL_ORACLE_TENSORS": status_of(one_step_oracle, "NOT_RUN"),
        "TADSR_ONE_STEP_JITTOR_TENSOR_RUN": nested_marker(one_step_jittor, "TADSR_ONE_STEP_JITTOR_TENSOR_RUN", "NOT_RUN"),
        "TADSR_ONE_STEP_TENSOR_STAGE_ALIGNMENT": nested_marker(one_step_jittor, "TADSR_ONE_STEP_TENSOR_STAGE_ALIGNMENT", "NOT_RUN"),
        "TADSR_ONE_STEP_TENSOR_ALIGNMENT_VALIDATION": nested_marker(one_step_validation, "TADSR_ONE_STEP_TENSOR_ALIGNMENT_VALIDATION", "NOT_RUN"),
        "TADSR_ONE_STEP_NO_IMAGE_VIDEO_GUARD": nested_marker(one_step_validation, "TADSR_ONE_STEP_NO_IMAGE_VIDEO_GUARD", "NOT_RUN"),
        "TADSR_ONE_STEP_RAW_TENSOR_POLICY": nested_marker(one_step_validation, "TADSR_ONE_STEP_RAW_TENSOR_POLICY", "NOT_RUN"),
        "TADSR_ONE_STEP_FULL_INFERENCE_GUARD_PRESERVED": nested_marker(one_step_validation, "TADSR_ONE_STEP_FULL_INFERENCE_GUARD_PRESERVED", "NOT_RUN"),
        "TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT": nested_marker(one_step_validation, "TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT", nested_marker(one_step_jittor, "TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT", "NOT_RUN")),
    }
    markers.update(one_step_markers)
    one_step_execution_statuses = list(one_step_markers.values())
    one_step_execution_ready = all(value == "PASS" for value in one_step_execution_statuses)
    one_step_execution_partial = any(value in {"PARTIAL", "BLOCKED", "NOT_RUN", "MISSING"} for value in one_step_execution_statuses)
    if one_step_execution_ready:
        markers["TADSR_FULL_INFERENCE_CONTRACT_READY_FOR_ONE_STEP"] = "PASS"
        markers["TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT_PLAN"] = "PASS"
        markers["TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT_PROTOCOL_READY"] = "PASS"
        markers["TADSR_PHASE3D_ONE_STEP_GATE_DECISION"] = "PASS"
        phase3_status = "PASS"
        markers["TADSR_PRODUCTION_COMPLETION_PHASE3_VALIDATION"] = "PASS"
        next_action = (
            "Controlled one-step tensor alignment passed. Next phase may plan tiny multi-step latent-only alignment; "
            "still do not generate images/videos, do not open production full inference, and keep JITTOR_FULL_INFERENCE NOT_COMPLETE."
        )
        blockers = []
        one_step_plan_marker = "PASS"
    actual_path_audit = load_json(ACTUAL_INFERENCE_PATH_JSON)
    postprocess_contract = load_json(POSTPROCESS_CONTRACT_JSON)
    multistep_decision = load_json(MULTISTEP_DECISION_JSON)
    experimental_cli_plan = load_json(EXPERIMENTAL_CLI_PLAN_JSON)
    phase5b_freeze_summary = load_json(PHASE5B_FREEZE_SUMMARY_JSON)
    diagnostic_image_smoke_plan = load_json(DIAGNOSTIC_IMAGE_SMOKE_PLAN_JSON)
    diagnostic_image_smoke_metrics = load_json(DIAGNOSTIC_IMAGE_SMOKE_METRICS_JSON)
    diagnostic_image_smoke_validation = load_json(DIAGNOSTIC_IMAGE_SMOKE_VALIDATION_JSON)
    one_step_diagnostic_cli = load_json(ONE_STEP_DIAGNOSTIC_CLI_JSON)
    timevae_closure = load_json(TIMEVAE_CLOSURE_JSON)
    runtime_lora_decision = load_json(RUNTIME_LORA_DECISION_JSON)
    phase5_markers = {
        "TADSR_OFFICIAL_ACTUAL_INFERENCE_PATH_AUDIT": nested_marker(actual_path_audit, "TADSR_OFFICIAL_ACTUAL_INFERENCE_PATH_AUDIT", status_of(actual_path_audit, "NOT_RUN")),
        "TADSR_OFFICIAL_INFERENCE_MULTISTEP_REQUIREMENT_AUDIT": nested_marker(actual_path_audit, "TADSR_OFFICIAL_INFERENCE_MULTISTEP_REQUIREMENT_AUDIT", "NOT_RUN"),
        "TADSR_OFFICIAL_ACTUAL_PATH_SINGLE_STEP_CONTRACT": nested_marker(actual_path_audit, "TADSR_OFFICIAL_ACTUAL_PATH_SINGLE_STEP_CONTRACT", "NOT_RUN"),
        "TADSR_POSTPROCESS_CONTRACT_AUDIT": nested_marker(postprocess_contract, "TADSR_POSTPROCESS_CONTRACT_AUDIT", status_of(postprocess_contract, "NOT_RUN")),
        "TADSR_OUTPUT_IMAGE_SAVE_POLICY_AUDIT": nested_marker(postprocess_contract, "TADSR_OUTPUT_IMAGE_SAVE_POLICY_AUDIT", "NOT_RUN"),
        "TADSR_POSTPROCESS_NOT_EXECUTED_GUARD": nested_marker(postprocess_contract, "TADSR_POSTPROCESS_NOT_EXECUTED_GUARD", "NOT_RUN"),
        "TADSR_TINY_MULTISTEP_ALIGNMENT_APPLICABILITY": nested_marker(multistep_decision, "TADSR_TINY_MULTISTEP_ALIGNMENT_APPLICABILITY", "NOT_RUN"),
        "TADSR_TINY_MULTISTEP_ALIGNMENT_DECISION": nested_marker(multistep_decision, "TADSR_TINY_MULTISTEP_ALIGNMENT_DECISION", status_of(multistep_decision, "NOT_RUN")),
        "TADSR_EXPERIMENTAL_CLI_METADATA_ONLY_PLAN": nested_marker(experimental_cli_plan, "TADSR_EXPERIMENTAL_CLI_METADATA_ONLY_PLAN", status_of(experimental_cli_plan, "NOT_RUN")),
        "TADSR_PHASE5B_SUBMISSION_FREEZE_SUMMARY": nested_marker(phase5b_freeze_summary, "TADSR_PHASE5B_SUBMISSION_FREEZE_SUMMARY", status_of(phase5b_freeze_summary, "NOT_RUN")),
        "TADSR_DIAGNOSTIC_IMAGE_SMOKE_PLAN": nested_marker(diagnostic_image_smoke_plan, "TADSR_DIAGNOSTIC_IMAGE_SMOKE_PLAN", status_of(diagnostic_image_smoke_plan, "NOT_RUN")),
        "TADSR_DIAGNOSTIC_IMAGE_SMOKE_READY": nested_marker(diagnostic_image_smoke_metrics, "TADSR_DIAGNOSTIC_IMAGE_SMOKE_READY", "MISSING"),
        "TADSR_DIAGNOSTIC_IMAGE_SMOKE_EXECUTED": nested_marker(diagnostic_image_smoke_metrics, "TADSR_DIAGNOSTIC_IMAGE_SMOKE_EXECUTED", nested_marker(diagnostic_image_smoke_plan, "TADSR_DIAGNOSTIC_IMAGE_SMOKE_EXECUTED", "NOT_RUN")),
        "TADSR_DIAGNOSTIC_IMAGE_SMOKE_ALIGNMENT": nested_marker(diagnostic_image_smoke_metrics, "TADSR_DIAGNOSTIC_IMAGE_SMOKE_ALIGNMENT", "MISSING"),
        "TADSR_DIAGNOSTIC_IMAGE_SMOKE_NOT_FULL_INFERENCE": nested_marker(diagnostic_image_smoke_metrics, "TADSR_DIAGNOSTIC_IMAGE_SMOKE_NOT_FULL_INFERENCE", "MISSING"),
        "TADSR_DIAGNOSTIC_IMAGE_SMOKE_NO_RAW_TENSORS": nested_marker(diagnostic_image_smoke_metrics, "TADSR_DIAGNOSTIC_IMAGE_SMOKE_NO_RAW_TENSORS", "MISSING"),
        "TADSR_DIAGNOSTIC_IMAGE_SMOKE_ARTIFACTS_READY": nested_marker(diagnostic_image_smoke_metrics, "TADSR_DIAGNOSTIC_IMAGE_SMOKE_ARTIFACTS_READY", "MISSING"),
        "TADSR_DIAGNOSTIC_IMAGE_SMOKE_VALIDATION": nested_marker(diagnostic_image_smoke_validation, "TADSR_DIAGNOSTIC_IMAGE_SMOKE_VALIDATION", status_of(diagnostic_image_smoke_validation, "MISSING")),
        "TADSR_DIAGNOSTIC_IMAGE_SMOKE_FALSE_CLAIM_GUARD": nested_marker(diagnostic_image_smoke_validation, "TADSR_DIAGNOSTIC_IMAGE_SMOKE_FALSE_CLAIM_GUARD", "MISSING"),
        "TADSR_DIAGNOSTIC_IMAGE_SMOKE_ARTIFACT_POLICY": nested_marker(diagnostic_image_smoke_validation, "TADSR_DIAGNOSTIC_IMAGE_SMOKE_ARTIFACT_POLICY", "MISSING"),
        "TADSR_ONE_STEP_DIAGNOSTIC_CLI_READY": nested_marker(one_step_diagnostic_cli, "TADSR_ONE_STEP_DIAGNOSTIC_CLI_READY", status_of(one_step_diagnostic_cli, "MISSING")),
        "TADSR_ONE_STEP_DIAGNOSTIC_CLI_NOT_FULL_INFERENCE": nested_marker(one_step_diagnostic_cli, "TADSR_ONE_STEP_DIAGNOSTIC_CLI_NOT_FULL_INFERENCE", "MISSING"),
        "TADSR_TIMEVAE_FULL_ALIGNMENT_CLOSURE_SUMMARY": nested_marker(timevae_closure, "TADSR_TIMEVAE_FULL_ALIGNMENT_CLOSURE_SUMMARY", status_of(timevae_closure, "MISSING")),
        "TADSR_TIMEVAE_FULL_ALIGNMENT_REMAINS_SCOPED": nested_marker(timevae_closure, "TADSR_TIMEVAE_FULL_ALIGNMENT_REMAINS_SCOPED", "MISSING"),
        "TADSR_RUNTIME_LORA_FINAL_DECISION_PROOF": nested_marker(runtime_lora_decision, "TADSR_RUNTIME_LORA_FINAL_DECISION_PROOF", status_of(runtime_lora_decision, "MISSING")),
        "TADSR_DYNAMIC_RUNTIME_LORA_NOT_REQUIRED_FOR_AUDITED_PATH": nested_marker(runtime_lora_decision, "TADSR_DYNAMIC_RUNTIME_LORA_NOT_REQUIRED_FOR_AUDITED_PATH", "MISSING"),
    }
    phase5_markers["TADSR_OFFICIAL_ACTUAL_SINGLE_STEP_PATH_DOCUMENTED"] = (
        "PASS"
        if phase5_markers["TADSR_OFFICIAL_ACTUAL_PATH_SINGLE_STEP_CONTRACT"] == "PASS"
        and actual_path_audit.get("official_actual_path_type") == "single_step_get_x0_from_res"
        else "BLOCKED"
    )
    phase5_markers["TADSR_MULTISTEP_NOT_REQUIRED_FOR_OFFICIAL_ACTUAL_PATH"] = (
        "PASS"
        if phase5_markers["TADSR_TINY_MULTISTEP_ALIGNMENT_APPLICABILITY"] == "NOT_REQUIRED_FOR_OFFICIAL_ACTUAL_INFERENCE"
        and actual_path_audit.get("multi_step_required_for_official_actual_inference") is False
        else "BLOCKED"
    )
    one_step_evidence = load_json(ONE_STEP_JITTOR_JSON)
    phase5_markers["TADSR_CONTROLLED_ONE_STEP_EVIDENCE_DOCUMENTED"] = (
        "PASS"
        if one_step_markers.get("TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT") == "PASS"
        and bool(one_step_evidence.get("stage_metrics"))
        else "BLOCKED"
    )
    phase5_markers["TADSR_FINAL_SUBMISSION_AFTER_PHASE5B_READY"] = (
        "PASS"
        if phase5_markers["TADSR_PHASE5B_SUBMISSION_FREEZE_SUMMARY"] == "PASS"
        and phase5_markers["TADSR_DIAGNOSTIC_IMAGE_SMOKE_PLAN"] == "PASS"
        and phase5_markers["TADSR_DIAGNOSTIC_IMAGE_SMOKE_EXECUTED"] in {"NOT_EXECUTED", "PASS"}
        and phase5_markers["TADSR_OFFICIAL_ACTUAL_SINGLE_STEP_PATH_DOCUMENTED"] == "PASS"
        and phase5_markers["TADSR_MULTISTEP_NOT_REQUIRED_FOR_OFFICIAL_ACTUAL_PATH"] == "PASS"
        and phase5_markers["TADSR_CONTROLLED_ONE_STEP_EVIDENCE_DOCUMENTED"] == "PASS"
        and must_remain_ok
        else "BLOCKED"
    )
    phase5_markers["TADSR_FINAL_SUBMISSION_WITH_DIAGNOSTIC_SMOKE_READY"] = (
        "PASS"
        if phase5_markers["TADSR_FINAL_SUBMISSION_AFTER_PHASE5B_READY"] == "PASS"
        and phase5_markers["TADSR_DIAGNOSTIC_IMAGE_SMOKE_VALIDATION"] in {"PASS", "PARTIAL", "BLOCKED"}
        and phase5_markers["TADSR_DIAGNOSTIC_IMAGE_SMOKE_FALSE_CLAIM_GUARD"] == "PASS"
        and phase5_markers["TADSR_DIAGNOSTIC_IMAGE_SMOKE_ARTIFACT_POLICY"] == "PASS"
        and phase5_markers["TADSR_ONE_STEP_DIAGNOSTIC_CLI_READY"] == "PASS"
        and phase5_markers["TADSR_ONE_STEP_DIAGNOSTIC_CLI_NOT_FULL_INFERENCE"] == "PASS"
        and phase5_markers["TADSR_TIMEVAE_FULL_ALIGNMENT_CLOSURE_SUMMARY"] == "PASS"
        and phase5_markers["TADSR_TIMEVAE_FULL_ALIGNMENT_REMAINS_SCOPED"] == "PASS"
        and phase5_markers["TADSR_RUNTIME_LORA_FINAL_DECISION_PROOF"] == "PASS"
        and phase5_markers["TADSR_DYNAMIC_RUNTIME_LORA_NOT_REQUIRED_FOR_AUDITED_PATH"] == "PASS"
        and must_remain_ok
        else "BLOCKED"
    )
    markers.update(phase5_markers)
    phase5_ready = (
        phase5_markers["TADSR_OFFICIAL_ACTUAL_INFERENCE_PATH_AUDIT"] == "PASS"
        and phase5_markers["TADSR_OFFICIAL_INFERENCE_MULTISTEP_REQUIREMENT_AUDIT"] == "PASS"
        and phase5_markers["TADSR_POSTPROCESS_CONTRACT_AUDIT"] in {"PASS", "PARTIAL"}
        and phase5_markers["TADSR_POSTPROCESS_NOT_EXECUTED_GUARD"] == "PASS"
        and phase5_markers["TADSR_TINY_MULTISTEP_ALIGNMENT_DECISION"] == "PASS"
        and phase5_markers["TADSR_EXPERIMENTAL_CLI_METADATA_ONLY_PLAN"] == "PASS"
        and phase5_markers["TADSR_FINAL_SUBMISSION_AFTER_PHASE5B_READY"] == "PASS"
        and phase5_markers["TADSR_FINAL_SUBMISSION_WITH_DIAGNOSTIC_SMOKE_READY"] == "PASS"
    )
    if phase5_ready:
        phase3_status = "PASS"
        markers["TADSR_PRODUCTION_COMPLETION_PHASE3_VALIDATION"] = "PASS"
        if phase5_markers["TADSR_TINY_MULTISTEP_ALIGNMENT_APPLICABILITY"] == "NOT_REQUIRED_FOR_OFFICIAL_ACTUAL_INFERENCE":
            if phase5_markers["TADSR_DIAGNOSTIC_IMAGE_SMOKE_EXECUTED"] == "PASS":
                next_action = (
                    "Diagnostic one-step image-smoke evidence is available, but it remains diagnostic only. "
                    "The final submission may proceed; keep full inference guarded."
                )
            else:
                next_action = (
                    "Diagnostic image-smoke is honestly blocked/partial because local tensors are not available in this checkout. "
                    "The final submission may proceed with documented blocker, or rerun the controlled local-tensor export on Linux before video if desired."
                )
        elif phase5_markers["TADSR_TINY_MULTISTEP_ALIGNMENT_APPLICABILITY"] == "REQUIRED_NEXT":
            next_action = "Official actual inference requires multi-step; next phase should be tiny two-step latent-only alignment with no image/video generation."
        else:
            next_action = "Resolve official actual inference path classification before any multi-step work."
    payload = {
        "status_marker": "TADSR_PRODUCTION_COMPLETION_PHASE3_VALIDATION",
        "status": phase3_status,
        "markers": markers,
        "official_env_resolution_status": env_status,
        "live_timevae_audit_status": timevae_audit_status,
        "timevae_oracle_metadata_status": oracle_status,
        "timevae_preflight_status": preflight_status,
        "lora_live_audit_status": lora_status,
        "full_inference_metadata_contract_status": contract_status,
        "remote_connectivity_status": remote.get("status", "NOT_RECORDED") if remote else "NOT_RECORDED",
        "remote_tcp_10022_succeeded": remote.get("tcp_10022_succeeded") if remote else None,
        "remote_ssh_authentication_available": remote.get("ssh_authentication_available") if remote else None,
        "phase3b_manual_execution_kit": manual_kit,
        "phase3b_manual_execution_kit_ready": manual_kit_ready,
        "phase3b_final_manual_handoff_status": markers["TADSR_PHASE3B_FINAL_MANUAL_HANDOFF_READY"],
        "phase3b_import_validation_status": markers["TADSR_PHASE3B_LIVE_RESULTS_IMPORT_VALIDATION"],
        "phase3b_import_no_raw_tensors_status": markers["TADSR_PHASE3B_LIVE_RESULTS_NO_RAW_TENSORS"],
        "timevae_metadata_partial_diagnosis_status": markers["TADSR_TIMEVAE_METADATA_PARTIAL_DIAGNOSIS"],
        "timevae_live_metadata_completion_status": markers["TADSR_TIMEVAE_LIVE_METADATA_COMPLETION"],
        "timevae_live_encode_metadata_status": markers["TADSR_TIMEVAE_LIVE_ENCODE_METADATA"],
        "timevae_live_decode_metadata_status": markers["TADSR_TIMEVAE_LIVE_DECODE_METADATA"],
        "timevae_live_safety_flags_status": markers["TADSR_TIMEVAE_LIVE_SAFETY_FLAGS"],
        "official_diffusers_overlay_ready_status": markers["TADSR_OFFICIAL_DIFFUSERS_OVERLAY_READY"],
        "official_runtime_dependency_repair_readiness_status": markers["TADSR_OFFICIAL_RUNTIME_DEPENDENCY_REPAIR_READINESS"],
        "official_runtime_dependency_overlay_active_status": markers["TADSR_OFFICIAL_RUNTIME_DEPENDENCY_OVERLAY_ACTIVE"],
        "official_runtime_dependency_diagnosis_status": markers["TADSR_OFFICIAL_RUNTIME_DEPENDENCY_DIAGNOSIS"],
        "timevae_metadata_repair_attempt_status": markers["TADSR_TIMEVAE_METADATA_REPAIR_ATTEMPT"],
        "timevae_metadata_fields_complete_status": markers["TADSR_TIMEVAE_METADATA_FIELDS_COMPLETE"],
        "timevae_metadata_ready_for_one_step_status": markers["TADSR_TIMEVAE_METADATA_READY_FOR_ONE_STEP"],
        "runtime_lora_fixed_inference_decision_status": markers["TADSR_RUNTIME_LORA_DECISION_FINALIZED_FOR_FIXED_INFERENCE"],
        "smoke_training_submission_summary_status": markers["TADSR_SMOKE_TRAINING_SUBMISSION_SUMMARY"],
        "phase3d_import_gate": phase3d,
        "one_step_execution_markers": one_step_markers,
        "phase5_actual_path_markers": phase5_markers,
        "official_actual_path_type": actual_path_audit.get("official_actual_path_type"),
        "multi_step_required_for_official_actual_inference": actual_path_audit.get("multi_step_required_for_official_actual_inference"),
        "phase5b_submission_freeze_status": phase5_markers["TADSR_PHASE5B_SUBMISSION_FREEZE_SUMMARY"],
        "diagnostic_image_smoke_plan_status": phase5_markers["TADSR_DIAGNOSTIC_IMAGE_SMOKE_PLAN"],
        "diagnostic_image_smoke_ready_status": phase5_markers["TADSR_DIAGNOSTIC_IMAGE_SMOKE_READY"],
        "diagnostic_image_smoke_executed_status": phase5_markers["TADSR_DIAGNOSTIC_IMAGE_SMOKE_EXECUTED"],
        "diagnostic_image_smoke_validation_status": phase5_markers["TADSR_DIAGNOSTIC_IMAGE_SMOKE_VALIDATION"],
        "one_step_diagnostic_cli_status": phase5_markers["TADSR_ONE_STEP_DIAGNOSTIC_CLI_READY"],
        "timevae_closure_status": phase5_markers["TADSR_TIMEVAE_FULL_ALIGNMENT_CLOSURE_SUMMARY"],
        "runtime_lora_decision_status": phase5_markers["TADSR_RUNTIME_LORA_FINAL_DECISION_PROOF"],
        "one_step_execution_ready": one_step_execution_ready,
        "one_step_execution_partial": one_step_execution_partial,
        "one_step_alignment_plan_status": one_step_plan_marker,
        "one_step_alignment_plan_json": str(ONE_STEP_PLAN_JSON.relative_to(ROOT)),
        "one_step_alignment_plan_md": str(ONE_STEP_PLAN_MD.relative_to(ROOT)),
        "ready_for_one_step_tensor_alignment": ready_for_one_step,
        "must_remain_statuses": must_remain,
        "must_remain_statuses_preserved": must_remain_ok,
        "metadata_only_safety_preserved": no_execution_claims,
        "blockers": blockers,
        "next_required_action": next_action,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_md(OUT_MD, "Production completion Phase 3 validation", payload, markers)

    blocker_payload = load_json(BLOCKER_JSON)
    blocker_payload.update(
        {
            "status_marker": "TADSR_PRODUCTION_COMPLETION_BLOCKER_REPORT",
            "status": "PASS",
            "official_env_resolution_status": env_status,
            "live_timevae_audit_status": timevae_audit_status,
            "timevae_oracle_metadata_status": oracle_status,
            "timevae_preflight_status": preflight_status,
            "lora_live_audit_status": lora_status,
            "full_inference_metadata_contract_status": contract_status,
            "remote_connectivity_status": remote.get("status", "NOT_RECORDED") if remote else "NOT_RECORDED",
            "remote_tcp_10022_succeeded": remote.get("tcp_10022_succeeded") if remote else None,
            "remote_ssh_authentication_available": remote.get("ssh_authentication_available") if remote else None,
            "phase3b_manual_execution_kit_ready": manual_kit_ready,
            "phase3b_final_manual_handoff_status": markers["TADSR_PHASE3B_FINAL_MANUAL_HANDOFF_READY"],
            "phase3b_import_validation_status": markers["TADSR_PHASE3B_LIVE_RESULTS_IMPORT_VALIDATION"],
            "phase3b_import_no_raw_tensors_status": markers["TADSR_PHASE3B_LIVE_RESULTS_NO_RAW_TENSORS"],
            "timevae_metadata_partial_diagnosis_status": markers["TADSR_TIMEVAE_METADATA_PARTIAL_DIAGNOSIS"],
            "timevae_live_metadata_completion_status": markers["TADSR_TIMEVAE_LIVE_METADATA_COMPLETION"],
            "timevae_live_encode_metadata_status": markers["TADSR_TIMEVAE_LIVE_ENCODE_METADATA"],
            "timevae_live_decode_metadata_status": markers["TADSR_TIMEVAE_LIVE_DECODE_METADATA"],
            "timevae_live_safety_flags_status": markers["TADSR_TIMEVAE_LIVE_SAFETY_FLAGS"],
            "official_diffusers_overlay_ready_status": markers["TADSR_OFFICIAL_DIFFUSERS_OVERLAY_READY"],
            "official_runtime_dependency_repair_readiness_status": markers["TADSR_OFFICIAL_RUNTIME_DEPENDENCY_REPAIR_READINESS"],
            "official_runtime_dependency_overlay_active_status": markers["TADSR_OFFICIAL_RUNTIME_DEPENDENCY_OVERLAY_ACTIVE"],
            "official_runtime_dependency_diagnosis_status": markers["TADSR_OFFICIAL_RUNTIME_DEPENDENCY_DIAGNOSIS"],
            "timevae_metadata_repair_attempt_status": markers["TADSR_TIMEVAE_METADATA_REPAIR_ATTEMPT"],
            "timevae_metadata_fields_complete_status": markers["TADSR_TIMEVAE_METADATA_FIELDS_COMPLETE"],
            "timevae_metadata_ready_for_one_step_status": markers["TADSR_TIMEVAE_METADATA_READY_FOR_ONE_STEP"],
            "runtime_lora_fixed_inference_decision_status": markers["TADSR_RUNTIME_LORA_DECISION_FINALIZED_FOR_FIXED_INFERENCE"],
            "smoke_training_submission_summary_status": markers["TADSR_SMOKE_TRAINING_SUBMISSION_SUMMARY"],
            "phase3d_import_gate": phase3d["markers"]["TADSR_PHASE3D_IMPORT_GATE"],
            "phase3d_one_step_gate_decision": phase3d["markers"]["TADSR_PHASE3D_ONE_STEP_GATE_DECISION"],
            "one_step_alignment_plan_status": one_step_plan_marker,
            "ready_for_one_step_tensor_alignment": ready_for_one_step,
            "next_required_action": next_action,
            "phase3_status": {
                "official_env_resolution_status": env_status,
                "live_timevae_audit_status": timevae_audit_status,
                "timevae_oracle_metadata_status": oracle_status,
                "timevae_preflight_status": preflight_status,
                "lora_live_audit_status": lora_status,
                "full_inference_metadata_contract_status": contract_status,
                "remote_connectivity_status": remote.get("status", "NOT_RECORDED") if remote else "NOT_RECORDED",
                "remote_tcp_10022_succeeded": remote.get("tcp_10022_succeeded") if remote else None,
                "remote_ssh_authentication_available": remote.get("ssh_authentication_available") if remote else None,
                "phase3b_manual_execution_kit_ready": manual_kit_ready,
                "phase3b_final_manual_handoff_status": markers["TADSR_PHASE3B_FINAL_MANUAL_HANDOFF_READY"],
                "phase3b_import_validation_status": markers["TADSR_PHASE3B_LIVE_RESULTS_IMPORT_VALIDATION"],
                "phase3b_import_no_raw_tensors_status": markers["TADSR_PHASE3B_LIVE_RESULTS_NO_RAW_TENSORS"],
                "timevae_metadata_partial_diagnosis_status": markers["TADSR_TIMEVAE_METADATA_PARTIAL_DIAGNOSIS"],
                "timevae_live_metadata_completion_status": markers["TADSR_TIMEVAE_LIVE_METADATA_COMPLETION"],
                "timevae_live_encode_metadata_status": markers["TADSR_TIMEVAE_LIVE_ENCODE_METADATA"],
                "timevae_live_decode_metadata_status": markers["TADSR_TIMEVAE_LIVE_DECODE_METADATA"],
                "timevae_live_safety_flags_status": markers["TADSR_TIMEVAE_LIVE_SAFETY_FLAGS"],
                "official_diffusers_overlay_ready_status": markers["TADSR_OFFICIAL_DIFFUSERS_OVERLAY_READY"],
                "official_runtime_dependency_repair_readiness_status": markers["TADSR_OFFICIAL_RUNTIME_DEPENDENCY_REPAIR_READINESS"],
                "official_runtime_dependency_overlay_active_status": markers["TADSR_OFFICIAL_RUNTIME_DEPENDENCY_OVERLAY_ACTIVE"],
                "official_runtime_dependency_diagnosis_status": markers["TADSR_OFFICIAL_RUNTIME_DEPENDENCY_DIAGNOSIS"],
                "timevae_metadata_repair_attempt_status": markers["TADSR_TIMEVAE_METADATA_REPAIR_ATTEMPT"],
                "timevae_metadata_fields_complete_status": markers["TADSR_TIMEVAE_METADATA_FIELDS_COMPLETE"],
                "timevae_metadata_ready_for_one_step_status": markers["TADSR_TIMEVAE_METADATA_READY_FOR_ONE_STEP"],
                "runtime_lora_fixed_inference_decision_status": markers["TADSR_RUNTIME_LORA_DECISION_FINALIZED_FOR_FIXED_INFERENCE"],
                "smoke_training_submission_summary_status": markers["TADSR_SMOKE_TRAINING_SUBMISSION_SUMMARY"],
                "phase3d_import_gate": phase3d["markers"]["TADSR_PHASE3D_IMPORT_GATE"],
                "phase3d_one_step_gate_decision": phase3d["markers"]["TADSR_PHASE3D_ONE_STEP_GATE_DECISION"],
                "ready_for_one_step_tensor_alignment": ready_for_one_step,
                "one_step_alignment_plan_status": one_step_plan_marker,
                "next_required_action": next_action,
            },
            "phase3_blockers": blockers,
        }
    )
    BLOCKER_JSON.parent.mkdir(parents=True, exist_ok=True)
    BLOCKER_JSON.write_text(json.dumps(blocker_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_md(BLOCKER_MD, "Production completion blocker report", blocker_payload, {"TADSR_PRODUCTION_COMPLETION_BLOCKER_REPORT": "PASS"})

    for key, value in markers.items():
        print(f"{key}: {value}")
    return 0 if phase3_status in {"PASS", "PASS_WITH_BLOCKERS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
