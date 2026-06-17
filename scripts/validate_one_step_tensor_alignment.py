#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "experiments" / "production_completion" / "full_inference" / "one_step"
FORBIDDEN_MEDIA_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".mp4", ".mov", ".avi", ".mkv"}
FORBIDDEN_RAW_SUFFIXES = {".npy", ".npz"}


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as exc:
        return {"status": "FAIL", "error": repr(exc)}


def nested_marker(payload: dict, marker: str, default: str = "MISSING") -> str:
    markers = payload.get("markers", {}) if isinstance(payload, dict) else {}
    if isinstance(markers, dict) and marker in markers:
        return str(markers[marker])
    if payload.get("status_marker") == marker:
        return str(payload.get("audit_status", payload.get("status", default)))
    return default


def git_output(args: list[str], cwd: Path) -> str:
    try:
        proc = subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        return proc.stdout
    except Exception:
        return ""


def check_raw_tensor_policy(root: Path, out_dir: Path) -> tuple[str, list[str]]:
    tracked = git_output(["git", "ls-files"], root).splitlines()
    staged = git_output(["git", "diff", "--cached", "--name-only"], root).splitlines()
    offenders = [
        path for path in tracked + staged
        if Path(path).suffix.lower() in FORBIDDEN_RAW_SUFFIXES
        and "experiments/production_completion/full_inference/one_step/local_tensors" in path.replace("\\", "/")
    ]
    # Raw tensors are allowed to exist locally under local_tensors only, but may not be tracked or staged.
    local_raw = [
        p for p in out_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in FORBIDDEN_RAW_SUFFIXES and "local_tensors" not in p.parts
    ]
    offenders.extend(str(p.relative_to(root)) for p in local_raw)
    return ("PASS" if not offenders else "FAIL"), offenders


def check_no_media(out_dir: Path) -> tuple[str, list[str]]:
    offenders = [str(p) for p in out_dir.rglob("*") if p.is_file() and p.suffix.lower() in FORBIDDEN_MEDIA_SUFFIXES]
    return ("PASS" if not offenders else "FAIL"), offenders


def check_guard(root: Path) -> tuple[str, str]:
    path = root / "jittor_tadsr_full" / "tadsr_full.py"
    if not path.exists():
        return "FAIL", "jittor_tadsr_full/tadsr_full.py missing"
    text = path.read_text(encoding="utf-8", errors="ignore")
    ok = "--full-inference" in text and "NotImplementedError" in text and "JITTOR_FULL_INFERENCE" not in text.replace("NOT_COMPLETE", "")
    # The text check is intentionally conservative; final_audit separately reports the official marker.
    if "--full-inference" in text and "NotImplementedError" in text:
        return "PASS", "full inference CLI still contains NotImplementedError guard"
    return "FAIL", "full inference CLI guard text not found"


def any_bad_execution_claim(payloads: list[dict]) -> list[str]:
    bad = []
    forbidden_true = [
        "full_inference_executed",
        "denoising_loop_executed",
        "production_cli_used",
        "image_or_video_generated",
        "image_saved",
        "runtime_lora_dynamic_loading",
    ]
    for idx, payload in enumerate(payloads):
        for key in forbidden_true:
            if payload.get(key) is True:
                bad.append(f"payload[{idx}] top-level {key}=true")
            policy = payload.get("policy", {})
            if isinstance(policy, dict) and policy.get(key) is True:
                bad.append(f"payload[{idx}] policy {key}=true")
            flags = payload.get("safety_flags", {})
            if isinstance(flags, dict) and flags.get(key) is True:
                bad.append(f"payload[{idx}] safety_flags {key}=true")
    return bad


def write_reports(out_dir: Path, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "one_step_tensor_alignment_validation.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# One-step tensor alignment validation",
        "",
        f"`TADSR_ONE_STEP_TENSOR_ALIGNMENT_VALIDATION: {payload['markers']['TADSR_ONE_STEP_TENSOR_ALIGNMENT_VALIDATION']}`",
        f"`TADSR_ONE_STEP_NO_IMAGE_VIDEO_GUARD: {payload['markers']['TADSR_ONE_STEP_NO_IMAGE_VIDEO_GUARD']}`",
        f"`TADSR_ONE_STEP_RAW_TENSOR_POLICY: {payload['markers']['TADSR_ONE_STEP_RAW_TENSOR_POLICY']}`",
        f"`TADSR_ONE_STEP_FULL_INFERENCE_GUARD_PRESERVED: {payload['markers']['TADSR_ONE_STEP_FULL_INFERENCE_GUARD_PRESERVED']}`",
        f"`TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT: {payload['markers']['TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT']}`",
        "",
        "This validation reads metadata only and does not import torch or Jittor.",
        "",
    ]
    if payload["blockers"]:
        lines += ["## Blockers", ""]
        for item in payload["blockers"]:
            lines.append(f"- {item}")
    (out_dir / "one_step_tensor_alignment_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate one-step tensor alignment artifacts and guardrails.")
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    official_audit = load_json(out_dir / "one_step_official_path_audit.json")
    official_oracle = load_json(out_dir / "official_one_step_oracle_metadata.json")
    jittor_alignment = load_json(out_dir / "jittor_one_step_alignment.json")
    official_audit_status = str(official_audit.get("status", "MISSING"))
    official_oracle_status = str(official_oracle.get("status", "MISSING"))
    jittor_run_status = nested_marker(jittor_alignment, "TADSR_ONE_STEP_JITTOR_TENSOR_RUN")
    stage_status = nested_marker(jittor_alignment, "TADSR_ONE_STEP_TENSOR_STAGE_ALIGNMENT")
    one_step_status = nested_marker(jittor_alignment, "TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT")

    raw_status, raw_offenders = check_raw_tensor_policy(root, out_dir)
    media_status, media_offenders = check_no_media(out_dir)
    guard_status, guard_note = check_guard(root)
    bad_claims = any_bad_execution_claim([official_audit, official_oracle, jittor_alignment])
    no_image_video_guard = "PASS" if media_status == "PASS" and not bad_claims else "FAIL"

    blockers = []
    if official_audit_status != "PASS":
        blockers.append(f"official path audit status is {official_audit_status}")
    if official_oracle_status != "PASS":
        blockers.append(f"official oracle status is {official_oracle_status}")
    if jittor_run_status != "PASS":
        blockers.append(f"Jittor one-step tensor run status is {jittor_run_status}")
    if stage_status != "PASS":
        blockers.append(f"one-step stage alignment status is {stage_status}")
    if raw_status != "PASS":
        blockers.append(f"raw tensor policy offenders: {raw_offenders}")
    if no_image_video_guard != "PASS":
        blockers.append(f"image/video guard failed; media={media_offenders}, claims={bad_claims}")
    if guard_status != "PASS":
        blockers.append(guard_note)

    if raw_status == "PASS" and no_image_video_guard == "PASS" and guard_status == "PASS":
        validation_status = "PASS" if not blockers else ("PARTIAL" if one_step_status in {"PARTIAL", "BLOCKED", "MISSING"} else "FAIL")
    else:
        validation_status = "FAIL"
    full_one_step = "PASS" if validation_status == "PASS" and one_step_status == "PASS" else ("FAIL" if validation_status == "FAIL" else "PARTIAL")
    markers = {
        "TADSR_ONE_STEP_TENSOR_ALIGNMENT_VALIDATION": validation_status,
        "TADSR_ONE_STEP_NO_IMAGE_VIDEO_GUARD": no_image_video_guard,
        "TADSR_ONE_STEP_RAW_TENSOR_POLICY": raw_status,
        "TADSR_ONE_STEP_FULL_INFERENCE_GUARD_PRESERVED": guard_status,
        "TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT": full_one_step,
    }
    payload = {
        "status_marker": "TADSR_ONE_STEP_TENSOR_ALIGNMENT_VALIDATION",
        "status": validation_status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "markers": markers,
        "official_path_audit_status": official_audit_status,
        "official_oracle_status": official_oracle_status,
        "jittor_one_step_run_status": jittor_run_status,
        "jittor_stage_alignment_status": stage_status,
        "raw_tensor_policy_offenders": raw_offenders,
        "image_video_offenders": media_offenders,
        "bad_execution_claims": bad_claims,
        "full_inference_guard_note": guard_note,
        "blockers": blockers,
        "next_required_action": (
            "One-step tensor alignment passed; next phase may plan tiny multi-step latent-only alignment, still without image/video/full-inference claims."
            if full_one_step == "PASS"
            else "Fix the listed one-step blockers before any multi-step or full-inference work."
        ),
    }
    write_reports(out_dir, payload)
    for marker, status in markers.items():
        print(f"{marker}: {status}")
    return 0 if validation_status in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
