#!/usr/bin/env python3
from __future__ import annotations

import json
import py_compile
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "experiments" / "production_completion" / "phase3b_manual_handoff.json"
OUT_MD = ROOT / "experiments" / "production_completion" / "phase3b_manual_handoff.md"

MANUAL_RUNBOOK = ROOT / "docs" / "production_completion" / "linux_phase3b_manual_runbook.md"
ENV_TEMPLATE = ROOT / "docs" / "production_completion" / "linux_official_env_template.sh"
LIVE_AUDIT_SCRIPT = ROOT / "scripts" / "run_phase3b_live_audit_linux.sh"
PACKAGER = ROOT / "scripts" / "package_phase3b_live_results.py"
IMPORTER = ROOT / "scripts" / "import_phase3b_live_results.py"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as exc:
        return {"status": "FAIL", "error": repr(exc)}


def audit_status(marker: str, default: str = "MISSING") -> str:
    report = load_json(ROOT / "experiments" / "final_audit_report.json")
    for row in report.get("rows", []):
        if row.get("check") == marker:
            return str(row.get("status", default))
    return default


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


def safe_text(path: Path) -> bool:
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
        "id_rsa",
        "id_ed25519",
    ]
    return not any(item in text for item in forbidden)


def write_reports(payload: dict) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    markers = payload["markers"]
    lines = [
        "# Phase 3-B manual handoff validation",
        "",
        f"`TADSR_PHASE3B_FINAL_MANUAL_HANDOFF_READY: {markers['TADSR_PHASE3B_FINAL_MANUAL_HANDOFF_READY']}`",
        f"`TADSR_PHASE3B_MANUAL_EXECUTION_KIT_READY: {markers['TADSR_PHASE3B_MANUAL_EXECUTION_KIT_READY']}`",
        "",
        "## 手动执行入口",
        "",
        "- Linux 登录由用户手动完成；本脚本不尝试 SSH，不保存密码、key 或 token。",
        "- Linux 端执行：`bash scripts/run_phase3b_live_audit_linux.sh`。",
        "- Linux 端打包：`python3 scripts/package_phase3b_live_results.py --repo-root . --output experiments/production_completion/phase3b_live_results_package.zip`。",
        "- Windows 端导入：`python scripts\\import_phase3b_live_results.py --package path\\to\\phase3b_live_results_package.zip --repo-root .`。",
        "",
        "## 文件检查",
        "",
    ]
    for item in payload["files"]:
        lines.append(f"- `{item['path']}`: `{item['status']}`")
    lines += [
        "",
        "## 必须保持的状态",
        "",
    ]
    for key, value in payload["must_remain_statuses"].items():
        lines.append(f"- `{key}`: `{value}`")
    if payload["blockers"]:
        lines += ["", "## Blockers", ""]
        for blocker in payload["blockers"]:
            lines.append(f"- {blocker}")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    files = [
        ("manual runbook", MANUAL_RUNBOOK, MANUAL_RUNBOOK.exists() and safe_text(MANUAL_RUNBOOK)),
        ("official env template", ENV_TEMPLATE, ENV_TEMPLATE.exists() and safe_text(ENV_TEMPLATE) and shell_syntax_ok(ENV_TEMPLATE)),
        ("Linux live audit script", LIVE_AUDIT_SCRIPT, LIVE_AUDIT_SCRIPT.exists() and safe_text(LIVE_AUDIT_SCRIPT) and shell_syntax_ok(LIVE_AUDIT_SCRIPT)),
        ("result packager", PACKAGER, PACKAGER.exists() and safe_text(PACKAGER) and python_syntax_ok(PACKAGER)),
        ("result importer", IMPORTER, IMPORTER.exists() and safe_text(IMPORTER) and python_syntax_ok(IMPORTER)),
    ]
    file_rows = [
        {
            "name": name,
            "path": str(path.relative_to(ROOT)),
            "exists": path.exists(),
            "status": "PASS" if ok else "BLOCKED",
        }
        for name, path, ok in files
    ]
    must_remain = {
        "JITTOR_FULL_INFERENCE": audit_status("JITTOR_FULL_INFERENCE"),
        "JITTOR_FULL_PORT": audit_status("JITTOR_FULL_PORT"),
        "TIME_VAE_FULL_ALIGNMENT": audit_status("TIME_VAE_FULL_ALIGNMENT"),
        "TADSR_DYNAMIC_RUNTIME_LORA_IMPLEMENTATION": audit_status("TADSR_DYNAMIC_RUNTIME_LORA_IMPLEMENTATION"),
    }
    must_remain_ok = (
        must_remain["JITTOR_FULL_INFERENCE"] == "NOT_COMPLETE"
        and must_remain["JITTOR_FULL_PORT"] == "PARTIAL"
        and must_remain["TIME_VAE_FULL_ALIGNMENT"] == "NOT_COMPLETE"
        and must_remain["TADSR_DYNAMIC_RUNTIME_LORA_IMPLEMENTATION"] == "NOT_IMPLEMENTED_BY_DESIGN"
    )
    blockers: list[str] = []
    if not all(row["status"] == "PASS" for row in file_rows):
        blockers.append("manual runbook / Linux script / packager / importer is missing or unsafe")
    if not must_remain_ok:
        blockers.append("must-remain NOT_COMPLETE / NOT_IMPLEMENTED markers changed")

    ready = not blockers
    markers = {
        "TADSR_PHASE3B_MANUAL_EXECUTION_KIT_READY": "PASS" if all(row["status"] == "PASS" for row in file_rows) else "BLOCKED",
        "TADSR_PHASE3B_FINAL_MANUAL_HANDOFF_READY": "PASS" if ready else "BLOCKED",
    }
    payload = {
        "status_marker": "TADSR_PHASE3B_FINAL_MANUAL_HANDOFF_READY",
        "status": markers["TADSR_PHASE3B_FINAL_MANUAL_HANDOFF_READY"],
        "markers": markers,
        "files": file_rows,
        "must_remain_statuses": must_remain,
        "must_remain_statuses_preserved": must_remain_ok,
        "blockers": blockers,
        "next_required_action": "User manually logs into Linux, runs Phase 3-B live audit, packages results, and imports them locally.",
    }
    write_reports(payload)
    for key, value in markers.items():
        print(f"{key}: {value}")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
