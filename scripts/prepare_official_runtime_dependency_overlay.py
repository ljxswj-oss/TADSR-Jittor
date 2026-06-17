#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "production_completion" / "env"
OUT_JSON = OUT_DIR / "official_runtime_dependency_overlay.json"
OUT_MD = OUT_DIR / "official_runtime_dependency_overlay.md"


def bool_arg(value: str | int | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def subprocess_env(overlay: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if overlay:
        old = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(overlay) + (os.pathsep + old if old else "")
    return env


def run_python_import(python: Path, module: str, overlay: Path | None = None) -> dict[str, Any]:
    if not python.exists():
        return {"module": module, "status": "BLOCKED", "version": None, "error": "official_python_missing"}
    code = (
        "import importlib, json; "
        f"m=importlib.import_module({module!r}); "
        "print(json.dumps({'version': getattr(m, '__version__', 'unknown')}))"
    )
    proc = subprocess.run(
        [str(python), "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        env=subprocess_env(overlay),
    )
    if proc.returncode == 0:
        try:
            parsed = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception:
            parsed = {"version": "unknown"}
        return {"module": module, "status": "PASS", "version": parsed.get("version", "unknown"), "error": None}
    return {"module": module, "status": "MISSING", "version": None, "error": proc.stdout.strip()[-2000:]}


def detect_version(official_repo: Path, package: str, requested: str) -> tuple[str | None, str]:
    if requested and requested != "auto":
        return requested, "cli"
    if not official_repo.exists():
        return None, "official_repo_missing"

    patterns = [
        re.compile(rf"{re.escape(package)}\s*==\s*([A-Za-z0-9_.!+\-]+)"),
        re.compile(rf"{re.escape(package)}\s*>=\s*([A-Za-z0-9_.!+\-]+)"),
        re.compile(rf"{re.escape(package)}\s*=\s*([A-Za-z0-9_.!+\-]+)"),
    ]
    candidate_files = []
    for name in ["requirements.txt", "environment.yml", "environment.yaml", "setup.py", "pyproject.toml"]:
        candidate_files.extend(official_repo.rglob(name))
    for path in candidate_files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                return match.group(1), f"{path.relative_to(official_repo)}"
    return None, "fallback_unpinned"


def pip_install_target(
    python: Path,
    overlay: Path,
    package: str,
    version: str | None,
    no_deps: bool,
    allow_deps: bool,
) -> dict[str, Any]:
    overlay.mkdir(parents=True, exist_ok=True)
    spec = f"{package}=={version}" if version else package
    cmd = [str(python), "-m", "pip", "install", "--target", str(overlay)]
    if no_deps and not allow_deps:
        cmd.append("--no-deps")
    cmd.append(spec)
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "status": "PASS" if proc.returncode == 0 else "BLOCKED",
        "output_tail": proc.stdout.strip()[-4000:],
    }


def write_reports(payload: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Official runtime dependency overlay",
        "",
        f"`TADSR_OFFICIAL_DIFFUSERS_OVERLAY_READY: {payload['markers']['TADSR_OFFICIAL_DIFFUSERS_OVERLAY_READY']}`",
        f"`TADSR_OFFICIAL_RUNTIME_DEPENDENCY_REPAIR_READINESS: {payload['markers']['TADSR_OFFICIAL_RUNTIME_DEPENDENCY_REPAIR_READINESS']}`",
        "",
        f"- `official_python`: `{payload['official_python']}`",
        f"- `overlay_dir`: `{payload['overlay_dir']}`",
        f"- `strict_env_modified`: `{payload['strict_env_modified']}`",
        f"- `execute`: `{payload['execute']}`",
        f"- `package`: `{payload['package']}`",
        f"- `version`: `{payload.get('diffusers_version')}`",
        f"- `version_source`: `{payload['version_source']}`",
        f"- `overlay_required`: `{payload['overlay_required']}`",
        f"- `official_import_diffusers_before`: `{payload['official_import_diffusers_before']['status']}`",
        f"- `official_import_diffusers_after_with_overlay`: `{payload['official_import_diffusers_after_with_overlay']['status']}`",
        f"- `safe_to_use_overlay_for_timevae_metadata`: `{payload['safe_to_use_overlay_for_timevae_metadata']}`",
        "",
        "## Planned / executed command",
        "",
        "```text",
        " ".join(payload["install_command"]) if payload["install_command"] else "not required",
        "```",
    ]
    if payload.get("blocker_reason"):
        lines += ["", "## Blocker", "", str(payload["blocker_reason"])]
    if payload.get("install_output_tail"):
        lines += ["", "## pip output tail", "", "```text", payload["install_output_tail"], "```"]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--official-weights", required=True)
    parser.add_argument("--official-python", required=True)
    parser.add_argument("--overlay-dir", default="/mnt/data/sj/tmp/tadsr_official_dependency_overlay_diffusers")
    parser.add_argument("--execute", default="0")
    parser.add_argument("--package", default="diffusers")
    parser.add_argument("--version", default="auto")
    parser.add_argument("--no-deps", default="1")
    parser.add_argument("--allow-deps", default="0")
    args = parser.parse_args()

    official_repo = Path(args.official_repo)
    official_weights = Path(args.official_weights)
    official_python = Path(args.official_python)
    overlay = Path(args.overlay_dir)
    execute = bool_arg(args.execute)
    no_deps = bool_arg(args.no_deps)
    allow_deps = bool_arg(args.allow_deps)

    torch_before = run_python_import(official_python, "torch")
    before = run_python_import(official_python, args.package)
    version, version_source = detect_version(official_repo, args.package, args.version)
    install_spec = f"{args.package}=={version}" if version else args.package
    planned = [str(official_python), "-m", "pip", "install", "--target", str(overlay)]
    if no_deps and not allow_deps:
        planned.append("--no-deps")
    planned.append(install_spec)

    install_result: dict[str, Any] = {
        "status": "NOT_REQUIRED" if before["status"] == "PASS" else ("PLAN_READY" if not execute else "NOT_RUN"),
        "command": planned,
        "returncode": None,
        "output_tail": "",
    }
    blocker = ""
    overlay_required = before["status"] != "PASS"

    if not official_python.exists():
        blocker = "official_python_missing"
    elif torch_before["status"] != "PASS":
        blocker = "official_python_cannot_import_torch"
    elif not official_repo.exists():
        blocker = "official_repo_missing"
    elif not official_weights.exists():
        blocker = "official_weights_missing"
    elif overlay_required and execute:
        install_result = pip_install_target(official_python, overlay, args.package, version, no_deps, allow_deps)
        if install_result["status"] != "PASS":
            blocker = "pip_install_failed"

    after = before if not overlay_required else run_python_import(official_python, args.package, overlay if overlay.exists() else None)
    if before["status"] == "PASS":
        overlay_ready = "NOT_REQUIRED"
        repair_ready = "PASS"
    elif after["status"] == "PASS":
        overlay_ready = "PASS"
        repair_ready = "PASS"
    elif not execute and not blocker:
        overlay_ready = "PARTIAL"
        repair_ready = "PARTIAL"
        blocker = "dry_run_only"
    else:
        overlay_ready = "BLOCKED"
        repair_ready = "BLOCKED"
        if not blocker:
            blocker = "overlay_import_failed"

    payload = {
        "status_marker": "TADSR_OFFICIAL_DIFFUSERS_OVERLAY_READY",
        "status": overlay_ready,
        "markers": {
            "TADSR_OFFICIAL_DIFFUSERS_OVERLAY_READY": overlay_ready,
            "TADSR_OFFICIAL_RUNTIME_DEPENDENCY_REPAIR_READINESS": repair_ready,
        },
        "official_repo": str(official_repo),
        "official_weights": str(official_weights),
        "official_python": str(official_python),
        "overlay_dir": str(overlay),
        "package": args.package,
        "requested_version": args.version,
        "diffusers_version": after.get("version") if after["status"] == "PASS" else version,
        "version_source": version_source,
        "official_import_torch": torch_before,
        "official_import_diffusers_before": before,
        "overlay_required": overlay_required,
        "execute": execute,
        "install_command": planned,
        "install_status": install_result["status"],
        "install_returncode": install_result.get("returncode"),
        "install_output_tail": install_result.get("output_tail", ""),
        "official_import_diffusers_after_with_overlay": after,
        "no_deps": no_deps,
        "allow_deps": allow_deps,
        "blocker_reason": blocker,
        "safe_to_use_overlay_for_timevae_metadata": after["status"] == "PASS",
        "strict_env_modified": False,
    }
    write_reports(payload)
    print(f"TADSR_OFFICIAL_DIFFUSERS_OVERLAY_READY: {overlay_ready}")
    print(f"TADSR_OFFICIAL_RUNTIME_DEPENDENCY_REPAIR_READINESS: {repair_ready}")
    return 0 if overlay_ready in {"PASS", "PARTIAL", "NOT_REQUIRED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
