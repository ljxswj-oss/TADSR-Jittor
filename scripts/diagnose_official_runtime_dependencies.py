#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "production_completion" / "env"
OUT_JSON = OUT_DIR / "official_runtime_dependency_diagnosis.json"
OUT_MD = OUT_DIR / "official_runtime_dependency_diagnosis.md"

IMPORTS = ["torch", "diffusers", "peft", "transformers", "accelerate", "safetensors", "PIL", "numpy", "cv2"]
TIMEVAE_HINTS = ["time", "vae", "autoencoder", "pipeline", "tadsr"]
WEIGHT_HINTS = {
    "scheduler": ["scheduler", "ddpm", "dpm"],
    "vae": ["vae", "time_vae", "autoencoder"],
    "unet": ["unet"],
    "lora": ["lora", "adapter"],
}


def overlay_env(overlay: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if overlay:
        old = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = overlay + (os.pathsep + old if old else "")
    return env


def run_import(python: str, module: str, pythonpath_overlay: str | None = None) -> dict[str, Any]:
    if not python or not Path(python).exists():
        return {"module": module, "status": "BLOCKED", "error": "official_python_missing"}
    code = (
        "import importlib, json; "
        f"m=importlib.import_module({module!r}); "
        "print(json.dumps({'ok': True, 'version': getattr(m, '__version__', 'unknown')}))"
    )
    proc = subprocess.run(
        [python, "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        env=overlay_env(pythonpath_overlay),
    )
    if proc.returncode == 0:
        try:
            parsed = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception:
            parsed = {"ok": True, "version": "unknown"}
        return {"module": module, "status": "PASS", "version": parsed.get("version", "unknown")}
    return {"module": module, "status": "MISSING", "error": proc.stdout.strip()[-1000:]}


def find_matching_files(root: Path, hints: list[str], limit: int = 50) -> list[str]:
    if not root.exists():
        return []
    matches: list[str] = []
    for path in root.rglob("*"):
        if len(matches) >= limit:
            break
        name = path.name.lower()
        rel = str(path.relative_to(root)).replace("\\", "/")
        if any(hint in name or hint in rel.lower() for hint in hints):
            matches.append(rel)
    return matches


def write_md(payload: dict[str, Any]) -> None:
    lines = [
        "# Official runtime dependency diagnosis",
        "",
        f"`TADSR_OFFICIAL_RUNTIME_DEPENDENCY_DIAGNOSIS: {payload['status']}`",
        "",
        f"- `official_repo`: `{payload['official_repo']}`",
        f"- `official_weights`: `{payload['official_weights']}`",
        f"- `official_python`: `{payload['official_python']}`",
        f"- `pythonpath_overlay`: `{payload.get('pythonpath_overlay')}`",
        f"- `strict_env_modified`: `{payload.get('strict_env_modified')}`",
        f"`TADSR_OFFICIAL_RUNTIME_DEPENDENCY_OVERLAY_ACTIVE: {payload['markers']['TADSR_OFFICIAL_RUNTIME_DEPENDENCY_OVERLAY_ACTIVE']}`",
        "",
        "## Python imports",
        "",
        "| Module | Status | Version / Error |",
        "|---|---|---|",
    ]
    for item in payload["imports"]:
        detail = item.get("version") or item.get("error", "")
        detail = str(detail).replace("\n", " ")[:180]
        lines.append(f"| `{item['module']}` | `{item['status']}` | {detail} |")
    lines += ["", "## Official repo evidence", ""]
    if payload["official_repo_timevae_candidates"]:
        for item in payload["official_repo_timevae_candidates"][:30]:
            lines.append(f"- `{item}`")
    else:
        lines.append("- no TimeVAE/TADSR candidates found")
    lines += ["", "## Weight candidates", ""]
    for group, items in payload["weight_candidates"].items():
        lines.append(f"- `{group}`: {len(items)} candidate(s)")
        for item in items[:10]:
            lines.append(f"  - `{item}`")
    if payload["dependency_blockers"]:
        lines += ["", "## Dependency blockers", ""]
        for item in payload["dependency_blockers"]:
            lines.append(f"- {item}")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--official-weights", required=True)
    parser.add_argument("--official-python", required=True)
    parser.add_argument("--pythonpath-overlay")
    args = parser.parse_args()

    official_repo = Path(args.official_repo)
    official_weights = Path(args.official_weights)
    official_python = Path(args.official_python)
    overlay = args.pythonpath_overlay
    imports = [run_import(str(official_python), module, overlay) for module in IMPORTS]
    import_status = {item["module"]: item["status"] for item in imports}
    diffusers_plain = run_import(str(official_python), "diffusers")
    overlay_active = "PASS" if overlay and import_status.get("diffusers") == "PASS" and diffusers_plain.get("status") != "PASS" else ("NOT_REQUIRED" if diffusers_plain.get("status") == "PASS" else ("BLOCKED" if overlay else "PARTIAL"))
    blockers: list[str] = []
    if not official_repo.exists():
        blockers.append("official_repo_missing")
    if not official_weights.exists():
        blockers.append("official_weights_missing")
    if not official_python.exists():
        blockers.append("official_python_missing")
    if import_status.get("torch") != "PASS":
        blockers.append("official_python_cannot_import_torch")
    if import_status.get("diffusers") != "PASS":
        blockers.append("official_python_cannot_import_diffusers")

    repo_candidates = find_matching_files(official_repo, TIMEVAE_HINTS)
    weight_candidates = {
        key: find_matching_files(official_weights, hints, limit=30)
        for key, hints in WEIGHT_HINTS.items()
    }
    if not repo_candidates:
        blockers.append("no_timevae_or_tadsr_source_candidates_found")
    if not any(weight_candidates.values()):
        blockers.append("no_scheduler_vae_unet_lora_weight_candidates_found")

    if not blockers:
        status = "PASS"
    elif official_repo.exists() and official_weights.exists() and official_python.exists() and import_status.get("torch") == "PASS":
        status = "PARTIAL"
    else:
        status = "BLOCKED"

    payload: dict[str, Any] = {
        "status_marker": "TADSR_OFFICIAL_RUNTIME_DEPENDENCY_DIAGNOSIS",
        "status": status,
        "markers": {
            "TADSR_OFFICIAL_RUNTIME_DEPENDENCY_DIAGNOSIS": status,
            "TADSR_OFFICIAL_RUNTIME_DEPENDENCY_OVERLAY_ACTIVE": overlay_active,
        },
        "official_repo": str(official_repo),
        "official_weights": str(official_weights),
        "official_python": str(official_python),
        "pythonpath_overlay": overlay,
        "strict_env_modified": False,
        "official_python_can_import_diffusers_without_overlay": diffusers_plain.get("status") == "PASS",
        "official_python_can_import_diffusers_with_overlay": import_status.get("diffusers") == "PASS",
        "imports": imports,
        "import_status": import_status,
        "diffusers_missing": import_status.get("diffusers") != "PASS",
        "official_repo_timevae_candidates": repo_candidates,
        "weight_candidates": weight_candidates,
        "dependency_blockers": blockers,
        "does_not_modify_environment": True,
        "pip_install_executed": False,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_md(payload)
    print(f"TADSR_OFFICIAL_RUNTIME_DEPENDENCY_DIAGNOSIS: {status}")
    return 0 if status in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
