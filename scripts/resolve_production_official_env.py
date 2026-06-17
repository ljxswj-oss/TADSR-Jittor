#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "production_completion" / "env"

HISTORICAL = {
    "official_repo": Path("/mnt/data/sj/projects/TADSR_official_pytorch"),
    "official_weights": Path("/mnt/data/sj/checkpoints/TADSR/preset/weights"),
    "official_python": Path("/mnt/data/sj/venvs/tadsr_official_strict_cu118/bin/python"),
}

EXPECTED_REPO_FILES = [
    "test_tadsr.py",
    "train_tadsr.py",
    "tadsr.py",
    "requirements.txt",
]

EXPECTED_WEIGHT_ENTRIES = [
    "time_vae",
    "unet",
    "vae",
    "text_encoder",
    "tokenizer",
    "scheduler",
    "feature_extractor",
    "bert-base-uncased",
    "DAPE.pth",
    "ram_swin_large_14m.pth",
    "tadsr.pkl",
]

CONFIG_CANDIDATES = [
    "model_index.json",
    "config.json",
    "scheduler/scheduler_config.json",
    "time_vae/config.json",
    "unet/config.json",
]

TIMEVAE_WEIGHT_PATTERNS = [
    "time_vae/*.bin",
    "time_vae/*.safetensors",
    "time_vae/*.pth",
    "time_vae/*.pkl",
    "**/*time*vae*.pth",
    "**/*time*vae*.pkl",
    "**/*time*vae*.safetensors",
]

LORA_WEIGHT_PATTERNS = [
    "**/*lora*.pth",
    "**/*lora*.pkl",
    "**/*lora*.safetensors",
    "**/*LoRA*.pth",
]


def resolve(cli_value: str | None, env_name: str, historical_key: str) -> tuple[Path | None, str]:
    if cli_value:
        return Path(cli_value), "cli"
    env_value = os.environ.get(env_name)
    if env_value:
        return Path(env_value), f"env:{env_name}"
    return HISTORICAL[historical_key], "historical_reference"


def run_python(python_path: Path, code: str) -> tuple[bool, str]:
    if not python_path.exists():
        return False, "python path does not exist"
    try:
        proc = subprocess.run(
            [str(python_path), "-c", code],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
    except Exception as exc:
        return False, repr(exc)
    return proc.returncode == 0, proc.stdout.strip()


def list_candidates(root: Path | None, patterns: list[str], limit: int = 25) -> list[str]:
    if not root or not root.exists():
        return []
    found: list[str] = []
    for pattern in patterns:
        for item in sorted(root.glob(pattern)):
            if item.is_file():
                try:
                    found.append(str(item.relative_to(root)))
                except ValueError:
                    found.append(str(item))
            if len(found) >= limit:
                return found
    return found


def exists_candidates(root: Path | None, candidates: list[str]) -> list[str]:
    if not root or not root.exists():
        return []
    present = []
    for name in candidates:
        if (root / name).exists():
            present.append(name)
    return present


def write(payload: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "official_env_resolution.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# Official production environment resolution",
        "",
        f"`TADSR_PRODUCTION_OFFICIAL_ENV_RESOLUTION: {payload['status']}`",
        "",
        "| Item | Status | Detail |",
        "|---|---|---|",
    ]
    for item in payload["checks"]:
        lines.append(f"| {item['name']} | {item['status']} | {item['detail']} |")
    if payload["blocker_reason"]:
        lines += ["", "## Blocker", "", payload["blocker_reason"]]
    lines += ["", "## Next action", "", payload["next_action"]]
    (OUT_DIR / "official_env_resolution.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo")
    parser.add_argument("--official-weights")
    parser.add_argument("--official-python")
    args = parser.parse_args()

    repo, repo_source = resolve(args.official_repo, "TADSR_OFFICIAL_REPO", "official_repo")
    weights, weights_source = resolve(args.official_weights, "TADSR_OFFICIAL_WEIGHTS", "official_weights")
    python_path, python_source = resolve(args.official_python, "TADSR_OFFICIAL_PYTHON", "official_python")

    repo_exists = repo.exists() if repo else False
    weights_exists = weights.exists() if weights else False
    python_exists = python_path.exists() if python_path else False
    repo_missing = [name for name in EXPECTED_REPO_FILES if not repo or not (repo / name).exists()]
    weights_missing = [name for name in EXPECTED_WEIGHT_ENTRIES if not weights or not (weights / name).exists()]
    config_candidates = exists_candidates(weights, CONFIG_CANDIDATES)
    repo_config_candidates = exists_candidates(repo, CONFIG_CANDIDATES)
    timevae_weight_candidates = list_candidates(weights, TIMEVAE_WEIGHT_PATTERNS)
    lora_weight_candidates = list_candidates(weights, LORA_WEIGHT_PATTERNS)
    scheduler_config_exists = bool(weights and (weights / "scheduler" / "scheduler_config.json").exists())

    py_version_ok, py_version = run_python(python_path, "import sys; print(sys.version.split()[0])") if python_path else (False, "missing")
    torch_ok, torch_detail = run_python(python_path, "import torch; print(torch.__version__)") if python_path else (False, "missing")
    diffusers_ok, diffusers_detail = run_python(python_path, "import diffusers; print(diffusers.__version__)") if python_path else (False, "missing")

    checks = [
        {"name": "official_repo_exists", "status": "PASS" if repo_exists else "BLOCKED", "detail": f"{repo} ({repo_source})"},
        {"name": "official_weights_exists", "status": "PASS" if weights_exists else "BLOCKED", "detail": f"{weights} ({weights_source})"},
        {"name": "official_python_exists", "status": "PASS" if python_exists else "BLOCKED", "detail": f"{python_path} ({python_source})"},
        {"name": "official_repo_has_expected_files", "status": "PASS" if repo_exists and not repo_missing else "PARTIAL" if repo_exists else "BLOCKED", "detail": ", ".join(repo_missing) if repo_missing else "all expected files present"},
        {"name": "official_weights_has_expected_files", "status": "PASS" if weights_exists and not weights_missing else "PARTIAL" if weights_exists else "BLOCKED", "detail": ", ".join(weights_missing) if weights_missing else "all expected weights present"},
        {"name": "official_python_version", "status": "PASS" if py_version_ok else "BLOCKED", "detail": py_version},
        {"name": "official_python_can_import_torch", "status": "PASS" if torch_ok else "BLOCKED", "detail": torch_detail},
        {"name": "official_python_can_import_diffusers", "status": "PASS" if diffusers_ok else "PARTIAL", "detail": diffusers_detail},
        {"name": "scheduler_config_exists", "status": "PASS" if scheduler_config_exists else "PARTIAL" if weights_exists else "BLOCKED", "detail": "scheduler/scheduler_config.json" if scheduler_config_exists else "not found"},
        {"name": "model_index_or_config_candidates", "status": "PASS" if config_candidates or repo_config_candidates else "PARTIAL", "detail": ", ".join(config_candidates + repo_config_candidates) or "no config candidates found"},
        {"name": "timevae_weight_candidates", "status": "PASS" if timevae_weight_candidates else "PARTIAL", "detail": ", ".join(timevae_weight_candidates[:5]) or "no TimeVAE weight candidates found"},
        {"name": "lora_weight_candidates", "status": "PASS" if lora_weight_candidates else "PARTIAL", "detail": ", ".join(lora_weight_candidates[:5]) or "no LoRA weight candidates found"},
    ]

    pass_ready = repo_exists and weights_exists and python_exists and torch_ok and not repo_missing and not weights_missing
    partial_ready = repo_exists or weights_exists or python_exists or torch_ok
    status = "PASS" if pass_ready else "PARTIAL" if partial_ready else "BLOCKED"
    blocker = "" if status == "PASS" else "official repo, weights, or strict Python environment are not fully available for live metadata oracle export"
    next_action = (
        "Proceed with live metadata-only TimeVAE oracle export."
        if status == "PASS"
        else "Set TADSR_OFFICIAL_REPO, TADSR_OFFICIAL_WEIGHTS, and TADSR_OFFICIAL_PYTHON, or pass them on the command line before live oracle export."
    )
    payload = {
        "status_marker": "TADSR_PRODUCTION_OFFICIAL_ENV_RESOLUTION",
        "status": status,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "python_executable": sys.executable,
        "official_repo": str(repo) if repo else "",
        "official_weights": str(weights) if weights else "",
        "official_python": str(python_path) if python_path else "",
        "official_python_version": py_version if py_version_ok else "",
        "official_torch_version": torch_detail if torch_ok else "",
        "official_diffusers_version": diffusers_detail if diffusers_ok else "",
        "official_repo_exists": repo_exists,
        "official_weights_exists": weights_exists,
        "official_python_exists": python_exists,
        "official_python_version_works": py_version_ok,
        "official_python_can_import_torch": torch_ok,
        "official_python_can_import_diffusers": diffusers_ok,
        "repo_missing_expected_files": repo_missing,
        "weights_missing_expected_entries": weights_missing,
        "scheduler_config_exists": scheduler_config_exists,
        "model_index_or_config_candidates": config_candidates + repo_config_candidates,
        "timevae_weight_candidates": timevae_weight_candidates,
        "lora_weight_candidates": lora_weight_candidates,
        "checks": checks,
        "blocker_reason": blocker,
        "next_action": next_action,
    }
    write(payload)
    print(f"TADSR_PRODUCTION_OFFICIAL_ENV_RESOLUTION: {status}")
    return 0 if status in {"PASS", "PARTIAL", "BLOCKED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
