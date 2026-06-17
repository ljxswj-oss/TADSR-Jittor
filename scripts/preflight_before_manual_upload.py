#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess
from pathlib import Path

OUT_MD = Path("experiments/full_repro/preflight_before_manual_upload.md")
OUT_JSON = Path("experiments/full_repro/preflight_before_manual_upload.json")


def sh(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.returncode, p.stdout.strip()


def main() -> int:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    code, commit = sh(["git", "rev-parse", "--short", "HEAD"])
    code2, status = sh(["git", "status", "--short"])
    final_code, final_out = sh(["python3", "scripts/final_audit.py"])
    checks = [
        ("git_commit", commit if code == 0 else "unknown", code == 0),
        ("git_worktree_clean", "clean" if not status else status, not bool(status)),
        ("official_venv_exists", "/mnt/data/sj/venvs/tadsr_official_pytorch", Path("/mnt/data/sj/venvs/tadsr_official_pytorch").exists()),
        ("official_env_installed", "env_check_strict.json PASS", json.loads(Path("experiments/full_repro/pytorch_official/env/env_check_strict.json").read_text()).get("status") == "PASS" if Path("experiments/full_repro/pytorch_official/env/env_check_strict.json").exists() else False),
        ("wheelhouse_exists", "/mnt/data/sj/wheelhouse/tadsr_official_pytorch", any(Path("/mnt/data/sj/wheelhouse/tadsr_official_pytorch").glob("*")) if Path("/mnt/data/sj/wheelhouse/tadsr_official_pytorch").exists() else False),
        ("incoming_dir_exists", "/mnt/data/sj/incoming", Path("/mnt/data/sj/incoming").exists()),
        ("checkpoints_dir_exists", "/mnt/data/sj/checkpoints/TADSR/preset/weights", Path("/mnt/data/sj/checkpoints/TADSR/preset/weights").exists()),
        ("datasets_dir_exists", "/mnt/data/sj/datasets/TADSR", Path("/mnt/data/sj/datasets/TADSR").exists()),
        ("run_after_assets_ready_exists", "scripts/run_after_assets_ready.sh", Path("scripts/run_after_assets_ready.sh").exists()),
        ("verify_official_assets_exists", "scripts/verify_official_assets.py", Path("scripts/verify_official_assets.py").exists()),
        ("final_audit_runs", "python3 scripts/final_audit.py", final_code == 0),
    ]
    data = {"checks": [{"name": n, "detail": d, "ok": ok} for n, d, ok in checks]}
    data["next_user_action"] = [
        "Download zty557/TADSR weights on a networked machine.",
        "Put them into /mnt/data/sj/incoming/TADSR_assets/TADSR_weights/",
        "If official environment is still missing, prepare wheelhouse in /mnt/data/sj/wheelhouse/tadsr_official_pytorch/",
        "Run: cd /mnt/data/sj/projects/TADSR-Jittor && bash scripts/run_after_assets_ready.sh",
    ]
    OUT_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")
    lines = ["# Preflight Before Manual Upload", "", "| Check | OK | Detail |", "|---|---:|---|"]
    for row in data["checks"]:
        lines.append(f"| {row['name']} | {row['ok']} | `{row['detail']}` |")
    lines += ["", "## NEXT USER ACTION", "", "1. Download zty557/TADSR weights on a networked machine.", "2. Put them into `/mnt/data/sj/incoming/TADSR_assets/TADSR_weights/`.", "3. If official environment is still missing, prepare wheelhouse in `/mnt/data/sj/wheelhouse/tadsr_official_pytorch/`.", "4. Run:", "", "```bash", "cd /mnt/data/sj/projects/TADSR-Jittor", "bash scripts/run_after_assets_ready.sh", "```"]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
