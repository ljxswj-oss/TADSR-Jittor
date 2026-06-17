#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import zipfile


ALLOWED_SUFFIXES = {".json", ".md", ".txt", ".log"}
FORBIDDEN_SUFFIXES = {".npy", ".npz"}
FORBIDDEN_PARTS = {".git", "__pycache__", "local_tensors", "checkpoints", "weights"}

INCLUDE_PATTERNS = [
    "experiments/production_completion/env/*",
    "experiments/production_completion/timevae_full/*",
    "experiments/production_completion/runtime_lora/*",
    "experiments/production_completion/full_inference/*",
    "experiments/production_completion/full_inference/one_step/*",
    "experiments/production_completion/phase3_validation.json",
    "experiments/production_completion/phase3_validation.md",
    "experiments/production_completion/blockers/*",
    "experiments/production_completion/live_audit_logs/*",
    "experiments/final_audit_report.json",
    "experiments/final_audit_report.md",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_forbidden(path: Path) -> bool:
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        return True
    return any(part in FORBIDDEN_PARTS for part in path.parts)


def collect_files(root: Path) -> list[Path]:
    files: dict[str, Path] = {}
    for pattern in INCLUDE_PATTERNS:
        for p in root.glob(pattern):
            if not p.is_file():
                continue
            rel = p.relative_to(root)
            if is_forbidden(rel):
                raise SystemExit(f"Forbidden artifact selected for package: {rel}")
            if p.suffix.lower() not in ALLOWED_SUFFIXES:
                continue
            files[str(rel).replace("\\", "/")] = p
    return [files[k] for k in sorted(files)]


def write_manifest(root: Path, output: Path, files: list[Path]) -> dict:
    included = []
    total_size = 0
    for p in files:
        rel = p.relative_to(root).as_posix()
        size = p.stat().st_size
        total_size += size
        included.append({"path": rel, "size_bytes": size, "sha256": sha256_file(p)})
    manifest = {
        "status_marker": "TADSR_PHASE3B_LIVE_RESULTS_PACKAGE",
        "status": "PASS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output": output.as_posix(),
        "file_count": len(files),
        "total_size_mb": round(total_size / (1024 * 1024), 6),
        "contains_raw_tensors": False,
        "contains_local_tensors": False,
        "contains_checkpoints": False,
        "included_files": included,
    }
    manifest_json = root / "experiments" / "production_completion" / "phase3b_live_results_manifest.json"
    manifest_md = root / "experiments" / "production_completion" / "phase3b_live_results_manifest.md"
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Phase 3-B live results package manifest",
        "",
        "`TADSR_PHASE3B_LIVE_RESULTS_PACKAGE: PASS`",
        "",
        f"- file_count: `{manifest['file_count']}`",
        f"- total_size_mb: `{manifest['total_size_mb']}`",
        "- contains_raw_tensors: `false`",
        "- contains_local_tensors: `false`",
        "- contains_checkpoints: `false`",
        "",
        "## Included files",
        "",
    ]
    for item in included:
        lines.append(f"- `{item['path']}` ({item['size_bytes']} bytes)")
    manifest_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Package Phase 3-B live audit JSON/Markdown/log results only.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)

    files = collect_files(root)
    manifest = write_manifest(root, output, files)

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            rel = p.relative_to(root).as_posix()
            zf.write(p, rel)
        for extra in [
            root / "experiments" / "production_completion" / "phase3b_live_results_manifest.json",
            root / "experiments" / "production_completion" / "phase3b_live_results_manifest.md",
        ]:
            if extra.exists():
                zf.write(extra, extra.relative_to(root).as_posix())

    digest = sha256_file(output)
    sha_path = output.with_suffix(".sha256.txt")
    sha_path.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    manifest["sha256"] = digest
    (root / "experiments" / "production_completion" / "phase3b_live_results_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("TADSR_PHASE3B_LIVE_RESULTS_PACKAGE: PASS")
    print(f"package: {output}")
    print(f"sha256: {digest}")
    print(f"file_count: {len(files)}")
    print(f"total_size_mb: {manifest['total_size_mb']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
