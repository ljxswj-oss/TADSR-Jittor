#!/usr/bin/env python3
"""Build a clean public release tree for manual GitHub upload.

This script only copies small public-facing files. It does not import model
frameworks, does not run inference, and does not create a zip archive.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


FORBIDDEN_SUFFIXES = {
    ".npy",
    ".npz",
    ".zip",
    ".pth",
    ".pt",
    ".ckpt",
    ".safetensors",
    ".pkl",
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".pyc",
}
FORBIDDEN_DIRS = {
    ".git",
    "__pycache__",
    "local_tensors",
    "weights",
    "checkpoints",
    ".mypy_cache",
    ".pytest_cache",
}
SECRET_TOKENS = ("secret", "token", "password", "passwd", "credential", "key.pem", "id_rsa")
EXPERIMENT_SUFFIXES = {".json", ".md", ".csv", ".png", ".jpg", ".jpeg", ".txt"}
FIGURE_SUFFIXES = {".png", ".jpg", ".jpeg", ".md", ".json"}
ALWAYS_INCLUDE_FILES = ["README.md", "requirements.txt", ".gitignore"]
INCLUDE_DIRS = [
    "jittor_tadsr_full",
    "jittor_tadsr",
    "scripts",
    "tools",
    "tests_jittor_alignment",
    "docs",
    "deliverables",
    "experiments",
    "figures",
]


def should_skip_path(path: Path, repo_root: Path, include_diagnostic_figures: bool) -> tuple[bool, str]:
    rel = path.relative_to(repo_root)
    parts = set(rel.parts)
    lowered_name = path.name.lower()
    if parts & FORBIDDEN_DIRS:
        return True, "forbidden_dir"
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        return True, "forbidden_suffix"
    if any(token in lowered_name for token in SECRET_TOKENS):
        return True, "secret_like_name"
    top = rel.parts[0] if rel.parts else ""
    if top == "experiments" and path.is_file() and path.suffix.lower() not in EXPERIMENT_SUFFIXES:
        return True, "experiment_non_public_suffix"
    if top == "figures":
        if not include_diagnostic_figures:
            return True, "figures_disabled"
        if path.is_file() and path.suffix.lower() not in FIGURE_SUFFIXES:
            return True, "figure_non_public_suffix"
    return False, ""


def safe_remove_output(output_dir: Path, repo_root: Path) -> None:
    resolved_output = output_dir.resolve()
    resolved_repo = repo_root.resolve()
    if resolved_output == resolved_repo or resolved_repo in resolved_output.parents:
        raise ValueError(f"Refusing to delete output inside repository: {resolved_output}")
    if resolved_output.exists():
        shutil.rmtree(resolved_output)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_package(
    repo_root: Path,
    output_dir: Path,
    overwrite: bool,
    include_diagnostic_figures: bool,
    include_deliverables: bool,
) -> dict:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        safe_remove_output(output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    skipped_patterns: dict[str, int] = {}
    copied_files: list[str] = []
    copied_top_level_dirs: set[str] = set()

    for rel_file in ALWAYS_INCLUDE_FILES:
        src = repo_root / rel_file
        if src.exists() and src.is_file():
            copy_file(src, output_dir / rel_file)
            copied_files.append(rel_file)

    for dirname in INCLUDE_DIRS:
        if dirname == "deliverables" and not include_deliverables:
            continue
        src_dir = repo_root / dirname
        if not src_dir.exists() or not src_dir.is_dir():
            continue
        copied_top_level_dirs.add(dirname)
        for src in src_dir.rglob("*"):
            if not src.is_file():
                continue
            skip, reason = should_skip_path(src, repo_root, include_diagnostic_figures)
            if skip:
                skipped_patterns[reason] = skipped_patterns.get(reason, 0) + 1
                continue
            rel = src.relative_to(repo_root)
            copy_file(src, output_dir / rel)
            copied_files.append(str(rel).replace("\\", "/"))

    forbidden_files, total_size, file_count = scan_release(output_dir)
    required = [
        "README.md",
        "requirements.txt",
        "jittor_tadsr_full",
        "scripts",
        "tests_jittor_alignment",
        "docs",
        "deliverables",
        "experiments/final_audit_report.md",
        "experiments/final_evidence_index.md",
    ]
    missing = [item for item in required if not (output_dir / item).exists()]
    size_mb = total_size / (1024 * 1024)
    ready = not forbidden_files and not missing and size_mb < 100
    status = "PASS" if ready else "PARTIAL"
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "output_dir": str(output_dir),
        "file_count": file_count,
        "copied_file_count": len(copied_files),
        "total_size_bytes": total_size,
        "total_size_mb": size_mb,
        "forbidden_count": len(forbidden_files),
        "forbidden_files": forbidden_files,
        "missing_required_files": missing,
        "copied_top_level_dirs": sorted(copied_top_level_dirs),
        "skipped_patterns": skipped_patterns,
        "includes_diagnostic_figures": include_diagnostic_figures,
        "includes_deliverables": include_deliverables,
        "ready_for_manual_github_upload": ready,
        "markers": {
            "TADSR_CLEAN_PUBLIC_RELEASE_PACKAGE_BUILD": status,
            "TADSR_CLEAN_PUBLIC_RELEASE_NO_RAW_TENSORS": "PASS" if not forbidden_files else "FAIL",
            "TADSR_CLEAN_PUBLIC_RELEASE_SIZE_AUDIT": "PASS" if size_mb < 100 else "PARTIAL",
        },
    }


def scan_release(release_dir: Path) -> tuple[list[dict[str, object]], int, int]:
    forbidden: list[dict[str, object]] = []
    total_size = 0
    file_count = 0
    for path in release_dir.rglob("*"):
        if not path.is_file():
            continue
        file_count += 1
        size = path.stat().st_size
        total_size += size
        rel = path.relative_to(release_dir)
        parts = set(rel.parts)
        lower_name = path.name.lower()
        if (
            path.suffix.lower() in FORBIDDEN_SUFFIXES
            or parts & FORBIDDEN_DIRS
            or any(token in lower_name for token in SECRET_TOKENS)
        ):
            forbidden.append({"path": str(rel).replace("\\", "/"), "size_bytes": size})
    return forbidden, total_size, file_count


def write_outputs(repo_root: Path, payload: dict) -> None:
    out_json = repo_root / "experiments/clean_public_release_package_manifest.json"
    out_md = repo_root / "experiments/clean_public_release_package_manifest.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Clean Public Release Package Manifest",
        "",
        f"Status: **{payload['status']}**",
        "",
        "## Markers",
        "",
        "| Marker | Status |",
        "|---|---|",
    ]
    for marker, status in payload["markers"].items():
        lines.append(f"| `{marker}` | `{status}` |")
    lines += [
        "",
        f"- Output dir: `{payload['output_dir']}`",
        f"- File count: `{payload['file_count']}`",
        f"- Total size: `{payload['total_size_mb']:.3f} MB`",
        f"- Forbidden count: `{payload['forbidden_count']}`",
        f"- Ready for manual GitHub upload: `{payload['ready_for_manual_github_upload']}`",
        "",
        "## Missing Required Files",
        "",
    ]
    if payload["missing_required_files"]:
        lines.extend(f"- `{item}`" for item in payload["missing_required_files"])
    else:
        lines.append("None.")
    lines += ["", "## Forbidden Files", ""]
    if payload["forbidden_files"]:
        lines.extend(f"- `{item['path']}`" for item in payload["forbidden_files"])
    else:
        lines.append("None.")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--output-dir",
        default=r"D:\HuaweiMoveData\Users\wangh\Documents\Playground\TADSR-Jittor_Public_Release_Clean",
    )
    parser.add_argument("--overwrite", default="0", choices=["0", "1"])
    parser.add_argument("--include-diagnostic-figures", default="1", choices=["0", "1"])
    parser.add_argument("--include-deliverables", default="1", choices=["0", "1"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    payload = build_package(
        repo_root=repo_root,
        output_dir=output_dir,
        overwrite=args.overwrite == "1",
        include_diagnostic_figures=args.include_diagnostic_figures == "1",
        include_deliverables=args.include_deliverables == "1",
    )
    write_outputs(repo_root, payload)
    for marker, status in payload["markers"].items():
        print(f"{marker}: {status}")
    print(f"output_dir: {payload['output_dir']}")
    print(f"file_count: {payload['file_count']}")
    print(f"total_size_mb: {payload['total_size_mb']:.3f}")
    print(f"forbidden_count: {payload['forbidden_count']}")
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
