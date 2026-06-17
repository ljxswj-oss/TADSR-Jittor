#!/usr/bin/env python3
"""Validate a clean public release tree for manual GitHub upload."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
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
}
FORBIDDEN_DIRS = {".git", "local_tensors", "__pycache__", "weights", "checkpoints"}
SECRET_TOKENS = ("secret", "token", "password", "passwd", "credential", "key.pem", "id_rsa")
FALSE_CLAIMS = [
    "full inference complete",
    "production pipeline complete",
    "image generation complete",
    "dynamic runtime LoRA complete",
    "full TADSR training complete",
]
NEGATION_ALLOWLIST = [
    "not complete",
    "not claimed",
    "do not claim",
    "does not claim",
    "NOT_COMPLETE",
    "NotImplementedError",
    "未完成",
    "不声明",
    "不要声明",
    "不允许",
    "不允许直接声明",
    "没有完成",
    "不是",
]
REQUIRED = [
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


def is_negated(line: str) -> bool:
    low = line.lower()
    return any(token.lower() in low for token in NEGATION_ALLOWLIST)


def scan_release(release_dir: Path) -> dict:
    forbidden: list[dict[str, object]] = []
    large_files: list[dict[str, object]] = []
    false_claim_hits: list[dict[str, object]] = []
    total_size = 0
    file_count = 0
    for path in release_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(release_dir)
        size = path.stat().st_size
        total_size += size
        file_count += 1
        lower_name = path.name.lower()
        if (
            path.suffix.lower() in FORBIDDEN_SUFFIXES
            or any(part in FORBIDDEN_DIRS for part in rel.parts)
            or any(token in lower_name for token in SECRET_TOKENS)
        ):
            forbidden.append({"path": str(rel).replace("\\", "/"), "size_bytes": size})
        if size > 50 * 1024 * 1024:
            large_files.append({"path": str(rel).replace("\\", "/"), "size_bytes": size})
        if path.suffix.lower() == ".md":
            text = path.read_text(encoding="utf-8", errors="ignore")
            for lineno, line in enumerate(text.splitlines(), 1):
                low = line.lower()
                for claim in FALSE_CLAIMS:
                    if claim.lower() in low and not is_negated(line):
                        false_claim_hits.append(
                            {
                                "path": str(rel).replace("\\", "/"),
                                "line": lineno,
                                "phrase": claim,
                                "text": line.strip(),
                            }
                        )
    return {
        "file_count": file_count,
        "total_size_bytes": total_size,
        "total_size_mb": total_size / (1024 * 1024),
        "forbidden_files": forbidden,
        "large_files": large_files,
        "false_claim_hits": false_claim_hits,
    }


def validate(release_dir: Path) -> dict:
    scan = scan_release(release_dir)
    missing = [rel for rel in REQUIRED if not (release_dir / rel).exists()]
    readme_text = (release_dir / "README.md").read_text(encoding="utf-8", errors="ignore") if (release_dir / "README.md").exists() else ""
    honest_statuses = all(
        phrase in readme_text
        for phrase in [
            "JITTOR_FULL_INFERENCE: NOT_COMPLETE",
            "TIME_VAE_FULL_ALIGNMENT: NOT_COMPLETE",
            "TADSR_DYNAMIC_RUNTIME_LORA_IMPLEMENTATION: NOT_IMPLEMENTED_BY_DESIGN",
        ]
    )
    size_ok = scan["total_size_mb"] < 100
    no_forbidden = not scan["forbidden_files"] and not scan["large_files"]
    no_false_claims = not scan["false_claim_hits"] and honest_statuses
    github_ready = release_dir.exists() and not missing and no_forbidden and size_ok and no_false_claims
    status = "PASS" if github_ready else "PARTIAL"
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "release_dir": str(release_dir),
        "missing_required_files": missing,
        "readme_honest_statuses_present": honest_statuses,
        **scan,
        "markers": {
            "TADSR_CLEAN_PUBLIC_RELEASE_PACKAGE_VALIDATION": status,
            "TADSR_CLEAN_PUBLIC_RELEASE_GITHUB_READY": "PASS" if github_ready else "PARTIAL",
            "TADSR_CLEAN_PUBLIC_RELEASE_FALSE_CLAIM_GUARD": "PASS" if no_false_claims else "FAIL",
        },
    }


def write_outputs(payload: dict) -> None:
    out_json = REPO_ROOT / "experiments/clean_public_release_package_validation.json"
    out_md = REPO_ROOT / "experiments/clean_public_release_package_validation.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Clean Public Release Package Validation",
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
        f"- Release dir: `{payload['release_dir']}`",
        f"- File count: `{payload['file_count']}`",
        f"- Total size: `{payload['total_size_mb']:.3f} MB`",
        f"- Forbidden files: `{len(payload['forbidden_files'])}`",
        f"- Large files: `{len(payload['large_files'])}`",
        f"- False claim hits: `{len(payload['false_claim_hits'])}`",
        "",
        "## Missing Required Files",
        "",
    ]
    if payload["missing_required_files"]:
        lines.extend(f"- `{item}`" for item in payload["missing_required_files"])
    else:
        lines.append("None.")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", required=True)
    args = parser.parse_args()
    payload = validate(Path(args.release_dir).resolve())
    write_outputs(payload)
    for marker, status in payload["markers"].items():
        print(f"{marker}: {status}")
    print(f"file_count: {payload['file_count']}")
    print(f"total_size_mb: {payload['total_size_mb']:.3f}")
    print(f"forbidden_count: {len(payload['forbidden_files'])}")
    print(f"large_file_count: {len(payload['large_files'])}")
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
