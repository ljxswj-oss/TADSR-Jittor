#!/usr/bin/env python3
"""Validate that final reviewer-facing materials are mainly Chinese.

The script allows technical English in commands, paths and markers, but checks
that key Markdown files contain substantial Chinese explanation and no obvious
mojibake. It does not import model frameworks.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "experiments/final_chinese_materials_validation.json"
OUT_MD = ROOT / "experiments/final_chinese_materials_validation.md"

FILES = [
    "README.md",
    "docs/03_ppt_outline.md",
    "docs/04_video_script.md",
    "docs/final_teacher_status_summary.md",
    "docs/final_defense_risk_response_pack.md",
    "docs/defense_short_answers_zh.md",
    "docs/defense_long_answers_zh.md",
    "docs/teacher_review_guide.md",
    "docs/final_submission_checklist.md",
    "docs/final_release_candidate_signoff.md",
    "docs/final_command_index.md",
    "docs/final_github_upload_checklist.md",
    "docs/final_human_submission_instructions.md",
    "docs/final_video_rehearsal_checklist.md",
    "docs/final_submission_freeze_tag.md",
    "docs/final_human_submission_lock_report.md",
    "docs/final_video_submission_check.md",
    "deliverables/TADSR-Jittor_submission_readme.md",
    "deliverables/TADSR-Jittor_video_recording_guide.md",
    "deliverables/TADSR-Jittor_final_presentation.md",
]

MOJIBAKE_PATTERNS = [
    "锟",
    "盲赂",
    "茅鈥",
    "鈥",
    "涓€",
    "鏃犮€",
    "鐘舵",
    "缂哄け",
    "鍗犱綅",
]

TECHNICAL_TERMS = {
    "TADSR",
    "Jittor",
    "PyTorch",
    "UNet",
    "TimeVAE",
    "VAEHook",
    "LoRA",
    "Scheduler",
    "README",
    "JSON",
    "Markdown",
}


def read_text(rel: str) -> str:
    path = ROOT / rel
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def strip_code(text: str) -> str:
    lines = []
    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if not in_code:
            lines.append(line)
    return "\n".join(lines)


def chinese_ratio(text: str) -> float:
    body = strip_code(text)
    chars = [ch for ch in body if not ch.isspace()]
    if not chars:
        return 0.0
    chinese = [ch for ch in chars if "\u4e00" <= ch <= "\u9fff"]
    return len(chinese) / len(chars)


def mojibake_hits(text: str) -> list[str]:
    hits = []
    for pattern in MOJIBAKE_PATTERNS:
        if pattern in text:
            hits.append(pattern)
    # Common UTF-8-as-GBK sequence: repeated mojibake-only characters.
    suspicious = re.findall(r"[閿鑴鐩鑼閳濡鏉鐠鍗]{2,}", text)
    hits.extend(seq[:20] for seq in suspicious[:5])
    return sorted(set(hits))


def validate() -> dict:
    rows = []
    for rel in FILES:
        path = ROOT / rel
        text = read_text(rel)
        ratio = chinese_ratio(text)
        hits = mojibake_hits(text)
        exists = path.exists()
        min_ratio = 0.12 if rel.endswith(".md") else 0.05
        if any(term in rel for term in TECHNICAL_TERMS):
            min_ratio = 0.08
        status = "PASS" if exists and ratio >= min_ratio and not hits else "PARTIAL"
        rows.append(
            {
                "path": rel,
                "exists": exists,
                "size_bytes": path.stat().st_size if exists else 0,
                "chinese_ratio": ratio,
                "min_required_ratio": min_ratio,
                "mojibake_hits": hits,
                "status": status,
            }
        )

    missing = [row for row in rows if not row["exists"]]
    partial = [row for row in rows if row["status"] != "PASS"]
    mojibake_rows = [row for row in rows if row["mojibake_hits"]]
    status = "PASS" if not missing and not partial else "PARTIAL"
    markers = {
        "TADSR_FINAL_CHINESE_MATERIALS_VALIDATION": status,
        "TADSR_FINAL_MOJIBAKE_SCAN": "PASS" if not mojibake_rows else "PARTIAL",
        "TADSR_FINAL_CHINESE_READABILITY_READY": "PASS" if not missing and not partial else "PARTIAL",
    }
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "files": rows,
        "markers": markers,
    }


def write_outputs(result: dict) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# 最终中文材料可读性检查",
        "",
        f"状态：**{result['status']}**",
        "",
        "## Markers",
        "",
        "| Marker | Status |",
        "|---|---|",
    ]
    for key, value in result["markers"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines += ["", "## 文件检查", "", "| 文件 | 中文比例 | 状态 | 乱码提示 |", "|---|---:|---|---|"]
    for row in result["files"]:
        hits = ", ".join(row["mojibake_hits"]) if row["mojibake_hits"] else "无"
        lines.append(f"| `{row['path']}` | {row['chinese_ratio']:.3f} | `{row['status']}` | {hits} |")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    result = validate()
    write_outputs(result)
    for marker, status in result["markers"].items():
        print(f"{marker}: {status}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
