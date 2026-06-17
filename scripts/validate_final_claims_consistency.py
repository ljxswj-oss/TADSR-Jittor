#!/usr/bin/env python3
"""Scan final materials for scope consistency and false-completion claims."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "experiments/final_claims_consistency_validation.json"
OUT_MD = ROOT / "experiments/final_claims_consistency_validation.md"

SCAN_ROOTS = ["README.md", "docs", "deliverables", "experiments"]
REQUIRED_PHRASES = [
    "JITTOR_FULL_INFERENCE: NOT_COMPLETE",
    "JITTOR_FULL_PORT: PARTIAL",
    "TIME_VAE_FULL_ALIGNMENT: NOT_COMPLETE",
    "TADSR_DYNAMIC_RUNTIME_LORA_IMPLEMENTATION: NOT_IMPLEMENTED_BY_DESIGN",
]
REQUIRED_MEANINGS = [
    ["diagnostic image smoke", "diagnostic image-smoke"],
    ["not full inference", "不是完整推理", "不是 full inference"],
    ["one-step tensor alignment", "one-step"],
    ["not full TADSR training", "不是 full TADSR training", "不是完整 TADSR 训练"],
]
FALSE_CLAIMS = [
    "full inference complete",
    "production pipeline complete",
    "image generation complete",
    "video generation complete",
    "full TADSR training complete",
    "dynamic runtime LoRA complete",
    "TimeVAE full alignment complete",
    "generated restoration results",
    "production full inference ready",
    "完整推理已完成",
    "生产级推理已完成",
    "最终图像生成已完成",
    "视频生成已完成",
    "dynamic runtime LoRA 已完成",
    "TimeVAE full alignment 已完成",
]
NEGATION = [
    "not",
    "no ",
    "without",
    "do not claim",
    "does not claim",
    "don't claim",
    "NOT_COMPLETE",
    "NotImplementedError",
    "不声明",
    "不声称",
    "不要",
    "不能",
    "不把",
    "不是",
    "未完成",
    "没有完成",
    "没有声称",
    "不等于",
    "误解",
    "仅作为",
    "diagnostic",
]


def collect_markdown() -> dict[str, str]:
    files: dict[str, str] = {}
    excluded = {
        OUT_MD.resolve(),
    }
    for root in SCAN_ROOTS:
        p = ROOT / root
        if p.is_file() and p.suffix.lower() == ".md":
            if p.resolve() in excluded:
                continue
            try:
                files[p.as_posix()] = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
        elif p.is_dir():
            for child in p.rglob("*.md"):
                if child.resolve() in excluded:
                    continue
                try:
                    files[child.as_posix()] = child.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
    return files


def negated(line: str) -> bool:
    low = line.lower()
    return any(token.lower() in low for token in NEGATION)


def scan_false_claims(files: dict[str, str]) -> list[dict]:
    hits = []
    for path, text in files.items():
        for lineno, line in enumerate(text.splitlines(), 1):
            low = line.lower()
            for phrase in FALSE_CLAIMS:
                if phrase.lower() in low and not negated(line):
                    hits.append({"path": path, "line": lineno, "phrase": phrase, "text": line.strip()})
    return hits


def build_report() -> dict:
    files = collect_markdown()
    combined = "\n".join(files.values())
    false_hits = scan_false_claims(files)
    phrase_status = {phrase: phrase in combined for phrase in REQUIRED_PHRASES}
    meaning_status = {
        " / ".join(group): any(token in combined for token in group)
        for group in REQUIRED_MEANINGS
    }
    markers = {
        "TADSR_FINAL_FALSE_CLAIM_GUARD": "PASS" if not false_hits else "FAIL",
        "TADSR_FINAL_SCOPE_CONSISTENCY": "PASS" if all(phrase_status.values()) and all(meaning_status.values()) else "PARTIAL",
    }
    markers["TADSR_FINAL_CLAIMS_CONSISTENCY"] = "PASS" if markers["TADSR_FINAL_FALSE_CLAIM_GUARD"] == "PASS" and markers["TADSR_FINAL_SCOPE_CONSISTENCY"] == "PASS" else "PARTIAL"
    return {
        "status": markers["TADSR_FINAL_CLAIMS_CONSISTENCY"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "markers": markers,
        "scanned_file_count": len(files),
        "required_phrase_status": phrase_status,
        "required_meaning_status": meaning_status,
        "false_claim_hits": false_hits,
    }


def write_markdown(data: dict) -> None:
    lines = [
        "# 最终表述一致性审计",
        "",
        f"总体状态：**{data['status']}**",
        "",
        "该审计扫描 README、docs、deliverables 和 experiments Markdown，检查是否存在未加否定语境的夸大表述。",
        "",
        "## Marker 汇总",
        "",
        "| Marker | Status |",
        "|---|---|",
    ]
    for k, v in data["markers"].items():
        lines.append(f"| `{k}` | `{v}` |")
    lines += ["", "## 必须出现的范围表述", "", "| 表述 | 是否出现 |", "|---|---:|"]
    for k, v in data["required_phrase_status"].items():
        lines.append(f"| `{k}` | `{v}` |")
    lines += ["", "## 必须覆盖的语义", "", "| 语义 | 是否覆盖 |", "|---|---:|"]
    for k, v in data["required_meaning_status"].items():
        lines.append(f"| {k} | `{v}` |")
    lines += ["", "## False-claim hits", ""]
    if data["false_claim_hits"]:
        for hit in data["false_claim_hits"]:
            lines.append(f"- `{hit['path']}`:{hit['line']} `{hit['phrase']}` -> {hit['text']}")
    else:
        lines.append("None.")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    data = build_report()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(data)
    for k, v in data["markers"].items():
        print(f"{k}: {v}")
    return 0 if data["markers"]["TADSR_FINAL_FALSE_CLAIM_GUARD"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
