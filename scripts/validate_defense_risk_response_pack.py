#!/usr/bin/env python3
"""Validate the Chinese final defense risk-response pack.

This validator is deliberately lightweight: it does not import PyTorch or
Jittor, does not run inference, and only audits submission-facing text.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "experiments/defense_risk_response_pack_validation.json"
OUT_MD = ROOT / "experiments/defense_risk_response_pack_validation.md"

DOCS = {
    "risk_pack": "docs/final_defense_risk_response_pack.md",
    "short_answers": "docs/defense_short_answers_zh.md",
    "long_answers": "docs/defense_long_answers_zh.md",
    "do_not_say": "docs/defense_do_not_say.md",
    "evidence_lookup": "docs/defense_evidence_lookup_table.md",
}

OPTIONAL_CONTEXT_DOCS = {
    "readme": "README.md",
    "ppt_outline": "docs/03_ppt_outline.md",
    "video_script": "docs/04_video_script.md",
    "submission_readme": "deliverables/TADSR-Jittor_submission_readme.md",
}

REQUIRED_TOPICS = {
    "full_inference_boundary": [
        "JITTOR_FULL_INFERENCE",
        "NOT_COMPLETE",
        "NotImplementedError",
        "full TADSR production inference",
    ],
    "timevae_boundary": [
        "TIME_VAE_ACTUAL_VAEHOOK_FULL_BOUNDARY_ALIGNMENT",
        "TIME_VAE_FULL_ALIGNMENT",
        "NOT_COMPLETE",
    ],
    "runtime_lora_boundary": [
        "TADSR_DYNAMIC_RUNTIME_LORA_IMPLEMENTATION",
        "NOT_IMPLEMENTED_BY_DESIGN",
        "static effective",
    ],
    "training_alignment": [
        "small-data",
        "smoke training",
        "train/validation loss",
        "multi-seed",
    ],
    "diagnostic_image_boundary": [
        "diagnostic",
        "tensor visualization",
        "restored image",
    ],
    "evidence_lookup": [
        "docs/final_evidence_index.md",
        "experiments/final_audit_report.md",
        "docs/teacher_review_guide.md",
    ],
    "future_roadmap": [
        "docs/full_engineering_completion_roadmap.md",
        "future work",
        "controlled",
    ],
}

REQUIRED_MARKERS = [
    "TADSR_UNET_FULL_FORWARD_ALIGNMENT",
    "TIME_VAE_ACTUAL_VAEHOOK_FULL_BOUNDARY_ALIGNMENT",
    "TADSR_STATIC_EFFECTIVE_LORA_COVERAGE_AUDIT",
    "TADSR_SCHEDULER_BOUNDARY_ALIGNMENT",
    "TADSR_MINIMAL_ONE_STEP_DECODE_DRY_RUN",
    "TADSR_SMALL_DATA_TRAINING_ALIGNMENT",
    "JITTOR_FULL_INFERENCE",
    "JITTOR_FULL_PORT",
    "TIME_VAE_FULL_ALIGNMENT",
    "TADSR_DYNAMIC_RUNTIME_LORA_IMPLEMENTATION",
]

FALSE_CLAIM_PATTERNS = [
    "full inference complete",
    "production pipeline complete",
    "image generation complete",
    "video generation complete",
    "image/video generation complete",
    "full TADSR training complete",
    "dynamic runtime LoRA complete",
    "runtime LoRA complete",
    "TimeVAE full alignment complete",
    "完整推理已完成",
    "生产级推理已完成",
    "完整生产流水线已完成",
    "最终图片生成已完成",
    "最终视频生成已完成",
    "动态 runtime LoRA 已完成",
    "完整 TADSR 训练已完成",
    "TimeVAE full alignment 已完成",
]

NEGATION_HINTS = [
    "不",
    "没有",
    "未",
    "不是",
    "不能",
    "不要",
    "禁止",
    "不声称",
    "仍是",
    "保持",
    "NOT_COMPLETE",
    "NOT_IMPLEMENTED",
    "NotImplementedError",
    "not",
    "no ",
    "without",
    "do not",
    "does not",
    "never",
]


def read_text(path: str | Path) -> str:
    p = ROOT / path
    return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""


def has_all(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return all(term.lower() in lowered for term in terms)


def has_any_negation(line: str) -> bool:
    lowered = line.lower()
    return any(token.lower() in lowered for token in NEGATION_HINTS)


def scan_false_claims(text_by_file: dict[str, str]) -> list[dict[str, object]]:
    hits: list[dict[str, object]] = []
    for name, text in text_by_file.items():
        for lineno, line in enumerate(text.splitlines(), 1):
            lowered = line.lower()
            in_do_not_say_table = name == "do_not_say" and ("不要说" in line or "|" in line)
            for pattern in FALSE_CLAIM_PATTERNS:
                if pattern.lower() in lowered and not has_any_negation(line) and not in_do_not_say_table:
                    hits.append(
                        {
                            "file": DOCS.get(name, OPTIONAL_CONTEXT_DOCS.get(name, name)),
                            "line": lineno,
                            "phrase": pattern,
                            "text": line.strip(),
                        }
                    )
            if "JITTOR_FULL_INFERENCE: PASS" in line or "TIME_VAE_FULL_ALIGNMENT: PASS" in line:
                hits.append(
                    {
                        "file": DOCS.get(name, OPTIONAL_CONTEXT_DOCS.get(name, name)),
                        "line": lineno,
                        "phrase": "forbidden PASS marker",
                        "text": line.strip(),
                    }
                )
    return hits


def validate() -> dict:
    file_status = {}
    text_by_file = {}
    for name, rel in {**DOCS, **OPTIONAL_CONTEXT_DOCS}.items():
        path = ROOT / rel
        file_status[name] = {
            "path": rel,
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
        }
        text_by_file[name] = read_text(rel)

    defense_text = "\n".join(text_by_file[name] for name in DOCS)
    topic_status = {name: has_all(defense_text, terms) for name, terms in REQUIRED_TOPICS.items()}
    marker_status = {marker: marker in defense_text for marker in REQUIRED_MARKERS}
    false_claim_hits = scan_false_claims(text_by_file)

    docs_exist = all(file_status[name]["exists"] and file_status[name]["size_bytes"] > 0 for name in DOCS)
    topics_ok = all(topic_status.values())
    markers_ok = all(marker_status.values())
    false_claims_ok = not false_claim_hits
    evidence_lookup_ready = file_status["evidence_lookup"]["exists"] and all(
        term in text_by_file["evidence_lookup"]
        for term in [
            "experiments/final_audit_report.md",
            "docs/final_evidence_index.md",
            "tests_jittor_alignment/test_smoke_training_artifacts.py",
        ]
    )
    short_answers_ready = file_status["short_answers"]["exists"] and text_by_file["short_answers"].count("## ") >= 10
    long_answers_ready = file_status["long_answers"]["exists"] and text_by_file["long_answers"].count("## ") >= 7

    overall = (
        docs_exist
        and topics_ok
        and markers_ok
        and false_claims_ok
        and evidence_lookup_ready
        and short_answers_ready
        and long_answers_ready
    )
    marker = lambda ok: "PASS" if ok else "PARTIAL"
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if overall else "PARTIAL",
        "files": file_status,
        "topic_status": topic_status,
        "required_marker_presence": marker_status,
        "false_claim_hits": false_claim_hits,
        "markers": {
            "TADSR_DEFENSE_RISK_RESPONSE_PACK": marker(docs_exist and topics_ok and markers_ok),
            "TADSR_DEFENSE_SHORT_ANSWERS_READY": marker(short_answers_ready),
            "TADSR_DEFENSE_LONG_ANSWERS_READY": marker(long_answers_ready),
            "TADSR_DEFENSE_FALSE_CLAIM_GUARD": marker(false_claims_ok),
            "TADSR_DEFENSE_EVIDENCE_LOOKUP_READY": marker(evidence_lookup_ready),
        },
    }


def write_outputs(result: dict) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# 答辩风险应对包校验",
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

    lines += ["", "## 主题覆盖", "", "| Topic | Covered |", "|---|---:|"]
    for key, value in result["topic_status"].items():
        lines.append(f"| `{key}` | `{value}` |")

    lines += ["", "## 必要状态标记", "", "| Marker | Present |", "|---|---:|"]
    for key, value in result["required_marker_presence"].items():
        lines.append(f"| `{key}` | `{value}` |")

    lines += ["", "## 误导性表述扫描", ""]
    if result["false_claim_hits"]:
        for hit in result["false_claim_hits"]:
            lines.append(f"- `{hit['file']}:{hit['line']}` `{hit['phrase']}`: {hit['text']}")
    else:
        lines.append("未发现未加限定的误导性完成声明。")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    result = validate()
    write_outputs(result)
    for key, value in result["markers"].items():
        print(f"{key}: {value}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
