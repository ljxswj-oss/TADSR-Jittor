#!/usr/bin/env python3
"""Validate diagnostic one-step image-smoke evidence without model execution."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FALSE_CLAIMS = (
    "full inference complete",
    "production pipeline complete",
    "image generation complete",
    "production restored image",
    "dynamic runtime LoRA complete",
)
NEGATIONS = (
    "not ",
    "no ",
    "without",
    "guard",
    "diagnostic",
    "NOT_COMPLETE",
    "NOT_EXECUTED",
    "not full inference",
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        return obj if isinstance(obj, dict) else {}
    except Exception as exc:
        return {"status": "FAIL", "error": repr(exc)}


def staged_bad_raw(repo_root: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if proc.returncode != 0:
        return []
    bad = []
    for line in proc.stdout.splitlines():
        p = Path(line)
        if p.suffix.lower() in {".npy", ".npz"} or "local_tensors" in p.parts:
            bad.append(line)
    return bad


def false_claim_hits(paths: list[Path]) -> list[dict[str, object]]:
    hits: list[dict[str, object]] = []
    for path in paths:
        if not path.exists() or path.suffix.lower() not in {".md", ".txt", ".json", ".csv"}:
            continue
        for idx, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            lower = line.lower()
            for phrase in FALSE_CLAIMS:
                if phrase.lower() in lower and not any(n.lower() in lower for n in NEGATIONS):
                    hits.append({"path": str(path), "line": idx, "phrase": phrase, "text": line.strip()})
    return hits


def write_reports(out_json: Path, out_md: Path, payload: dict[str, Any]) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Diagnostic Image-Smoke Validation",
        "",
        f"Status: **{payload['status']}**",
        "",
        "This validator reads diagnostic artifacts only. It does not run full inference, does not run a denoising loop, and does not generate production images.",
        "",
        "## Markers",
        "",
        "| Marker | Status |",
        "|---|---|",
    ]
    for key, value in payload["markers"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines += [
        "",
        "## Artifact check",
        "",
        f"- Metrics JSON exists: `{payload['metrics_json_exists']}`",
        f"- Diagnostic PNG count: `{payload['diagnostic_png_count']}`",
        f"- Staged raw tensor count: `{len(payload['staged_raw_tensors'])}`",
        "",
        "## Blocker",
        "",
        payload.get("blocker") or "None.",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", default="experiments/production_completion/diagnostic_image_smoke")
    parser.add_argument("--figure-dir", default="figures/diagnostic_image_smoke")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    figure_dir = (repo_root / args.figure_dir).resolve()
    metrics_json = output_dir / "diagnostic_image_smoke_metrics.json"
    metrics = load_json(metrics_json)
    markers_in = metrics.get("markers", {}) if isinstance(metrics.get("markers"), dict) else {}
    pngs = sorted(figure_dir.glob("*.png")) if figure_dir.exists() else []
    staged_raw = staged_bad_raw(repo_root)
    paths_to_scan = [
        metrics_json,
        output_dir / "diagnostic_image_smoke_metrics.md",
        output_dir / "diagnostic_image_smoke_metrics.csv",
    ]
    hits = false_claim_hits(paths_to_scan)
    safety = metrics.get("safety", {}) if isinstance(metrics.get("safety"), dict) else {}
    executed = str(markers_in.get("TADSR_DIAGNOSTIC_IMAGE_SMOKE_EXECUTED", "NOT_EXECUTED"))
    status = str(metrics.get("status", "MISSING")) if metrics else "MISSING"
    if not metrics:
        validation_status = "BLOCKED"
    elif hits or staged_raw or safety.get("full_inference_executed") or safety.get("denoising_loop_executed"):
        validation_status = "FAIL"
    elif executed == "PASS":
        required = {
            "official_one_step_clamped_output.png",
            "jittor_one_step_clamped_output.png",
            "absolute_difference_heatmap.png",
            "side_by_side_diagnostic_grid.png",
        }
        validation_status = "PASS" if required.issubset({p.name for p in pngs}) and status == "PASS" else "PARTIAL"
    else:
        validation_status = "PARTIAL"

    markers = {
        "TADSR_DIAGNOSTIC_IMAGE_SMOKE_VALIDATION": validation_status,
        "TADSR_DIAGNOSTIC_IMAGE_SMOKE_FALSE_CLAIM_GUARD": "PASS" if not hits else "FAIL",
        "TADSR_DIAGNOSTIC_IMAGE_SMOKE_ARTIFACT_POLICY": "PASS" if not staged_raw else "FAIL",
    }
    for key in [
        "TADSR_DIAGNOSTIC_IMAGE_SMOKE_READY",
        "TADSR_DIAGNOSTIC_IMAGE_SMOKE_EXECUTED",
        "TADSR_DIAGNOSTIC_IMAGE_SMOKE_ALIGNMENT",
        "TADSR_DIAGNOSTIC_IMAGE_SMOKE_NOT_FULL_INFERENCE",
        "TADSR_DIAGNOSTIC_IMAGE_SMOKE_NO_RAW_TENSORS",
        "TADSR_DIAGNOSTIC_IMAGE_SMOKE_ARTIFACTS_READY",
    ]:
        markers[key] = str(markers_in.get(key, "MISSING"))

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": validation_status,
        "metrics_json_exists": metrics_json.exists(),
        "diagnostic_png_count": len(pngs),
        "diagnostic_pngs": [str(p.relative_to(repo_root)) for p in pngs],
        "staged_raw_tensors": staged_raw,
        "false_claim_hits": hits,
        "blocker": metrics.get("blocker") if metrics else "metrics JSON missing",
        "markers": markers,
    }
    write_reports(
        output_dir / "diagnostic_image_smoke_validation.json",
        output_dir / "diagnostic_image_smoke_validation.md",
        payload,
    )
    for key, value in markers.items():
        print(f"{key}: {value}")
    return 1 if validation_status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
