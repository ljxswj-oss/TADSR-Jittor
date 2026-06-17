#!/usr/bin/env python3
"""Select at most one idle GPU for smoke training, or recommend CPU fallback."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def query_gpus() -> list[dict[str, object]]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except Exception:
        return []
    rows = []
    for raw in out.splitlines():
        cells = [x.strip() for x in next(csv.reader([raw]))]
        if len(cells) != 5:
            continue
        rows.append(
            {
                "index": int(cells[0]),
                "name": cells[1],
                "memory_used_mb": float(cells[2]),
                "memory_total_mb": float(cells[3]),
                "utilization_gpu_percent": float(cells[4]),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-memory-used-mb", type=float, default=2000.0)
    parser.add_argument("--max-utilization", type=float, default=20.0)
    parser.add_argument("--output-dir", default="experiments/smoke_training/output_tail")
    args = parser.parse_args()

    gpus = query_gpus()
    candidates = [
        g for g in gpus
        if g["memory_used_mb"] < args.max_memory_used_mb and g["utilization_gpu_percent"] <= args.max_utilization
    ]
    candidates.sort(key=lambda g: (g["memory_used_mb"], g["utilization_gpu_percent"]))
    selected = candidates[0] if candidates else None
    result = {
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpus": gpus,
        "selected_gpu": None if selected is None else int(selected["index"]),
        "selected_gpu_name": None if selected is None else selected["name"],
        "cpu_fallback": selected is None,
        "rule": {
            "max_memory_used_mb": args.max_memory_used_mb,
            "max_utilization": args.max_utilization,
        },
    }
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gpu_selection.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Smoke Training GPU Selection",
        "",
        f"Selected GPU: `{result['selected_gpu']}`",
        f"CPU fallback: `{result['cpu_fallback']}`",
        "",
        "| GPU | Name | Memory Used MB | Total MB | Util % |",
        "|---:|---|---:|---:|---:|",
    ]
    for gpu in gpus:
        lines.append(
            f"| {gpu['index']} | {gpu['name']} | {gpu['memory_used_mb']:.0f} | "
            f"{gpu['memory_total_mb']:.0f} | {gpu['utilization_gpu_percent']:.0f} |"
        )
    (out_dir / "gpu_selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("TADSR_SMOKE_TRAINING_GPU_SELECTION: PASS")
    print(f"selected_gpu={result['selected_gpu']}")
    print(f"cpu_fallback={result['cpu_fallback']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
