#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from jittor_tadsr_full.load_weights import summarize_all
from jittor_tadsr_full.utils import write_json

OUT = Path("experiments/full_repro/jittor_alignment")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    result = summarize_all()
    write_json(OUT / "weight_loading_alignment.json", result)
    md = ["# Weight Loading Alignment", "", f"Status: **{result['status']}**", "", "| Component | Status | Key count | Path |", "|---|---|---:|---|"]
    for name, comp in result.get("components", {}).items():
        md.append(f"| `{name}` | {comp.get('status')} | {comp.get('key_count', 0)} | `{comp.get('path')}` |")
    (OUT / "weight_loading_alignment.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Weight loading alignment: {result['status']}")
    return 0 if result["status"] in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
