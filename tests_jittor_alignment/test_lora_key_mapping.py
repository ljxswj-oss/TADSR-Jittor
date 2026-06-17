#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from jittor_tadsr_full.lora import find_lora_keys
from jittor_tadsr_full.utils import write_json

OUT = Path("experiments/full_repro/jittor_alignment")
TADSR = Path("/mnt/data/sj/checkpoints/TADSR/jittor_converted/tadsr_weights.npz")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not TADSR.exists():
        result = {"status": "BLOCKED", "reason": "tadsr_weights.npz missing", "rows": []}
    else:
        rows = find_lora_keys(TADSR, max_rows=1000)
        result = {"status": "PARTIAL" if rows else "TODO", "npz": str(TADSR), "lora_key_count_sampled": len(rows), "rows": rows, "note": "Key inspection only; LoRA forward/merge is not claimed as complete."}
    write_json(OUT / "lora_key_mapping.json", result)
    md = ["# LoRA Key Mapping", "", f"Status: **{result['status']}**", "", result.get("note", ""), "", "| Key | Shape | Dtype |", "|---|---|---|"]
    for row in result.get("rows", [])[:300]:
        md.append(f"| `{row['key']}` | `{row['shape']}` | `{row['dtype']}` |")
    (OUT / "lora_key_mapping.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"LoRA key mapping: {result['status']} keys={len(result.get('rows', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
