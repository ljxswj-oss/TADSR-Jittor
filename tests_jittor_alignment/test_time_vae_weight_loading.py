#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from jittor_tadsr_full.vae_time_aware import time_vae_key_groups
from jittor_tadsr_full.utils import write_json

OUT = Path("experiments/full_repro/jittor_alignment")
TIME_VAE = Path("/mnt/data/sj/checkpoints/TADSR/jittor_converted/diffusers/time_vae_weights.npz")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    result = time_vae_key_groups(TIME_VAE)
    write_json(OUT / "time_vae_loading.json", result)
    md = ["# Time-VAE Weight Loading", "", f"Status: **{result['status']}**", "", result.get("note", ""), "", "| Group | Count |", "|---|---:|"]
    for group, count in result.get("counts", {}).items():
        md.append(f"| `{group}` | {count} |")
    (OUT / "time_vae_loading.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Time-VAE loading: {result['status']}")
    return 0 if result["status"] in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
