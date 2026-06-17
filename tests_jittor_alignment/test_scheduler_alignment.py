#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np

from jittor_tadsr_full.scheduler import MinimalDDPMScheduler
from jittor_tadsr_full.utils import compare_arrays, write_json

OUT = Path("experiments/full_repro/jittor_alignment")
ORACLE = Path("experiments/full_repro/oracle_tensors/smoke/scheduler_tensors.npz")
CONFIG = Path("/mnt/data/sj/checkpoints/TADSR/preset/weights/scheduler/scheduler_config.json")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not ORACLE.exists():
        result = {"status": "BLOCKED", "reason": "oracle scheduler tensors missing"}
    else:
        oracle = np.load(ORACLE)
        sched = MinimalDDPMScheduler.from_config(CONFIG)
        rows = []
        ok = True
        for key, arr in sched.state_dict().items():
            cmp = compare_arrays(arr, oracle[key])
            rows.append({"tensor": key, **cmp})
            ok = ok and cmp.get("shape_match") and cmp.get("max_abs_error", 1.0) <= 1e-8
        result = {"status": "PASS" if ok else "FAIL", "rows": rows}
    write_json(OUT / "scheduler_alignment.json", result)
    md = ["# Scheduler Alignment", "", f"Status: **{result['status']}**", "", "| Tensor | Max abs error | Mean abs error | Relative error |", "|---|---:|---:|---:|"]
    for r in result.get("rows", []):
        md.append(f"| `{r.get('tensor','')}` | {r.get('max_abs_error')} | {r.get('mean_abs_error')} | {r.get('relative_error')} |")
    if result.get("reason"):
        md += ["", f"Reason: {result['reason']}"]
    (OUT / "scheduler_alignment.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Scheduler alignment: {result['status']}")
    return 0 if result["status"] in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
