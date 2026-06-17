#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from jittor_tadsr_full.preprocess import preprocess_lq, ram_normalize_from_lq
from jittor_tadsr_full.utils import compare_arrays, write_json

OUT = Path("experiments/full_repro/jittor_alignment")
ORACLE = Path("experiments/full_repro/oracle_tensors/smoke")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = ORACLE / "input_manifest.json"
    oracle_path = ORACLE / "preprocess_tensors.npz"
    if not manifest_path.exists() or not oracle_path.exists():
        result = {"status": "BLOCKED", "reason": "oracle preprocess tensors missing"}
    else:
        manifest = json.loads(manifest_path.read_text())
        oracle = np.load(oracle_path)
        rows = []
        ok = True
        for img in manifest.get("images", []):
            idx = int(img["index"])
            prefix = f"sample_{idx:04d}"
            got = preprocess_lq(img["path"])
            ram = ram_normalize_from_lq(got["lq_0_1"])
            for name, arr in [("lq_0_1", got["lq_0_1"]), ("lq_minus1_1", got["lq_minus1_1"]), ("ram_normalized", ram)]:
                cmp = compare_arrays(arr, oracle[f"{prefix}_{name}"])
                rows.append({"sample": prefix, "tensor": name, **cmp})
                ok = ok and cmp.get("shape_match") and cmp.get("max_abs_error", 1.0) <= 1e-6
        result = {"status": "PASS" if ok else "FAIL", "rows": rows}
    write_json(OUT / "preprocess_alignment.json", result)
    md = ["# Preprocess Alignment", "", f"Status: **{result['status']}**", "", "| Sample | Tensor | Max abs error | Mean abs error | Relative error |", "|---|---|---:|---:|---:|"]
    for r in result.get("rows", []):
        md.append(f"| `{r.get('sample','')}` | `{r.get('tensor','')}` | {r.get('max_abs_error')} | {r.get('mean_abs_error')} | {r.get('relative_error')} |")
    if result.get("reason"):
        md += ["", f"Reason: {result['reason']}"]
    (OUT / "preprocess_alignment.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Preprocess alignment: {result['status']}")
    return 0 if result["status"] in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
