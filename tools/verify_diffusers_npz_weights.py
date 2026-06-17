#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def stats(array: np.ndarray):
    x = array.astype(np.float64, copy=False)
    return {"shape": list(array.shape), "dtype": str(array.dtype), "mean": float(x.mean()) if x.size else 0.0, "std": float(x.std()) if x.size else 0.0, "min": float(x.min()) if x.size else 0.0, "max": float(x.max()) if x.size else 0.0, "numel": int(array.size)}


def close(a: float, b: float, tol: float = 1e-7) -> bool:
    return abs(float(a) - float(b)) <= tol


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="experiments/full_repro/weights/diffusers_conversion_manifest.json")
    parser.add_argument("--report_dir", default="experiments/full_repro/weights")
    args = parser.parse_args()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        result = {"status": "BLOCKED", "reason": "diffusers conversion manifest missing"}
    else:
        manifest = json.loads(manifest_path.read_text())
        rows = []
        ok = True
        for item in manifest.get("converted_files", []):
            target = Path(item.get("target", ""))
            if item.get("status") != "converted" or not target.exists():
                rows.append({"component": item.get("name"), "target": str(target), "status": "missing_or_not_converted"})
                ok = False
                continue
            npz = np.load(target)
            tensors = {t["npz_key"]: t for t in item.get("tensors", [])}
            for key in npz.files:
                current = stats(npz[key])
                expected = tensors.get(key)
                match = expected is not None and current["shape"] == expected.get("shape") and current["dtype"] == expected.get("dtype")
                if expected:
                    for stat_key in ["mean", "std", "min", "max"]:
                        if not close(current[stat_key], expected.get(stat_key, 0.0)):
                            match = False
                rows.append({"component": item.get("name"), "key": key, "match": bool(match), "current": current, "expected": expected})
                ok = ok and bool(match)
        result = {"status": "PASS" if ok else "FAIL", "rows_checked": len(rows), "rows": rows[:5000]}
    (report_dir / "diffusers_weight_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = ["# Diffusers Weight Verification", "", f"Status: **{result.get('status')}**", "", f"Rows checked: {result.get('rows_checked', 0)}", "", "| Component | Key | Match | Shape | Dtype |", "|---|---|---:|---|---|"]
    for row in result.get("rows", [])[:300]:
        cur = row.get("current", {})
        md.append(f"| `{row.get('component','')}` | `{row.get('key','')}` | {row.get('match', False)} | `{cur.get('shape','')}` | `{cur.get('dtype','')}` |")
    if result.get("reason"):
        md += ["", f"Reason: {result['reason']}"]
    (report_dir / "diffusers_weight_verification.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Diffusers weight verification: {result.get('status')} rows={result.get('rows_checked', 0)}")
    return 0 if result.get("status") in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
