#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pytorch_dir", required=True)
    ap.add_argument("--jittor_dir", required=True)
    ap.add_argument("--output", default="experiments/full_repro/full_alignment/compare_full_outputs.json")
    args = ap.parse_args()
    p = Path(args.pytorch_dir)
    j = Path(args.jittor_dir)
    result = {
        "pytorch_dir_exists": p.exists(),
        "jittor_dir_exists": j.exists(),
        "status": "BLOCKED" if not (p.exists() and j.exists()) else "TODO_COMPARE_IMAGES",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
if __name__ == "__main__":
    main()
