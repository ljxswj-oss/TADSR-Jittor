#!/usr/bin/env python3
from __future__ import annotations
import argparse, shutil
from pathlib import Path

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
NOTE = "Smoke input is only for official pipeline verification, not for paper benchmark metrics."

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/lq")
    ap.add_argument("--output", default="/mnt/data/sj/datasets/TADSR/smoke/input")
    ap.add_argument("--num_images", type=int, default=4)
    ap.add_argument("--manifest", default="experiments/full_repro/pytorch_official/smoke/input_manifest.md")
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    candidates = []
    for root in [Path(args.source), Path("data")]:
        if root.exists():
            candidates += [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT]
    copied = []
    for p in sorted(dict.fromkeys(candidates))[: args.num_images]:
        dst = out / p.name
        shutil.copy2(p, dst)
        copied.append((p, dst))
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Official Smoke Input Manifest", "", NOTE, "", f"Output directory: `{out}`", "", "| Source | Copied to |", "|---|---|"]
    for src, dst in copied:
        lines.append(f"| `{src}` | `{dst}` |")
    if not copied:
        lines += ["", "## Status", "", "BLOCKED: no local image found under `data/` or the requested source."]
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("No local image found to prepare smoke inputs. Please put images under data/ or pass --source.")
        return 2
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Prepared {len(copied)} smoke input images at {out}")
    print(NOTE)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
