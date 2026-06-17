#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--gt_dir", default=None)
    ap.add_argument("--metrics_json", default=None)
    args = ap.parse_args()
    out = Path(args.output_dir)
    imgs = [p for p in out.rglob("*") if p.suffix.lower() in IMG_EXT] if out.exists() else []
    metrics = {
        "output_dir": str(out),
        "num_output_images": len(imgs),
        "total_output_bytes": sum(p.stat().st_size for p in imgs),
        "gt_dir": args.gt_dir,
        "psnr": None,
        "ssim": None,
        "note": "PSNR/SSIM are only computed when GT support is implemented and gt_dir is provided.",
    }
    target = Path(args.metrics_json) if args.metrics_json else out / "metrics.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
if __name__ == "__main__":
    main()
