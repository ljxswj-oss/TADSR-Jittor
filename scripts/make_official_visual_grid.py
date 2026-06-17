#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--thumb", type=int, default=160)
    args = ap.parse_args()
    imgs = [p for p in Path(args.input_dir).rglob("*") if p.suffix.lower() in IMG_EXT]
    if not imgs:
        print("No output images for visual grid.")
        return 2
    thumbs = []
    for p in imgs[:8]:
        im = Image.open(p).convert("RGB")
        im.thumbnail((args.thumb, args.thumb))
        canvas = Image.new("RGB", (args.thumb, args.thumb + 24), "white")
        canvas.paste(im, ((args.thumb - im.width)//2, 0))
        ImageDraw.Draw(canvas).text((4, args.thumb + 4), p.name[:24], fill=(0,0,0))
        thumbs.append(canvas)
    grid = Image.new("RGB", (args.thumb * len(thumbs), args.thumb + 24), "white")
    for i, im in enumerate(thumbs):
        grid.paste(im, (i * args.thumb, 0))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.output)
    print(f"Saved {args.output}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
