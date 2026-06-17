import argparse, random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def synth(i, size):
    rng = np.random.default_rng(1000 + i)
    y, x = np.mgrid[0:size, 0:size]
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[..., 0] = (x * (i + 3) + y * 2) % 256
    arr[..., 1] = (y * (i + 5) + 40 * np.sin(x / 9.0)) % 256
    arr[..., 2] = ((x // (4 + i % 5) + y // (5 + i % 7)) % 2) * 220
    img = Image.fromarray(arr, "RGB")
    draw = ImageDraw.Draw(img)
    for _ in range(8):
        x0, y0 = int(rng.integers(0, size - 20)), int(rng.integers(0, size - 20))
        x1, y1 = min(size - 1, x0 + int(rng.integers(8, 45))), min(size - 1, y0 + int(rng.integers(8, 45)))
        color = tuple(int(v) for v in rng.integers(30, 255, size=3))
        (draw.rectangle if rng.random() < 0.5 else draw.ellipse)([x0, y0, x1, y1], outline=color, width=2)
    return img

def degrade(hr, scale):
    w, h = hr.size
    small = hr.resize((w // scale, h // scale), Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(0.6))
    lr = small.resize((w, h), Image.Resampling.BICUBIC)
    arr = np.asarray(lr).astype(np.float32) + np.random.normal(0, 3, (h, w, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

def grid(pairs, out, cell=128):
    n = min(8, len(pairs))
    can = Image.new("RGB", (cell * n, cell * 2), "white")
    for i, (hr, lr) in enumerate(pairs[:n]):
        can.paste(lr.resize((cell, cell)), (i * cell, 0))
        can.paste(hr.resize((cell, cell)), (i * cell, cell))
    can.save(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr_dir")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--num_images", type=int, default=32)
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--make_synthetic", action="store_true")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed)
    out = Path(a.out_dir); hr_o = out / "mini_hr"; lr_o = out / "mini_lr"
    hr_o.mkdir(parents=True, exist_ok=True); lr_o.mkdir(parents=True, exist_ok=True)
    src = []
    if a.hr_dir and not a.make_synthetic:
        src = [p for p in sorted(Path(a.hr_dir).iterdir()) if p.suffix.lower() in EXT][:a.num_images]
    pairs, lines = [], []
    for i in range(a.num_images):
        hr = Image.open(src[i % len(src)]).convert("RGB").resize((a.image_size, a.image_size)) if src else synth(i, a.image_size)
        lr = degrade(hr, a.scale)
        hp, lp = hr_o / f"{i:04d}.png", lr_o / f"{i:04d}.png"
        hr.save(hp); lr.save(lp); pairs.append((hr, lr))
        lines.append(f'{hp.as_posix()} {lp.as_posix()} "a low-quality real-world image"\n')
    (out / "dataset_list.txt").write_text("".join(lines), encoding="utf-8")
    grid(pairs, out / "visual_degradation_grid.png")
    print(f"Prepared {len(lines)} image pairs under {out}")

if __name__ == "__main__":
    main()
