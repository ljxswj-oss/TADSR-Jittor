#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path

REQUIRED = [
    "time_vae", "unet", "vae", "text_encoder", "tokenizer", "scheduler",
    "feature_extractor", "bert-base-uncased", "DAPE.pth",
    "ram_swin_large_14m.pth", "tadsr.pkl",
]

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def dir_stats(path: Path) -> tuple[int, int]:
    total = 0
    count = 0
    for p in path.rglob("*"):
        if p.is_file():
            count += 1
            total += p.stat().st_size
    return count, total

def human(n: int) -> str:
    value = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.3f} {unit}"
        value /= 1024

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="/mnt/data/sj/incoming/TADSR_assets/TADSR_weights")
    ap.add_argument("--out_dir", default="experiments/full_repro/assets")
    args = ap.parse_args()
    root = Path(args.input_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    missing = []
    for name in REQUIRED:
        p = root / name
        item = {"name": name, "path": str(p), "exists": p.exists()}
        if not p.exists():
            missing.append(name)
        elif p.is_file():
            item.update({"type": "file", "size_bytes": p.stat().st_size, "size": human(p.stat().st_size), "sha256": sha256_file(p)})
        elif p.is_dir():
            count, total = dir_stats(p)
            item.update({"type": "dir", "file_count": count, "size_bytes": total, "size": human(total)})
        entries.append(item)
    manifest = {"input_dir": str(root), "required": REQUIRED, "missing": missing, "items": entries}
    (out / "offline_assets_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = ["# Offline Assets Manifest", "", f"Input directory: `{root}`", "", "| Item | Exists | Type | Size | Extra |", "|---|---:|---|---:|---|"]
    for item in entries:
        extra = item.get("sha256", f"files={item.get('file_count','')}")
        lines.append(f"| `{item['name']}` | {item['exists']} | {item.get('type','missing')} | {item.get('size','-')} | `{extra}` |")
    if missing:
        lines += ["", "## Missing Items", ""] + [f"- `{m}`" for m in missing]
    (out / "offline_assets_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if missing:
        print("Official assets are missing:")
        for m in missing:
            print(f"  - {m}")
        print("Please follow docs/manual_asset_download_guide.md and place files in /mnt/data/sj/incoming/TADSR_assets/TADSR_weights/")
        return 2
    print("Official assets verified.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
