#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

IN_WEIGHTS="${IN_WEIGHTS:-/mnt/data/sj/incoming/TADSR_assets/TADSR_weights}"
OUT_WEIGHTS="${OUT_WEIGHTS:-/mnt/data/sj/checkpoints/TADSR/preset/weights}"
IN_DATA="${IN_DATA:-/mnt/data/sj/incoming/TADSR_assets/datasets}"
OUT_DATA="${OUT_DATA:-/mnt/data/sj/datasets/TADSR}"
ASSET_OUT="experiments/full_repro/assets"
mkdir -p "$ASSET_OUT" "$OUT_WEIGHTS" "$OUT_DATA"

python3 scripts/verify_official_assets.py --input_dir "$IN_WEIGHTS" --out_dir "$ASSET_OUT"

required=(time_vae unet vae text_encoder tokenizer scheduler feature_extractor bert-base-uncased DAPE.pth ram_swin_large_14m.pth tadsr.pkl)
{
  echo "# Prepared Official Assets Manifest"
  echo
  echo "Input weights: \`$IN_WEIGHTS\`"
  echo
  echo "Output weights: \`$OUT_WEIGHTS\`"
  echo
  echo "| Item | Source exists | Target | Link target | Broken symlink |"
  echo "|---|---:|---|---|---:|"
} > "$ASSET_OUT/prepared_assets_manifest.md"

for item in "${required[@]}"; do
  src="$IN_WEIGHTS/$item"
  dst="$OUT_WEIGHTS/$item"
  rm -rf "$dst"
  ln -s "$src" "$dst"
  broken=false
  if [[ ! -e "$dst" ]]; then broken=true; fi
  printf '| `%s` | %s | `%s` | `%s` | %s |\n' "$item" "$([[ -e "$src" ]] && echo true || echo false)" "$dst" "$src" "$broken" >> "$ASSET_OUT/prepared_assets_manifest.md"
  if [[ "$broken" == true ]]; then
    echo "BROKEN_SYMLINK: $dst -> $src" >&2
    exit 2
  fi
done

for ds in RealSR DRealSR RealLR200; do
  src="$IN_DATA/$ds"
  dst="$OUT_DATA/$ds"
  if [[ -d "$src" ]]; then
    rm -rf "$dst"
    ln -s "$src" "$dst"
    echo "Linked dataset $ds"
  else
    echo "WARNING: dataset $ds not found at $src; official benchmark remains blocked."
  fi
done

find -L /mnt/data/sj/checkpoints/TADSR -maxdepth 5 -printf '%y %p -> %l\n' | sort > "$ASSET_OUT/prepared_assets_tree.txt"
if find "$OUT_WEIGHTS" -xtype l | grep -q .; then
  echo "BROKEN_SYMLINK_DETECTED under $OUT_WEIGHTS" >&2
  find "$OUT_WEIGHTS" -xtype l >&2
  exit 2
fi

echo "Prepared official assets under $OUT_WEIGHTS"
