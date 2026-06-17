#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
DATASET="${DATASET:-RealSR}"
NUM_IMAGES="${NUM_IMAGES:-20}"
ROOT="${DATASET_ROOT:-/mnt/data/sj/datasets/TADSR/$DATASET}"
OUT="experiments/full_repro/pytorch_official/${DATASET}_subset_${NUM_IMAGES}"
TMP="/mnt/data/sj/datasets/TADSR/${DATASET}_subset_${NUM_IMAGES}/input"
mkdir -p "$OUT"
write_subset_status() {
  local status="$1"; local reason="$2"
  python3 - <<PY
import json
from pathlib import Path
out=Path("$OUT"); out.mkdir(parents=True, exist_ok=True)
data={"dataset":"$DATASET","num_images":"$NUM_IMAGES","status":status if False else "$status","reason":"$reason"}
json.dump(data, open(out/"subset_status.json", "w"), indent=2)
with open(out/"subset_status.md", "w") as f:
    f.write("# Official Subset Status\n\n")
    f.write("Dataset: $DATASET\n\n")
    f.write("Status: **" + data["status"] + "**\n\n")
    f.write("Reason: " + data["reason"] + "\n")
PY
}
if [[ ! -d "$ROOT" ]]; then echo "BLOCKED_DATASET_MISSING: $DATASET missing at $ROOT" | tee "$OUT/BLOCKED_DATASET_MISSING.txt"; write_subset_status "BLOCKED_DATASET_MISSING" "$DATASET missing at $ROOT"; exit 2; fi
candidates=(LR lr input LQ images .)
SRC=""
for c in "${candidates[@]}"; do
  if [[ -d "$ROOT/$c" ]] && find "$ROOT/$c" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.bmp' \) | grep -q .; then SRC="$ROOT/$c"; break; fi
done
if [[ -z "$SRC" ]]; then echo "BLOCKED_DATASET_IMAGES_MISSING: no LR/input images found under $ROOT" | tee "$OUT/BLOCKED_DATASET_MISSING.txt"; write_subset_status "BLOCKED_DATASET_IMAGES_MISSING" "no LR/input images found under $ROOT"; exit 2; fi
rm -rf "$TMP"; mkdir -p "$TMP"
if [[ "$NUM_IMAGES" == "all" ]]; then limit=100000000; else limit="$NUM_IMAGES"; fi
find -L "$SRC" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.bmp' \) | sort | head -n "$limit" | while read -r f; do cp -n "$f" "$TMP/" || true; done
GT=""; for c in HR hr GT gt target; do [[ -d "$ROOT/$c" ]] && GT="$ROOT/$c" && break; done
{ echo "DATASET=$DATASET"; echo "ROOT=$ROOT"; echo "SRC=$SRC"; echo "GT=$GT"; echo "NUM_IMAGES=$NUM_IMAGES"; echo "TMP=$TMP"; echo "OUT=$OUT"; } > "$OUT/dataset_selection.log"
OUT_DIR="$OUT" INPUT="$TMP" bash scripts/run_official_pytorch_smoke.sh
