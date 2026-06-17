#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p experiments/full_repro/assets /mnt/data/sj/incoming/TADSR_assets/TADSR_weights
LOG=experiments/full_repro/assets/download_failure.log

if [[ "${ALLOW_NETWORK:-0}" != "1" ]]; then
  {
    echo "Network download is disabled by default."
    echo "Please follow docs/manual_asset_download_guide.md and place files in /mnt/data/sj/incoming/TADSR_assets/."
    echo "To explicitly try network download, run: ALLOW_NETWORK=1 bash scripts/download_official_assets.sh"
  } | tee "$LOG"
  exit 2
fi

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli not found. Install huggingface_hub on a networked machine or current venv." | tee "$LOG"
  exit 2
fi

timeout 120 huggingface-cli download zty557/TADSR \
  --local-dir /mnt/data/sj/incoming/TADSR_assets/TADSR_weights \
  --local-dir-use-symlinks False \
  --resume-download 2>&1 | tee "$LOG"
