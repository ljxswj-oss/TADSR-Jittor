#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
WHEELHOUSE="${WHEELHOUSE:-/mnt/data/sj/wheelhouse/tadsr_official_pytorch}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/mnt/data/sj/.cache/pip}"
REQ="${REQ:-requirements_official_pytorch.txt}"
LOG_DIR="experiments/full_repro/pytorch_official/env"
LOG="$LOG_DIR/build_official_wheelhouse.log"
mkdir -p "$WHEELHOUSE" "$PIP_CACHE_DIR" "$LOG_DIR"
{
  echo "=== build_official_wheelhouse ==="
  date
  echo "REQ=$REQ"
  echo "WHEELHOUSE=$WHEELHOUSE"
  if [[ "${ALLOW_NETWORK:-0}" != "1" ]]; then
    echo "BLOCKED_NETWORK_DISABLED: set ALLOW_NETWORK=1 to download wheels."
    exit 2
  fi
  python3 -m pip download --cache-dir "$PIP_CACHE_DIR" --dest "$WHEELHOUSE" -r "$REQ"
  find "$WHEELHOUSE" -maxdepth 1 -type f | sort > "$LOG_DIR/wheelhouse_files.txt"
  echo "Wheelhouse built at $WHEELHOUSE"
} 2>&1 | tee "$LOG"
