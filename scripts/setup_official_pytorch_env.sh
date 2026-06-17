#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PIP_CACHE_DIR=/mnt/data/sj/.cache/pip
mkdir -p "$PIP_CACHE_DIR" experiments/full_repro/pytorch_official
python3 scripts/check_official_pytorch_env.py || {
  echo "Official PyTorch environment is incomplete. If network is unavailable, install the missing wheels offline."
  exit 2
}
