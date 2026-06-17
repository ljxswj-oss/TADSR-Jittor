#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
VENV_DIR="${VENV_DIR:-/mnt/data/sj/venvs/tadsr_official_pytorch}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/mnt/data/sj/.cache/pip}"
LOG_DIR="experiments/full_repro/pytorch_official/env"
LOG="$LOG_DIR/create_official_pytorch_venv.log"
mkdir -p "$LOG_DIR" "$PIP_CACHE_DIR" /mnt/data/sj/venvs
{
  echo "=== create_official_pytorch_venv ==="
  date
  echo "VENV_DIR=$VENV_DIR"
  echo "PIP_CACHE_DIR=$PIP_CACHE_DIR"
  if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
    echo "Created venv: $VENV_DIR"
  else
    echo "Venv already exists: $VENV_DIR"
  fi
  source "$VENV_DIR/bin/activate"
  python -V
  python -m pip --version
  if [[ "${ALLOW_NETWORK:-0}" == "1" ]]; then
    echo "ALLOW_NETWORK=1: installing strict official requirements."
    python -m pip install --cache-dir "$PIP_CACHE_DIR" -r requirements_official_pytorch.txt
  else
    echo "ALLOW_NETWORK is not 1, so no packages were installed."
    echo "Next: build/copy wheelhouse, then run scripts/install_official_from_wheelhouse.sh."
  fi
} 2>&1 | tee "$LOG"
