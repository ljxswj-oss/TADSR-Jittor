#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

SELECTED_ENV="experiments/full_repro/pytorch_official/env_matrix/selected_env.sh"
if [[ -f "$SELECTED_ENV" ]]; then
  source "$SELECTED_ENV"
fi
if [[ -n "${OFFICIAL_PYTORCH_VENV:-}" ]]; then
  VENV_DIR="$OFFICIAL_PYTORCH_VENV"
else
  VENV_DIR="${VENV_DIR:-/mnt/data/sj/venvs/tadsr_official_pytorch}"
fi

WHEELHOUSE="${WHEELHOUSE:-/mnt/data/sj/wheelhouse/tadsr_official_pytorch}"
REQ="${REQ:-requirements_official_pytorch.txt}"
LOG_DIR="experiments/full_repro/pytorch_official/env"
LOG="$LOG_DIR/install_official_from_wheelhouse.log"
mkdir -p "$LOG_DIR"
{
  echo "=== install_official_from_wheelhouse ==="
  date
  echo "VENV_DIR=$VENV_DIR"
  echo "WHEELHOUSE=$WHEELHOUSE"
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then echo "BLOCKED_VENV_MISSING: run bash scripts/install_official_env_matrix.sh first."; exit 2; fi
  if [[ ! -d "$WHEELHOUSE" ]] || ! find "$WHEELHOUSE" -maxdepth 1 -type f | grep -q .; then echo "BLOCKED_WHEELHOUSE_EMPTY: copy wheels to $WHEELHOUSE or run offline_pack/build_offline_pack.sh on a networked machine."; exit 2; fi
  source "$VENV_DIR/bin/activate"
  python -m pip install --no-index --find-links "$WHEELHOUSE" -r "$REQ"
} 2>&1 | tee "$LOG"
