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

VENV_NAME="${VENV_NAME:-${OFFICIAL_PYTORCH_ENV_KIND:-selected}}"
LOG_DIR="experiments/full_repro/pytorch_official/env"
mkdir -p "$LOG_DIR"
SAFE_NAME="${VENV_NAME//\//_}"
LOG="$LOG_DIR/env_check_${SAFE_NAME}.stdout.log"
OUT_JSON="$LOG_DIR/env_check_${SAFE_NAME}.json"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "BLOCKED_VENV_MISSING: $VENV_DIR does not exist. Run bash scripts/install_official_env_matrix.sh" | tee "$LOG"
  cat > "$OUT_JSON" <<JSON
{"venv_name":"$VENV_NAME","status":"BLOCKED_VENV_MISSING","python_executable":"$VENV_DIR/bin/python","missing_packages":["venv"],"critical_missing":["venv"],"version_mismatches":[]}
JSON
  exit 2
fi
source "$VENV_DIR/bin/activate"
python scripts/check_official_pytorch_env.py --venv-name "$VENV_NAME" --output "$OUT_JSON" 2>&1 | tee "$LOG"
exit ${PIPESTATUS[0]}
