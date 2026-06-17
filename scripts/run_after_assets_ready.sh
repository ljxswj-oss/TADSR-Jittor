#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
LOG_DIR="experiments/full_repro/run_after_assets_ready"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/run.log"
: > "$MASTER_LOG"
run_step() {
  local name="$1"; shift
  echo "=== $name ===" | tee -a "$MASTER_LOG"
  printf '%q ' "$@" > "$LOG_DIR/${name}.command.sh"; echo >> "$LOG_DIR/${name}.command.sh"
  set +e
  "$@" > "$LOG_DIR/${name}.stdout.log" 2> "$LOG_DIR/${name}.stderr.log"
  local status=$?
  set -e
  echo "$name status=$status" | tee -a "$MASTER_LOG"
  if [[ $status -ne 0 ]]; then echo "STOPPED_AT=$name" | tee -a "$MASTER_LOG"; exit $status; fi
}
if [[ ! -f experiments/full_repro/pytorch_official/env_matrix/selected_env.sh ]]; then
  run_step install_official_env_matrix bash scripts/install_official_env_matrix.sh
fi
run_step verify_official_assets python3 scripts/verify_official_assets.py --input_dir /mnt/data/sj/incoming/TADSR_assets/TADSR_weights --out_dir experiments/full_repro/assets
run_step prepare_official_assets bash scripts/prepare_official_assets_offline.sh
run_step check_official_pytorch_env bash scripts/check_official_pytorch_env.sh
run_step check_official_repo_imports python3 scripts/check_official_repo_imports.py
run_step prepare_official_test_inputs python3 scripts/prepare_official_test_inputs.py
run_step run_official_pytorch_smoke bash scripts/run_official_pytorch_smoke.sh
if [[ "${SKIP_WEIGHT_CONVERSION:-0}" == "1" ]]; then
  echo "SKIP_WEIGHT_CONVERSION=1: skipping inspect/convert/verify weight steps" | tee -a "$MASTER_LOG"
else
  run_step inspect_official_weights python3 tools/inspect_official_weights.py
  run_step convert_official_weights python3 tools/convert_official_weights_to_npz.py
  run_step verify_converted_weights python3 tools/verify_converted_weights.py
fi
run_step final_audit python3 scripts/final_audit.py
echo "RUN_AFTER_ASSETS_READY_PASS" | tee -a "$MASTER_LOG"
