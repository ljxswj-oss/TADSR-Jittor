#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
OUT="experiments/full_repro/pytorch_official/dryrun"
mkdir -p "$OUT"
SELECTED_ENV="experiments/full_repro/pytorch_official/env_matrix/selected_env.sh"
if [[ -f "$SELECTED_ENV" ]]; then source "$SELECTED_ENV"; fi
OFFICIAL_REPO="${OFFICIAL_REPO:-/mnt/data/sj/projects/TADSR_official_pytorch}"
if [[ -z "${OFFICIAL_PYTORCH_VENV:-}" ]] || [[ ! -x "$OFFICIAL_PYTORCH_VENV/bin/python" ]]; then
  echo '{"status":"BLOCKED_NO_SELECTED_ENV"}' > "$OUT/status.json"
  echo "BLOCKED_NO_SELECTED_ENV"
  exit 2
fi
VENV_DIR="$OFFICIAL_PYTORCH_VENV"
if [[ ! -f "$OFFICIAL_REPO/test_tadsr.py" ]]; then
  echo '{"status":"BLOCKED_OFFICIAL_REPO_MISSING"}' > "$OUT/status.json"
  echo "BLOCKED_OFFICIAL_REPO_MISSING"
  exit 2
fi
source "$VENV_DIR/bin/activate"
set +e
(cd "$OFFICIAL_REPO" && python test_tadsr.py --help) > "$OUT/stdout.log" 2> "$OUT/stderr.log"
status=$?
if [[ $status -ne 0 ]]; then
  (cd "$OFFICIAL_REPO" && python test_tadsr.py --definitely_invalid_arg_for_dryrun) >> "$OUT/stdout.log" 2>> "$OUT/stderr.log"
  status=$?
fi
set -e
python3 - <<PY
import json
from pathlib import Path
out=Path('$OUT')
stdout=(out/'stdout.log').read_text(errors='ignore') if (out/'stdout.log').exists() else ''
stderr=(out/'stderr.log').read_text(errors='ignore') if (out/'stderr.log').exists() else ''
text=(stdout+'\n'+stderr).lower()
cli_checked = ('usage' in text) or ('error' in text) or ('options' in text)
status = 'PASS' if cli_checked else 'FAIL'
json.dump({'status': status, 'returncode': $status, 'cli_checked': cli_checked}, open(out/'status.json','w'), indent=2)
PY
cat "$OUT/status.json"
