#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
 -m pip install --upgrade pip
 -m pip install -r requirements.txt
mkdir -p .jittor_cache experiments/jittor_tiny/logs
echo "setup_env completed at $(date)" | tee experiments/env_setup.log
