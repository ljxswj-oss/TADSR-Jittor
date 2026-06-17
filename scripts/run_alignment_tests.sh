#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p experiments/full_repro/alignment_tests
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python3 tests/test_preprocess_alignment.py | tee experiments/full_repro/alignment_tests/preprocess.log
python3 tests/test_scheduler_alignment.py | tee experiments/full_repro/alignment_tests/scheduler.log
