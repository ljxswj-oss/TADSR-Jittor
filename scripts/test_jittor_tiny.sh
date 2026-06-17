#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p experiments/jittor_tiny/logs experiments/jittor_tiny/results
source .venv/bin/activate 2>/dev/null || true
export JITTOR_HOME="$ROOT/.jittor_cache"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
 -m jittor_tadsr.test --dataset_list data/dataset_list.txt --checkpoint experiments/jittor_tiny/checkpoints/latest.pkl --output_dir experiments/jittor_tiny/results --image_size 128 --timesteps 50,200,500 --use_cuda ${USE_CUDA:-0} 2>&1 | tee experiments/jittor_tiny/logs/test_stdout.log
