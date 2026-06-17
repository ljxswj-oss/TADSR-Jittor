#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p experiments/jittor_tiny/logs
source .venv/bin/activate 2>/dev/null || true
export JITTOR_HOME="$ROOT/.jittor_cache"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
 -m jittor_tadsr.train --dataset_list data/dataset_list.txt --output_dir experiments/jittor_tiny --image_size 128 --batch_size 2 --max_steps ${MAX_STEPS:-300} --lr 1e-4 --seed 123 --use_cuda ${USE_CUDA:-0} 2>&1 | tee experiments/jittor_tiny/logs/train_stdout.log
