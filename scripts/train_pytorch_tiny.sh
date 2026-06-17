#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p experiments/pytorch_tiny/logs
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
 -m pytorch_tiny_baseline.train --dataset_list data/dataset_list.txt --output_dir experiments/pytorch_tiny --image_size 128 --batch_size 2 --max_steps ${MAX_STEPS:-300} --lr 1e-4 --seed 123 --use_cuda ${USE_CUDA:-1} 2>&1 | tee experiments/pytorch_tiny/logs/train_stdout.log
 -m pytorch_tiny_baseline.test --dataset_list data/dataset_list.txt --checkpoint experiments/pytorch_tiny/checkpoints/latest.pt --output_dir experiments/pytorch_tiny/results --image_size 128 --use_cuda ${USE_CUDA:-1} 2>&1 | tee experiments/pytorch_tiny/logs/test_stdout.log
