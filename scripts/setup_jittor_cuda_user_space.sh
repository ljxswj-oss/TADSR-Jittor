#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p /mnt/data/sj/tmp /mnt/data/sj/.cache/jittor experiments/full_repro/gpu_setup /mnt/data/sj/local/cuda_user/bin /mnt/data/sj/local/cuda_user/include /mnt/data/sj/local/cuda_user/lib64
LOG=experiments/full_repro/gpu_setup/jittor_cuda_user_space.log
PYTHON_BIN=${PYTHON:-.venv/bin/python}; [[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=python3
{
  echo "=== Jittor CUDA user-space setup probe ==="; date; echo "PYTHON_BIN=$PYTHON_BIN"
  cat > /mnt/data/sj/local/cuda_user/bin/nvcc <<'NVCC'
#!/usr/bin/env bash
exec /usr/local/cuda/bin/nvcc "$@"
NVCC
  chmod +x /mnt/data/sj/local/cuda_user/bin/nvcc
  find /usr/local/cuda/include -maxdepth 1 -type f -print0 2>/dev/null | while IFS= read -r -d '' f; do ln -sfn "$f" "/mnt/data/sj/local/cuda_user/include/$(basename "$f")"; done
  find /usr/local/cuda/lib64 -maxdepth 1 \( -type f -o -type l \) -print0 2>/dev/null | while IFS= read -r -d '' f; do ln -sfn "$f" "/mnt/data/sj/local/cuda_user/lib64/$(basename "$f")"; done
  CUDNN_INCLUDE=""; CUDNN_LIB=""
  if [[ -n "${CUDNN8_HOME:-}" ]]; then
    [[ -f "$CUDNN8_HOME/include/cudnn.h" ]] && CUDNN_INCLUDE="$CUDNN8_HOME/include"
    [[ -d "$CUDNN8_HOME/lib" ]] && CUDNN_LIB="$CUDNN8_HOME/lib"
    [[ -d "$CUDNN8_HOME/lib64" ]] && CUDNN_LIB="$CUDNN8_HOME/lib64"
    if [[ -z "$CUDNN_INCLUDE" || -z "$CUDNN_LIB" ]]; then echo "BLOCKED_CUDNN8_LAYOUT_INVALID"; exit 2; fi
    echo "Using CUDNN8_HOME=$CUDNN8_HOME"
  else
    "$PYTHON_BIN" scripts/probe_cudnn_user_space.py || true
    CUDNN_LIB=$(python3 - <<'PY'
import json
from pathlib import Path
p=Path('experiments/full_repro/gpu_setup/cudnn_probe.json')
if p.exists():
    d=json.loads(p.read_text()); print(str(Path(d['found_libs'][0]).parent) if d.get('found_libs') else '')
PY
)
    if [[ -n "$CUDNN_LIB" ]] && ls "$CUDNN_LIB"/libcudnn*.so.9* >/dev/null 2>&1; then
      echo "BLOCKED_CUDNN9_INCOMPATIBLE: cuDNN 9 incompatible with current Jittor wrapper. Install cuDNN8 at /mnt/data/sj/local/cudnn8."
      exit 2
    fi
  fi
  [[ -n "$CUDNN_INCLUDE" ]] && find "$CUDNN_INCLUDE" -maxdepth 1 -type f -print0 | while IFS= read -r -d '' f; do ln -sfn "$f" "/mnt/data/sj/local/cuda_user/include/$(basename "$f")"; done
  [[ -n "$CUDNN_LIB" ]] && find "$CUDNN_LIB" -maxdepth 1 \( -type f -o -type l \) -print0 | while IFS= read -r -d '' f; do ln -sfn "$f" "/mnt/data/sj/local/cuda_user/lib64/$(basename "$f")"; done
  export TMPDIR=/mnt/data/sj/tmp JITTOR_HOME=/mnt/data/sj/.cache/jittor CUDA_HOME=/mnt/data/sj/local/cuda_user nvcc_path=/mnt/data/sj/local/cuda_user/bin/nvcc
  export PATH=/mnt/data/sj/local/cuda_user/bin:/usr/local/cuda/bin:$PATH
  export CPATH="/mnt/data/sj/local/cuda_user/include:${CUDNN_INCLUDE}:${CPATH:-}"
  export C_INCLUDE_PATH="$CPATH" CPLUS_INCLUDE_PATH="$CPATH" LD_LIBRARY_PATH="/mnt/data/sj/local/cuda_user/lib64:${CUDNN_LIB}:${LD_LIBRARY_PATH:-}"
  USE_CUDA=1 "$PYTHON_BIN" scripts/check_env.py
  status=$?; echo "Jittor CUDA status: $status"; exit $status
} 2>&1 | tee "$LOG"
exit ${PIPESTATUS[0]}
