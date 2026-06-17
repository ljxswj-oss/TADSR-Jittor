#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."

VENV_DIR="${VENV_DIR:-/mnt/data/sj/venvs/tadsr_official_pytorch}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/mnt/data/sj/.cache/pip}"
TMPDIR="${TMPDIR:-/mnt/data/sj/tmp}"
LOG_DIR="experiments/full_repro/pytorch_official/env"
mkdir -p "$LOG_DIR" "$PIP_CACHE_DIR" "$TMPDIR"
NETWORK_LOG="$LOG_DIR/network_probe.log"
STRICT_LOG="$LOG_DIR/install_strict.log"
RELAXED_LOG="$LOG_DIR/install_relaxed.log"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Official venv missing; creating it at $VENV_DIR"
  bash scripts/create_official_pytorch_venv.sh || exit $?
fi

source "$VENV_DIR/bin/activate"
export PIP_CACHE_DIR TMPDIR

python - <<'PY' > "$NETWORK_LOG" 2>&1
from __future__ import annotations
import json, time, urllib.request
urls = ["https://pypi.org/simple", "https://download.pytorch.org"]
rows = []
for url in urls:
    t0 = time.time()
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            rows.append({"url": url, "ok": True, "status": getattr(r, "status", None), "elapsed_sec": round(time.time() - t0, 3)})
    except Exception as exc:
        rows.append({"url": url, "ok": False, "error": repr(exc), "elapsed_sec": round(time.time() - t0, 3)})
print(json.dumps({"rows": rows, "all_ok": all(r["ok"] for r in rows)}, indent=2))
raise SystemExit(0 if all(r["ok"] for r in rows) else 2)
PY
network_status=$?
if [[ $network_status -ne 0 ]]; then
  cat > docs/official_pytorch_env_blocker.md <<'EOF'
# Official PyTorch Environment Blocker

Network probing failed, so the server cannot install the official PyTorch oracle environment directly.

Generated log:

```text
experiments/full_repro/pytorch_official/env/network_probe.log
```

Use the offline pack workflow instead:

```bash
# On a networked machine:
bash offline_pack/build_offline_pack.sh

# Upload to server:
scp -P 10022 TADSR_offline_pack.tar.gz sj@10.195.160.95:/mnt/data/sj/incoming/

# On the server:
cd /mnt/data/sj/projects/TADSR-Jittor
bash offline_pack/unpack_on_server.sh /mnt/data/sj/incoming/TADSR_offline_pack.tar.gz
```
EOF
  echo "PYTORCH_OFFICIAL_ENV_BLOCKED_NETWORK; see $NETWORK_LOG"
  exit 2
fi

echo "Network probe passed. Trying strict official install..."
{
  date
  python -m pip install --cache-dir "$PIP_CACHE_DIR" --extra-index-url https://download.pytorch.org/whl/cu118 \
    torch==2.0.1 torchvision==0.15.2
  python -m pip install --cache-dir "$PIP_CACHE_DIR" -r requirements_official_pytorch_no_torch.txt
} > "$STRICT_LOG" 2>&1
strict_status=$?

if [[ $strict_status -eq 0 ]]; then
  echo "STRICT_INSTALL_PASS"
  VENV_NAME=strict bash scripts/check_official_pytorch_env.sh || true
  exit 0
fi

echo "STRICT_INSTALL_FAILED; see $STRICT_LOG"
echo "Trying relaxed install without forcing torch replacement..."
{
  date
  python -m pip install --cache-dir "$PIP_CACHE_DIR" -r requirements_official_pytorch_relaxed.txt
} > "$RELAXED_LOG" 2>&1
relaxed_status=$?

VENV_NAME=relaxed bash scripts/check_official_pytorch_env.sh || true
if [[ $relaxed_status -eq 0 ]]; then
  echo "RELAXED_INSTALL_ATTEMPT_DONE"
  exit 0
fi
echo "RELAXED_INSTALL_FAILED; see $RELAXED_LOG"
exit 1
