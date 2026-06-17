#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: this script is intended for Linux only." >&2
  exit 2
fi

if [[ ! -f "scripts/validate_production_completion_phase3.py" || ! -d "experiments/production_completion" ]]; then
  echo "ERROR: run this script from the TADSR-Jittor repository root." >&2
  exit 2
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
log_dir="experiments/production_completion/live_audit_logs"
mkdir -p "$log_dir"
log_file="$log_dir/phase3b_live_audit_${timestamp}.log"
exec > >(tee -a "$log_file") 2>&1

echo "TADSR_PHASE3B_LIVE_AUDIT_SCRIPT_STARTED: $(date -Is)"
echo "Repository: $(pwd)"
echo "Git branch: $(git branch --show-current 2>/dev/null || true)"
echo "Git head: $(git log -1 --oneline 2>/dev/null || true)"

required_env=(TADSR_OFFICIAL_REPO TADSR_OFFICIAL_WEIGHTS TADSR_OFFICIAL_PYTHON)
for name in "${required_env[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: missing required environment variable: ${name}" >&2
    echo "BLOCKED_BY_OFFICIAL_ENV"
    python3 scripts/validate_production_completion_phase3.py || true
    exit 2
  fi
  echo "${name}=${!name}"
done

for path in "$TADSR_OFFICIAL_REPO" "$TADSR_OFFICIAL_WEIGHTS" "$TADSR_OFFICIAL_PYTHON"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: required path does not exist: $path" >&2
    echo "BLOCKED_BY_OFFICIAL_ENV"
    python3 scripts/resolve_production_official_env.py \
      --official-repo "$TADSR_OFFICIAL_REPO" \
      --official-weights "$TADSR_OFFICIAL_WEIGHTS" \
      --official-python "$TADSR_OFFICIAL_PYTHON" || true
    python3 scripts/validate_production_completion_phase3.py || true
    exit 2
  fi
done

"$TADSR_OFFICIAL_PYTHON" -c "import sys; print('official_python', sys.executable); print(sys.version)"
"$TADSR_OFFICIAL_PYTHON" -c "import torch; print('torch', torch.__version__)"
"$TADSR_OFFICIAL_PYTHON" -c "import diffusers; print('diffusers', diffusers.__version__)" || true

echo "== official env resolution =="
python3 scripts/resolve_production_official_env.py \
  --official-repo "$TADSR_OFFICIAL_REPO" \
  --official-weights "$TADSR_OFFICIAL_WEIGHTS" \
  --official-python "$TADSR_OFFICIAL_PYTHON"

echo "== TimeVAE production path audit =="
python3 tools/audit_timevae_full_production_path.py \
  --official-repo "$TADSR_OFFICIAL_REPO" \
  --official-weights "$TADSR_OFFICIAL_WEIGHTS" \
  --official-python "$TADSR_OFFICIAL_PYTHON" \
  --output-dir experiments/production_completion/timevae_full \
  --metadata-only 1

echo "== TimeVAE production metadata oracle =="
python3 tools/export_timevae_production_metadata_oracle.py \
  --official-repo "$TADSR_OFFICIAL_REPO" \
  --official-weights "$TADSR_OFFICIAL_WEIGHTS" \
  --official-python "$TADSR_OFFICIAL_PYTHON" \
  --output-dir experiments/production_completion/timevae_full \
  --metadata-only 1 \
  --num-samples 1 \
  --height 64 \
  --width 64 \
  --seed 1234

echo "== Jittor TimeVAE production alignment preflight =="
python3 tests_jittor_alignment/test_timevae_production_alignment_preflight.py

echo "== official runtime LoRA behavior audit =="
python3 tools/audit_official_runtime_lora_behavior.py \
  --official-repo "$TADSR_OFFICIAL_REPO" \
  --official-weights "$TADSR_OFFICIAL_WEIGHTS" \
  --official-python "$TADSR_OFFICIAL_PYTHON" \
  --output-dir experiments/production_completion/runtime_lora \
  --metadata-only 1

echo "== metadata contract / Phase3 validator / final audit =="
python3 scripts/validate_full_inference_metadata_contract.py
python3 scripts/validate_production_completion_phase3.py
python3 scripts/final_audit.py

echo "== full inference guard =="
guard_output="$(python3 -m jittor_tadsr_full.tadsr_full --full-inference 2>&1 || true)"
echo "$guard_output"
if ! grep -q "NotImplementedError" <<<"$guard_output"; then
  echo "ERROR: full inference guard did not raise NotImplementedError." >&2
  exit 1
fi

echo "== runtime torch import scan =="
python3 - <<'PY'
from pathlib import Path
import re

hits = []
pat = re.compile(r"^\s*(import\s+torch\b|from\s+torch\b)")
for root in ["jittor_tadsr_full", "jittor_tadsr"]:
    base = Path(root)
    if not base.exists():
        continue
    for p in base.rglob("*.py"):
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            if pat.search(line):
                hits.append(f"{p}:{i}:{line}")
print("TORCH_IMPORT_HITS:", len(hits))
if hits:
    print("\n".join(hits))
    raise SystemExit(1)
PY

echo "== git safety =="
git status --short
git diff --check
git diff --cached --check
if git diff --cached --name-only | grep -E '\.(npy|npz)$|(^|/)local_tensors(/|$)'; then
  echo "ERROR: staged raw tensor artifact detected." >&2
  exit 1
fi

echo "== Phase3 live audit summary =="
python3 - <<'PY'
import json
from pathlib import Path

def load(path):
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8", errors="ignore"))

phase3 = load("experiments/production_completion/phase3_validation.json")
final = load("experiments/final_audit_report.json")
statuses = {r.get("check"): r.get("status") for r in final.get("rows", [])}
for key in [
    "official_env_resolution_status",
    "timevae_oracle_metadata_status",
    "lora_live_audit_status",
    "full_inference_metadata_contract_status",
    "ready_for_one_step_tensor_alignment",
    "one_step_alignment_plan_status",
]:
    print(f"{key}: {phase3.get(key)}")
for key in [
    "JITTOR_FULL_INFERENCE",
    "JITTOR_FULL_PORT",
    "TIME_VAE_FULL_ALIGNMENT",
    "TADSR_DYNAMIC_RUNTIME_LORA_IMPLEMENTATION",
]:
    print(f"{key}: {statuses.get(key)}")
print("next_required_action:", phase3.get("next_required_action"))
PY

echo "TADSR_PHASE3B_LIVE_AUDIT_SCRIPT_FINISHED: $(date -Is)"
echo "Log file: $log_file"
