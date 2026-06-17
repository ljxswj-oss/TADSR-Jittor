#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/mnt/data/sj/.cache/pip}"
export TMPDIR="${TMPDIR:-/mnt/data/sj/tmp}"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-60}"
export OFFICIAL_PYTORCH_REPO="${OFFICIAL_PYTORCH_REPO:-/mnt/data/sj/projects/TADSR_official_pytorch}"
export PYTHONPATH="$OFFICIAL_PYTORCH_REPO:${PYTHONPATH:-}"
PER_PIP_TIMEOUT="${PER_PIP_TIMEOUT:-1800}"
LOG_DIR="experiments/full_repro/pytorch_official/env_matrix"
mkdir -p "$LOG_DIR" "$PIP_CACHE_DIR" "$TMPDIR" /mnt/data/sj/venvs

probe_network() {
  python3 - <<'PY' > "$LOG_DIR/network_probe.json" 2> "$LOG_DIR/network_probe.stderr.log"
from __future__ import annotations
import json, time, urllib.request
urls=["https://pypi.org/simple","https://download.pytorch.org/whl/cu118","https://download.pytorch.org/whl/torch_stable.html"]
rows=[]
for url in urls:
    t=time.time()
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            rows.append({"url":url,"ok":True,"status":getattr(r,"status",None),"elapsed_sec":round(time.time()-t,3)})
    except Exception as e:
        rows.append({"url":url,"ok":False,"error":repr(e),"elapsed_sec":round(time.time()-t,3)})
print(json.dumps({"rows":rows,"pypi_ok":rows[0]["ok"]}, indent=2))
PY
}

create_env() {
  local venv="$1"
  if [[ ! -x "$venv/bin/python" ]]; then python3.10 -m venv "$venv"; fi
  source "$venv/bin/activate"
  python -m pip install -U pip wheel
  python -m pip install setuptools==69.5.1
}
run_check() {
  local name="$1"; local venv="$2"
  source "$venv/bin/activate"
  python scripts/check_official_pytorch_env.py --venv-name "$name" --output "$LOG_DIR/env_check_${name}.json" > "$LOG_DIR/env_check_${name}.stdout.log" 2>&1 || true
}
run_import_smoke() {
  local name="$1"; local venv="$2"
  source "$venv/bin/activate"
  set +e
  python - <<PY > "$LOG_DIR/import_smoke_${name}.log" 2>&1
import json
mods=["torch","torchvision","diffusers","transformers","peft","cv2","PIL","numpy","xformers","pyiqa","basicsr","loralib","fairscale","lpips"]
missing=[]; versions={}
for m in mods:
    try:
        mod=__import__(m); versions[m]=getattr(mod,"__version__","unknown")
    except Exception as e:
        missing.append({"module":m,"error":repr(e)})
print("IMPORT_SMOKE_OK" if not missing else "IMPORT_SMOKE_FAIL")
try:
    import torch
    print("torch", torch.__version__); print("cuda available", torch.cuda.is_available()); print("torch cuda", torch.version.cuda); print("cudnn", torch.backends.cudnn.version())
except Exception as e:
    print("torch probe failed", repr(e))
print(json.dumps({"missing":missing,"versions":versions}, indent=2))
raise SystemExit(0 if not missing else 2)
PY
  local smoke_status=$?
  set -e
  echo "$smoke_status" > "$LOG_DIR/import_smoke_${name}.status"
}
install_one() {
  local name="$1"; local venv="$2"; local mode="$3"; local log="$LOG_DIR/install_${name}.log"
  echo "=== install $name ===" | tee "$log"
  echo "venv=$venv mode=$mode timeout=$PER_PIP_TIMEOUT" | tee -a "$log"
  set +e
  {
    create_env "$venv"
    if [[ "$mode" == "strict-pypi" ]]; then
      timeout "$PER_PIP_TIMEOUT" python -m pip install torch==2.0.1 torchvision==0.15.2
      timeout "$PER_PIP_TIMEOUT" python -m pip install -r requirements_official_pytorch_no_torch.txt || timeout "$PER_PIP_TIMEOUT" python -m pip install --no-build-isolation -r requirements_official_pytorch_no_torch.txt
    elif [[ "$mode" == "strict-cu118" ]]; then
      timeout "$PER_PIP_TIMEOUT" python -m pip install --extra-index-url https://download.pytorch.org/whl/cu118 torch==2.0.1 torchvision==0.15.2
      timeout "$PER_PIP_TIMEOUT" python -m pip install -r requirements_official_pytorch_no_torch.txt || timeout "$PER_PIP_TIMEOUT" python -m pip install --no-build-isolation -r requirements_official_pytorch_no_torch.txt
    elif [[ "$mode" == "relaxed-pypi" ]]; then
      timeout "$PER_PIP_TIMEOUT" python -m pip install torch torchvision
      timeout "$PER_PIP_TIMEOUT" python -m pip install -r requirements_official_pytorch_relaxed.txt || timeout "$PER_PIP_TIMEOUT" python -m pip install --no-build-isolation -r requirements_official_pytorch_relaxed.txt
    else
      echo "unknown mode $mode"; exit 2
    fi
  } >> "$log" 2>&1
  local status=$?
  python -m pip install setuptools==69.5.1 >> "$log" 2>&1 || true
  set -e
  echo "$status" > "$LOG_DIR/install_${name}.status"
  run_check "$name" "$venv"
  run_import_smoke "$name" "$venv"
}
probe_network || true
install_one strict-pypi /mnt/data/sj/venvs/tadsr_official_strict_pypi strict-pypi
install_one strict-cu118 /mnt/data/sj/venvs/tadsr_official_strict_cu118 strict-cu118
install_one relaxed-pypi /mnt/data/sj/venvs/tadsr_official_relaxed_pypi relaxed-pypi
python3 - <<'PY'
from __future__ import annotations
import json
from pathlib import Path
log=Path('experiments/full_repro/pytorch_official/env_matrix')
order=['strict-cu118','strict-pypi','relaxed-pypi']
venvs={'strict-cu118':'/mnt/data/sj/venvs/tadsr_official_strict_cu118','strict-pypi':'/mnt/data/sj/venvs/tadsr_official_strict_pypi','relaxed-pypi':'/mnt/data/sj/venvs/tadsr_official_relaxed_pypi'}
rows=[]
for name in ['strict-pypi','strict-cu118','relaxed-pypi']:
    install_status=(log/f'install_{name}.status').read_text().strip() if (log/f'install_{name}.status').exists() else 'missing'
    smoke_status=(log/f'import_smoke_{name}.status').read_text().strip() if (log/f'import_smoke_{name}.status').exists() else 'missing'
    check=json.loads((log/f'env_check_{name}.json').read_text()) if (log/f'env_check_{name}.json').exists() else {}
    rows.append({'env':name,'venv':venvs[name],'install_status':'PASS' if install_status=='0' else 'FAIL','install_exit_code':install_status,'import_smoke':'PASS' if smoke_status=='0' else 'FAIL','torch':next((p.get('version') for p in check.get('packages',[]) if p.get('dist_name')=='torch'),None),'torch_cuda':check.get('torch_cuda',{}).get('torch_cuda'),'cuda_available':check.get('torch_cuda',{}).get('available'),'missing_packages':check.get('missing_packages') or check.get('missing_required') or [],'critical_missing':check.get('critical_missing') or [],'version_mismatches':check.get('version_mismatches') or check.get('version_mismatch') or [],'check_status':check.get('status','MISSING_CHECK'),'log':str(log/f'install_{name}.log')})
selected=None
for name in order:
    row=next(r for r in rows if r['env']==name)
    if row['install_status']=='PASS' and row['import_smoke']=='PASS' and row['check_status']=='PASS': selected=row; break
if selected is None:
    row=next(r for r in rows if r['env']=='relaxed-pypi')
    if row['install_status']=='PASS' and row['import_smoke']=='PASS': selected=row
selected_env=log/'selected_env.sh'
if selected: selected_env.write_text(f"export OFFICIAL_PYTORCH_VENV={selected['venv']}\nexport OFFICIAL_PYTORCH_ENV_KIND={selected['env']}\n", encoding='utf-8')
else: selected_env.write_text('export OFFICIAL_PYTORCH_VENV=\nexport OFFICIAL_PYTORCH_ENV_KIND=NONE\n', encoding='utf-8')
summary={'rows':rows,'selected_env':selected,'selected_env_sh':str(selected_env)}
(log/'env_matrix_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
md=['# Official PyTorch Environment Matrix','','| Env | Install Status | Import Smoke | Torch | Torch CUDA | CUDA Available | Missing Packages | Version Mismatches | Log |','|---|---|---|---|---|---:|---|---|---|']
for r in rows:
    mm='; '.join(r['missing_packages']) if r['missing_packages'] else '-'
    vm='; '.join([f"{x.get('package')} {x.get('actual')} != {x.get('expected')}" for x in r['version_mismatches']]) if r['version_mismatches'] else '-'
    md.append(f"| {r['env']} | {r['install_status']} / {r['check_status']} | {r['import_smoke']} | {r['torch'] or '-'} | {r['torch_cuda'] or '-'} | {r['cuda_available']} | {mm} | {vm} | `{r['log']}` |")
md += ['', '## Selected Environment', '', f"`{selected['env']}` at `{selected['venv']}`" if selected else '`NONE`']
(log/'env_matrix_summary.md').write_text('\n'.join(md)+'\n', encoding='utf-8')
print('\n'.join(md))
PY
