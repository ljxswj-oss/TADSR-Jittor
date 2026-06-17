#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess
from pathlib import Path

ROOT = Path('/mnt/data/sj/projects/TADSR-Jittor')
OFFICIAL = Path('/mnt/data/sj/projects/TADSR_official_pytorch')
OUT = ROOT / 'experiments/full_repro/pytorch_official/env_matrix'
SELECTED = OUT / 'selected_env.sh'
TARGETS = ['tadsr', 'test_tadsr', 'train_tadsr', 'dataloaders', 'models', 'ram', 'my_utils', 'diffusers']

def selected_venv() -> tuple[str, str]:
    if not SELECTED.exists():
        return '', 'NONE'
    env = ''
    kind = 'NONE'
    for line in SELECTED.read_text().splitlines():
        if line.startswith('export OFFICIAL_PYTORCH_VENV='):
            env = line.split('=', 1)[1].strip().strip('"')
        if line.startswith('export OFFICIAL_PYTORCH_ENV_KIND='):
            kind = line.split('=', 1)[1].strip().strip('"')
    return env, kind

def classify(output: str) -> str:
    low = output.lower()
    if any(x in low for x in ['no such file', 'checkpoint', 'weight', 'pretrained', 'dataset', 'data root']):
        return 'IMPORT_WITH_RUNTIME_DEPENDENCY'
    if 'modulenotfounderror' in low or 'importerror' in low:
        return 'IMPORT_FAIL'
    return 'IMPORT_FAIL'

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    venv, kind = selected_venv()
    rows = []
    if not venv or not (Path(venv) / 'bin/python').exists():
        data = {'status': 'BLOCKED_NO_SELECTED_ENV', 'selected_env': venv, 'env_kind': kind, 'rows': []}
        (OUT/'official_repo_imports.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
        (OUT/'official_repo_imports.md').write_text('# Official Repo Imports\n\nStatus: **BLOCKED_NO_SELECTED_ENV**\n', encoding='utf-8')
        print('BLOCKED_NO_SELECTED_ENV')
        return 2
    if not OFFICIAL.exists():
        data = {'status': 'BLOCKED_OFFICIAL_REPO_MISSING', 'official_repo': str(OFFICIAL), 'rows': []}
        (OUT/'official_repo_imports.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
        return 2
    py = str(Path(venv) / 'bin/python')
    for target in TARGETS:
        cmd = [py, '-c', f'import {target}; print("IMPORT_OK:{target}")']
        p = subprocess.run(cmd, cwd=OFFICIAL, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        combined = (p.stdout or '') + (p.stderr or '')
        rows.append({'target': target, 'returncode': p.returncode, 'status': 'PASS' if p.returncode == 0 else classify(combined), 'stdout': p.stdout, 'stderr': p.stderr})
    pass_count = sum(1 for r in rows if r['status'] == 'PASS')
    if pass_count == len(rows): status = 'PASS'
    elif pass_count > 0: status = 'PARTIAL'
    else: status = 'BLOCKED'
    data = {'status': status, 'selected_env': venv, 'env_kind': kind, 'official_repo': str(OFFICIAL), 'rows': rows}
    (OUT/'official_repo_imports.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    md = ['# Official Repo Imports', '', f'Status: **{status}**', '', f'Selected env: `{venv}` ({kind})', '', '| Target | Status | Return code |', '|---|---|---:|']
    for r in rows:
        md.append(f"| `{r['target']}` | {r['status']} | {r['returncode']} |")
    (OUT/'official_repo_imports.md').write_text('\n'.join(md)+'\n', encoding='utf-8')
    print(status)
    return 0 if status in {'PASS','PARTIAL'} else 2

if __name__ == '__main__':
    raise SystemExit(main())
