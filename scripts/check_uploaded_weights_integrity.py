#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REQUIRED_DIRS = [
    'time_vae', 'unet', 'vae', 'text_encoder', 'tokenizer', 'scheduler',
    'feature_extractor', 'bert-base-uncased',
]
REQUIRED_FILES = ['DAPE.pth', 'ram_swin_large_14m.pth', 'tadsr.pkl']
MODEL_FILE_OPTIONS = {
    'unet': ['diffusion_pytorch_model.safetensors', 'diffusion_pytorch_model.bin'],
    'vae': ['diffusion_pytorch_model.safetensors', 'diffusion_pytorch_model.bin'],
    'time_vae': ['diffusion_pytorch_model.safetensors', 'diffusion_pytorch_model.bin'],
}

def human(n: int) -> str:
    value = float(n)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if value < 1024 or unit == 'TB':
            return f'{value:.3f} {unit}'
        value /= 1024

def dir_stats(path: Path) -> tuple[int, int]:
    count = 0
    total = 0
    for item in path.rglob('*'):
        if item.is_file():
            count += 1
            total += item.stat().st_size
    return count, total

def parse_selected_env(path: Path) -> str:
    if not path.exists(): return ''
    for line in path.read_text().splitlines():
        if line.startswith('export OFFICIAL_PYTORCH_VENV='):
            return line.split('=', 1)[1].strip().strip('"')
    return ''

def safe_open_check(file_path: Path, python_exe: str | None) -> dict:
    if python_exe is None:
        return {'path': str(file_path), 'status': 'SKIPPED_NO_PYTHON'}
    code = """
import json, sys
from safetensors import safe_open
p=sys.argv[1]
try:
    with safe_open(p, framework='pt', device='cpu') as f:
        keys=list(f.keys())
    print(json.dumps({'status':'PASS','key_count':len(keys),'first_keys':keys[:10]}))
except Exception as e:
    print(json.dumps({'status':'FAIL','error':repr(e)}))
    raise SystemExit(2)
"""
    proc = subprocess.run([python_exe, '-c', code, str(file_path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        data = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        data = {'status': 'FAIL', 'stdout': proc.stdout, 'stderr': proc.stderr, 'returncode': proc.returncode}
    data['path'] = str(file_path)
    if proc.returncode != 0 and data.get('status') != 'FAIL':
        data['status'] = 'FAIL'
    return data

def torch_load_keys(file_path: Path, python_exe: str | None) -> dict:
    if python_exe is None:
        return {'path': str(file_path), 'status': 'SKIPPED_NO_PYTHON'}
    code = """
import json, sys, torch
p=sys.argv[1]
try:
    obj=torch.load(p, map_location='cpu')
    if hasattr(obj, 'keys'):
        keys=[str(k) for k in list(obj.keys())]
    elif hasattr(obj, 'state_dict'):
        keys=[str(k) for k in list(obj.state_dict().keys())]
    else:
        keys=[]
    print(json.dumps({'status':'PASS','type':type(obj).__name__,'key_count':len(keys),'first_keys':keys[:20]}))
except Exception as e:
    print(json.dumps({'status':'FAIL','error':repr(e)}))
    raise SystemExit(2)
"""
    proc = subprocess.run([python_exe, '-c', code, str(file_path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        data = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        data = {'status': 'FAIL', 'stdout': proc.stdout, 'stderr': proc.stderr, 'returncode': proc.returncode}
    data['path'] = str(file_path)
    if proc.returncode != 0 and data.get('status') != 'FAIL':
        data['status'] = 'FAIL'
    return data

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--upload_dir', default='/mnt/data/sj/incoming/TADSR_assets/TADSR_weights')
    ap.add_argument('--out_dir', default='experiments/full_repro/assets')
    ap.add_argument('--selected_env', default='experiments/full_repro/pytorch_official/env_matrix/selected_env.sh')
    args = ap.parse_args()
    root = Path(args.upload_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    venv = parse_selected_env(Path(args.selected_env))
    python_exe = str(Path(venv) / 'bin/python') if venv and (Path(venv) / 'bin/python').exists() else None
    report = {
        'upload_dir': str(root),
        'selected_venv': venv,
        'python_executable': python_exe,
        'missing': [],
        'empty_dirs': [],
        'zero_size_files': [],
        'suspicious': [],
        'items': [],
        'safetensors_checks': [],
        'torch_load_checks': [],
    }
    if not root.exists():
        report['status'] = 'FAIL'
        report['missing'].append(str(root))
    else:
        for name in REQUIRED_DIRS:
            p = root / name
            item = {'name': name, 'path': str(p), 'expected_type': 'dir', 'exists': p.exists(), 'is_dir': p.is_dir()}
            if not p.exists() or not p.is_dir():
                report['missing'].append(name)
            else:
                count, total = dir_stats(p)
                item.update({'file_count': count, 'size_bytes': total, 'size': human(total)})
                if count == 0 or total == 0:
                    report['empty_dirs'].append(name)
            report['items'].append(item)
        for name in REQUIRED_FILES:
            p = root / name
            item = {'name': name, 'path': str(p), 'expected_type': 'file', 'exists': p.exists(), 'is_file': p.is_file()}
            if not p.exists() or not p.is_file():
                report['missing'].append(name)
            else:
                size = p.stat().st_size
                item.update({'size_bytes': size, 'size': human(size)})
                if size == 0:
                    report['zero_size_files'].append(name)
            report['items'].append(item)
        for dirname, names in MODEL_FILE_OPTIONS.items():
            found = [root / dirname / n for n in names if (root / dirname / n).exists()]
            if not found:
                report['missing'].append(f'{dirname}/model_file')
            else:
                for fp in found:
                    if fp.stat().st_size == 0:
                        report['zero_size_files'].append(str(fp.relative_to(root)))
        for fp in root.rglob('*.safetensors'):
            report['safetensors_checks'].append(safe_open_check(fp, python_exe))
        if (root / 'tadsr.pkl').exists():
            report['torch_load_checks'].append(torch_load_keys(root / 'tadsr.pkl', python_exe))
    failures = []
    for key in ['missing', 'empty_dirs', 'zero_size_files', 'suspicious']:
        failures.extend(report.get(key, []))
    failures.extend([x for x in report['safetensors_checks'] if x.get('status') != 'PASS'])
    failures.extend([x for x in report['torch_load_checks'] if x.get('status') != 'PASS'])
    report['status'] = 'PASS' if not failures else 'FAIL'
    json_path = out / 'upload_integrity_report.json'
    md_path = out / 'upload_integrity_report.md'
    json_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# Uploaded TADSR Weights Integrity Report', '', f'Status: **{report["status"]}**', '', f'Upload dir: `{root}`', f'Selected venv: `{venv or "NONE"}`', '', '## Required Items', '', '| Item | Expected | Exists | File count | Size |', '|---|---|---:|---:|---:|']
    for item in report['items']:
        lines.append(f"| `{item['name']}` | {item['expected_type']} | {item.get('exists')} | {item.get('file_count', '-')} | {item.get('size', '-')} |")
    for title, key in [('Missing', 'missing'), ('Empty dirs', 'empty_dirs'), ('Zero-size files', 'zero_size_files'), ('Suspicious', 'suspicious')]:
        lines += ['', f'## {title}', '']
        vals = report.get(key, [])
        lines += [f'- `{x}`' for x in vals] if vals else ['None']
    lines += ['', '## Safetensors Checks', '', '| File | Status | Key count |', '|---|---|---:|']
    for x in report['safetensors_checks']:
        lines.append(f"| `{x.get('path')}` | {x.get('status')} | {x.get('key_count', '-')} |")
    lines += ['', '## torch.load Checks', '', '| File | Status | Type | Key count | First keys |', '|---|---|---|---:|---|']
    for x in report['torch_load_checks']:
        lines.append(f"| `{x.get('path')}` | {x.get('status')} | {x.get('type', '-')} | {x.get('key_count', '-')} | `{', '.join(x.get('first_keys', [])[:5])}` |")
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Upload integrity: {report["status"]}')
    print(f'JSON: {json_path}')
    print(f'Markdown: {md_path}')
    if report['status'] != 'PASS':
        print('Missing/corrupt/suspicious items detected; do not continue official smoke.')
        return 2
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
