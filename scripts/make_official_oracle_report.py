#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path('experiments/full_repro')
OUT = ROOT / 'pytorch_official' / 'official_oracle_report.md'

def load_json(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception as exc:
        return {'status': 'READ_ERROR', 'error': repr(exc)}

def selected_env() -> tuple[str, str]:
    p = ROOT / 'pytorch_official/env_matrix/selected_env.sh'
    env, kind = '', 'NONE'
    if p.exists():
        for line in p.read_text().splitlines():
            if line.startswith('export OFFICIAL_PYTORCH_VENV='):
                env = line.split('=', 1)[1].strip().strip('"')
            if line.startswith('export OFFICIAL_PYTORCH_ENV_KIND='):
                kind = line.split('=', 1)[1].strip().strip('"')
    return env, kind

def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    env, kind = selected_env()
    env_check = load_json(ROOT / 'pytorch_official/env/env_check_strict-cu118.json') or load_json(ROOT / 'pytorch_official/env_matrix/env_check_strict-cu118.json')
    assets = load_json(ROOT / 'assets/offline_assets_manifest.json')
    smoke = load_json(ROOT / 'pytorch_official/smoke/smoke_status.json')
    audit = load_json('experiments/final_audit_report.json')
    subset_rows = []
    for ds in ['RealSR', 'DRealSR', 'RealLR200']:
        matches = sorted((ROOT / 'pytorch_official').glob(f'{ds}_subset_*/subset_status.json'))
        subset_rows.append((ds, load_json(matches[-1]) if matches else {'status': 'BLOCKED_DATASET_MISSING', 'reason': 'not run or dataset missing'}))
    torch_cuda = env_check.get('torch_cuda', {})
    lines = [
        '# Official PyTorch Oracle Report', '',
        'This report records the official PyTorch oracle state. It does not claim a full Jittor reproduction.', '',
        '## 1. Official Environment', '',
        f'- Selected env kind: `{kind}`',
        f'- Selected env path: `{env or "NONE"}`',
        f'- Environment check status: `{env_check.get("status", "MISSING")}`',
        f'- Torch CUDA: `{torch_cuda.get("torch_cuda", "-")}`',
        f'- cuDNN: `{torch_cuda.get("cudnn", "-")}`',
        f'- CUDA available: `{torch_cuda.get("available", "-")}`',
        f'- GPU devices: `{", ".join(torch_cuda.get("devices", [])) if torch_cuda.get("devices") else "-"}`', '',
        '## 2. Official Assets', '',
        f'- Assets input dir: `{assets.get("input_dir", "/mnt/data/sj/incoming/TADSR_assets/TADSR_weights")}`',
        f'- Missing items: `{", ".join(assets.get("missing", [])) if assets.get("missing") else "none"}`', '',
        '| Required item | Exists | Type | Size |', '|---|---:|---|---:|'
    ]
    for item in assets.get('items', []):
        lines.append(f"| `{item.get('name')}` | {item.get('exists')} | {item.get('type', 'missing')} | {item.get('size', '-')} |")
    lines += ['', '## 3. Smoke Inference', '']
    if smoke:
        lines += [f'- Status: `{smoke.get("status")}`', f'- Reason: `{smoke.get("reason", "-")}`', f'- Input count: `{smoke.get("input_count", "-")}`', f'- Output count: `{smoke.get("output_count", "-")}`', f'- Runtime sec: `{smoke.get("runtime_sec", "-")}`', f'- Visual grid: `experiments/full_repro/pytorch_official/smoke/visual_grid.png`']
    else:
        lines += ['- Status: `BLOCKED`', '- Reason: official weights are missing, so smoke was not run.']
    lines += ['', '## 4. Subset Benchmark', '', '| Dataset | Status | Reason |', '|---|---|---|']
    for ds, data in subset_rows:
        lines.append(f"| `{ds}` | `{data.get('status', 'MISSING')}` | {data.get('reason', 'not run')} |")
    lines += ['', '## 5. Current Blockers', '']
    if audit:
        for row in audit.get('rows', []):
            if row.get('status') in {'BLOCKED', 'PARTIAL', 'FAIL'}:
                lines.append(f"- `{row.get('check')}`: {row.get('status')} - {row.get('note')}")
    lines += ['', '## 6. Next Jittor Migration Action', '',
              '1. Upload RealSR/DRealSR/RealLR200 benchmark datasets if paper-level metrics are needed.',
              '2. Continue Jittor module-level alignment from the official PyTorch smoke oracle.',
              '3. Keep official weights under /mnt/data/sj and avoid copying them into git.',
              '4. Use the official PyTorch oracle for module-level Jittor alignment: scheduler, preprocessing, VAE, LoRA, UNet blocks, and full inference.',
              '5. Do not claim full Jittor reproduction until oracle-aligned outputs and benchmarks pass.']
    OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(OUT)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
