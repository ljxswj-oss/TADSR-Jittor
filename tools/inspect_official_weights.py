#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
from pathlib import Path
EXTS={'.pkl','.pth','.bin','.safetensors'}
def tensor_summary(name,t):
    if hasattr(t,'detach'):
        x=t.detach().cpu().float(); n=x.numel()
        return {'key':name,'shape':list(t.shape),'dtype':str(t.dtype),'mean':float(x.mean().item()) if n else 0.0,'std':float(x.std(unbiased=False).item()) if n else 0.0,'min':float(x.min().item()) if n else 0.0,'max':float(x.max().item()) if n else 0.0}
    return None
def summarize(obj,prefix=''):
    rows=[]
    if hasattr(obj,'state_dict'): obj=obj.state_dict()
    if isinstance(obj,dict):
        for k,v in obj.items():
            name=f'{prefix}.{k}' if prefix else str(k)
            r=tensor_summary(name,v)
            if r: rows.append(r)
            elif isinstance(v,dict): rows.extend(summarize(v,name))
    return rows
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--weights_dir',default='/mnt/data/sj/checkpoints/TADSR/preset/weights'); ap.add_argument('--out_dir',default='experiments/full_repro/weights'); args=ap.parse_args()
    weights=Path(args.weights_dir); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    if not weights.exists():
        msg=f'BLOCKED_WEIGHTS_DIR_MISSING: {weights}'; (out/'official_weight_inspection.md').write_text('# Official Weight Inspection\n\n'+msg+'\n',encoding='utf-8'); (out/'official_weight_inspection.json').write_text(json.dumps({'status':'BLOCKED','reason':msg},indent=2),encoding='utf-8'); print(msg); return 2
    all_files=[]
    for root, _, names in os.walk(weights, followlinks=True):
        for name in names:
            all_files.append(Path(root)/name)
    files=sorted([p for p in all_files if p.is_file() and p.suffix in EXTS])
    configs=sorted([p for p in all_files if p.name.endswith('.json') or p.name in {'config.json','model_index.json'}])
    if not files and not configs:
        msg=f'BLOCKED_NO_OFFICIAL_WEIGHT_FILES: {weights}'; (out/'official_weight_inspection.md').write_text('# Official Weight Inspection\n\n'+msg+'\n',encoding='utf-8'); (out/'official_weight_inspection.json').write_text(json.dumps({'status':'BLOCKED','reason':msg},indent=2),encoding='utf-8'); print(msg); return 2
    try: import torch
    except Exception as e: torch=None; torch_err=repr(e)
    try:
        from safetensors.torch import load_file as st_load
    except Exception as e: st_load=None; st_err=repr(e)
    report={'status':'PASS','weights_dir':str(weights),'configs':[str(p) for p in configs],'files':[]}
    if torch is None: report['torch_import_error']=torch_err
    if st_load is None: report['safetensors_import_error']=st_err
    for p in files:
        item={'path':str(p),'size_bytes':p.stat().st_size,'suffix':p.suffix,'status':'not_loaded','tensors':[]}
        try:
            if p.suffix=='.safetensors':
                if st_load is None: item['status']='blocked_safetensors_unavailable'
                else: item['tensors']=summarize(st_load(str(p),device='cpu'))[:2000]; item['status']='loaded'
            elif torch is None: item['status']='blocked_torch_unavailable'
            else: item['tensors']=summarize(torch.load(p,map_location='cpu'))[:2000]; item['status']='loaded'
        except Exception as e: item['status']='load_failed'; item['error']=repr(e)
        report['files'].append(item)
    (out/'official_weight_inspection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# Official Weight Inspection','',f'Weights directory: `{weights}`','',f'Config/model index files found: {len(configs)}','','| File | Status | Tensors shown | Size MB |','|---|---|---:|---:|']
    for it in report['files']: lines.append(f"| `{it['path']}` | {it['status']} | {len(it.get('tensors',[]))} | {it['size_bytes']/1024/1024:.3f} |")
    if configs: lines += ['','## Diffusers/config files','']+[f'- `{p}`' for p in configs]
    (out/'official_weight_inspection.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Inspected {len(files)} official weight files; configs={len(configs)}'); return 0
if __name__=='__main__': raise SystemExit(main())
