#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
TARGETS=['tadsr.pkl','DAPE.pth','ram_swin_large_14m.pth']
def flatten(obj,prefix=''):
    rows=[]
    if hasattr(obj,'state_dict'): obj=obj.state_dict()
    if isinstance(obj,dict):
        for k,v in obj.items():
            name=f'{prefix}.{k}' if prefix else str(k)
            if hasattr(v,'detach'): rows.append((name,v.detach().cpu().numpy()))
            elif isinstance(v,dict): rows.extend(flatten(v,name))
    return rows
def key(k): return k.replace('.','__').replace('/','__').replace(':','__')
def stats(a):
    x=a.astype(np.float64,copy=False); return {'shape':list(a.shape),'dtype':str(a.dtype),'mean':float(x.mean()) if x.size else 0.0,'std':float(x.std()) if x.size else 0.0,'min':float(x.min()) if x.size else 0.0,'max':float(x.max()) if x.size else 0.0}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--weights_dir',default='/mnt/data/sj/checkpoints/TADSR/preset/weights'); ap.add_argument('--output_dir',default='/mnt/data/sj/checkpoints/TADSR/jittor_converted'); ap.add_argument('--report_dir',default='experiments/full_repro/weights'); args=ap.parse_args()
    w=Path(args.weights_dir); out=Path(args.output_dir); rep=Path(args.report_dir); out.mkdir(parents=True,exist_ok=True); rep.mkdir(parents=True,exist_ok=True)
    missing=[n for n in TARGETS if not (w/n).exists()]
    if missing:
        msg='BLOCKED_MISSING_WEIGHT_FILES: '+', '.join(missing); (rep/'conversion_manifest.md').write_text('# Conversion Manifest\n\n'+msg+'\n',encoding='utf-8'); (rep/'conversion_manifest.json').write_text(json.dumps({'status':'BLOCKED','missing':missing},indent=2),encoding='utf-8'); print(msg); return 2
    try: import torch
    except Exception as e:
        msg=f'BLOCKED_TORCH_IMPORT_FAILED: {e!r}'; (rep/'conversion_manifest.md').write_text('# Conversion Manifest\n\n'+msg+'\n',encoding='utf-8'); print(msg); return 2
    manifest={'status':'PASS','converted_files':[]}
    for n in TARGETS:
        p=w/n; item={'source':str(p),'target':str(out/(p.stem+'_weights.npz')),'tensors':[]}
        try:
            arrays={}
            for orig,a in flatten(torch.load(p,map_location='cpu')):
                sk=key(orig); arrays[sk]=a; item['tensors'].append({'original_key':orig,'npz_key':sk,**stats(a)})
            np.savez(item['target'],**arrays); item['status']='converted'
        except Exception as e: item['status']='load_or_convert_failed'; item['error']=repr(e)
        manifest['converted_files'].append(item)
    (rep/'conversion_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    lines=['# Conversion Manifest','','| Source | Status | Tensor count | Target |','|---|---|---:|---|']
    for it in manifest['converted_files']: lines.append(f"| `{it['source']}` | {it['status']} | {len(it.get('tensors',[]))} | `{it['target']}` |")
    (rep/'conversion_manifest.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    failed=[i for i in manifest['converted_files'] if i.get('status')!='converted']; print(f'Converted {len(manifest["converted_files"])-len(failed)} files; failed={len(failed)}'); return 1 if failed else 0
if __name__=='__main__': raise SystemExit(main())
