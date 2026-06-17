#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
def stats(a):
    x=a.astype(np.float64,copy=False); return {'shape':list(a.shape),'dtype':str(a.dtype),'mean':float(x.mean()) if x.size else 0.0,'std':float(x.std()) if x.size else 0.0,'min':float(x.min()) if x.size else 0.0,'max':float(x.max()) if x.size else 0.0}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--conversion_manifest',default='experiments/full_repro/weights/conversion_manifest.json'); ap.add_argument('--report_dir',default='experiments/full_repro/weights'); args=ap.parse_args(); rep=Path(args.report_dir); rep.mkdir(parents=True,exist_ok=True)
    mp=Path(args.conversion_manifest)
    if not mp.exists(): msg='BLOCKED: conversion_manifest.json does not exist.'; (rep/'weight_conversion_report.md').write_text('# Weight Conversion Report\n\n'+msg+'\n',encoding='utf-8'); print(msg); return 2
    man=json.loads(mp.read_text())
    if man.get('status')=='BLOCKED': msg='BLOCKED: conversion did not run because official weights are missing.'; (rep/'weight_conversion_report.md').write_text('# Weight Conversion Report\n\n'+msg+'\n',encoding='utf-8'); print(msg); return 2
    rows=[]; ok=True
    for it in man.get('converted_files',[]):
        target=Path(it.get('target',''))
        if it.get('status')!='converted' or not target.exists(): rows.append({'source':it.get('source'),'status':'missing_npz'}); ok=False; continue
        npz=np.load(target); by={t['npz_key']:t for t in it.get('tensors',[])}
        for k in npz.files[:5000]:
            cur=stats(npz[k]); orig=by.get(k,{}); match=cur.get('shape')==orig.get('shape') and cur.get('dtype')==orig.get('dtype')
            for s in ['mean','std','min','max']:
                if abs(float(cur.get(s,0))-float(orig.get(s,0)))>1e-8: match=False
            rows.append({'source':it.get('source'),'key':k,'match':match,'current':cur,'original':orig}); ok=ok and match
    result={'status':'PASS' if ok else 'FAIL','rows_checked':len(rows),'rows':rows[:2000]}; (rep/'weight_conversion_report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    md=['# Weight Conversion Report','',f"Status: **{result['status']}**",'',f'Rows checked: {len(rows)}','','| Source | Key | Match | Shape | Dtype |','|---|---|---:|---|---|']
    for r in rows[:300]: md.append(f"| `{r.get('source','')}` | `{r.get('key','')}` | {r.get('match',False)} | `{r.get('current',{}).get('shape','')}` | `{r.get('current',{}).get('dtype','')}` |")
    (rep/'weight_conversion_report.md').write_text('\n'.join(md)+'\n',encoding='utf-8'); print(f"Weight conversion verification: {result['status']} rows={len(rows)}"); return 0 if ok else 1
if __name__=='__main__': raise SystemExit(main())
