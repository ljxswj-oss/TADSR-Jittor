#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python - <<'PY'
from pathlib import Path
import json, pandas as pd, matplotlib.pyplot as plt
root=Path(".")
jlog=root/"experiments/jittor_tiny/logs/train_log.csv"; plog=root/"experiments/pytorch_tiny/logs/train_log.csv"
if jlog.exists() and plog.exists():
    jd=pd.read_csv(jlog); pdlog=pd.read_csv(plog)
    plt.figure(); plt.plot(jd.step,jd.total_loss,label="Jittor"); plt.plot(pdlog.step,pdlog.total_loss,label="PyTorch"); plt.legend(); plt.xlabel("step"); plt.ylabel("loss"); plt.tight_layout(); plt.savefig(root/"experiments/compare_loss_jittor_vs_pytorch.png",dpi=160); plt.close()
    plt.figure(); plt.plot(jd.step,jd.psnr,label="Jittor"); plt.plot(pdlog.step,pdlog.psnr,label="PyTorch"); plt.legend(); plt.xlabel("step"); plt.ylabel("PSNR"); plt.tight_layout(); plt.savefig(root/"experiments/compare_psnr_jittor_vs_pytorch.png",dpi=160); plt.close()
rows=[]
for fw,p in [("Jittor",root/"experiments/jittor_tiny/logs/test_metrics.json"),("PyTorch",root/"experiments/pytorch_tiny/logs/test_metrics.json")]:
    if p.exists():
        d=json.loads(p.read_text()); d["framework"]=fw; rows.append(d)
md=["# Jittor vs PyTorch Tiny Alignment Report",""]
if rows:
    df=pd.DataFrame(rows); cols=[c for c in ["framework","psnr","ssim","latency_ms_per_image","params","checkpoint_size_mb"] if c in df]
    md.append(df[cols].to_markdown(index=False))
md.append("\n结论：这里比较的是 tiny faithful reproduction 的训练趋势、输出形态和指标量级，不声称达到官方 full Stable Diffusion TADSR 的 SOTA 性能。")
(root/"experiments/alignment_report.md").write_text("\n".join(md),encoding="utf-8")
print("\n".join(md))
PY
