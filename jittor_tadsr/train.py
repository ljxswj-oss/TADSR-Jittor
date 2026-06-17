import argparse, os, time
from pathlib import Path
import numpy as np
os.environ.setdefault("JITTOR_HOME", os.path.abspath(".jittor_cache"))
import jittor as jt
from jittor import nn
from .dataset import read_dataset_list, sample_batch
from .models import TinyTADSR
from .models.losses import l1_loss, gradient_loss, tavsd_like_loss
from .metrics import psnr, count_params, gpu_memory_text
from .utils import set_seed, ensure_dirs, append_csv, save_grid

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset_list",required=True); ap.add_argument("--output_dir",required=True); ap.add_argument("--image_size",type=int,default=128); ap.add_argument("--batch_size",type=int,default=2); ap.add_argument("--max_steps",type=int,default=300); ap.add_argument("--lr",type=float,default=1e-4); ap.add_argument("--seed",type=int,default=123); ap.add_argument("--use_cuda",type=int,default=1)
    a=ap.parse_args(); set_seed(a.seed)
    if a.use_cuda: jt.flags.use_cuda=1
    out=Path(a.output_dir); ensure_dirs(out/"logs",out/"checkpoints",out/"curves",out/"visuals")
    items=read_dataset_list(a.dataset_list); model=TinyTADSR(); teacher=TinyTADSR(); opt=nn.Adam(model.parameters(),lr=a.lr)
    print(f"TRAIN_START items={len(items)} steps={a.max_steps} params={count_params(model)} gpu={gpu_memory_text()}")
    t0=time.time()
    for step in range(1,a.max_steps+1):
        lr_np,hr_np,_,_=sample_batch(items,a.batch_size,a.image_size); lr=jt.array(lr_np); hr=jt.array(hr_np); tv=jt.array(np.random.randint(1,900,size=(a.batch_size,),dtype="int32"))
        sr,_,pred=model(lr,tv)
        with jt.no_grad(): _,_,tp=teacher(lr,tv)
        rec=l1_loss(sr,hr); edge=gradient_loss(sr,hr); tavsd=tavsd_like_loss(pred,tp); loss=rec+0.05*edge+0.01*tavsd
        opt.step(loss)
        if step%10==0 or step==1:
            row={"step":step,"total_loss":float(loss.data),"rec_loss":float(rec.data),"edge_loss":float(edge.data),"tavsd_loss":float(tavsd.data),"psnr":psnr(sr.data[0],hr_np[0]),"lr":a.lr,"time_sec":time.time()-t0}
            append_csv(out/"logs/train_log.csv",row); print("LOG",row,flush=True)
        if step%50==0 or step==a.max_steps: save_grid(lr_np[0],sr.data[0],hr_np[0],out/f"visuals/step_{step:04d}_grid.png")
        if step%100==0 or step==a.max_steps: jt.save(model.state_dict(),str(out/f"checkpoints/step_{step:04d}.pkl")); jt.save(model.state_dict(),str(out/"checkpoints/latest.pkl"))
    import subprocess, sys; subprocess.call([sys.executable,"scripts/plot_loss.py","--csv",str(out/"logs/train_log.csv"),"--out_dir",str(out/"curves")])
    print("TRAIN_DONE",out/"checkpoints/latest.pkl")
if __name__=="__main__": main()
