import argparse, os, time, json
from pathlib import Path
import numpy as np
os.environ.setdefault("JITTOR_HOME", os.path.abspath(".jittor_cache"))
import jittor as jt
from .dataset import read_dataset_list, iter_items
from .models import TinyTADSR
from .metrics import psnr, ssim, count_params
from .utils import ensure_dirs, save_image, save_grid, write_json

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset_list",required=True); ap.add_argument("--checkpoint",required=True); ap.add_argument("--output_dir",required=True); ap.add_argument("--image_size",type=int,default=128); ap.add_argument("--timesteps",default="50,200,500"); ap.add_argument("--use_cuda",type=int,default=1)
    a=ap.parse_args()
    if a.use_cuda: jt.flags.use_cuda=1
    out=Path(a.output_dir); ensure_dirs(out,out.parent/"logs")
    items=read_dataset_list(a.dataset_list); model=TinyTADSR(); model.load_state_dict(jt.load(a.checkpoint)); model.eval(); ts=[int(x) for x in a.timesteps.split(",")]
    ps,ss,lat=[],[],[]; first=False
    for i,(lr_np,hr_np,item) in enumerate(iter_items(items,a.image_size)):
        lr=jt.array(lr_np); variants=[]; st=time.time()
        for tv in ts:
            sr,_,_=model(lr,jt.array(np.array([tv],dtype="int32"))); arr=sr.data[0]; variants.append(arr); save_image(arr,out/f"{i:04d}_t{tv}.png")
        lat.append((time.time()-st)*1000/len(ts)); ps.append(psnr(variants[0],hr_np[0])); ss.append(ssim(variants[0],hr_np[0]))
        if not first: save_grid(lr_np[0],variants[0],hr_np[0],out/"compare_grid.png"); first=True
    metrics={"psnr":float(np.mean(ps)),"ssim":float(np.mean(ss)),"latency_ms_per_image":float(np.mean(lat)),"params":count_params(model),"checkpoint_size_mb":Path(a.checkpoint).stat().st_size/1024/1024,"num_images":len(items)}
    write_json(out.parent/"logs/test_metrics.json",metrics); print("TEST_METRICS",json.dumps(metrics,indent=2))
if __name__=="__main__": main()
