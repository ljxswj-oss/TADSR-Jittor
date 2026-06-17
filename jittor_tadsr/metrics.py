import math, subprocess
import numpy as np
from skimage.metrics import structural_similarity
def psnr(p,t):
    mse=float(np.mean((np.clip(p,0,1)-np.clip(t,0,1))**2)); return 99.0 if mse==0 else 10*math.log10(1.0/mse)
def ssim(p,t):
    return float(structural_similarity(np.clip(t.transpose(1,2,0),0,1),np.clip(p.transpose(1,2,0),0,1),channel_axis=2,data_range=1.0))
def count_params(m): return int(sum(int(np.prod(p.shape)) for p in m.parameters()))
def gpu_memory_text():
    try: return subprocess.check_output("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits",shell=True,text=True,timeout=5).strip()
    except Exception: return "unavailable"
