import os, subprocess, sys
os.environ.setdefault("JITTOR_HOME", os.path.abspath(".jittor_cache"))

def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True, timeout=20).strip()
    except Exception as exc:
        return f"ERROR: {exc}"

print("python", sys.version.replace("\n", " "))
print("hostname", sh("hostname"))
print("gpu", sh("nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits | head -6"))
import jittor as jt
from jittor import nn
print("jittor", jt.__version__)
if int(os.environ.get("USE_CUDA", "1")):
    jt.flags.use_cuda = 1
x = jt.randn((2, 3, 16, 16))
m = nn.Conv(3, 4, 3, padding=1)
y = m(x)
loss = (y * y).mean()
opt = nn.Adam(m.parameters(), lr=1e-3)
opt.step(loss)
print("jittor_forward_backward_ok", tuple(y.shape), float(loss.data))
print("CHECK_ENV_OK")
