import jittor as jt
from jittor import nn

class FiLMConv(nn.Module):
    def __init__(self, ic, oc, td, stride=1):
        super().__init__(); self.conv=nn.Conv(ic,oc,3,stride=stride,padding=1); self.aff=nn.Linear(td,oc*2); self.act=nn.ReLU()
    def execute(self,x,e):
        h=self.conv(x); gb=self.aff(e); g,b=jt.chunk(gb,2,dim=1)
        return self.act(h*(1+g.reshape((g.shape[0],g.shape[1],1,1)))+b.reshape((b.shape[0],b.shape[1],1,1)))

class TimeAwareEncoder(nn.Module):
    def __init__(self, in_ch=3, base=32, latent_ch=32, time_dim=128):
        super().__init__(); self.b1=FiLMConv(in_ch,base,time_dim); self.b2=FiLMConv(base,base*2,time_dim,stride=2); self.b3=FiLMConv(base*2,latent_ch,time_dim)
    def execute(self,x,e):
        h1=self.b1(x,e); return self.b3(self.b2(h1,e),e), h1
