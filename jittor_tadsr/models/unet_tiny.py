from jittor import nn
from .time_aware_encoder import FiLMConv

class TinyUNet(nn.Module):
    def __init__(self,ch=32,time_dim=128):
        super().__init__(); self.down=FiLMConv(ch,ch*2,time_dim,stride=2); self.mid=FiLMConv(ch*2,ch*2,time_dim); self.up=FiLMConv(ch*2,ch,time_dim); self.out=nn.Conv(ch,ch,3,padding=1)
    def execute(self,z,e):
        d=self.down(z,e); m=self.mid(d,e); u=nn.interpolate(m,size=z.shape[-2:],mode="bilinear"); return self.out(self.up(u,e)+z)
