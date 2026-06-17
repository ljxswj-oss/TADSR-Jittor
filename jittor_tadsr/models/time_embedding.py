import math
import jittor as jt
from jittor import nn

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim=64, hidden_dim=128):
        super().__init__(); self.dim=dim
        self.mlp=nn.Sequential(nn.Linear(dim,hidden_dim), nn.ReLU(), nn.Linear(hidden_dim,hidden_dim), nn.ReLU())
    def execute(self,t):
        if len(t.shape)==0: t=t.reshape((1,))
        half=self.dim//2
        freqs=jt.exp(jt.arange(half).float()*(-math.log(10000.0)/(half-1)))
        args=t.float().unsqueeze(1)*freqs.unsqueeze(0)
        emb=jt.concat([jt.sin(args),jt.cos(args)],dim=1)
        return self.mlp(emb)
