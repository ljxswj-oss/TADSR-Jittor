import jittor as jt

class TinyDDPMScheduler:
    def __init__(self,n=1000):
        b=jt.linspace(1e-4,0.02,n); self.alphas_cumprod=jt.cumprod(1-b,dim=0)
    def _a(self,t,like):
        return self.alphas_cumprod[t].reshape((t.shape[0],1,1,1)).broadcast(like.shape)
    def get_x0_from_res(self, latent, pred, t):
        a=self._a(t,latent); return latent/jt.sqrt(a)-pred
