import jittor as jt
from jittor import nn
from .time_embedding import SinusoidalTimeEmbedding
from .time_aware_encoder import TimeAwareEncoder, FiLMConv
from .unet_tiny import TinyUNet
from .scheduler import TinyDDPMScheduler

class TinyDecoder(nn.Module):
    def __init__(self, latent_ch=32, base=32, td=128):
        super().__init__(); self.b1=FiLMConv(latent_ch,base,td); self.b2=FiLMConv(base,base,td); self.out=nn.Conv(base,3,3,padding=1)
    def execute(self,z,e,out_size):
        h=nn.interpolate(z,size=out_size,mode="bilinear"); h=self.b1(h,e); h=self.b2(h,e); return jt.sigmoid(self.out(h))

class TinyTADSR(nn.Module):
    def __init__(self, base=32, latent_ch=32, td=128):
        super().__init__(); self.time_embed=SinusoidalTimeEmbedding(64,td); self.encoder=TimeAwareEncoder(3,base,latent_ch,td); self.unet=TinyUNet(latent_ch,td); self.decoder=TinyDecoder(latent_ch,base,td); self.scheduler=TinyDDPMScheduler()
    def execute(self,lr,t):
        e=self.time_embed(t); latent,_=self.encoder(lr,e); pred=self.unet(latent,e); x0=self.scheduler.get_x0_from_res(latent,pred,t); sr=self.decoder(x0,e,lr.shape[-2:]); return sr,latent,pred
