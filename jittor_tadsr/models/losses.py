import jittor as jt
def l1_loss(a,b): return jt.abs(a-b).mean()
def gradient_loss(a,b):
    ax=jt.abs(a[:,:,:,1:]-a[:,:,:,:-1]); bx=jt.abs(b[:,:,:,1:]-b[:,:,:,:-1])
    ay=jt.abs(a[:,:,1:,:]-a[:,:,:-1,:]); by=jt.abs(b[:,:,1:,:]-b[:,:,:-1,:])
    return jt.abs(ax-bx).mean()+jt.abs(ay-by).mean()
def tavsd_like_loss(a,b): return ((a-b.stop_grad())*(a-b.stop_grad())).mean()
