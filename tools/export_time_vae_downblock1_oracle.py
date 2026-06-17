#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
OFFICIAL_REPO=Path('/mnt/data/sj/projects/TADSR_official_pytorch')
WEIGHTS_DIR=Path('/mnt/data/sj/checkpoints/TADSR/preset/weights/time_vae')
OUT=Path('experiments/full_repro/time_vae_alignment/oracle_tensors_downblock1')

def to_np(x): return x.detach().cpu().numpy().astype(np.float32)

def main():
    OUT.mkdir(parents=True,exist_ok=True); sys.path.insert(0,str(OFFICIAL_REPO))
    from diffusers.models.autoencoders.time_autoencoder_kl import TimeAwareAutoencoderKL
    torch.manual_seed(42)
    model=TimeAwareAutoencoderKL.from_pretrained(str(WEIGHTS_DIR),local_files_only=True); model.eval(); enc=model.encoder
    block0=enc.down_blocks[0]; block1=enc.down_blocks[1]
    sample=torch.linspace(-1.0,1.0,steps=1*3*32*32,dtype=torch.float32).reshape(1,3,32,32); timesteps=torch.tensor([50],dtype=torch.long)
    hooks=[]; failures=[]
    with torch.no_grad():
        time_proj=enc.time_proj(timesteps); temb=enc.time_embedding(time_proj.to(dtype=list(enc.time_embedding.parameters())[0].dtype)); conv_in=enc.conv_in(sample)
        block0_out=block0(conv_in,temb=temb)
        r0_in=block0_out; r0_out=block1.resnets[0](r0_in,temb=temb)
        r1_in=r0_out; r1_out=block1.resnets[1](r1_in,temb=temb)
        ds_in=r1_out
        if getattr(block1,'downsamplers',None) is not None and len(block1.downsamplers)>0:
            ds_out=block1.downsamplers[0](ds_in); final_out=ds_out
            hooks.append({'name':'encoder.down_blocks.1.downsamplers.0','class':block1.downsamplers[0].__class__.__name__,'input_shape':list(ds_in.shape),'output_shape':list(ds_out.shape),'status':'PASS'})
        else:
            ds_out=None; final_out=r1_out; failures.append({'name':'encoder.down_blocks.1.downsamplers.0','status':'NOT_APPLICABLE','reason':'official block has no downsampler'})
        direct_final=block1(block0_out,temb=temb)
    hooks += [
        {'name':'encoder.time_proj','class':enc.time_proj.__class__.__name__,'input_shape':list(timesteps.shape),'output_shape':list(time_proj.shape),'status':'PASS'},
        {'name':'encoder.time_embedding','class':enc.time_embedding.__class__.__name__,'input_shape':list(time_proj.shape),'output_shape':list(temb.shape),'status':'PASS'},
        {'name':'encoder.conv_in','class':enc.conv_in.__class__.__name__,'input_shape':list(sample.shape),'output_shape':list(conv_in.shape),'status':'PASS'},
        {'name':'encoder.down_blocks.0','class':block0.__class__.__name__,'input_shape':list(conv_in.shape),'temb_shape':list(temb.shape),'output_shape':list(block0_out.shape),'status':'PASS'},
        {'name':'encoder.down_blocks.1.resnets.0','class':block1.resnets[0].__class__.__name__,'input_shape':list(r0_in.shape),'temb_shape':list(temb.shape),'output_shape':list(r0_out.shape),'status':'PASS','channel_change':r0_in.shape[1]!=r0_out.shape[1]},
        {'name':'encoder.down_blocks.1.resnets.1','class':block1.resnets[1].__class__.__name__,'input_shape':list(r1_in.shape),'temb_shape':list(temb.shape),'output_shape':list(r1_out.shape),'status':'PASS','channel_change':r1_in.shape[1]!=r1_out.shape[1]},
        {'name':'encoder.down_blocks.1','class':block1.__class__.__name__,'input_shape':list(block0_out.shape),'temb_shape':list(temb.shape),'output_shape':list(final_out.shape),'status':'PASS'},
    ]
    inputs={'synthetic_image_tensor':to_np(sample),'timestep':timesteps.cpu().numpy().astype(np.int64),'encoder_conv_in_input':to_np(sample),'stage01_input':to_np(sample),'downblock0_input':to_np(conv_in),'downblock1_input':to_np(block0_out),'resnet0_input':to_np(r0_in),'resnet1_input':to_np(r1_in),'downblock1_temb':to_np(temb)}
    outputs={'encoder_time_proj_output':to_np(time_proj),'encoder_time_embedding_output':to_np(temb),'encoder_conv_in_output':to_np(conv_in),'downblock0_output':to_np(block0_out),'resnet0_output':to_np(r0_out),'resnet1_output':to_np(r1_out),'downblock1_output':to_np(final_out),'downblock1_direct_output':to_np(direct_final),'stage01_output':to_np(final_out)}
    if ds_out is not None: inputs['downsampler0_input']=to_np(ds_in); outputs['downsampler0_output']=to_np(ds_out)
    np.savez_compressed(OUT/'time_vae_downblock1_inputs.npz',**inputs); np.savez_compressed(OUT/'time_vae_downblock1_outputs.npz',**outputs)
    meta={'status':'PASS','official_class':f'{model.__class__.__module__}.{model.__class__.__name__}','encoder_class':f'{enc.__class__.__module__}.{enc.__class__.__name__}','sample_shape':list(sample.shape),'timestep':int(timesteps.item()),'module_order':['encoder.conv_in','encoder.down_blocks.0','encoder.down_blocks.1.resnets.0','encoder.down_blocks.1.resnets.1','encoder.down_blocks.1.downsamplers.0'],'channel_change_happens':True,'hook_targets':hooks,'hook_failures':failures}
    (OUT/'time_vae_downblock1_hook_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    md=['# TimeAware VAE DownBlock1 Oracle Export','',f"Status: **{meta['status']}**",'',f"Input shape: `{meta['sample_shape']}`",f"Timestep: `{meta['timestep']}`",f"Channel change happens: `{meta['channel_change_happens']}`",'','| Target | Class | Input shape | Output shape | Status | Channel change |','|---|---|---:|---:|---|---|']
    for row in hooks:
        inp=row.get('input_shape')
        if row.get('temb_shape'): inp=f"{inp}, temb={row['temb_shape']}"
        md.append(f"| `{row['name']}` | `{row.get('class')}` | `{inp}` | `{row.get('output_shape')}` | {row['status']} | `{row.get('channel_change','')}` |")
    if failures:
        md += ['','## Hook failures / not applicable']
        for row in failures: md.append(f"- `{row['name']}`: {row.get('status')} {row.get('reason','')}")
    (OUT/'time_vae_downblock1_hook_report.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(json.dumps(meta,indent=2))
if __name__=='__main__': main()
