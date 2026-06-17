#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
OFFICIAL_REPO=Path('/mnt/data/sj/projects/TADSR_official_pytorch')
WEIGHTS_DIR=Path('/mnt/data/sj/checkpoints/TADSR/preset/weights/time_vae')
OUT=Path('experiments/full_repro/time_vae_alignment')
BLOCK_INDEX=1

def module_tree(module, prefix=f'encoder.down_blocks.{BLOCK_INDEX}'):
    rows=[]
    for name, child in module.named_modules():
        full=prefix if not name else f'{prefix}.{name}'
        row={'name':full,'class':child.__class__.__name__,'module':f'{child.__class__.__module__}.{child.__class__.__name__}','child_count':len(list(child.children()))}
        for attr in ['channels','out_channels','use_conv','padding','name','time_embedding_norm','output_scale_factor','in_channels']:
            if hasattr(child, attr):
                value=getattr(child, attr)
                row[attr]=str(value) if not isinstance(value,(int,float,str,bool,type(None))) else value
        rows.append(row)
    return rows

def state_rows(model):
    rows=[]; prefix=f'encoder.down_blocks.{BLOCK_INDEX}.'
    for key,tensor in model.state_dict().items():
        if key.startswith(prefix): rows.append({'key':key,'shape':list(tensor.shape),'dtype':str(tensor.dtype),'numel':int(tensor.numel())})
    return rows

def resnet_info(block, idx):
    r=block.resnets[idx]; shortcut=getattr(r,'conv_shortcut',None); tp=getattr(r,'time_emb_proj',None)
    return {'module_name':f'encoder.down_blocks.{BLOCK_INDEX}.resnets.{idx}','class_name':r.__class__.__name__,'class_module':r.__class__.__module__,'in_channels':getattr(r,'in_channels',None),'out_channels':getattr(r,'out_channels',None),'time_embedding_norm':getattr(r,'time_embedding_norm',None),'output_scale_factor':getattr(r,'output_scale_factor',None),'conv1_weight_shape':list(r.conv1.weight.shape),'conv2_weight_shape':list(r.conv2.weight.shape),'time_emb_proj_weight_shape':list(tp.weight.shape) if tp is not None else None,'has_conv_shortcut':shortcut is not None,'conv_shortcut_weight_shape':list(shortcut.weight.shape) if shortcut is not None else None}

def main():
    OUT.mkdir(parents=True,exist_ok=True); sys.path.insert(0,str(OFFICIAL_REPO))
    from diffusers.models.autoencoders.time_autoencoder_kl import TimeAwareAutoencoderKL
    model=TimeAwareAutoencoderKL.from_pretrained(str(WEIGHTS_DIR),local_files_only=True); model.eval(); block=model.encoder.down_blocks[BLOCK_INDEX]
    rows=module_tree(block); states=state_rows(model); has_down=getattr(block,'downsamplers',None) is not None and len(block.downsamplers)>0
    ds_info=None
    if has_down:
        ds=block.downsamplers[0]; conv=getattr(ds,'conv',None)
        ds_info={'module_name':f'encoder.down_blocks.{BLOCK_INDEX}.downsamplers.0','class_name':ds.__class__.__name__,'class_module':ds.__class__.__module__,'use_conv':getattr(ds,'use_conv',None),'padding':getattr(ds,'padding',None),'channels':getattr(ds,'channels',None),'out_channels':getattr(ds,'out_channels',None),'conv_class':conv.__class__.__name__ if conv is not None else None,'conv_kernel_size':list(conv.kernel_size) if conv is not None else None,'conv_stride':list(conv.stride) if conv is not None else None,'conv_padding':list(conv.padding) if conv is not None else None,'asymmetric_padding_when_padding_zero':getattr(ds,'padding',None)==0,'asymmetric_padding_note':'PyTorch Downsample2D applies F.pad(hidden_states, (0, 1, 0, 1)) before conv when use_conv=True and padding=0.' if getattr(ds,'padding',None)==0 else ''}
    resnets=[resnet_info(block,i) for i in range(len(block.resnets))]
    mapping=[{'pytorch_key':r['key'],'jittor_npz_key':r['key'].replace('.','__'),'shape':r['shape']} for r in states]
    result={'status':'PASS','block_index':BLOCK_INDEX,'block_class':block.__class__.__name__,'block_module':block.__class__.__module__,'resnet_count':len(block.resnets),'has_resnet0':len(block.resnets)>0,'has_resnet1':len(block.resnets)>1,'has_downsampler':has_down,'resnet_info':resnets,'downsampler_info':ds_info,'channel_change_detected':any(x.get('in_channels')!=x.get('out_channels') for x in resnets),'module_tree':rows,'state_keys':states,'key_mapping':mapping}
    (OUT/'downblock1_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (OUT/'downblock1_key_mapping.json').write_text(json.dumps({'status':result['status'],'rows':mapping},indent=2),encoding='utf-8')
    md=['# Official TimeAware VAE DownBlock1 Audit','',f"Status: **{result['status']}**",'',f"Block class: `{result['block_module']}.{result['block_class']}`",f"ResNet count: `{result['resnet_count']}`",f"Has downsampler: `{result['has_downsampler']}`",f"Channel change detected: `{result['channel_change_detected']}`",'','## ResNet modules','','| Module | In | Out | Time norm | Time proj shape | Conv shortcut | Shortcut shape |','|---|---:|---:|---|---:|---|---:|']
    for info in resnets: md.append(f"| `{info['module_name']}` | {info.get('in_channels')} | {info.get('out_channels')} | `{info.get('time_embedding_norm')}` | `{info.get('time_emb_proj_weight_shape')}` | {info.get('has_conv_shortcut')} | `{info.get('conv_shortcut_weight_shape')}` |")
    if ds_info:
        md += ['','## Downsampler','','| Field | Value |','|---|---|']
        for k,v in ds_info.items(): md.append(f'| {k} | `{v}` |')
    md += ['','## State Keys','','| Key | Shape | DType |','|---|---:|---|']
    for r in states: md.append(f"| `{r['key']}` | `{r['shape']}` | `{r['dtype']}` |")
    (OUT/'downblock1_audit.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    map_md=['# DownBlock1 PyTorch-to-NPZ Key Mapping','','| PyTorch key | Jittor NPZ key | Shape |','|---|---|---:|']
    for r in mapping: map_md.append(f"| `{r['pytorch_key']}` | `{r['jittor_npz_key']}` | `{r['shape']}` |")
    (OUT/'downblock1_key_mapping.md').write_text('\n'.join(map_md)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','resnet_count','has_downsampler','channel_change_detected','resnet_info','downsampler_info']},indent=2))
if __name__=='__main__': main()
