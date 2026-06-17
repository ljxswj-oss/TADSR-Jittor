# 中文阅读说明：official_time_vae_module_tree.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# Official TimeAware VAE Module Tree

Status: **PASS**

Class: `diffusers.models.autoencoders.time_autoencoder_kl.TimeAwareAutoencoderKL`
Encoder: `diffusers.models.autoencoders.vae.TimeAwareEncoder`

## Config

```json
{
  "in_channels": 3,
  "out_channels": 3,
  "down_block_types": [
    "DownEncoderBlock2D",
    "DownEncoderBlock2D",
    "DownEncoderBlock2D",
    "DownEncoderBlock2D"
  ],
  "up_block_types": [
    "UpDecoderBlock2D",
    "UpDecoderBlock2D",
    "UpDecoderBlock2D",
    "UpDecoderBlock2D"
  ],
  "block_out_channels": [
    128,
    256,
    512,
    512
  ],
  "layers_per_block": 2,
  "act_fn": "silu",
  "latent_channels": 4,
  "norm_num_groups": 32,
  "sample_size": 768,
  "scaling_factor": 0.18215,
  "force_upcast": true,
  "time_scale_shift": "scale_shift",
  "_use_default_values": [
    "force_upcast",
    "scaling_factor",
    "time_scale_shift"
  ],
  "_class_name": "TimeAwareAutoencoderKL",
  "_diffusers_version": "0.10.0.dev0",
  "_name_or_path": "/mnt/data/sj/checkpoints/TADSR/preset/weights/time_vae"
}
```

## Modules

| Name | Class | Children |
|---|---|---:|
| `<root>` | `TimeAwareAutoencoderKL` | 4 |
| `encoder` | `TimeAwareEncoder` | 8 |
| `encoder.time_proj` | `Timesteps` | 0 |
| `encoder.time_embedding` | `TimestepEmbedding` | 3 |
| `encoder.time_embedding.linear_1` | `LoRACompatibleLinear` | 0 |
| `encoder.time_embedding.act` | `SiLU` | 0 |
| `encoder.time_embedding.linear_2` | `LoRACompatibleLinear` | 0 |
| `encoder.conv_in` | `Conv2d` | 0 |
| `encoder.down_blocks` | `ModuleList` | 4 |
| `encoder.down_blocks.0` | `DownEncoderBlock2D` | 2 |
| `encoder.down_blocks.0.resnets` | `ModuleList` | 2 |
| `encoder.down_blocks.0.resnets.0` | `ResnetBlock2D` | 7 |
| `encoder.down_blocks.0.resnets.0.norm1` | `GroupNorm` | 0 |
| `encoder.down_blocks.0.resnets.0.conv1` | `LoRACompatibleConv` | 0 |
| `encoder.down_blocks.0.resnets.0.time_emb_proj` | `LoRACompatibleLinear` | 0 |
| `encoder.down_blocks.0.resnets.0.norm2` | `GroupNorm` | 0 |
| `encoder.down_blocks.0.resnets.0.dropout` | `Dropout` | 0 |
| `encoder.down_blocks.0.resnets.0.conv2` | `LoRACompatibleConv` | 0 |
| `encoder.down_blocks.0.resnets.1` | `ResnetBlock2D` | 7 |
| `encoder.down_blocks.0.resnets.1.norm1` | `GroupNorm` | 0 |
| `encoder.down_blocks.0.resnets.1.conv1` | `LoRACompatibleConv` | 0 |
| `encoder.down_blocks.0.resnets.1.time_emb_proj` | `LoRACompatibleLinear` | 0 |
| `encoder.down_blocks.0.resnets.1.norm2` | `GroupNorm` | 0 |
| `encoder.down_blocks.0.resnets.1.dropout` | `Dropout` | 0 |
| `encoder.down_blocks.0.resnets.1.conv2` | `LoRACompatibleConv` | 0 |
| `encoder.down_blocks.0.downsamplers` | `ModuleList` | 1 |
| `encoder.down_blocks.0.downsamplers.0` | `Downsample2D` | 1 |
| `encoder.down_blocks.0.downsamplers.0.conv` | `LoRACompatibleConv` | 0 |
| `encoder.down_blocks.1` | `DownEncoderBlock2D` | 2 |
| `encoder.down_blocks.1.resnets` | `ModuleList` | 2 |
| `encoder.down_blocks.1.resnets.0` | `ResnetBlock2D` | 8 |
| `encoder.down_blocks.1.resnets.0.norm1` | `GroupNorm` | 0 |
| `encoder.down_blocks.1.resnets.0.conv1` | `LoRACompatibleConv` | 0 |
| `encoder.down_blocks.1.resnets.0.time_emb_proj` | `LoRACompatibleLinear` | 0 |
| `encoder.down_blocks.1.resnets.0.norm2` | `GroupNorm` | 0 |
| `encoder.down_blocks.1.resnets.0.dropout` | `Dropout` | 0 |
| `encoder.down_blocks.1.resnets.0.conv2` | `LoRACompatibleConv` | 0 |
| `encoder.down_blocks.1.resnets.0.conv_shortcut` | `LoRACompatibleConv` | 0 |
| `encoder.down_blocks.1.resnets.1` | `ResnetBlock2D` | 7 |
| `encoder.down_blocks.1.resnets.1.norm1` | `GroupNorm` | 0 |
| `encoder.down_blocks.1.resnets.1.conv1` | `LoRACompatibleConv` | 0 |
| `encoder.down_blocks.1.resnets.1.time_emb_proj` | `LoRACompatibleLinear` | 0 |
| `encoder.down_blocks.1.resnets.1.norm2` | `GroupNorm` | 0 |
| `encoder.down_blocks.1.resnets.1.dropout` | `Dropout` | 0 |
| `encoder.down_blocks.1.resnets.1.conv2` | `LoRACompatibleConv` | 0 |
| `encoder.down_blocks.1.downsamplers` | `ModuleList` | 1 |
| `encoder.down_blocks.1.downsamplers.0` | `Downsample2D` | 1 |
| `encoder.down_blocks.1.downsamplers.0.conv` | `LoRACompatibleConv` | 0 |
| `encoder.down_blocks.2` | `DownEncoderBlock2D` | 2 |
| `encoder.down_blocks.2.resnets` | `ModuleList` | 2 |
| `encoder.down_blocks.2.resnets.0` | `ResnetBlock2D` | 8 |
| `encoder.down_blocks.2.resnets.0.norm1` | `GroupNorm` | 0 |
| `encoder.down_blocks.2.resnets.0.conv1` | `LoRACompatibleConv` | 0 |
| `encoder.down_blocks.2.resnets.0.time_emb_proj` | `LoRACompatibleLinear` | 0 |
| `encoder.down_blocks.2.resnets.0.norm2` | `GroupNorm` | 0 |
| `encoder.down_blocks.2.resnets.0.dropout` | `Dropout` | 0 |
| `encoder.down_blocks.2.resnets.0.conv2` | `LoRACompatibleConv` | 0 |
| `encoder.down_blocks.2.resnets.0.conv_shortcut` | `LoRACompatibleConv` | 0 |
| `encoder.down_blocks.2.resnets.1` | `ResnetBlock2D` | 7 |
| `encoder.down_blocks.2.resnets.1.norm1` | `GroupNorm` | 0 |
| `encoder.down_blocks.2.resnets.1.conv1` | `LoRACompatibleConv` | 0 |
| `encoder.down_blocks.2.resnets.1.time_emb_proj` | `LoRACompatibleLinear` | 0 |
| `encoder.down_blocks.2.resnets.1.norm2` | `GroupNorm` | 0 |
| `encoder.down_blocks.2.resnets.1.dropout` | `Dropout` | 0 |
| `encoder.down_blocks.2.resnets.1.conv2` | `LoRACompatibleConv` | 0 |
| `encoder.down_blocks.2.downsamplers` | `ModuleList` | 1 |
| `encoder.down_blocks.2.downsamplers.0` | `Downsample2D` | 1 |
| `encoder.down_blocks.2.downsamplers.0.conv` | `LoRACompatibleConv` | 0 |
| `encoder.down_blocks.3` | `DownEncoderBlock2D` | 1 |
| `encoder.down_blocks.3.resnets` | `ModuleList` | 2 |
| `encoder.down_blocks.3.resnets.0` | `ResnetBlock2D` | 7 |
| `encoder.down_blocks.3.resnets.0.norm1` | `GroupNorm` | 0 |
| `encoder.down_blocks.3.resnets.0.conv1` | `LoRACompatibleConv` | 0 |
| `encoder.down_blocks.3.resnets.0.time_emb_proj` | `LoRACompatibleLinear` | 0 |
| `encoder.down_blocks.3.resnets.0.norm2` | `GroupNorm` | 0 |
| `encoder.down_blocks.3.resnets.0.dropout` | `Dropout` | 0 |
| `encoder.down_blocks.3.resnets.0.conv2` | `LoRACompatibleConv` | 0 |
| `encoder.down_blocks.3.resnets.1` | `ResnetBlock2D` | 7 |
| `encoder.down_blocks.3.resnets.1.norm1` | `GroupNorm` | 0 |
| `encoder.down_blocks.3.resnets.1.conv1` | `LoRACompatibleConv` | 0 |
| `encoder.down_blocks.3.resnets.1.time_emb_proj` | `LoRACompatibleLinear` | 0 |
| `encoder.down_blocks.3.resnets.1.norm2` | `GroupNorm` | 0 |
| `encoder.down_blocks.3.resnets.1.dropout` | `Dropout` | 0 |
| `encoder.down_blocks.3.resnets.1.conv2` | `LoRACompatibleConv` | 0 |
| `encoder.mid_block` | `UNetMidBlock2D` | 2 |
| `encoder.mid_block.attentions` | `ModuleList` | 1 |
| `encoder.mid_block.attentions.0` | `Attention` | 5 |
| `encoder.mid_block.attentions.0.group_norm` | `GroupNorm` | 0 |
| `encoder.mid_block.attentions.0.to_q` | `LoRACompatibleLinear` | 0 |
| `encoder.mid_block.attentions.0.to_k` | `LoRACompatibleLinear` | 0 |
| `encoder.mid_block.attentions.0.to_v` | `LoRACompatibleLinear` | 0 |
| `encoder.mid_block.attentions.0.to_out` | `ModuleList` | 2 |
| `encoder.mid_block.attentions.0.to_out.0` | `LoRACompatibleLinear` | 0 |
| `encoder.mid_block.attentions.0.to_out.1` | `Dropout` | 0 |
| `encoder.mid_block.resnets` | `ModuleList` | 2 |
| `encoder.mid_block.resnets.0` | `ResnetBlock2D` | 7 |
| `encoder.mid_block.resnets.0.norm1` | `GroupNorm` | 0 |
| `encoder.mid_block.resnets.0.conv1` | `LoRACompatibleConv` | 0 |
| `encoder.mid_block.resnets.0.time_emb_proj` | `LoRACompatibleLinear` | 0 |
| `encoder.mid_block.resnets.0.norm2` | `GroupNorm` | 0 |
| `encoder.mid_block.resnets.0.dropout` | `Dropout` | 0 |
| `encoder.mid_block.resnets.0.conv2` | `LoRACompatibleConv` | 0 |
| `encoder.mid_block.resnets.1` | `ResnetBlock2D` | 7 |
| `encoder.mid_block.resnets.1.norm1` | `GroupNorm` | 0 |
| `encoder.mid_block.resnets.1.conv1` | `LoRACompatibleConv` | 0 |
| `encoder.mid_block.resnets.1.time_emb_proj` | `LoRACompatibleLinear` | 0 |
| `encoder.mid_block.resnets.1.norm2` | `GroupNorm` | 0 |
| `encoder.mid_block.resnets.1.dropout` | `Dropout` | 0 |
| `encoder.mid_block.resnets.1.conv2` | `LoRACompatibleConv` | 0 |
| `encoder.conv_norm_out` | `GroupNorm` | 0 |
| `encoder.conv_act` | `SiLU` | 0 |
| `encoder.conv_out` | `Conv2d` | 0 |
| `decoder` | `Decoder` | 6 |
| `decoder.conv_in` | `Conv2d` | 0 |
| `decoder.up_blocks` | `ModuleList` | 4 |
| `decoder.up_blocks.0` | `UpDecoderBlock2D` | 2 |
| `decoder.up_blocks.0.resnets` | `ModuleList` | 3 |
| `decoder.up_blocks.0.resnets.0` | `ResnetBlock2D` | 6 |
| `decoder.up_blocks.0.resnets.0.norm1` | `GroupNorm` | 0 |
| `decoder.up_blocks.0.resnets.0.conv1` | `LoRACompatibleConv` | 0 |
| `decoder.up_blocks.0.resnets.0.norm2` | `GroupNorm` | 0 |
| `decoder.up_blocks.0.resnets.0.dropout` | `Dropout` | 0 |
| `decoder.up_blocks.0.resnets.0.conv2` | `LoRACompatibleConv` | 0 |
| `decoder.up_blocks.0.resnets.1` | `ResnetBlock2D` | 6 |
| `decoder.up_blocks.0.resnets.1.norm1` | `GroupNorm` | 0 |
| `decoder.up_blocks.0.resnets.1.conv1` | `LoRACompatibleConv` | 0 |
| `decoder.up_blocks.0.resnets.1.norm2` | `GroupNorm` | 0 |
| `decoder.up_blocks.0.resnets.1.dropout` | `Dropout` | 0 |
| `decoder.up_blocks.0.resnets.1.conv2` | `LoRACompatibleConv` | 0 |
| `decoder.up_blocks.0.resnets.2` | `ResnetBlock2D` | 6 |
| `decoder.up_blocks.0.resnets.2.norm1` | `GroupNorm` | 0 |
| `decoder.up_blocks.0.resnets.2.conv1` | `LoRACompatibleConv` | 0 |
| `decoder.up_blocks.0.resnets.2.norm2` | `GroupNorm` | 0 |
| `decoder.up_blocks.0.resnets.2.dropout` | `Dropout` | 0 |
| `decoder.up_blocks.0.resnets.2.conv2` | `LoRACompatibleConv` | 0 |
| `decoder.up_blocks.0.upsamplers` | `ModuleList` | 1 |
| `decoder.up_blocks.0.upsamplers.0` | `Upsample2D` | 1 |
| `decoder.up_blocks.0.upsamplers.0.conv` | `LoRACompatibleConv` | 0 |
| `decoder.up_blocks.1` | `UpDecoderBlock2D` | 2 |
| `decoder.up_blocks.1.resnets` | `ModuleList` | 3 |
| `decoder.up_blocks.1.resnets.0` | `ResnetBlock2D` | 6 |
| `decoder.up_blocks.1.resnets.0.norm1` | `GroupNorm` | 0 |
| `decoder.up_blocks.1.resnets.0.conv1` | `LoRACompatibleConv` | 0 |
| `decoder.up_blocks.1.resnets.0.norm2` | `GroupNorm` | 0 |
| `decoder.up_blocks.1.resnets.0.dropout` | `Dropout` | 0 |
| `decoder.up_blocks.1.resnets.0.conv2` | `LoRACompatibleConv` | 0 |
| `decoder.up_blocks.1.resnets.1` | `ResnetBlock2D` | 6 |
| `decoder.up_blocks.1.resnets.1.norm1` | `GroupNorm` | 0 |
| `decoder.up_blocks.1.resnets.1.conv1` | `LoRACompatibleConv` | 0 |
| `decoder.up_blocks.1.resnets.1.norm2` | `GroupNorm` | 0 |
| `decoder.up_blocks.1.resnets.1.dropout` | `Dropout` | 0 |
| `decoder.up_blocks.1.resnets.1.conv2` | `LoRACompatibleConv` | 0 |
| `decoder.up_blocks.1.resnets.2` | `ResnetBlock2D` | 6 |
| `decoder.up_blocks.1.resnets.2.norm1` | `GroupNorm` | 0 |
| `decoder.up_blocks.1.resnets.2.conv1` | `LoRACompatibleConv` | 0 |
| `decoder.up_blocks.1.resnets.2.norm2` | `GroupNorm` | 0 |
| `decoder.up_blocks.1.resnets.2.dropout` | `Dropout` | 0 |
| `decoder.up_blocks.1.resnets.2.conv2` | `LoRACompatibleConv` | 0 |
| `decoder.up_blocks.1.upsamplers` | `ModuleList` | 1 |
| `decoder.up_blocks.1.upsamplers.0` | `Upsample2D` | 1 |
| `decoder.up_blocks.1.upsamplers.0.conv` | `LoRACompatibleConv` | 0 |
| `decoder.up_blocks.2` | `UpDecoderBlock2D` | 2 |
| `decoder.up_blocks.2.resnets` | `ModuleList` | 3 |
| `decoder.up_blocks.2.resnets.0` | `ResnetBlock2D` | 7 |
| `decoder.up_blocks.2.resnets.0.norm1` | `GroupNorm` | 0 |
| `decoder.up_blocks.2.resnets.0.conv1` | `LoRACompatibleConv` | 0 |
| `decoder.up_blocks.2.resnets.0.norm2` | `GroupNorm` | 0 |
| `decoder.up_blocks.2.resnets.0.dropout` | `Dropout` | 0 |
| `decoder.up_blocks.2.resnets.0.conv2` | `LoRACompatibleConv` | 0 |
| `decoder.up_blocks.2.resnets.0.conv_shortcut` | `LoRACompatibleConv` | 0 |
| `decoder.up_blocks.2.resnets.1` | `ResnetBlock2D` | 6 |
| `decoder.up_blocks.2.resnets.1.norm1` | `GroupNorm` | 0 |
| `decoder.up_blocks.2.resnets.1.conv1` | `LoRACompatibleConv` | 0 |
| `decoder.up_blocks.2.resnets.1.norm2` | `GroupNorm` | 0 |
| `decoder.up_blocks.2.resnets.1.dropout` | `Dropout` | 0 |
| `decoder.up_blocks.2.resnets.1.conv2` | `LoRACompatibleConv` | 0 |
| `decoder.up_blocks.2.resnets.2` | `ResnetBlock2D` | 6 |
| `decoder.up_blocks.2.resnets.2.norm1` | `GroupNorm` | 0 |
| `decoder.up_blocks.2.resnets.2.conv1` | `LoRACompatibleConv` | 0 |
| `decoder.up_blocks.2.resnets.2.norm2` | `GroupNorm` | 0 |
| `decoder.up_blocks.2.resnets.2.dropout` | `Dropout` | 0 |
| `decoder.up_blocks.2.resnets.2.conv2` | `LoRACompatibleConv` | 0 |
| `decoder.up_blocks.2.upsamplers` | `ModuleList` | 1 |
| `decoder.up_blocks.2.upsamplers.0` | `Upsample2D` | 1 |
| `decoder.up_blocks.2.upsamplers.0.conv` | `LoRACompatibleConv` | 0 |
| `decoder.up_blocks.3` | `UpDecoderBlock2D` | 1 |
| `decoder.up_blocks.3.resnets` | `ModuleList` | 3 |
| `decoder.up_blocks.3.resnets.0` | `ResnetBlock2D` | 7 |
| `decoder.up_blocks.3.resnets.0.norm1` | `GroupNorm` | 0 |
| `decoder.up_blocks.3.resnets.0.conv1` | `LoRACompatibleConv` | 0 |
| `decoder.up_blocks.3.resnets.0.norm2` | `GroupNorm` | 0 |
| `decoder.up_blocks.3.resnets.0.dropout` | `Dropout` | 0 |
| `decoder.up_blocks.3.resnets.0.conv2` | `LoRACompatibleConv` | 0 |
| `decoder.up_blocks.3.resnets.0.conv_shortcut` | `LoRACompatibleConv` | 0 |
| `decoder.up_blocks.3.resnets.1` | `ResnetBlock2D` | 6 |
| `decoder.up_blocks.3.resnets.1.norm1` | `GroupNorm` | 0 |
| `decoder.up_blocks.3.resnets.1.conv1` | `LoRACompatibleConv` | 0 |
| `decoder.up_blocks.3.resnets.1.norm2` | `GroupNorm` | 0 |
| `decoder.up_blocks.3.resnets.1.dropout` | `Dropout` | 0 |
| `decoder.up_blocks.3.resnets.1.conv2` | `LoRACompatibleConv` | 0 |
| `decoder.up_blocks.3.resnets.2` | `ResnetBlock2D` | 6 |
| `decoder.up_blocks.3.resnets.2.norm1` | `GroupNorm` | 0 |
| `decoder.up_blocks.3.resnets.2.conv1` | `LoRACompatibleConv` | 0 |
| `decoder.up_blocks.3.resnets.2.norm2` | `GroupNorm` | 0 |
| `decoder.up_blocks.3.resnets.2.dropout` | `Dropout` | 0 |
| `decoder.up_blocks.3.resnets.2.conv2` | `LoRACompatibleConv` | 0 |
| `decoder.mid_block` | `UNetMidBlock2D` | 2 |
| `decoder.mid_block.attentions` | `ModuleList` | 1 |
| `decoder.mid_block.attentions.0` | `Attention` | 5 |
| `decoder.mid_block.attentions.0.group_norm` | `GroupNorm` | 0 |
| `decoder.mid_block.attentions.0.to_q` | `LoRACompatibleLinear` | 0 |
| `decoder.mid_block.attentions.0.to_k` | `LoRACompatibleLinear` | 0 |
| `decoder.mid_block.attentions.0.to_v` | `LoRACompatibleLinear` | 0 |
| `decoder.mid_block.attentions.0.to_out` | `ModuleList` | 2 |
| `decoder.mid_block.attentions.0.to_out.0` | `LoRACompatibleLinear` | 0 |
| `decoder.mid_block.attentions.0.to_out.1` | `Dropout` | 0 |
| `decoder.mid_block.resnets` | `ModuleList` | 2 |
| `decoder.mid_block.resnets.0` | `ResnetBlock2D` | 6 |
| `decoder.mid_block.resnets.0.norm1` | `GroupNorm` | 0 |
| `decoder.mid_block.resnets.0.conv1` | `LoRACompatibleConv` | 0 |
| `decoder.mid_block.resnets.0.norm2` | `GroupNorm` | 0 |
| `decoder.mid_block.resnets.0.dropout` | `Dropout` | 0 |
| `decoder.mid_block.resnets.0.conv2` | `LoRACompatibleConv` | 0 |
| `decoder.mid_block.resnets.1` | `ResnetBlock2D` | 6 |
| `decoder.mid_block.resnets.1.norm1` | `GroupNorm` | 0 |
| `decoder.mid_block.resnets.1.conv1` | `LoRACompatibleConv` | 0 |
| `decoder.mid_block.resnets.1.norm2` | `GroupNorm` | 0 |
| `decoder.mid_block.resnets.1.dropout` | `Dropout` | 0 |
| `decoder.mid_block.resnets.1.conv2` | `LoRACompatibleConv` | 0 |
| `decoder.conv_norm_out` | `GroupNorm` | 0 |
| `decoder.conv_act` | `SiLU` | 0 |
| `decoder.conv_out` | `Conv2d` | 0 |
| `quant_conv` | `Conv2d` | 0 |
| `post_quant_conv` | `Conv2d` | 0 |
