# 中文阅读说明：official_time_vae_state_keys.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# Official TimeAware VAE State Dict Keys

Status: **PASS**

| Key | Shape | DType | Numel |
|---|---:|---|---:|
| `encoder.time_embedding.linear_1.weight` | `[256, 128]` | `torch.float32` | 32768 |
| `encoder.time_embedding.linear_1.bias` | `[256]` | `torch.float32` | 256 |
| `encoder.time_embedding.linear_2.weight` | `[256, 256]` | `torch.float32` | 65536 |
| `encoder.time_embedding.linear_2.bias` | `[256]` | `torch.float32` | 256 |
| `encoder.conv_in.weight` | `[128, 3, 3, 3]` | `torch.float32` | 3456 |
| `encoder.conv_in.bias` | `[128]` | `torch.float32` | 128 |
| `encoder.down_blocks.0.resnets.0.norm1.weight` | `[128]` | `torch.float32` | 128 |
| `encoder.down_blocks.0.resnets.0.norm1.bias` | `[128]` | `torch.float32` | 128 |
| `encoder.down_blocks.0.resnets.0.conv1.weight` | `[128, 128, 3, 3]` | `torch.float32` | 147456 |
| `encoder.down_blocks.0.resnets.0.conv1.bias` | `[128]` | `torch.float32` | 128 |
| `encoder.down_blocks.0.resnets.0.time_emb_proj.weight` | `[256, 256]` | `torch.float32` | 65536 |
| `encoder.down_blocks.0.resnets.0.time_emb_proj.bias` | `[256]` | `torch.float32` | 256 |
| `encoder.down_blocks.0.resnets.0.norm2.weight` | `[128]` | `torch.float32` | 128 |
| `encoder.down_blocks.0.resnets.0.norm2.bias` | `[128]` | `torch.float32` | 128 |
| `encoder.down_blocks.0.resnets.0.conv2.weight` | `[128, 128, 3, 3]` | `torch.float32` | 147456 |
| `encoder.down_blocks.0.resnets.0.conv2.bias` | `[128]` | `torch.float32` | 128 |
| `encoder.down_blocks.0.resnets.1.norm1.weight` | `[128]` | `torch.float32` | 128 |
| `encoder.down_blocks.0.resnets.1.norm1.bias` | `[128]` | `torch.float32` | 128 |
| `encoder.down_blocks.0.resnets.1.conv1.weight` | `[128, 128, 3, 3]` | `torch.float32` | 147456 |
| `encoder.down_blocks.0.resnets.1.conv1.bias` | `[128]` | `torch.float32` | 128 |
| `encoder.down_blocks.0.resnets.1.time_emb_proj.weight` | `[256, 256]` | `torch.float32` | 65536 |
| `encoder.down_blocks.0.resnets.1.time_emb_proj.bias` | `[256]` | `torch.float32` | 256 |
| `encoder.down_blocks.0.resnets.1.norm2.weight` | `[128]` | `torch.float32` | 128 |
| `encoder.down_blocks.0.resnets.1.norm2.bias` | `[128]` | `torch.float32` | 128 |
| `encoder.down_blocks.0.resnets.1.conv2.weight` | `[128, 128, 3, 3]` | `torch.float32` | 147456 |
| `encoder.down_blocks.0.resnets.1.conv2.bias` | `[128]` | `torch.float32` | 128 |
| `encoder.down_blocks.0.downsamplers.0.conv.weight` | `[128, 128, 3, 3]` | `torch.float32` | 147456 |
| `encoder.down_blocks.0.downsamplers.0.conv.bias` | `[128]` | `torch.float32` | 128 |
| `encoder.down_blocks.1.resnets.0.norm1.weight` | `[128]` | `torch.float32` | 128 |
| `encoder.down_blocks.1.resnets.0.norm1.bias` | `[128]` | `torch.float32` | 128 |
| `encoder.down_blocks.1.resnets.0.conv1.weight` | `[256, 128, 3, 3]` | `torch.float32` | 294912 |
| `encoder.down_blocks.1.resnets.0.conv1.bias` | `[256]` | `torch.float32` | 256 |
| `encoder.down_blocks.1.resnets.0.time_emb_proj.weight` | `[512, 256]` | `torch.float32` | 131072 |
| `encoder.down_blocks.1.resnets.0.time_emb_proj.bias` | `[512]` | `torch.float32` | 512 |
| `encoder.down_blocks.1.resnets.0.norm2.weight` | `[256]` | `torch.float32` | 256 |
| `encoder.down_blocks.1.resnets.0.norm2.bias` | `[256]` | `torch.float32` | 256 |
| `encoder.down_blocks.1.resnets.0.conv2.weight` | `[256, 256, 3, 3]` | `torch.float32` | 589824 |
| `encoder.down_blocks.1.resnets.0.conv2.bias` | `[256]` | `torch.float32` | 256 |
| `encoder.down_blocks.1.resnets.0.conv_shortcut.weight` | `[256, 128, 1, 1]` | `torch.float32` | 32768 |
| `encoder.down_blocks.1.resnets.0.conv_shortcut.bias` | `[256]` | `torch.float32` | 256 |
| `encoder.down_blocks.1.resnets.1.norm1.weight` | `[256]` | `torch.float32` | 256 |
| `encoder.down_blocks.1.resnets.1.norm1.bias` | `[256]` | `torch.float32` | 256 |
| `encoder.down_blocks.1.resnets.1.conv1.weight` | `[256, 256, 3, 3]` | `torch.float32` | 589824 |
| `encoder.down_blocks.1.resnets.1.conv1.bias` | `[256]` | `torch.float32` | 256 |
| `encoder.down_blocks.1.resnets.1.time_emb_proj.weight` | `[512, 256]` | `torch.float32` | 131072 |
| `encoder.down_blocks.1.resnets.1.time_emb_proj.bias` | `[512]` | `torch.float32` | 512 |
| `encoder.down_blocks.1.resnets.1.norm2.weight` | `[256]` | `torch.float32` | 256 |
| `encoder.down_blocks.1.resnets.1.norm2.bias` | `[256]` | `torch.float32` | 256 |
| `encoder.down_blocks.1.resnets.1.conv2.weight` | `[256, 256, 3, 3]` | `torch.float32` | 589824 |
| `encoder.down_blocks.1.resnets.1.conv2.bias` | `[256]` | `torch.float32` | 256 |
| `encoder.down_blocks.1.downsamplers.0.conv.weight` | `[256, 256, 3, 3]` | `torch.float32` | 589824 |
| `encoder.down_blocks.1.downsamplers.0.conv.bias` | `[256]` | `torch.float32` | 256 |
| `encoder.down_blocks.2.resnets.0.norm1.weight` | `[256]` | `torch.float32` | 256 |
| `encoder.down_blocks.2.resnets.0.norm1.bias` | `[256]` | `torch.float32` | 256 |
| `encoder.down_blocks.2.resnets.0.conv1.weight` | `[512, 256, 3, 3]` | `torch.float32` | 1179648 |
| `encoder.down_blocks.2.resnets.0.conv1.bias` | `[512]` | `torch.float32` | 512 |
| `encoder.down_blocks.2.resnets.0.time_emb_proj.weight` | `[1024, 256]` | `torch.float32` | 262144 |
| `encoder.down_blocks.2.resnets.0.time_emb_proj.bias` | `[1024]` | `torch.float32` | 1024 |
| `encoder.down_blocks.2.resnets.0.norm2.weight` | `[512]` | `torch.float32` | 512 |
| `encoder.down_blocks.2.resnets.0.norm2.bias` | `[512]` | `torch.float32` | 512 |
| `encoder.down_blocks.2.resnets.0.conv2.weight` | `[512, 512, 3, 3]` | `torch.float32` | 2359296 |
| `encoder.down_blocks.2.resnets.0.conv2.bias` | `[512]` | `torch.float32` | 512 |
| `encoder.down_blocks.2.resnets.0.conv_shortcut.weight` | `[512, 256, 1, 1]` | `torch.float32` | 131072 |
| `encoder.down_blocks.2.resnets.0.conv_shortcut.bias` | `[512]` | `torch.float32` | 512 |
| `encoder.down_blocks.2.resnets.1.norm1.weight` | `[512]` | `torch.float32` | 512 |
| `encoder.down_blocks.2.resnets.1.norm1.bias` | `[512]` | `torch.float32` | 512 |
| `encoder.down_blocks.2.resnets.1.conv1.weight` | `[512, 512, 3, 3]` | `torch.float32` | 2359296 |
| `encoder.down_blocks.2.resnets.1.conv1.bias` | `[512]` | `torch.float32` | 512 |
| `encoder.down_blocks.2.resnets.1.time_emb_proj.weight` | `[1024, 256]` | `torch.float32` | 262144 |
| `encoder.down_blocks.2.resnets.1.time_emb_proj.bias` | `[1024]` | `torch.float32` | 1024 |
| `encoder.down_blocks.2.resnets.1.norm2.weight` | `[512]` | `torch.float32` | 512 |
| `encoder.down_blocks.2.resnets.1.norm2.bias` | `[512]` | `torch.float32` | 512 |
| `encoder.down_blocks.2.resnets.1.conv2.weight` | `[512, 512, 3, 3]` | `torch.float32` | 2359296 |
| `encoder.down_blocks.2.resnets.1.conv2.bias` | `[512]` | `torch.float32` | 512 |
| `encoder.down_blocks.2.downsamplers.0.conv.weight` | `[512, 512, 3, 3]` | `torch.float32` | 2359296 |
| `encoder.down_blocks.2.downsamplers.0.conv.bias` | `[512]` | `torch.float32` | 512 |
| `encoder.down_blocks.3.resnets.0.norm1.weight` | `[512]` | `torch.float32` | 512 |
| `encoder.down_blocks.3.resnets.0.norm1.bias` | `[512]` | `torch.float32` | 512 |
| `encoder.down_blocks.3.resnets.0.conv1.weight` | `[512, 512, 3, 3]` | `torch.float32` | 2359296 |
| `encoder.down_blocks.3.resnets.0.conv1.bias` | `[512]` | `torch.float32` | 512 |
| `encoder.down_blocks.3.resnets.0.time_emb_proj.weight` | `[1024, 256]` | `torch.float32` | 262144 |
| `encoder.down_blocks.3.resnets.0.time_emb_proj.bias` | `[1024]` | `torch.float32` | 1024 |
| `encoder.down_blocks.3.resnets.0.norm2.weight` | `[512]` | `torch.float32` | 512 |
| `encoder.down_blocks.3.resnets.0.norm2.bias` | `[512]` | `torch.float32` | 512 |
| `encoder.down_blocks.3.resnets.0.conv2.weight` | `[512, 512, 3, 3]` | `torch.float32` | 2359296 |
| `encoder.down_blocks.3.resnets.0.conv2.bias` | `[512]` | `torch.float32` | 512 |
| `encoder.down_blocks.3.resnets.1.norm1.weight` | `[512]` | `torch.float32` | 512 |
| `encoder.down_blocks.3.resnets.1.norm1.bias` | `[512]` | `torch.float32` | 512 |
| `encoder.down_blocks.3.resnets.1.conv1.weight` | `[512, 512, 3, 3]` | `torch.float32` | 2359296 |
| `encoder.down_blocks.3.resnets.1.conv1.bias` | `[512]` | `torch.float32` | 512 |
| `encoder.down_blocks.3.resnets.1.time_emb_proj.weight` | `[1024, 256]` | `torch.float32` | 262144 |
| `encoder.down_blocks.3.resnets.1.time_emb_proj.bias` | `[1024]` | `torch.float32` | 1024 |
| `encoder.down_blocks.3.resnets.1.norm2.weight` | `[512]` | `torch.float32` | 512 |
| `encoder.down_blocks.3.resnets.1.norm2.bias` | `[512]` | `torch.float32` | 512 |
| `encoder.down_blocks.3.resnets.1.conv2.weight` | `[512, 512, 3, 3]` | `torch.float32` | 2359296 |
| `encoder.down_blocks.3.resnets.1.conv2.bias` | `[512]` | `torch.float32` | 512 |
| `encoder.mid_block.attentions.0.group_norm.weight` | `[512]` | `torch.float32` | 512 |
| `encoder.mid_block.attentions.0.group_norm.bias` | `[512]` | `torch.float32` | 512 |
| `encoder.mid_block.attentions.0.to_q.weight` | `[512, 512]` | `torch.float32` | 262144 |
| `encoder.mid_block.attentions.0.to_q.bias` | `[512]` | `torch.float32` | 512 |
| `encoder.mid_block.attentions.0.to_k.weight` | `[512, 512]` | `torch.float32` | 262144 |
| `encoder.mid_block.attentions.0.to_k.bias` | `[512]` | `torch.float32` | 512 |
| `encoder.mid_block.attentions.0.to_v.weight` | `[512, 512]` | `torch.float32` | 262144 |
| `encoder.mid_block.attentions.0.to_v.bias` | `[512]` | `torch.float32` | 512 |
| `encoder.mid_block.attentions.0.to_out.0.weight` | `[512, 512]` | `torch.float32` | 262144 |
| `encoder.mid_block.attentions.0.to_out.0.bias` | `[512]` | `torch.float32` | 512 |
| `encoder.mid_block.resnets.0.norm1.weight` | `[512]` | `torch.float32` | 512 |
| `encoder.mid_block.resnets.0.norm1.bias` | `[512]` | `torch.float32` | 512 |
| `encoder.mid_block.resnets.0.conv1.weight` | `[512, 512, 3, 3]` | `torch.float32` | 2359296 |
| `encoder.mid_block.resnets.0.conv1.bias` | `[512]` | `torch.float32` | 512 |
| `encoder.mid_block.resnets.0.time_emb_proj.weight` | `[1024, 256]` | `torch.float32` | 262144 |
| `encoder.mid_block.resnets.0.time_emb_proj.bias` | `[1024]` | `torch.float32` | 1024 |
| `encoder.mid_block.resnets.0.norm2.weight` | `[512]` | `torch.float32` | 512 |
| `encoder.mid_block.resnets.0.norm2.bias` | `[512]` | `torch.float32` | 512 |
| `encoder.mid_block.resnets.0.conv2.weight` | `[512, 512, 3, 3]` | `torch.float32` | 2359296 |
| `encoder.mid_block.resnets.0.conv2.bias` | `[512]` | `torch.float32` | 512 |
| `encoder.mid_block.resnets.1.norm1.weight` | `[512]` | `torch.float32` | 512 |
| `encoder.mid_block.resnets.1.norm1.bias` | `[512]` | `torch.float32` | 512 |
| `encoder.mid_block.resnets.1.conv1.weight` | `[512, 512, 3, 3]` | `torch.float32` | 2359296 |
| `encoder.mid_block.resnets.1.conv1.bias` | `[512]` | `torch.float32` | 512 |
| `encoder.mid_block.resnets.1.time_emb_proj.weight` | `[1024, 256]` | `torch.float32` | 262144 |
| `encoder.mid_block.resnets.1.time_emb_proj.bias` | `[1024]` | `torch.float32` | 1024 |
| `encoder.mid_block.resnets.1.norm2.weight` | `[512]` | `torch.float32` | 512 |
| `encoder.mid_block.resnets.1.norm2.bias` | `[512]` | `torch.float32` | 512 |
| `encoder.mid_block.resnets.1.conv2.weight` | `[512, 512, 3, 3]` | `torch.float32` | 2359296 |
| `encoder.mid_block.resnets.1.conv2.bias` | `[512]` | `torch.float32` | 512 |
| `encoder.conv_norm_out.weight` | `[512]` | `torch.float32` | 512 |
| `encoder.conv_norm_out.bias` | `[512]` | `torch.float32` | 512 |
| `encoder.conv_out.weight` | `[8, 512, 3, 3]` | `torch.float32` | 36864 |
| `encoder.conv_out.bias` | `[8]` | `torch.float32` | 8 |
| `decoder.conv_in.weight` | `[512, 4, 3, 3]` | `torch.float32` | 18432 |
| `decoder.conv_in.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.0.resnets.0.norm1.weight` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.0.resnets.0.norm1.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.0.resnets.0.conv1.weight` | `[512, 512, 3, 3]` | `torch.float32` | 2359296 |
| `decoder.up_blocks.0.resnets.0.conv1.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.0.resnets.0.norm2.weight` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.0.resnets.0.norm2.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.0.resnets.0.conv2.weight` | `[512, 512, 3, 3]` | `torch.float32` | 2359296 |
| `decoder.up_blocks.0.resnets.0.conv2.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.0.resnets.1.norm1.weight` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.0.resnets.1.norm1.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.0.resnets.1.conv1.weight` | `[512, 512, 3, 3]` | `torch.float32` | 2359296 |
| `decoder.up_blocks.0.resnets.1.conv1.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.0.resnets.1.norm2.weight` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.0.resnets.1.norm2.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.0.resnets.1.conv2.weight` | `[512, 512, 3, 3]` | `torch.float32` | 2359296 |
| `decoder.up_blocks.0.resnets.1.conv2.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.0.resnets.2.norm1.weight` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.0.resnets.2.norm1.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.0.resnets.2.conv1.weight` | `[512, 512, 3, 3]` | `torch.float32` | 2359296 |
| `decoder.up_blocks.0.resnets.2.conv1.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.0.resnets.2.norm2.weight` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.0.resnets.2.norm2.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.0.resnets.2.conv2.weight` | `[512, 512, 3, 3]` | `torch.float32` | 2359296 |
| `decoder.up_blocks.0.resnets.2.conv2.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.0.upsamplers.0.conv.weight` | `[512, 512, 3, 3]` | `torch.float32` | 2359296 |
| `decoder.up_blocks.0.upsamplers.0.conv.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.1.resnets.0.norm1.weight` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.1.resnets.0.norm1.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.1.resnets.0.conv1.weight` | `[512, 512, 3, 3]` | `torch.float32` | 2359296 |
| `decoder.up_blocks.1.resnets.0.conv1.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.1.resnets.0.norm2.weight` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.1.resnets.0.norm2.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.1.resnets.0.conv2.weight` | `[512, 512, 3, 3]` | `torch.float32` | 2359296 |
| `decoder.up_blocks.1.resnets.0.conv2.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.1.resnets.1.norm1.weight` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.1.resnets.1.norm1.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.1.resnets.1.conv1.weight` | `[512, 512, 3, 3]` | `torch.float32` | 2359296 |
| `decoder.up_blocks.1.resnets.1.conv1.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.1.resnets.1.norm2.weight` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.1.resnets.1.norm2.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.1.resnets.1.conv2.weight` | `[512, 512, 3, 3]` | `torch.float32` | 2359296 |
| `decoder.up_blocks.1.resnets.1.conv2.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.1.resnets.2.norm1.weight` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.1.resnets.2.norm1.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.1.resnets.2.conv1.weight` | `[512, 512, 3, 3]` | `torch.float32` | 2359296 |
| `decoder.up_blocks.1.resnets.2.conv1.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.1.resnets.2.norm2.weight` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.1.resnets.2.norm2.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.1.resnets.2.conv2.weight` | `[512, 512, 3, 3]` | `torch.float32` | 2359296 |
| `decoder.up_blocks.1.resnets.2.conv2.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.1.upsamplers.0.conv.weight` | `[512, 512, 3, 3]` | `torch.float32` | 2359296 |
| `decoder.up_blocks.1.upsamplers.0.conv.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.2.resnets.0.norm1.weight` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.2.resnets.0.norm1.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.up_blocks.2.resnets.0.conv1.weight` | `[256, 512, 3, 3]` | `torch.float32` | 1179648 |
| `decoder.up_blocks.2.resnets.0.conv1.bias` | `[256]` | `torch.float32` | 256 |
| `decoder.up_blocks.2.resnets.0.norm2.weight` | `[256]` | `torch.float32` | 256 |
| `decoder.up_blocks.2.resnets.0.norm2.bias` | `[256]` | `torch.float32` | 256 |
| `decoder.up_blocks.2.resnets.0.conv2.weight` | `[256, 256, 3, 3]` | `torch.float32` | 589824 |
| `decoder.up_blocks.2.resnets.0.conv2.bias` | `[256]` | `torch.float32` | 256 |
| `decoder.up_blocks.2.resnets.0.conv_shortcut.weight` | `[256, 512, 1, 1]` | `torch.float32` | 131072 |
| `decoder.up_blocks.2.resnets.0.conv_shortcut.bias` | `[256]` | `torch.float32` | 256 |
| `decoder.up_blocks.2.resnets.1.norm1.weight` | `[256]` | `torch.float32` | 256 |
| `decoder.up_blocks.2.resnets.1.norm1.bias` | `[256]` | `torch.float32` | 256 |
| `decoder.up_blocks.2.resnets.1.conv1.weight` | `[256, 256, 3, 3]` | `torch.float32` | 589824 |
| `decoder.up_blocks.2.resnets.1.conv1.bias` | `[256]` | `torch.float32` | 256 |
| `decoder.up_blocks.2.resnets.1.norm2.weight` | `[256]` | `torch.float32` | 256 |
| `decoder.up_blocks.2.resnets.1.norm2.bias` | `[256]` | `torch.float32` | 256 |
| `decoder.up_blocks.2.resnets.1.conv2.weight` | `[256, 256, 3, 3]` | `torch.float32` | 589824 |
| `decoder.up_blocks.2.resnets.1.conv2.bias` | `[256]` | `torch.float32` | 256 |
| `decoder.up_blocks.2.resnets.2.norm1.weight` | `[256]` | `torch.float32` | 256 |
| `decoder.up_blocks.2.resnets.2.norm1.bias` | `[256]` | `torch.float32` | 256 |
| `decoder.up_blocks.2.resnets.2.conv1.weight` | `[256, 256, 3, 3]` | `torch.float32` | 589824 |
| `decoder.up_blocks.2.resnets.2.conv1.bias` | `[256]` | `torch.float32` | 256 |
| `decoder.up_blocks.2.resnets.2.norm2.weight` | `[256]` | `torch.float32` | 256 |
| `decoder.up_blocks.2.resnets.2.norm2.bias` | `[256]` | `torch.float32` | 256 |
| `decoder.up_blocks.2.resnets.2.conv2.weight` | `[256, 256, 3, 3]` | `torch.float32` | 589824 |
| `decoder.up_blocks.2.resnets.2.conv2.bias` | `[256]` | `torch.float32` | 256 |
| `decoder.up_blocks.2.upsamplers.0.conv.weight` | `[256, 256, 3, 3]` | `torch.float32` | 589824 |
| `decoder.up_blocks.2.upsamplers.0.conv.bias` | `[256]` | `torch.float32` | 256 |
| `decoder.up_blocks.3.resnets.0.norm1.weight` | `[256]` | `torch.float32` | 256 |
| `decoder.up_blocks.3.resnets.0.norm1.bias` | `[256]` | `torch.float32` | 256 |
| `decoder.up_blocks.3.resnets.0.conv1.weight` | `[128, 256, 3, 3]` | `torch.float32` | 294912 |
| `decoder.up_blocks.3.resnets.0.conv1.bias` | `[128]` | `torch.float32` | 128 |
| `decoder.up_blocks.3.resnets.0.norm2.weight` | `[128]` | `torch.float32` | 128 |
| `decoder.up_blocks.3.resnets.0.norm2.bias` | `[128]` | `torch.float32` | 128 |
| `decoder.up_blocks.3.resnets.0.conv2.weight` | `[128, 128, 3, 3]` | `torch.float32` | 147456 |
| `decoder.up_blocks.3.resnets.0.conv2.bias` | `[128]` | `torch.float32` | 128 |
| `decoder.up_blocks.3.resnets.0.conv_shortcut.weight` | `[128, 256, 1, 1]` | `torch.float32` | 32768 |
| `decoder.up_blocks.3.resnets.0.conv_shortcut.bias` | `[128]` | `torch.float32` | 128 |
| `decoder.up_blocks.3.resnets.1.norm1.weight` | `[128]` | `torch.float32` | 128 |
| `decoder.up_blocks.3.resnets.1.norm1.bias` | `[128]` | `torch.float32` | 128 |
| `decoder.up_blocks.3.resnets.1.conv1.weight` | `[128, 128, 3, 3]` | `torch.float32` | 147456 |
| `decoder.up_blocks.3.resnets.1.conv1.bias` | `[128]` | `torch.float32` | 128 |
| `decoder.up_blocks.3.resnets.1.norm2.weight` | `[128]` | `torch.float32` | 128 |
| `decoder.up_blocks.3.resnets.1.norm2.bias` | `[128]` | `torch.float32` | 128 |
| `decoder.up_blocks.3.resnets.1.conv2.weight` | `[128, 128, 3, 3]` | `torch.float32` | 147456 |
| `decoder.up_blocks.3.resnets.1.conv2.bias` | `[128]` | `torch.float32` | 128 |
| `decoder.up_blocks.3.resnets.2.norm1.weight` | `[128]` | `torch.float32` | 128 |
| `decoder.up_blocks.3.resnets.2.norm1.bias` | `[128]` | `torch.float32` | 128 |
| `decoder.up_blocks.3.resnets.2.conv1.weight` | `[128, 128, 3, 3]` | `torch.float32` | 147456 |
| `decoder.up_blocks.3.resnets.2.conv1.bias` | `[128]` | `torch.float32` | 128 |
| `decoder.up_blocks.3.resnets.2.norm2.weight` | `[128]` | `torch.float32` | 128 |
| `decoder.up_blocks.3.resnets.2.norm2.bias` | `[128]` | `torch.float32` | 128 |
| `decoder.up_blocks.3.resnets.2.conv2.weight` | `[128, 128, 3, 3]` | `torch.float32` | 147456 |
| `decoder.up_blocks.3.resnets.2.conv2.bias` | `[128]` | `torch.float32` | 128 |
| `decoder.mid_block.attentions.0.group_norm.weight` | `[512]` | `torch.float32` | 512 |
| `decoder.mid_block.attentions.0.group_norm.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.mid_block.attentions.0.to_q.weight` | `[512, 512]` | `torch.float32` | 262144 |
| `decoder.mid_block.attentions.0.to_q.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.mid_block.attentions.0.to_k.weight` | `[512, 512]` | `torch.float32` | 262144 |
| `decoder.mid_block.attentions.0.to_k.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.mid_block.attentions.0.to_v.weight` | `[512, 512]` | `torch.float32` | 262144 |
| `decoder.mid_block.attentions.0.to_v.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.mid_block.attentions.0.to_out.0.weight` | `[512, 512]` | `torch.float32` | 262144 |
| `decoder.mid_block.attentions.0.to_out.0.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.mid_block.resnets.0.norm1.weight` | `[512]` | `torch.float32` | 512 |
| `decoder.mid_block.resnets.0.norm1.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.mid_block.resnets.0.conv1.weight` | `[512, 512, 3, 3]` | `torch.float32` | 2359296 |
| `decoder.mid_block.resnets.0.conv1.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.mid_block.resnets.0.norm2.weight` | `[512]` | `torch.float32` | 512 |
| `decoder.mid_block.resnets.0.norm2.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.mid_block.resnets.0.conv2.weight` | `[512, 512, 3, 3]` | `torch.float32` | 2359296 |
| `decoder.mid_block.resnets.0.conv2.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.mid_block.resnets.1.norm1.weight` | `[512]` | `torch.float32` | 512 |
| `decoder.mid_block.resnets.1.norm1.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.mid_block.resnets.1.conv1.weight` | `[512, 512, 3, 3]` | `torch.float32` | 2359296 |
| `decoder.mid_block.resnets.1.conv1.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.mid_block.resnets.1.norm2.weight` | `[512]` | `torch.float32` | 512 |
| `decoder.mid_block.resnets.1.norm2.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.mid_block.resnets.1.conv2.weight` | `[512, 512, 3, 3]` | `torch.float32` | 2359296 |
| `decoder.mid_block.resnets.1.conv2.bias` | `[512]` | `torch.float32` | 512 |
| `decoder.conv_norm_out.weight` | `[128]` | `torch.float32` | 128 |
| `decoder.conv_norm_out.bias` | `[128]` | `torch.float32` | 128 |
| `decoder.conv_out.weight` | `[3, 128, 3, 3]` | `torch.float32` | 3456 |
| `decoder.conv_out.bias` | `[3]` | `torch.float32` | 3 |
| `quant_conv.weight` | `[8, 8, 1, 1]` | `torch.float32` | 64 |
| `quant_conv.bias` | `[8]` | `torch.float32` | 8 |
| `post_quant_conv.weight` | `[4, 4, 1, 1]` | `torch.float32` | 16 |
| `post_quant_conv.bias` | `[4]` | `torch.float32` | 4 |
