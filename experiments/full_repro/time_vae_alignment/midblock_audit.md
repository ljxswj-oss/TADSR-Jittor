# 中文阅读说明：midblock_audit.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# Official TimeAware VAE MidBlock Audit

Status: **PASS**

MidBlock class: `diffusers.models.unet_2d_blocks.UNetMidBlock2D`
ResNet count: `2`
Attention count: `1`

## ResNet modules

| Module | In | Out | Time norm | Time proj shape | Conv shortcut | Shortcut shape | NIN shortcut |
|---|---:|---:|---|---:|---|---:|---|
| `encoder.mid_block.resnets.0` | 512 | 512 | `scale_shift` | `[1024, 256]` | False | `None` | False |
| `encoder.mid_block.resnets.1` | 512 | 512 | `scale_shift` | `[1024, 256]` | False | `None` | False |

## Attention modules

| Module | Heads | Head dim | GroupNorm | Q/K/V shape | Out shape | Residual | Rescale |
|---|---:|---:|---|---|---|---|---:|
| `encoder.mid_block.attentions.0` | 1 | 512 | True | `[512, 512]` / `[512, 512]` / `[512, 512]` | `[512, 512]` | True | `1` |

## State Keys

| Key | Shape | DType |
|---|---:|---|
| `encoder.mid_block.attentions.0.group_norm.weight` | `[512]` | `torch.float32` |
| `encoder.mid_block.attentions.0.group_norm.bias` | `[512]` | `torch.float32` |
| `encoder.mid_block.attentions.0.to_q.weight` | `[512, 512]` | `torch.float32` |
| `encoder.mid_block.attentions.0.to_q.bias` | `[512]` | `torch.float32` |
| `encoder.mid_block.attentions.0.to_k.weight` | `[512, 512]` | `torch.float32` |
| `encoder.mid_block.attentions.0.to_k.bias` | `[512]` | `torch.float32` |
| `encoder.mid_block.attentions.0.to_v.weight` | `[512, 512]` | `torch.float32` |
| `encoder.mid_block.attentions.0.to_v.bias` | `[512]` | `torch.float32` |
| `encoder.mid_block.attentions.0.to_out.0.weight` | `[512, 512]` | `torch.float32` |
| `encoder.mid_block.attentions.0.to_out.0.bias` | `[512]` | `torch.float32` |
| `encoder.mid_block.resnets.0.norm1.weight` | `[512]` | `torch.float32` |
| `encoder.mid_block.resnets.0.norm1.bias` | `[512]` | `torch.float32` |
| `encoder.mid_block.resnets.0.conv1.weight` | `[512, 512, 3, 3]` | `torch.float32` |
| `encoder.mid_block.resnets.0.conv1.bias` | `[512]` | `torch.float32` |
| `encoder.mid_block.resnets.0.time_emb_proj.weight` | `[1024, 256]` | `torch.float32` |
| `encoder.mid_block.resnets.0.time_emb_proj.bias` | `[1024]` | `torch.float32` |
| `encoder.mid_block.resnets.0.norm2.weight` | `[512]` | `torch.float32` |
| `encoder.mid_block.resnets.0.norm2.bias` | `[512]` | `torch.float32` |
| `encoder.mid_block.resnets.0.conv2.weight` | `[512, 512, 3, 3]` | `torch.float32` |
| `encoder.mid_block.resnets.0.conv2.bias` | `[512]` | `torch.float32` |
| `encoder.mid_block.resnets.1.norm1.weight` | `[512]` | `torch.float32` |
| `encoder.mid_block.resnets.1.norm1.bias` | `[512]` | `torch.float32` |
| `encoder.mid_block.resnets.1.conv1.weight` | `[512, 512, 3, 3]` | `torch.float32` |
| `encoder.mid_block.resnets.1.conv1.bias` | `[512]` | `torch.float32` |
| `encoder.mid_block.resnets.1.time_emb_proj.weight` | `[1024, 256]` | `torch.float32` |
| `encoder.mid_block.resnets.1.time_emb_proj.bias` | `[1024]` | `torch.float32` |
| `encoder.mid_block.resnets.1.norm2.weight` | `[512]` | `torch.float32` |
| `encoder.mid_block.resnets.1.norm2.bias` | `[512]` | `torch.float32` |
| `encoder.mid_block.resnets.1.conv2.weight` | `[512, 512, 3, 3]` | `torch.float32` |
| `encoder.mid_block.resnets.1.conv2.bias` | `[512]` | `torch.float32` |
