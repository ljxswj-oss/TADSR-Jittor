# 中文阅读说明：downblock3_audit.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# Official TimeAware VAE DownBlock3 Audit

Status: **PASS**

Block class: `diffusers.models.unet_2d_blocks.DownEncoderBlock2D`
ResNet count: `2`
Has downsampler: `False`
Channel change detected: `False`

## ResNet modules

| Module | In | Out | Time norm | Time proj shape | Conv shortcut | Shortcut shape | NIN shortcut |
|---|---:|---:|---|---:|---|---:|---|
| `encoder.down_blocks.3.resnets.0` | 512 | 512 | `scale_shift` | `[1024, 256]` | False | `None` | False |
| `encoder.down_blocks.3.resnets.1` | 512 | 512 | `scale_shift` | `[1024, 256]` | False | `None` | False |

## State Keys

| Key | Shape | DType |
|---|---:|---|
| `encoder.down_blocks.3.resnets.0.norm1.weight` | `[512]` | `torch.float32` |
| `encoder.down_blocks.3.resnets.0.norm1.bias` | `[512]` | `torch.float32` |
| `encoder.down_blocks.3.resnets.0.conv1.weight` | `[512, 512, 3, 3]` | `torch.float32` |
| `encoder.down_blocks.3.resnets.0.conv1.bias` | `[512]` | `torch.float32` |
| `encoder.down_blocks.3.resnets.0.time_emb_proj.weight` | `[1024, 256]` | `torch.float32` |
| `encoder.down_blocks.3.resnets.0.time_emb_proj.bias` | `[1024]` | `torch.float32` |
| `encoder.down_blocks.3.resnets.0.norm2.weight` | `[512]` | `torch.float32` |
| `encoder.down_blocks.3.resnets.0.norm2.bias` | `[512]` | `torch.float32` |
| `encoder.down_blocks.3.resnets.0.conv2.weight` | `[512, 512, 3, 3]` | `torch.float32` |
| `encoder.down_blocks.3.resnets.0.conv2.bias` | `[512]` | `torch.float32` |
| `encoder.down_blocks.3.resnets.1.norm1.weight` | `[512]` | `torch.float32` |
| `encoder.down_blocks.3.resnets.1.norm1.bias` | `[512]` | `torch.float32` |
| `encoder.down_blocks.3.resnets.1.conv1.weight` | `[512, 512, 3, 3]` | `torch.float32` |
| `encoder.down_blocks.3.resnets.1.conv1.bias` | `[512]` | `torch.float32` |
| `encoder.down_blocks.3.resnets.1.time_emb_proj.weight` | `[1024, 256]` | `torch.float32` |
| `encoder.down_blocks.3.resnets.1.time_emb_proj.bias` | `[1024]` | `torch.float32` |
| `encoder.down_blocks.3.resnets.1.norm2.weight` | `[512]` | `torch.float32` |
| `encoder.down_blocks.3.resnets.1.norm2.bias` | `[512]` | `torch.float32` |
| `encoder.down_blocks.3.resnets.1.conv2.weight` | `[512, 512, 3, 3]` | `torch.float32` |
| `encoder.down_blocks.3.resnets.1.conv2.bias` | `[512]` | `torch.float32` |
