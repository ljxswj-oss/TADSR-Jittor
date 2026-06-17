# 中文阅读说明：downblock1_audit.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# Official TimeAware VAE DownBlock1 Audit

Status: **PASS**

Block class: `diffusers.models.unet_2d_blocks.DownEncoderBlock2D`
ResNet count: `2`
Has downsampler: `True`
Channel change detected: `True`

## ResNet modules

| Module | In | Out | Time norm | Time proj shape | Conv shortcut | Shortcut shape |
|---|---:|---:|---|---:|---|---:|
| `encoder.down_blocks.1.resnets.0` | 128 | 256 | `scale_shift` | `[512, 256]` | True | `[256, 128, 1, 1]` |
| `encoder.down_blocks.1.resnets.1` | 256 | 256 | `scale_shift` | `[512, 256]` | False | `None` |

## Downsampler

| Field | Value |
|---|---|
| module_name | `encoder.down_blocks.1.downsamplers.0` |
| class_name | `Downsample2D` |
| class_module | `diffusers.models.downsampling` |
| use_conv | `True` |
| padding | `0` |
| channels | `256` |
| out_channels | `256` |
| conv_class | `LoRACompatibleConv` |
| conv_kernel_size | `[3, 3]` |
| conv_stride | `[2, 2]` |
| conv_padding | `[0, 0]` |
| asymmetric_padding_when_padding_zero | `True` |
| asymmetric_padding_note | `PyTorch Downsample2D applies F.pad(hidden_states, (0, 1, 0, 1)) before conv when use_conv=True and padding=0.` |

## State Keys

| Key | Shape | DType |
|---|---:|---|
| `encoder.down_blocks.1.resnets.0.norm1.weight` | `[128]` | `torch.float32` |
| `encoder.down_blocks.1.resnets.0.norm1.bias` | `[128]` | `torch.float32` |
| `encoder.down_blocks.1.resnets.0.conv1.weight` | `[256, 128, 3, 3]` | `torch.float32` |
| `encoder.down_blocks.1.resnets.0.conv1.bias` | `[256]` | `torch.float32` |
| `encoder.down_blocks.1.resnets.0.time_emb_proj.weight` | `[512, 256]` | `torch.float32` |
| `encoder.down_blocks.1.resnets.0.time_emb_proj.bias` | `[512]` | `torch.float32` |
| `encoder.down_blocks.1.resnets.0.norm2.weight` | `[256]` | `torch.float32` |
| `encoder.down_blocks.1.resnets.0.norm2.bias` | `[256]` | `torch.float32` |
| `encoder.down_blocks.1.resnets.0.conv2.weight` | `[256, 256, 3, 3]` | `torch.float32` |
| `encoder.down_blocks.1.resnets.0.conv2.bias` | `[256]` | `torch.float32` |
| `encoder.down_blocks.1.resnets.0.conv_shortcut.weight` | `[256, 128, 1, 1]` | `torch.float32` |
| `encoder.down_blocks.1.resnets.0.conv_shortcut.bias` | `[256]` | `torch.float32` |
| `encoder.down_blocks.1.resnets.1.norm1.weight` | `[256]` | `torch.float32` |
| `encoder.down_blocks.1.resnets.1.norm1.bias` | `[256]` | `torch.float32` |
| `encoder.down_blocks.1.resnets.1.conv1.weight` | `[256, 256, 3, 3]` | `torch.float32` |
| `encoder.down_blocks.1.resnets.1.conv1.bias` | `[256]` | `torch.float32` |
| `encoder.down_blocks.1.resnets.1.time_emb_proj.weight` | `[512, 256]` | `torch.float32` |
| `encoder.down_blocks.1.resnets.1.time_emb_proj.bias` | `[512]` | `torch.float32` |
| `encoder.down_blocks.1.resnets.1.norm2.weight` | `[256]` | `torch.float32` |
| `encoder.down_blocks.1.resnets.1.norm2.bias` | `[256]` | `torch.float32` |
| `encoder.down_blocks.1.resnets.1.conv2.weight` | `[256, 256, 3, 3]` | `torch.float32` |
| `encoder.down_blocks.1.resnets.1.conv2.bias` | `[256]` | `torch.float32` |
| `encoder.down_blocks.1.downsamplers.0.conv.weight` | `[256, 256, 3, 3]` | `torch.float32` |
| `encoder.down_blocks.1.downsamplers.0.conv.bias` | `[256]` | `torch.float32` |
