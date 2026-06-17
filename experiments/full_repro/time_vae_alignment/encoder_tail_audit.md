# 中文阅读说明：encoder_tail_audit.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# Official TimeAware VAE Encoder Tail + QuantConv Audit

Status: **PASS**

## Encoder tail

| Module | Class | Key facts |
|---|---|---|
| `encoder.conv_norm_out` | `torch.nn.modules.normalization.GroupNorm` | groups=32, channels=512, eps=1e-06, affine=True |
| `encoder.conv_act` | `torch.nn.modules.activation.SiLU` | activation=SiLU |
| `encoder.conv_out` | `torch.nn.modules.conv.Conv2d` | weight=[8, 512, 3, 3], stride=[1, 1], padding=[1, 1], out_channels=8 |

## Quant conv

| Module | Class | Weight | Kernel | Stride | Padding | Channels |
|---|---|---:|---:|---:|---:|---|
| `quant_conv` | `torch.nn.modules.conv.Conv2d` | `[8, 8, 1, 1]` | `[1, 1]` | `[1, 1]` | `[0, 0]` | 8 -> 8 |
| `post_quant_conv` | `torch.nn.modules.conv.Conv2d` | `[4, 4, 1, 1]` | `[1, 1]` | `[1, 1]` | `[0, 0]` | 4 -> 4 |

## State Keys

| Key | Shape | DType |
|---|---:|---|
| `encoder.conv_norm_out.weight` | `[512]` | `torch.float32` |
| `encoder.conv_norm_out.bias` | `[512]` | `torch.float32` |
| `encoder.conv_out.weight` | `[8, 512, 3, 3]` | `torch.float32` |
| `encoder.conv_out.bias` | `[8]` | `torch.float32` |
| `quant_conv.weight` | `[8, 8, 1, 1]` | `torch.float32` |
| `quant_conv.bias` | `[8]` | `torch.float32` |
| `post_quant_conv.weight` | `[4, 4, 1, 1]` | `torch.float32` |
| `post_quant_conv.bias` | `[4]` | `torch.float32` |
