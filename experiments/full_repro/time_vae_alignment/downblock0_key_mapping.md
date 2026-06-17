# 中文阅读说明：downblock0_key_mapping.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# DownBlock0 PyTorch-to-NPZ Key Mapping

| PyTorch key | Jittor NPZ key | Shape |
|---|---|---:|
| `encoder.down_blocks.0.resnets.0.norm1.weight` | `encoder__down_blocks__0__resnets__0__norm1__weight` | `[128]` |
| `encoder.down_blocks.0.resnets.0.norm1.bias` | `encoder__down_blocks__0__resnets__0__norm1__bias` | `[128]` |
| `encoder.down_blocks.0.resnets.0.conv1.weight` | `encoder__down_blocks__0__resnets__0__conv1__weight` | `[128, 128, 3, 3]` |
| `encoder.down_blocks.0.resnets.0.conv1.bias` | `encoder__down_blocks__0__resnets__0__conv1__bias` | `[128]` |
| `encoder.down_blocks.0.resnets.0.time_emb_proj.weight` | `encoder__down_blocks__0__resnets__0__time_emb_proj__weight` | `[256, 256]` |
| `encoder.down_blocks.0.resnets.0.time_emb_proj.bias` | `encoder__down_blocks__0__resnets__0__time_emb_proj__bias` | `[256]` |
| `encoder.down_blocks.0.resnets.0.norm2.weight` | `encoder__down_blocks__0__resnets__0__norm2__weight` | `[128]` |
| `encoder.down_blocks.0.resnets.0.norm2.bias` | `encoder__down_blocks__0__resnets__0__norm2__bias` | `[128]` |
| `encoder.down_blocks.0.resnets.0.conv2.weight` | `encoder__down_blocks__0__resnets__0__conv2__weight` | `[128, 128, 3, 3]` |
| `encoder.down_blocks.0.resnets.0.conv2.bias` | `encoder__down_blocks__0__resnets__0__conv2__bias` | `[128]` |
| `encoder.down_blocks.0.resnets.1.norm1.weight` | `encoder__down_blocks__0__resnets__1__norm1__weight` | `[128]` |
| `encoder.down_blocks.0.resnets.1.norm1.bias` | `encoder__down_blocks__0__resnets__1__norm1__bias` | `[128]` |
| `encoder.down_blocks.0.resnets.1.conv1.weight` | `encoder__down_blocks__0__resnets__1__conv1__weight` | `[128, 128, 3, 3]` |
| `encoder.down_blocks.0.resnets.1.conv1.bias` | `encoder__down_blocks__0__resnets__1__conv1__bias` | `[128]` |
| `encoder.down_blocks.0.resnets.1.time_emb_proj.weight` | `encoder__down_blocks__0__resnets__1__time_emb_proj__weight` | `[256, 256]` |
| `encoder.down_blocks.0.resnets.1.time_emb_proj.bias` | `encoder__down_blocks__0__resnets__1__time_emb_proj__bias` | `[256]` |
| `encoder.down_blocks.0.resnets.1.norm2.weight` | `encoder__down_blocks__0__resnets__1__norm2__weight` | `[128]` |
| `encoder.down_blocks.0.resnets.1.norm2.bias` | `encoder__down_blocks__0__resnets__1__norm2__bias` | `[128]` |
| `encoder.down_blocks.0.resnets.1.conv2.weight` | `encoder__down_blocks__0__resnets__1__conv2__weight` | `[128, 128, 3, 3]` |
| `encoder.down_blocks.0.resnets.1.conv2.bias` | `encoder__down_blocks__0__resnets__1__conv2__bias` | `[128]` |
| `encoder.down_blocks.0.downsamplers.0.conv.weight` | `encoder__down_blocks__0__downsamplers__0__conv__weight` | `[128, 128, 3, 3]` |
| `encoder.down_blocks.0.downsamplers.0.conv.bias` | `encoder__down_blocks__0__downsamplers__0__conv__bias` | `[128]` |
