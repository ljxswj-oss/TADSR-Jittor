# 中文阅读说明：time_vae_downblock1_hook_report.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TimeAware VAE DownBlock1 Oracle Export

Status: **PASS**

Input shape: `[1, 3, 32, 32]`
Timestep: `50`
Channel change happens: `True`

| Target | Class | Input shape | Output shape | Status | Channel change |
|---|---|---:|---:|---|---|
| `encoder.down_blocks.1.downsamplers.0` | `Downsample2D` | `[1, 256, 16, 16]` | `[1, 256, 8, 8]` | PASS | `` |
| `encoder.time_proj` | `Timesteps` | `[1]` | `[1, 128]` | PASS | `` |
| `encoder.time_embedding` | `TimestepEmbedding` | `[1, 128]` | `[1, 256]` | PASS | `` |
| `encoder.conv_in` | `Conv2d` | `[1, 3, 32, 32]` | `[1, 128, 32, 32]` | PASS | `` |
| `encoder.down_blocks.0` | `DownEncoderBlock2D` | `[1, 128, 32, 32], temb=[1, 256]` | `[1, 128, 16, 16]` | PASS | `` |
| `encoder.down_blocks.1.resnets.0` | `ResnetBlock2D` | `[1, 128, 16, 16], temb=[1, 256]` | `[1, 256, 16, 16]` | PASS | `True` |
| `encoder.down_blocks.1.resnets.1` | `ResnetBlock2D` | `[1, 256, 16, 16], temb=[1, 256]` | `[1, 256, 16, 16]` | PASS | `False` |
| `encoder.down_blocks.1` | `DownEncoderBlock2D` | `[1, 128, 16, 16], temb=[1, 256]` | `[1, 256, 8, 8]` | PASS | `` |
