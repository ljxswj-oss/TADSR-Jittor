# 中文阅读说明：time_vae_port_plan.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TimeAware VAE Port Plan

The plan is based on the real official PyTorch `TimeAwareAutoencoderKL` module tree and converted weight shapes.

| Priority | Target | Reason |
|---:|---|---|
| 1 | `encoder.conv_in` | leaf Conv2d, no timestep dependency, smallest real forward alignment target |
| 2 | `encoder.time_proj + encoder.time_embedding` | required before time-aware ResNet blocks |
| 3 | `encoder.down_blocks.0.resnets.0` | first real TimeAware ResnetBlock2D with scale-shift timestep conditioning |
| 4 | `encoder.down_blocks.0` | first DownEncoderBlock2D including two resnets and downsample |
| 5 | `encoder.mid_block` | contains attention and time-aware ResNet blocks |
| 6 | `quant_conv / post_quant_conv` | latent bottleneck conversion |
| 7 | `decoder blocks` | decoder is standard VAE decoder after latent path is aligned |

Important: full TimeAware VAE forward remains **not complete** until encoder/decoder block-level alignment is finished.
