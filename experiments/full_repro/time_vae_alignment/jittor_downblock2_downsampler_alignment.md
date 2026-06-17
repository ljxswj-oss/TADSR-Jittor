# 中文阅读说明：jittor_downblock2_downsampler_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TimeAware VAE DownBlock2 Downsampler Alignment

Status: **PASS**

## Metrics

| Metric | Value |
|---|---:|
| shape_match | `True` |
| shape | `[1, 512, 4, 4]` |
| max_abs_error | `3.0517578125e-05` |
| mean_abs_error | `2.4967854415081092e-06` |
| relative_error | `1.1100365112356033e-07` |
| cosine_similarity | `0.9999999999999929` |
| tolerance | `0.0001` |

Padding note: Diffusers Downsample2D uses right/bottom asymmetric zero padding before stride-2 conv when padding=0.
