# 中文阅读说明：jittor_decoder_midblock_upblock0_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TimeAware VAE Decoder MidBlock To UpBlock0 Alignment

Status: **PASS**

## Metrics

| Metric | Value |
|---|---:|
| shape_match | `True` |
| shape | `[1, 512, 8, 8]` |
| max_abs_error | `1.239776611328125e-05` |
| mean_abs_error | `1.6619595442080026e-06` |
| relative_error | `5.573097562830429e-07` |
| cosine_similarity | `0.999999999999856` |
| tolerance | `0.0001` |

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine |
|---|---:|---:|---:|---:|
| decoder_midblock | `2.3096799850463867e-06` | `3.492893938528141e-07` | `3.5995961093502767e-07` | `0.9999999999999342` |
| decoder_upblock0 | `1.239776611328125e-05` | `1.6619595442080026e-06` | `5.573097562830429e-07` | `0.999999999999856` |

Note: synthetic latent decoder.conv_in output; decoder temb is None
