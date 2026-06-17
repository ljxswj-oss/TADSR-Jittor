# 中文阅读说明：jittor_decoder_midblock_synthetic_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TimeAware VAE Decoder MidBlock Synthetic Alignment

Status: **PASS**

## Metrics

| Metric | Value |
|---|---:|
| shape_match | `True` |
| shape | `[1, 512, 4, 4]` |
| max_abs_error | `1.5974044799804688e-05` |
| mean_abs_error | `2.900418621720746e-06` |
| relative_error | `2.3293420972058104e-06` |
| cosine_similarity | `0.9999999999973104` |
| tolerance | `0.0001` |

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine |
|---|---:|---:|---:|---:|
| resnet0 | `9.59634780883789e-06` | `1.5699989717177232e-06` | `1.8885144625196398e-06` | `0.9999999999981068` |
| attention | `1.5497207641601562e-05` | `2.4765263333392795e-06` | `2.1720359053997265e-06` | `0.9999999999975752` |
| resnet1 | `1.5974044799804688e-05` | `2.900418621720746e-06` | `2.3293420972058104e-06` | `0.9999999999973104` |
| output | `1.5974044799804688e-05` | `2.900418621720746e-06` | `2.3293420972058104e-06` | `0.9999999999973104` |

Note: isolated deterministic synthetic hidden; decoder temb is None
