# 中文阅读说明：jittor_decoder_upblock1_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TimeAware VAE Decoder UpBlock1 Alignment

Status: **PASS**

## Metrics

| Metric | Value |
|---|---:|
| shape_match | `True` |
| shape | `[1, 512, 16, 16]` |
| max_abs_error | `6.103515625e-05` |
| mean_abs_error | `4.61722751765592e-06` |
| relative_error | `3.382536784406058e-07` |
| cosine_similarity | `0.9999999999999603` |
| tolerance | `0.0001` |

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine |
|---|---:|---:|---:|---:|
| resnet0 | `3.814697265625e-06` | `5.505788180926174e-07` | `1.270972112448151e-07` | `0.999999999999994` |
| resnet1 | `7.62939453125e-06` | `9.135429195339384e-07` | `2.0093320124868208e-07` | `0.9999999999999847` |
| resnet2 | `9.5367431640625e-06` | `1.272552026421181e-06` | `3.4447453424044364e-07` | `0.999999999999966` |
| pre_upsampler | `9.5367431640625e-06` | `1.272552026421181e-06` | `3.4447453424044364e-07` | `0.999999999999966` |
| upsampler0 | `6.103515625e-05` | `4.61722751765592e-06` | `3.382536784406058e-07` | `0.9999999999999603` |
| output | `6.103515625e-05` | `4.61722751765592e-06` | `3.382536784406058e-07` | `0.9999999999999603` |

Note: decoder temb is None; upsampler is nearest 2x followed by 3x3 conv
