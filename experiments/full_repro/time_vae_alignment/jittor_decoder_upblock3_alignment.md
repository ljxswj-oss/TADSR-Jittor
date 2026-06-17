# 中文阅读说明：jittor_decoder_upblock3_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TimeAware VAE Decoder UpBlock3 Alignment

Status: **PASS**

## Metrics

| Metric | Value |
|---|---:|
| shape_match | `True` |
| shape | `[1, 128, 32, 32]` |
| max_abs_error | `9.1552734375e-05` |
| mean_abs_error | `5.020616782758225e-06` |
| relative_error | `1.81613615821278e-07` |
| cosine_similarity | `0.9999999999999858` |
| tolerance | `0.001` |

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine |
|---|---:|---:|---:|---:|
| resnet0 | `7.62939453125e-05` | `3.7603155789156517e-06` | `1.5298880999992405e-07` | `0.9999999999999888` |
| resnet1 | `7.62939453125e-05` | `4.342165212278815e-06` | `1.6963950016243558e-07` | `0.999999999999987` |
| resnet2 | `9.1552734375e-05` | `5.020616782758225e-06` | `1.81613615821278e-07` | `0.9999999999999858` |
| pre_upsampler | `9.1552734375e-05` | `5.020616782758225e-06` | `1.81613615821278e-07` | `0.9999999999999858` |
| output | `9.1552734375e-05` | `5.020616782758225e-06` | `1.81613615821278e-07` | `0.9999999999999858` |

Note: decoder temb is None; resnet0 changes 256 channels to 128 through conv_shortcut; official upblock3 has no upsampler
