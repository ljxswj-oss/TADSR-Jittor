# 中文阅读说明：jittor_decoder_upblock2_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TimeAware VAE Decoder UpBlock2 Alignment

Status: **PASS**

## Metrics

| Metric | Value |
|---|---:|
| shape_match | `True` |
| shape | `[1, 256, 32, 32]` |
| max_abs_error | `0.0001678466796875` |
| mean_abs_error | `9.56894639259076e-06` |
| relative_error | `4.35112879911289e-07` |
| cosine_similarity | `0.9999999999999332` |
| tolerance | `0.001` |

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine |
|---|---:|---:|---:|---:|
| resnet0 | `0.0001068115234375` | `2.3815127860871144e-06` | `3.2559158835211904e-07` | `0.9999999999999588` |
| resnet1 | `0.0001220703125` | `2.8715757594000024e-06` | `3.8993729912072327e-07` | `0.999999999999952` |
| resnet2 | `0.0001220703125` | `3.3766652336453262e-06` | `5.349511757679872e-07` | `0.9999999999999412` |
| pre_upsampler | `0.0001220703125` | `3.3766652336453262e-06` | `5.349511757679872e-07` | `0.9999999999999412` |
| upsampler0 | `0.0001678466796875` | `9.56894639259076e-06` | `4.35112879911289e-07` | `0.9999999999999332` |
| output | `0.0001678466796875` | `9.56894639259076e-06` | `4.35112879911289e-07` | `0.9999999999999332` |

Note: decoder temb is None; resnet0 changes 512 channels to 256 through conv_shortcut; upsampler is nearest 2x followed by 3x3 conv
