# 中文阅读说明：jittor_decoder_upblock1_synthetic_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TimeAware VAE Decoder UpBlock1 Synthetic Alignment

Status: **PASS**

## Metrics

| Metric | Value |
|---|---:|
| shape_match | `True` |
| shape | `[1, 512, 16, 16]` |
| max_abs_error | `0.00063323974609375` |
| mean_abs_error | `9.810523469511168e-05` |
| relative_error | `1.1767920165247865e-05` |
| cosine_similarity | `0.9999999999345668` |
| tolerance | `0.001` |

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine |
|---|---:|---:|---:|---:|
| resnet0 | `0.00014400482177734375` | `1.7385355960186644e-05` | `8.789544480536033e-06` | `0.9999999999625597` |
| resnet1 | `0.00018787384033203125` | `3.0464780309102935e-05` | `1.1122483274820569e-05` | `0.9999999999385983` |
| resnet2 | `0.000240325927734375` | `3.586780542264023e-05` | `1.119896526485909e-05` | `0.999999999938995` |
| pre_upsampler | `0.000240325927734375` | `3.586780542264023e-05` | `1.119896526485909e-05` | `0.999999999938995` |
| upsampler0 | `0.00063323974609375` | `9.810523469511168e-05` | `1.1767920165247865e-05` | `0.9999999999345668` |
| output | `0.00063323974609375` | `9.810523469511168e-05` | `1.1767920165247865e-05` | `0.9999999999345668` |

Note: isolated deterministic synthetic upblock1 pressure test; uses 1e-3 tolerance for large linspace activations
