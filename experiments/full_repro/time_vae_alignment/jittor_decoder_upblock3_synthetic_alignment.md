# 中文阅读说明：jittor_decoder_upblock3_synthetic_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TimeAware VAE Decoder UpBlock3 Synthetic Alignment

Status: **PASS**

## Metrics

| Metric | Value |
|---|---:|
| shape_match | `True` |
| shape | `[1, 128, 32, 32]` |
| max_abs_error | `0.0002288818359375` |
| mean_abs_error | `2.627187774351114e-05` |
| relative_error | `1.9539194299983966e-06` |
| cosine_similarity | `0.9999999999989243` |
| tolerance | `0.001` |

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine |
|---|---:|---:|---:|---:|
| resnet0 | `8.58306884765625e-05` | `1.7295496384406306e-05` | `2.2302023065617683e-06` | `0.9999999999985113` |
| resnet1 | `0.000152587890625` | `2.0161251399031244e-05` | `1.9815713071947083e-06` | `0.9999999999987447` |
| resnet2 | `0.0002288818359375` | `2.627187774351114e-05` | `1.9539194299983966e-06` | `0.9999999999989243` |
| pre_upsampler | `0.0002288818359375` | `2.627187774351114e-05` | `1.9539194299983966e-06` | `0.9999999999989243` |
| output | `0.0002288818359375` | `2.627187774351114e-05` | `1.9539194299983966e-06` | `0.9999999999989243` |

Note: isolated deterministic synthetic upblock3 pressure test; uses 1e-3 tolerance for large linspace activations and the 256->128 shortcut; official upblock3 has no upsampler
