# 中文阅读说明：jittor_decoder_upblock2_synthetic_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TimeAware VAE Decoder UpBlock2 Synthetic Alignment

Status: **PASS**

## Metrics

| Metric | Value |
|---|---:|
| shape_match | `True` |
| shape | `[1, 256, 32, 32]` |
| max_abs_error | `0.00037860870361328125` |
| mean_abs_error | `4.2609574690821717e-05` |
| relative_error | `4.592745931157008e-06` |
| cosine_similarity | `0.9999999999896408` |
| tolerance | `0.001` |

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine |
|---|---:|---:|---:|---:|
| resnet0 | `4.476308822631836e-05` | `7.814919939619358e-06` | `2.519599208541337e-06` | `0.9999999999970048` |
| resnet1 | `5.841255187988281e-05` | `9.711023324143753e-06` | `2.713779560738637e-06` | `0.9999999999964135` |
| resnet2 | `0.00012063980102539062` | `1.8586631597372616e-05` | `4.5580631473300894e-06` | `0.9999999999898312` |
| pre_upsampler | `0.00012063980102539062` | `1.8586631597372616e-05` | `4.5580631473300894e-06` | `0.9999999999898312` |
| upsampler0 | `0.00037860870361328125` | `4.2609574690821717e-05` | `4.592745931157008e-06` | `0.9999999999896408` |
| output | `0.00037860870361328125` | `4.2609574690821717e-05` | `4.592745931157008e-06` | `0.9999999999896408` |

Note: isolated deterministic synthetic upblock2 pressure test; uses 1e-3 tolerance for large linspace activations and channel-changing shortcut
