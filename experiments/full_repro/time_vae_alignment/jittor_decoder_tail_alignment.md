# 中文阅读说明：jittor_decoder_tail_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TimeAware VAE Decoder Tail Alignment

Status: **PASS**

## Metrics

| Metric | Value |
|---|---:|
| shape_match | `True` |
| shape | `[1, 3, 32, 32]` |
| max_abs_error | `1.7285346984863281e-06` |
| mean_abs_error | `6.843505010086423e-07` |
| relative_error | `1.8593110967628196e-06` |
| cosine_similarity | `0.9999999999987242` |
| tolerance | `0.0001` |

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| norm_out | `9.298324584960938e-06` | `1.8709834789104596e-06` | `2.755312612653112e-06` | `0.9999999999937493` | `0.0001` |
| act | `1.0251998901367188e-05` | `9.220940512463804e-07` | `2.8234234962256467e-06` | `0.9999999999915102` | `0.0001` |
| conv_out | `1.7285346984863281e-06` | `6.843505010086423e-07` | `1.8593110967628196e-06` | `0.9999999999987242` | `0.0001` |
| tail | `1.7285346984863281e-06` | `6.843505010086423e-07` | `1.8593110967628196e-06` | `0.9999999999987242` | `0.0001` |

Note: isolated synthetic hidden -> full decoder tail alignment; no scaling_factor, tanh, clamp, or pipeline postprocess is applied
