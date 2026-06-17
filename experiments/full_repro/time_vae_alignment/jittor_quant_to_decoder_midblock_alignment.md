# 中文阅读说明：jittor_quant_to_decoder_midblock_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TimeAware VAE Quant Moments To Decoder MidBlock Alignment

Status: **PASS**

## Metrics

| Metric | Value |
|---|---:|
| shape_match | `True` |
| shape | `[1, 512, 4, 4]` |
| max_abs_error | `3.814697265625e-06` |
| mean_abs_error | `5.609419702068408e-07` |
| relative_error | `4.7478154466565144e-07` |
| cosine_similarity | `0.999999999999912` |
| tolerance | `0.0001` |

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine |
|---|---:|---:|---:|---:|
| posterior_mode | `0.0` | `0.0` | `0.0` | `1.0` |
| post_quant_conv | `4.76837158203125e-07` | `4.516914486885071e-08` | `1.6216094289621725e-08` | `0.9999999999999997` |
| decoder_conv_in | `1.9073486328125e-06` | `9.555520819048979e-08` | `9.694945281911466e-08` | `0.9999999999999947` |
| decoder_midblock | `3.814697265625e-06` | `5.609419702068408e-07` | `4.7478154466565144e-07` | `0.999999999999912` |

Note: deterministic posterior mode/mean; decoder temb is None; no sampling
