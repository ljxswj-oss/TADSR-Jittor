# 中文阅读说明：jittor_decoder_midblock_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TimeAware VAE Decoder MidBlock Alignment

Status: **PASS**

## Metrics

| Metric | Value |
|---|---:|
| shape_match | `True` |
| shape | `[1, 512, 4, 4]` |
| max_abs_error | `3.814697265625e-06` |
| mean_abs_error | `5.09384832980686e-07` |
| relative_error | `4.3114355972088685e-07` |
| cosine_similarity | `0.9999999999999267` |
| tolerance | `0.0001` |

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine |
|---|---:|---:|---:|---:|
| resnet0 | `2.86102294921875e-06` | `2.1092500901431777e-07` | `2.0579742258424653e-07` | `0.9999999999999829` |
| attention | `2.86102294921875e-06` | `3.9340693547273986e-07` | `3.7365078309720756e-07` | `0.9999999999999458` |
| resnet1 | `3.814697265625e-06` | `5.09384832980686e-07` | `4.3114355972088685e-07` | `0.9999999999999267` |
| output | `3.814697265625e-06` | `5.09384832980686e-07` | `4.3114355972088685e-07` | `0.9999999999999267` |

Note: real encoder-derived decoder.conv_in output; decoder temb is None
