# 中文阅读说明：jittor_encoder_to_decoder_upblock1_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TimeAware VAE Encoder To Decoder UpBlock1 Alignment

Status: **PASS**

## Metrics

| Metric | Value |
|---|---:|
| shape_match | `True` |
| shape | `[1, 512, 16, 16]` |
| max_abs_error | `8.0108642578125e-05` |
| mean_abs_error | `8.99469925741414e-06` |
| relative_error | `6.589430775618209e-07` |
| cosine_similarity | `0.9999999999998507` |
| tolerance | `0.0001` |

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine |
|---|---:|---:|---:|---:|
| moments | `9.5367431640625e-06` | `1.689302735030651e-06` | `1.6456072783499378e-07` | `0.9999999999999774` |
| posterior_mode | `2.86102294921875e-06` | `6.665941327810287e-07` | `1.2711904631882547e-07` | `0.9999999999999906` |
| decoder_conv_in | `1.9073486328125e-06` | `1.9293725017632823e-07` | `1.9575239473845134e-07` | `0.9999999999999838` |
| decoder_midblock | `4.500150680541992e-06` | `6.389774540593862e-07` | `5.408308145189686e-07` | `0.9999999999998869` |
| decoder_upblock0 | `2.288818359375e-05` | `2.426172670766391e-06` | `5.853031475402665e-07` | `0.9999999999998798` |
| decoder_upblock1 | `8.0108642578125e-05` | `8.99469925741414e-06` | `6.589430775618209e-07` | `0.9999999999998507` |

Note: deterministic full bridge through encoder and partial decoder, ending at decoder.up_blocks.1
