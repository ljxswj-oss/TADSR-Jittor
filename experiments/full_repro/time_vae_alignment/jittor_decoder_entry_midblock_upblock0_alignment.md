# 中文阅读说明：jittor_decoder_entry_midblock_upblock0_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TimeAware VAE Decoder Entry MidBlock UpBlock0 Alignment

Status: **PASS**

## Metrics

| Metric | Value |
|---|---:|
| shape_match | `True` |
| shape | `[1, 512, 8, 8]` |
| max_abs_error | `1.33514404296875e-05` |
| mean_abs_error | `1.7265455696247045e-06` |
| relative_error | `5.789675771425907e-07` |
| cosine_similarity | `0.9999999999998436` |
| tolerance | `0.0001` |

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine |
|---|---:|---:|---:|---:|
| post_quant_conv | `5.960464477539063e-08` | `4.190951585769653e-09` | `1.4472931614908453e-08` | `0.9999999999999994` |
| decoder_conv_in | `2.384185791015625e-07` | `1.1058745030823047e-08` | `5.798317694904203e-08` | `0.9999999999999974` |
| decoder_midblock | `2.384185791015625e-06` | `3.6261644709156826e-07` | `3.73693783753209e-07` | `0.9999999999999295` |
| decoder_upblock0 | `1.33514404296875e-05` | `1.7265455696247045e-06` | `5.789675771425907e-07` | `0.9999999999998436` |

Note: synthetic latent; deterministic posterior not involved; decoder temb is None
