# 中文阅读说明：jittor_decoder_upblocks0123_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TimeAware VAE Decoder UpBlocks0-3 Alignment

Status: **PASS**

## Metrics

| Metric | Value |
|---|---:|
| shape_match | `True` |
| shape | `[1, 128, 32, 32]` |
| max_abs_error | `0.00048828125` |
| mean_abs_error | `1.7898237473445988e-05` |
| relative_error | `8.239870910820068e-07` |
| cosine_similarity | `0.999999999999662` |
| tolerance | `0.001` |

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine |
|---|---:|---:|---:|---:|
| decoder_upblock0 | `1.1444091796875e-05` | `1.265898209368288e-06` | `4.244973501315773e-07` | `0.9999999999999165` |
| decoder_upblock1 | `5.030632019042969e-05` | `6.402784205050693e-06` | `5.91286974577679e-07` | `0.9999999999998331` |
| decoder_upblock2 | `0.00014591217041015625` | `1.2784620821548742e-05` | `8.997569789783459e-07` | `0.9999999999996184` |
| decoder_upblock3 | `0.00048828125` | `1.7898237473445988e-05` | `8.239870910820068e-07` | `0.999999999999662` |

Note: bridge test for consecutive decoder upblocks through the final channel-changing upblock3; decoder tail remains out of scope
