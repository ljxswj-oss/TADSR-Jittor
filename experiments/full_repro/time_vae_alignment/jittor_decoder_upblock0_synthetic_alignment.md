# 中文阅读说明：jittor_decoder_upblock0_synthetic_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TimeAware VAE Decoder UpBlock0 Synthetic Alignment

Status: **PASS**

## Metrics

| Metric | Value |
|---|---:|
| shape_match | `True` |
| shape | `[1, 512, 8, 8]` |
| max_abs_error | `7.891654968261719e-05` |
| mean_abs_error | `5.974327631719234e-06` |
| relative_error | `2.9031277305676893e-06` |
| cosine_similarity | `0.9999999999955662` |
| tolerance | `0.0001` |

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine |
|---|---:|---:|---:|---:|
| resnet0 | `1.6689300537109375e-05` | `2.1308310351741966e-06` | `2.1773247131569553e-06` | `0.9999999999975442` |
| resnet1 | `1.811981201171875e-05` | `2.608704107842641e-06` | `2.5272694269215835e-06` | `0.9999999999967893` |
| resnet2 | `1.9550323486328125e-05` | `3.0789342417847365e-06` | `2.8911444998563612e-06` | `0.9999999999957743` |
| upsampler0 | `7.891654968261719e-05` | `5.974327631719234e-06` | `2.9031277305676893e-06` | `0.9999999999955662` |
| output | `7.891654968261719e-05` | `5.974327631719234e-06` | `2.9031277305676893e-06` | `0.9999999999955662` |

Note: isolated deterministic synthetic hidden; decoder temb is None
