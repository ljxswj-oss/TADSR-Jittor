# 中文阅读说明：jittor_decoder_upblock0_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TimeAware VAE Decoder UpBlock0 Alignment

Status: **PASS**

## Metrics

| Metric | Value |
|---|---:|
| shape_match | `True` |
| shape | `[1, 512, 8, 8]` |
| max_abs_error | `1.52587890625e-05` |
| mean_abs_error | `1.2780401164036448e-06` |
| relative_error | `3.0832137869951657e-07` |
| cosine_similarity | `0.9999999999999653` |
| tolerance | `0.0001` |

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine |
|---|---:|---:|---:|---:|
| resnet0 | `1.430511474609375e-06` | `2.1671326067007612e-07` | `1.608601506294337e-07` | `0.9999999999999893` |
| resnet1 | `2.1457672119140625e-06` | `3.585385002224939e-07` | `2.491752888035756e-07` | `0.999999999999974` |
| resnet2 | `3.0994415283203125e-06` | `5.099022928334307e-07` | `4.0906609697166503e-07` | `0.9999999999999413` |
| pre_upsampler | `3.0994415283203125e-06` | `5.099022928334307e-07` | `4.0906609697166503e-07` | `0.9999999999999413` |
| upsampler0 | `1.52587890625e-05` | `1.2780401164036448e-06` | `3.0832137869951657e-07` | `0.9999999999999653` |
| output | `1.52587890625e-05` | `1.2780401164036448e-06` | `3.0832137869951657e-07` | `0.9999999999999653` |

Note: decoder temb is None; upsampler is nearest 2x followed by 3x3 conv
