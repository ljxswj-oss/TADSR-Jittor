# 中文阅读说明：jittor_unet_output_tail_synthetic_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet Output Tail Synthetic Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| synthetic_norm | `1.0967254638671875e-05` | `5.591842791316325e-07` | `2.0041135284393322e-06` | `0.9999999999989374` | `0.0001` |
| synthetic_act | `1.6987323760986328e-06` | `2.2332164061992498e-07` | `2.709401090909763e-06` | `0.9999999999947106` | `0.0001` |
| synthetic_conv_out | `3.203749656677246e-06` | `9.551721404932323e-07` | `1.9612306029867895e-05` | `0.9999999997624285` | `0.0002` |

Note: Synthetic hidden tensor -> output tail alignment; no full UNet forward or full inference.
