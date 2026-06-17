# 中文阅读说明：jittor_downblock2_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TimeAware VAE Full DownBlock2 Alignment

Status: **PASS**

## Metrics

| Metric | Value |
|---|---:|
| shape_match | `True` |
| shape | `[1, 512, 4, 4]` |
| max_abs_error | `4.57763671875e-05` |
| mean_abs_error | `3.5400487377046375e-06` |
| relative_error | `1.5738570423704925e-07` |
| cosine_similarity | `0.9999999999999885` |
| tolerance | `0.0001` |

## Layer-by-layer diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine |
|---|---:|---:|---:|---:|
| after_resnet0 | `1.52587890625e-05` | `7.525580940637155e-07` | `2.656870115600867e-07` | `0.9999999999999628` |
| after_resnet1 | `1.52587890625e-05` | `8.383405827316892e-07` | `3.2518129018333035e-07` | `0.9999999999999544` |
| after_downsampler | `4.57763671875e-05` | `3.5400487377046375e-06` | `1.5738570423704925e-07` | `0.9999999999999885` |
| final_downblock2 | `4.57763671875e-05` | `3.5400487377046375e-06` | `1.5738570423704925e-07` | `0.9999999999999885` |
| direct_pytorch_downblock2_reference | `0.0` | `0.0` | `0.0` | `1.0` |
