# 中文阅读说明：jittor_downblock1_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TimeAware VAE Full DownBlock1 Alignment

Status: **PASS**

## Final metrics

| Metric | Value |
|---|---:|
| shape_match | `True` |
| shape | `[1, 256, 8, 8]` |
| max_abs_error | `9.5367431640625e-06` |
| mean_abs_error | `8.436727796379273e-07` |
| relative_error | `2.0347485433614864e-07` |
| cosine_similarity | `0.999999999999981` |
| tolerance | `0.0001` |

## Layer-by-layer diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine |
|---|---:|---:|---:|---:|
| after_resnet0 | `6.67572021484375e-06` | `2.82310594457158e-07` | `2.1589291821496559e-07` | `0.9999999999999786` |
| after_resnet1 | `6.67572021484375e-06` | `3.4412779115200465e-07` | `3.003308144721771e-07` | `0.9999999999999677` |
| after_downsampler | `9.5367431640625e-06` | `8.436727796379273e-07` | `2.0347485433614864e-07` | `0.999999999999981` |
| final_downblock1 | `9.5367431640625e-06` | `8.436727796379273e-07` | `2.0347485433614864e-07` | `0.999999999999981` |
| direct_pytorch_downblock1_reference | `0.0` | `0.0` | `0.0` | `1.0` |
