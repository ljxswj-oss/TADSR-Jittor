# 中文阅读说明：jittor_downblock0_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TimeAware VAE Full DownBlock0 Alignment

Status: **PASS**

## Final metrics

| Metric | Value |
|---|---:|
| shape_match | `True` |
| shape | `[1, 128, 16, 16]` |
| max_abs_error | `3.337860107421875e-06` |
| mean_abs_error | `4.0526577649302453e-07` |
| relative_error | `1.8031701151913138e-07` |
| cosine_similarity | `0.9999999999999829` |
| tolerance | `0.0001` |

## Layer-by-layer diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine |
|---|---:|---:|---:|---:|
| after_resnet0 | `1.430511474609375e-06` | `1.1833916602199679e-07` | `2.0397433283820883e-07` | `0.9999999999999832` |
| after_resnet1 | `1.430511474609375e-06` | `1.7957881226493555e-07` | `2.7642880894354596e-07` | `0.9999999999999727` |
| after_downsampler | `3.337860107421875e-06` | `4.0526577649302453e-07` | `1.8031701151913138e-07` | `0.9999999999999829` |
| final_downblock0 | `3.337860107421875e-06` | `4.0526577649302453e-07` | `1.8031701151913138e-07` | `0.9999999999999829` |
| direct_pytorch_downblock0_reference | `0.0` | `0.0` | `0.0` | `0.9999999999999998` |
