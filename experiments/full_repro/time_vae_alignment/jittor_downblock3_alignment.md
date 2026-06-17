# 中文阅读说明：jittor_downblock3_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TimeAware VAE Full DownBlock3 Alignment

Status: **PASS**

## Metrics

| Metric | Value |
|---|---:|
| shape_match | `True` |
| shape | `[1, 512, 4, 4]` |
| max_abs_error | `1.52587890625e-05` |
| mean_abs_error | `1.186162990052253e-06` |
| relative_error | `5.0843952115127e-08` |
| cosine_similarity | `0.9999999999999988` |
| tolerance | `0.0001` |

## Layer-by-layer diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine |
|---|---:|---:|---:|---:|
| after_resnet0 | `1.52587890625e-05` | `7.079052011249587e-07` | `3.123139230253033e-08` | `0.9999999999999991` |
| after_resnet1 | `1.52587890625e-05` | `1.186162990052253e-06` | `5.0843952115127e-08` | `0.9999999999999988` |
| final_downblock3 | `1.52587890625e-05` | `1.186162990052253e-06` | `5.0843952115127e-08` | `0.9999999999999988` |
| direct_pytorch_downblock3_reference | `0.0` | `0.0` | `0.0` | `1.0000000000000002` |
