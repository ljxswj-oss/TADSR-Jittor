# 中文阅读说明：jittor_unet_time_embedding_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet time_embedding Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| linear1 | `9.5367431640625e-07` | `2.1003188521717675e-08` | `9.810873964028384e-08` | `0.9999999999999969` | `0.0001` |
| act | `4.76837158203125e-07` | `5.591943086358242e-09` | `1.2210132837621203e-07` | `0.9999999999999989` | `0.0001` |
| full | `4.76837158203125e-07` | `4.34172306995606e-09` | `2.5231813777698347e-07` | `0.9999999999999917` | `0.001` |

Note: Tests only the timestep MLP; no down/up/mid/cross-attention blocks are executed.
