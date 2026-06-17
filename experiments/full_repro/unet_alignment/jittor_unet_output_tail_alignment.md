# 中文阅读说明：jittor_unet_output_tail_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet Output Tail Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| norm | `9.5367431640625e-07` | `1.9101597324035808e-08` | `4.682003268718505e-08` | `0.9999999999999987` | `0.0001` |
| act | `1.7881393432617188e-07` | `5.8304399278383635e-09` | `8.787306465458178e-08` | `0.9999999999999944` | `0.0001` |
| conv_out | `1.4901161193847656e-07` | `3.231585310459195e-08` | `3.8489308860744733e-07` | `0.9999999999999357` | `0.0002` |

Note: Isolated output-tail validation from the PyTorch up_blocks.3 output tensor. This executes only conv_norm_out -> conv_act -> conv_out and does not call official UNet.forward or full TADSR inference.
