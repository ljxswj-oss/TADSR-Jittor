# 中文阅读说明：jittor_unet_downblock3_resnet1_synthetic_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet down_blocks.3.resnets.1 Synthetic Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| synthetic_output | `3.337860107421875e-05` | `4.352007135821623e-06` | `3.5230673119959438e-06` | `0.9999999999945104` | `0.0001` |
| synthetic_conv2 | `3.337860107421875e-05` | `4.351937150204321e-06` | `3.7815391777205005e-06` | `0.9999999999939876` | `0.0001` |
| synthetic_shortcut | `0.0` | `0.0` | `0.0` | `1.0` | `0.0001` |

Note: Synthetic isolated hidden/temb -> down_blocks.3.resnets.1 only.
