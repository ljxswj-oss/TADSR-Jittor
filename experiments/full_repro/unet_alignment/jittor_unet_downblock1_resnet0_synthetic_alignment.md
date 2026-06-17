# 中文阅读说明：jittor_unet_downblock1_resnet0_synthetic_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet down_blocks.1.resnets.0 Synthetic Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| synthetic_output | `0.00011277198791503906` | `9.41769759705835e-06` | `1.3991437328460526e-05` | `0.9999999999017049` | `0.0002` |
| synthetic_shortcut | `1.3560056686401367e-06` | `8.377605595910608e-08` | `3.5560451907018035e-07` | `0.9999999999999156` | `0.0001` |

Note: Synthetic isolated hidden/temb path for channel-changing down_blocks.1.resnets.0.
