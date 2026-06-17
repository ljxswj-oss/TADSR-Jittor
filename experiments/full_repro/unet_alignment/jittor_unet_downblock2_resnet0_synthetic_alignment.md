# 中文阅读说明：jittor_unet_downblock2_resnet0_synthetic_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet down_blocks.2.resnets.0 Synthetic Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| synthetic_output | `8.118152618408203e-05` | `1.3766117834279613e-05` | `1.3500070940498983e-05` | `0.9999999999168652` | `0.0002` |
| synthetic_shortcut | `1.7657876014709473e-06` | `1.0067037994332396e-07` | `4.339627854245507e-07` | `0.999999999999876` | `0.0001` |

Note: Synthetic isolated hidden/temb path for down_blocks.2.resnets.0.
