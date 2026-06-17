# 中文阅读说明：jittor_unet_full_forward_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet Full Forward Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| jittor_vs_official | `2.2277235984802246e-06` | `5.021606739319395e-07` | `5.980908879035516e-06` | `0.9999999999827354` | `0.002` |
| jittor_vs_manual | `2.2277235984802246e-06` | `5.021606739319395e-07` | `5.980908879035516e-06` | `0.9999999999827354` | `0.002` |
| tensor_return_vs_intermediate | `0.0` | `0.0` | `0.0` | `1.0000000000000002` | `0.0` |
| dict_return_vs_intermediate | `0.0` | `0.0` | `0.0` | `1.0000000000000002` | `0.0` |

Note: Alignment-only full forward wrapper matches the official PyTorch UNet.forward oracle. This is UNet numerical alignment only; it does not run scheduler, VAE, full TADSR inference, image generation, or generic runtime LoRA.
