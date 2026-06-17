# 中文阅读说明：jittor_unet_entry_downblock0_downblock1_downblock2_downblock3_resnet0_resnet1_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet Entry + DownBlock0 + DownBlock1 + DownBlock2 + DownBlock3 ResNet0 + ResNet1 Bridge Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| conv_in | `8.344650268554688e-07` | `1.2484131417522804e-08` | `8.945803719865577e-08` | `0.9999999999999959` | `0.0001` |
| time_embedding | `4.76837158203125e-07` | `4.2749661588459276e-09` | `2.484385767769718e-07` | `0.9999999999999917` | `0.0001` |
| downblock0_output | `1.2159347534179688e-05` | `1.706075145335717e-06` | `1.1445761910420367e-06` | `0.9999999999994224` | `0.002` |
| downblock1_output | `2.2277235984802246e-05` | `2.579873682861944e-06` | `1.0776458816026287e-06` | `0.9999999999994817` | `0.002` |
| downblock2_output | `3.0517578125e-05` | `3.670575767955597e-06` | `1.3387986697652404e-06` | `0.9999999999991991` | `0.002` |
| downblock3_resnet0_output | `2.765655517578125e-05` | `3.6863247260043862e-06` | `1.3878085568194012e-06` | `0.9999999999991651` | `0.002` |
| downblock3_resnet1_output | `2.47955322265625e-05` | `3.7183403492235813e-06` | `1.4007089970714264e-06` | `0.9999999999991385` | `0.002` |

Note: Local bridge: entry -> full down_blocks.0 -> full down_blocks.1 -> full down_blocks.2 -> down_blocks.3.resnets.0 -> down_blocks.3.resnets.1. Full UNet is not executed.
