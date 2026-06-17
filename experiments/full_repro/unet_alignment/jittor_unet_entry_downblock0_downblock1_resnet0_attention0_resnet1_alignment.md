# 中文阅读说明：jittor_unet_entry_downblock0_downblock1_resnet0_attention0_resnet1_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet Entry + DownBlock0 + DownBlock1 ResNet0 + Attention0 + ResNet1 Bridge Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| conv_in | `8.344650268554688e-07` | `1.2484131417522804e-08` | `8.945803719865577e-08` | `0.9999999999999959` | `0.0001` |
| time_embedding | `4.76837158203125e-07` | `4.2749661588459276e-09` | `2.484385767769718e-07` | `0.9999999999999917` | `0.0001` |
| downblock0_output | `1.2159347534179688e-05` | `1.706075145335717e-06` | `1.1445761910420367e-06` | `0.9999999999994224` | `0.002` |
| downblock1_resnet0_output | `1.0609626770019531e-05` | `1.077069811117326e-06` | `1.109991285338639e-06` | `0.9999999999994256` | `0.002` |
| downblock1_attention0_output | `1.329183578491211e-05` | `1.3588772276307282e-06` | `1.2827697305785698e-06` | `0.9999999999991661` | `0.002` |
| downblock1_resnet1_output | `1.3828277587890625e-05` | `1.571142846046314e-06` | `1.5137789527172096e-06` | `0.9999999999988827` | `0.002` |

Note: Local bridge stops after down_blocks.1.resnets.1. Full down_blocks.1 and full UNet are not executed.
