# 中文阅读说明：jittor_unet_downblock0_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet Full Local down_blocks.0 Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| conv_in | `8.344650268554688e-07` | `1.2484131417522804e-08` | `8.945803719865577e-08` | `0.9999999999999959` | `0.0001` |
| time_embedding | `4.76837158203125e-07` | `4.2749661588459276e-09` | `2.484385767769718e-07` | `0.9999999999999917` | `0.0001` |
| resnet0_output | `1.1920928955078125e-06` | `1.068939880610742e-07` | `2.321830538716543e-07` | `0.9999999999999781` | `0.0001` |
| attention0_output | `3.2186508178710938e-06` | `4.2014392249711817e-07` | `8.921249154252471e-07` | `0.9999999999996211` | `0.002` |
| resnet1_output | `4.552304744720459e-06` | `7.01634245459104e-07` | `1.179647133478816e-06` | `0.9999999999993504` | `0.002` |
| attention1_output | `6.67572021484375e-06` | `8.942489181151814e-07` | `1.4646243150770818e-06` | `0.9999999999990075` | `0.002` |
| downsampler_output | `1.2159347534179688e-05` | `1.706075145335717e-06` | `1.1445761910420367e-06` | `0.9999999999994224` | `0.002` |
| block_forward_output | `1.2159347534179688e-05` | `1.706075145335717e-06` | `1.1445761910420367e-06` | `0.9999999999994224` | `0.002` |

Note: Completes local down_blocks.0 only. Full UNet forward, later blocks and full TADSR inference remain NotImplemented.
