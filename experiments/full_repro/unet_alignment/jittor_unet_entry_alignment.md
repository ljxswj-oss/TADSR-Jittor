# 中文阅读说明：jittor_unet_entry_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet Entry Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| center_input | `0.0` | `0.0` | `0.0` | `1.0000000000000002` | `0.0` |
| conv_in | `8.344650268554688e-07` | `1.2484131417522804e-08` | `8.945803719865577e-08` | `0.9999999999999959` | `0.0001` |
| time_proj | `1.1920928955078125e-07` | `4.547564458334819e-09` | `8.438172602820315e-09` | `0.9999999999999998` | `1e-05` |
| time_embedding_linear1 | `9.5367431640625e-07` | `2.0802190192625858e-08` | `9.716984920864244e-08` | `0.9999999999999968` | `0.0001` |
| time_embedding_act | `4.76837158203125e-07` | `5.502738709139976e-09` | `1.2015353084195342e-07` | `0.999999999999999` | `0.0001` |
| time_embedding | `4.76837158203125e-07` | `4.2749661588459276e-09` | `2.484385767769718e-07` | `0.9999999999999917` | `0.001` |

Note: Entry aggregate stops before down_blocks.0. This is not full UNet forward and not full TADSR inference.
